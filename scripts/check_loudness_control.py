#!/usr/bin/env python
"""Loudness-controlled separation test: do the embeddings separate same-label
from different-label sections BEYOND what loudness alone explains?

Why: on SongFormBench a loudness-only baseline already reaches per-song AUC
0.735 (HarmonixSet) and 0.875 (Chinese pop), close to the embeddings' 0.845 /
0.919. Choruses are often simply louder. This script removes loudness in four
ways and reports what separation survives, per dataset.

  1. MATCHED AUC. Within a song, compare a same-label pair with a
     different-label pair only when their loudness differences are within
     --match_db of each other (e.g. a verse-verse pair 1.2 dB apart vs a
     verse-bridge pair 1.5 dB apart). Loudness cannot decide those comparisons,
     so loudness scores ~0.5 on them by construction (printed as a check).
  2. RESIDUAL AUC. Within a song, regress embedding distance on loudness
     difference and score the residuals. Conservative: it also removes any
     real structure that happens to go with loudness.
  3. ADDED VALUE. A logistic model fitted across songs (5-fold, split by song)
     predicts same/different from loudness difference, then from loudness
     difference + embedding distance. Reports the per-song AUC of each and the
     share of songs where adding the embeddings helps.
  4. (--normalize, needs GPU) LOUDNESS-NORMALIZED AUDIO. Every section is
     gain-matched to the song's average level (50 ms crossfades), embeddings
     are re-extracted, and the ordinary per-song AUC is recomputed. Loudness
     on that audio should drop to ~0.5 (printed as a check).

Section loudness = median frame level in dB (frames at the manifest rate);
loudness difference = |level_i - level_j|.

    python scripts/check_loudness_control.py \
        --manifest data/msa/songformbench_manifest.json \
        --audio_dir $DATA/songformbench/audio --limit 0 --permutation 1000 \
        --save_json results/sfb_loudness.json

    # strongest version: re-extract embeddings on loudness-normalized audio
    python scripts/check_loudness_control.py ... --normalize \
        --norm_cache $DATA/cache/loudnorm --save_json results/sfb_loudness_norm.json

How to read it: the embeddings see more than loudness if (1) they beat
loudness on the loudness-matched comparisons in most songs (sign test p small),
(2) the residual AUC stays above 0.5, and (4) the AUC on normalized audio stays
well above 0.5 while loudness there drops to ~0.5. The label-shuffle p alone is
NOT enough: matching within a band leaves a little loudness, which the sign
test against loudness on the same comparisons accounts for.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from formadherence import audio_io  # noqa: E402
from formadherence.config import FormAdherenceConfig  # noqa: E402
from formadherence.manifest import by_dataset, load_manifests  # noqa: E402
from formadherence.metric import hard_auc  # noqa: E402
from diagnose_formulation import pair_distances, song_segments  # noqa: E402


# --------------------------------------------------------------------------
# loudness
# --------------------------------------------------------------------------

def frame_db(x, sr, frame_rate):
    """Frame level in dB at frame_rate (window = 2 hops), like the baseline."""
    hop = int(round(sr / frame_rate))
    n = int(np.ceil(len(x) / hop))
    out = np.empty(n)
    for i in range(n):
        seg = x[i * hop: i * hop + 2 * hop]
        out[i] = 20 * np.log10(np.sqrt(np.mean(seg ** 2)) + 1e-8) if len(seg) else -160.0
    return out


def section_levels(db, segs, decimals=None):
    """Median frame level per section (dB). decimals rounds it (used for the
    after-normalization check, where float noise would otherwise order pairs)."""
    v = np.array([np.median(db[s.start:s.end]) if s.end > s.start else np.nan for s in segs])
    return np.round(v, decimals) if decimals is not None else v


def normalize_sections(x, sr, segs, frame_rate, max_gain_db=20.0, fade_s=0.05):
    """Gain-match every section to the song's overall RMS level."""
    target = np.sqrt(np.mean(x ** 2)) + 1e-9
    gains = np.ones(len(x))
    for s in segs:
        a, b = int(round(s.start / frame_rate * sr)), int(round(s.end / frame_rate * sr))
        b = min(b, len(x))
        if b <= a:
            continue
        r = np.sqrt(np.mean(x[a:b] ** 2)) + 1e-9
        g = np.clip(target / r, 10 ** (-max_gain_db / 20), 10 ** (max_gain_db / 20))
        gains[a:b] = g
    k = max(1, int(fade_s * sr))                         # smooth gain steps
    gains = np.convolve(gains, np.ones(k) / k, mode="same")
    return x * gains


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------

def matched_auc(score, dl, same, tol, min_cmp):
    """AUC over (same, diff) comparisons whose loudness differences match."""
    S, T = np.where(same)[0], np.where(~same)[0]
    if len(S) == 0 or len(T) == 0:
        return None, 0.0
    M = np.abs(dl[S][:, None] - dl[T][None, :]) <= tol
    if M.sum() < min_cmp:
        return None, float(M.mean())
    w = (score[T][None, :] > score[S][:, None]) + 0.5 * (score[T][None, :] == score[S][:, None])
    return float(w[M].mean()), float(M.mean())


def sign_test(a, b):
    """One-sided paired sign test: is a > b in more songs than chance (ties
    dropped)? p = P(at least n_pos wins out of n under a fair coin)."""
    from math import comb
    diff = np.asarray(a, float) - np.asarray(b, float)
    n_pos, n = int((diff > 0).sum()), int((diff != 0).sum())
    if n == 0:
        return 1.0, 0, 0
    p = sum(comb(n, i) for i in range(n_pos, n + 1)) / 2 ** n
    return float(p), n_pos, n


def residual_auc(d, dl, same):
    if len(d) < 4 or np.std(dl) < 1e-9:
        return None
    A = np.column_stack([np.ones_like(dl), dl])
    coef, *_ = np.linalg.lstsq(A, d, rcond=None)
    r = d - A @ coef
    return hard_auc(r[same], r[~same])


def zscore(v):
    sd = v.std()
    return (v - v.mean()) / sd if sd > 1e-12 else v * 0.0


def fit_logistic(X, y, iters=50, l2=1e-3):
    X = np.column_stack([np.ones(len(X)), X])
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(X @ w, -30, 30)))
        g = X.T @ (p - y) + l2 * w
        H = X.T @ (X * (p * (1 - p))[:, None]) + l2 * np.eye(len(w))
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return w


def added_value(songs, folds=5, seed=0):
    """songs: list of dicts with d, dl, same. Grouped CV by song."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(songs))
    fold_of = {int(k): f % folds for f, k in enumerate(idx)}
    feats = [(np.column_stack([zscore(s["dl"]), zscore(s["d"])]), (~s["same"]).astype(float))
             for s in songs]
    a_l, a_le, coefs = [], [], []
    for f in range(folds):
        tr = [k for k in range(len(songs)) if fold_of[k] != f]
        te = [k for k in range(len(songs)) if fold_of[k] == f]
        if not tr or not te:
            continue
        X = np.vstack([feats[k][0] for k in tr])
        y = np.concatenate([feats[k][1] for k in tr])
        wl = fit_logistic(X[:, :1], y)
        wle = fit_logistic(X, y)
        coefs.append(wle[1:].tolist())
        for k in te:
            Xk, yk = feats[k]
            same = yk == 0
            sl = wl[0] + Xk[:, :1] @ wl[1:]
            sle = wle[0] + Xk @ wle[1:]
            al, ale = hard_auc(sl[same], sl[~same]), hard_auc(sle[same], sle[~same])
            if al is not None and ale is not None:
                a_l.append(al)
                a_le.append(ale)
    a_l, a_le = np.array(a_l), np.array(a_le)
    c = np.array(coefs).mean(0) if coefs else [np.nan, np.nan]
    return {"songs": int(len(a_l)),
            "loudness_only_median_auc": float(np.median(a_l)),
            "loudness_plus_embedding_median_auc": float(np.median(a_le)),
            "frac_songs_improved": float((a_le > a_l).mean()),
            "mean_coef_loudness": float(c[0]), "mean_coef_embedding": float(c[1])}


def summarize(vals):
    v = np.array([x for x in vals if x is not None], dtype=float)
    if not len(v):
        return None
    return {"songs": int(len(v)), "median": float(np.median(v)),
            "frac_above_half": float((v > 0.5).mean())}


def perm_matched(songs, tol, min_cmp, n_perm, seed=0):
    rng = np.random.default_rng(seed)
    obs = [matched_auc(s["d"], s["dl"], s["same"], tol, min_cmp)[0] for s in songs]
    observed = float(np.median([o for o in obs if o is not None]))
    null = []
    for _ in range(n_perm):
        meds = []
        for s in songs:
            labs = rng.permutation(s["labs"])
            same = labs[s["i"]] == labs[s["j"]]
            a, _ = matched_auc(s["d"], s["dl"], same, tol, min_cmp)
            if a is not None:
                meds.append(a)
        if meds:
            null.append(float(np.median(meds)))
    null = np.array(null)
    return {"observed_median": observed, "null_median": float(np.median(null)),
            "null_95": float(np.quantile(null, 0.95)),
            "p_value": float((1 + (null >= observed).sum()) / (1 + len(null))),
            "n_perm": n_perm}


# --------------------------------------------------------------------------

def song_record(emb, segs, labs, levels, distance, cfg, rng):
    D = pair_distances(emb, segs, distance, cfg, "cosine", rng)
    keys = [k for k in sorted(D) if not (np.isnan(levels[k[0]]) or np.isnan(levels[k[1]]))]
    if not keys:
        return None
    i = np.array([k[0] for k in keys])
    j = np.array([k[1] for k in keys])
    labs = np.array(labs)
    same = labs[i] == labs[j]
    if same.sum() == 0 or (~same).sum() == 0:
        return None
    return {"d": np.array([D[k] for k in keys]), "dl": np.abs(levels[i] - levels[j]),
            "same": same, "i": i, "j": j, "labs": labs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/msa/songformbench_manifest.json")
    ap.add_argument("--audio_dir", required=True)
    ap.add_argument("--limit", type=int, default=0, help="per manifest; 0 = all")
    ap.add_argument("--distance", default="ot,mean", help="ot, mean, or both")
    ap.add_argument("--ot_backend", default="sinkhorn", choices=["exact", "sinkhorn"])
    ap.add_argument("--match_db", type=float, default=1.0,
                    help="loudness differences within this many dB count as matched")
    ap.add_argument("--min_comparisons", type=int, default=5)
    ap.add_argument("--permutation", type=int, default=0)
    ap.add_argument("--normalize", action="store_true",
                    help="also re-extract embeddings on loudness-normalized audio (GPU)")
    ap.add_argument("--norm_cache", default="cache/loudnorm",
                    help="where normalized-audio embeddings are cached")
    ap.add_argument("--save_json", default="")
    audio_io.add_extractor_args(ap)
    args = ap.parse_args()

    cfg = FormAdherenceConfig(ot_backend=args.ot_backend)
    distances = [d.strip() for d in args.distance.split(",") if d.strip()]
    audio = audio_io.index_audio_dir(args.audio_dir)
    items = [it for it in load_manifests(args.manifest, limit=args.limit)
             if audio.get(audio_io.track_id(it) or "")]
    fr = args.frame_rate
    print(f"[data] {len(items)} songs with audio; frame rate {fr}; match within "
          f"{args.match_db} dB; distances {distances}")
    extractor = spec = None
    if args.normalize:
        spec = audio_io.spec_from_args(args)
        extractor = audio_io.build_extractor(spec)
        os.makedirs(args.norm_cache, exist_ok=True)
        print(f"[normalize] extractor {spec.describe()}, cache {args.norm_cache}")

    results, t0 = {}, time.time()
    for dname, grp in by_dataset(items).items():
        print(f"\n===== {dname}: {len(grp)} songs =====", flush=True)
        recs = {d: [] for d in distances}
        loud_auc, norm_recs, norm_loud = [], [], []
        rng = np.random.default_rng(0)
        for k, it in enumerate(grp, 1):
            tid = audio_io.track_id(it)
            emb = np.load(it["emb"])
            wav, sr = audio_io.read_audio(audio[tid])
            x = np.asarray(wav, dtype=np.float64)
            x = x.mean(axis=1) if x.ndim == 2 else x
            db = frame_db(x, sr, fr)
            T = min(len(emb), len(db))
            segs, labs = song_segments(emb[:T], it["boundaries"], it["labels"])
            levels = section_levels(db[:T], segs)
            r0 = None
            for d in distances:
                r = song_record(emb[:T], segs, labs, levels, d, cfg, rng)
                if r is not None:
                    recs[d].append(r)
                    r0 = r0 or r
            if r0 is not None:
                loud_auc.append(hard_auc(r0["dl"][r0["same"]], r0["dl"][~r0["same"]]))
            if extractor is not None:
                cpath = os.path.join(args.norm_cache, f"{tid}.{spec.tag()}.npy")
                xn = normalize_sections(x, sr, segs, fr)
                if os.path.exists(cpath):
                    En = np.load(cpath)
                else:
                    En = np.asarray(extractor.extract(xn.astype(np.float32), sr))
                    np.save(cpath, En)
                dbn = frame_db(xn, sr, fr)
                Tn = min(len(En), len(dbn))
                segs_n, labs_n = song_segments(En[:Tn], it["boundaries"], it["labels"])
                lv_n = section_levels(dbn[:Tn], segs_n, decimals=1)
                rn = song_record(En[:Tn], segs_n, labs_n, lv_n, distances[0], cfg, rng)
                if rn is not None:
                    norm_recs.append(rn)
                    norm_loud.append(hard_auc(rn["dl"][rn["same"]], rn["dl"][~rn["same"]]))
            if k % max(1, len(grp) // 10) == 0 or k == len(grp):
                print(f"  {k}/{len(grp)} songs ({time.time() - t0:.0f}s)", flush=True)

        res = {"songs": len(grp), "loudness_diff_auc": summarize(loud_auc)}
        la = res["loudness_diff_auc"]
        print(f"\n  loudness difference alone            per-song AUC median "
              f"{la['median']:.3f}  ({la['frac_above_half']:.0%} of songs > 0.5)")
        for d in distances:
            S = recs[d]
            raw = summarize([hard_auc(s["d"][s["same"]], s["d"][~s["same"]]) for s in S])
            mt = [matched_auc(s["d"], s["dl"], s["same"], args.match_db, args.min_comparisons)
                  for s in S]
            ml = [matched_auc(s["dl"], s["dl"], s["same"], args.match_db, args.min_comparisons)
                  for s in S]
            matched = summarize([m[0] for m in mt])
            matched_loud = summarize([m[0] for m in ml])
            pairs = [(a[0], b[0]) for a, b in zip(mt, ml) if a[0] is not None and b[0] is not None]
            excess = None
            if pairs:
                pe, ple = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
                sp, npos, nn = sign_test(pe, ple)
                excess = {"median_excess": float(np.median(pe - ple)),
                          "songs_embedding_better": npos, "songs_compared": nn,
                          "sign_test_p": sp}
            kept = float(np.median([m[1] for m in mt])) if mt else float("nan")
            resid = summarize([residual_auc(s["d"], s["dl"], s["same"]) for s in S])
            added = added_value(S)
            r = {"raw_auc": raw, "matched_auc": matched,
                 "matched_auc_loudness_check": matched_loud,
                 "matched_share_of_comparisons": kept, "matched_excess": excess,
                 "residual_auc": resid, "added_value": added}
            print(f"  [{d.upper():4}] embedding, no control                AUC median "
                  f"{raw['median']:.3f}  ({raw['frac_above_half']:.0%} > 0.5)")
            if not matched:
                print(f"  [{d.upper():4}] 1. loudness-MATCHED comparisons: too few in every "
                      f"song (loudness alone separates almost perfectly); try --match_db 2")
            if matched:
                print(f"  [{d.upper():4}] 1. loudness-MATCHED comparisons      AUC median "
                      f"{matched['median']:.3f}  ({matched['frac_above_half']:.0%} > 0.5, "
                      f"{matched['songs']} songs, {kept:.0%} of comparisons matched)")
                print(f"         loudness on the same comparisons          AUC median "
                      f"{matched_loud['median']:.3f}  (leftover loudness inside the band)")
                if excess:
                    print(f"         embedding minus loudness, per song: median "
                          f"{excess['median_excess']:+.3f}; embedding better in "
                          f"{excess['songs_embedding_better']}/{excess['songs_compared']} songs, "
                          f"sign test p = {excess['sign_test_p']:.4f}   <- the key number")
            if args.permutation and matched:
                pt = perm_matched(S, args.match_db, args.min_comparisons, args.permutation)
                r["matched_perm"] = pt
                print(f"         shuffled labels: median {pt['null_median']:.3f} "
                      f"(95% {pt['null_95']:.3f}), p = {pt['p_value']:.4f}")
            if resid:
                print(f"  [{d.upper():4}] 2. loudness REGRESSED OUT              AUC median "
                      f"{resid['median']:.3f}  ({resid['frac_above_half']:.0%} > 0.5)")
            print(f"  [{d.upper():4}] 3. ADDED VALUE (5-fold by song): loudness "
                  f"{added['loudness_only_median_auc']:.3f} -> loudness+embedding "
                  f"{added['loudness_plus_embedding_median_auc']:.3f}; better in "
                  f"{added['frac_songs_improved']:.0%} of songs (weights: loudness "
                  f"{added['mean_coef_loudness']:+.2f}, embedding {added['mean_coef_embedding']:+.2f})")
            res[d] = r
        if norm_recs:
            nraw = summarize([hard_auc(s["d"][s["same"]], s["d"][~s["same"]]) for s in norm_recs])
            nl = summarize(norm_loud)
            res["normalized"] = {"distance": distances[0], "raw_auc": nraw,
                                 "loudness_diff_auc_after": nl}
            print(f"  [{distances[0].upper():4}] 4. loudness-NORMALIZED audio, re-extracted  "
                  f"AUC median {nraw['median']:.3f}  ({nraw['frac_above_half']:.0%} > 0.5)")
            print(f"         check: loudness difference after normalizing  AUC median "
                  f"{nl['median']:.3f}  (should be ~0.5)")
        results[dname] = res

    print("\nHow to read: the embeddings see more than loudness if they beat loudness on")
    print("matched comparisons in most songs (sign test p small), the residual AUC stays")
    print("above 0.5, and the normalized-audio AUC stays well above 0.5 while loudness")
    print("there drops to ~0.5. 3 shows how much the embeddings add on top of loudness.")
    if args.save_json:
        os.makedirs(os.path.dirname(args.save_json) or ".", exist_ok=True)
        with open(args.save_json, "w") as f:
            json.dump({"settings": {k: v for k, v in vars(args).items()},
                       "results": results}, f, indent=2)
        print(f"[saved] {args.save_json}")


if __name__ == "__main__":
    main()
