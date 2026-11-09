#!/usr/bin/env python
"""Within-song separation of same-label vs different-label sections.

This is the experiment behind the proposal's preliminary result (per-song AUC).
For every song: OT (or mean-pool) distance between every pair of annotated
sections, split into same-label and different-label pairs, then
  * per-song AUC = P(diff distance > same distance); 0.5 chance, 1.0 perfect
  * pooled gap/spread (old measure; pools songs, so scale differences leak in)

Fixes / additions vs the old script:
  * several manifests at once, results reported PER DATASET (no silent pooling)
  * --permutation N : shuffle labels within each song (keeping how many of
    each letter) N times. Gives the real chance level and a p-value for the
    median AUC, instead of assuming 0.5.
  * --rms_audio_dir : loudness-only baseline. If plain loudness already gets a
    high AUC, the embedding result is mostly intro/outro/quiet-vs-loud contrast.
  * debiased Sinkhorn by default (--no_debias reproduces old numbers)
  * every setting printed and saved (--save_json), so a result can be cited

Examples:
  # the headline number, per dataset, with a permutation test
  python scripts/diagnose_formulation.py --mode real \
      --manifest data/msa/salami_manifest.json,data/msa/songformbench_manifest.json \
      --limit 0 --auc --permutation 1000 --save_json results/auc_main.json

  # same, plus loudness baseline (needs the audio)
  python scripts/diagnose_formulation.py --mode real \
      --manifest data/msa/songformbench_manifest.json --auc \
      --rms_audio_dir /storage/ssd1/richtsai1103/form-adherence-research/songformbench/audio

  # synthetic control (clean, and with a shared component like real music)
  python scripts/diagnose_formulation.py --mode synthetic --auc
  python scripts/diagnose_formulation.py --mode synthetic --auc --shared_component 3
"""

import argparse
import json
import os
import sys
from itertools import combinations

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from formadherence.config import FormAdherenceConfig
from formadherence.ot import segment_distance
from formadherence.metric import hard_auc
from formadherence.segmentation import boundaries_to_segments
from formadherence.manifest import load_manifests, by_dataset
from formadherence.audio_io import read_audio, index_audio_dir, track_id


# ---- per-song pieces ------------------------------------------------------

def apply_center(emb, mode):
    if mode == "none":
        return emb
    mu = emb.mean(axis=0, keepdims=True)
    if mode == "mean":
        return emb - mu
    if mode == "zscore":
        return (emb - mu) / (emb.std(axis=0, keepdims=True) + 1e-8)
    raise ValueError(mode)


def song_segments(emb, boundaries, labels):
    T = len(emb)
    b = [x for x in boundaries if 0 < x < T]
    segs = boundaries_to_segments(b, T, labels=None)
    labs = [str(l) for l in labels][:len(segs)]
    while len(labs) < len(segs):
        labs.append(labs[-1] if labs else "A")
    return segs, labs


def pair_distances(emb, segs, distance, cfg, metric, rng):
    """Upper-triangle dict (i,j) -> distance between segment frame clouds."""
    clouds = []
    for s in segs:
        fr = emb[s.start:s.end]
        if len(fr) > cfg.max_frames_per_segment:
            idx = np.sort(rng.choice(len(fr), cfg.max_frames_per_segment, replace=False))
            fr = fr[idx]
        clouds.append(fr if len(fr) else None)
    D = {}
    for i, j in combinations(range(len(segs)), 2):
        a, b = clouds[i], clouds[j]
        if a is None or b is None:
            continue
        if distance == "ot":
            D[(i, j)] = segment_distance(a, b, cfg, metric=metric)
        else:
            ma, mb = a.mean(0), b.mean(0)
            if metric == "euclidean":
                D[(i, j)] = float(np.linalg.norm(ma - mb))
            else:
                ma = ma / (np.linalg.norm(ma) + 1e-12)
                mb = mb / (np.linalg.norm(mb) + 1e-12)
                D[(i, j)] = float(1 - ma @ mb)
    return D


def split_same_diff(D, labs):
    same = [d for (i, j), d in D.items() if labs[i] == labs[j]]
    diff = [d for (i, j), d in D.items() if labs[i] != labs[j]]
    return same, diff


def compute_songs(songs, distance, cfg, center, metric, tag=""):
    """songs: list of (emb, boundaries, labels). Returns list of (D, labs)."""
    import time
    rng = np.random.default_rng(0)
    out = []
    t0 = time.time()
    n = len(songs)
    every = max(1, n // 10)
    for k, (emb, bounds, labels) in enumerate(songs, 1):
        emb = apply_center(np.asarray(emb, dtype=float), center)
        segs, labs = song_segments(emb, bounds, labels)
        out.append((pair_distances(emb, segs, distance, cfg, metric, rng), labs))
        if n >= 20 and (k % every == 0 or k == n):
            el = time.time() - t0
            print(f"  [{tag} {distance}] {k}/{n} songs  {el:.0f}s elapsed, "
                  f"~{el / k * (n - k):.0f}s left", flush=True)
    return out


# ---- statistics -------------------------------------------------------------

def pooled_separation(computed, per_song_z=False):
    same, diff = [], []
    for D, labs in computed:
        s, d = split_same_diff(D, labs)
        if per_song_z:
            alld = np.array(s + d, dtype=float)
            if len(alld) < 2:
                continue
            mu, sd = alld.mean(), alld.std() + 1e-9
            s = [(x - mu) / sd for x in s]
            d = [(x - mu) / sd for x in d]
        same += s
        diff += d
    if not same or not diff:
        return None
    same, diff = np.array(same), np.array(diff)
    ms, md = float(np.median(same)), float(np.median(diff))
    spread = float(np.quantile(same, .95) - np.quantile(same, .05))
    return dict(n_same=len(same), n_diff=len(diff), same_med=ms, diff_med=md,
                gap=md - ms, spread=spread, gap_over_spread=(md - ms) / max(spread, 1e-9))


def song_aucs(computed, labs_override=None):
    aucs = []
    for k, (D, labs) in enumerate(computed):
        labs = labs_override[k] if labs_override is not None else labs
        a = hard_auc(*split_same_diff(D, labs))
        if a is not None:
            aucs.append(a)
    return aucs


def auc_summary(aucs):
    if not aucs:
        return None
    a = np.asarray(aucs)
    return dict(songs=len(a), median=float(np.median(a)), mean=float(a.mean()),
                frac_above_half=float((a > 0.5).mean()),
                n_above_half=int((a > 0.5).sum()))


def permutation_test(computed, n_perm, seed=0):
    """Shuffle labels within each song; null distribution of the median AUC."""
    rng = np.random.default_rng(seed)
    observed = float(np.median(song_aucs(computed)))
    null = []
    for _ in range(n_perm):
        perm = [list(rng.permutation(labs)) for _, labs in computed]
        a = song_aucs(computed, perm)
        if a:
            null.append(float(np.median(a)))
    null = np.asarray(null)
    p = float((1 + (null >= observed).sum()) / (1 + len(null)))
    return dict(observed_median=observed, null_median=float(np.median(null)),
                null_95=float(np.quantile(null, .95)), p_value=p, n_perm=n_perm)


# ---- loudness baseline ----------------------------------------------------

def rms_frames(path, frame_rate):
    wav, sr = read_audio(path)
    x = wav.mean(axis=1) if wav.ndim == 2 else wav
    hop = int(round(sr / frame_rate))
    n = int(np.ceil(len(x) / hop))
    out = np.empty((n, 1))
    for i in range(n):
        seg = x[i * hop: i * hop + 2 * hop]
        out[i, 0] = 20 * np.log10(np.sqrt(np.mean(seg ** 2)) + 1e-8) if len(seg) else -160
    return out


# ---- data -----------------------------------------------------------------

def load_synthetic(forms, dim, seg_len, variation_scale, copy_paste, n_per_form,
                   shared_component):
    from formadherence.synthetic import make_song_embeddings
    songs = []
    for fi, form in enumerate(forms):
        for k in range(n_per_form):
            emb, bounds = make_song_embeddings(
                form, dim=dim, seg_len=seg_len, variation_scale=variation_scale,
                copy_paste=copy_paste, shared_component=shared_component,
                seed=fi * 100 + k)
            songs.append((emb, bounds, list(form)))
    return songs


def run_block(tag, songs, args, cfg, results, metric=None, distances=("ot", "mean")):
    metric = metric or args.metric
    res = {"songs": len(songs)}
    for dist in distances:
        computed = compute_songs(songs, dist, cfg, args.center, metric, tag=tag)
        sep = pooled_separation(computed, args.per_song_z)
        res[f"{dist}_pooled"] = sep
        line = f"[{tag} {dist.upper():4}] "
        line += ("not enough pairs" if sep is None else
                 f"same_med={sep['same_med']:.4f} diff_med={sep['diff_med']:.4f} "
                 f"gap/spread={sep['gap_over_spread']:.2f}")
        if args.auc:
            s = auc_summary(song_aucs(computed))
            res[f"{dist}_auc"] = s
            if s:
                line += (f" | AUC median={s['median']:.3f} mean={s['mean']:.3f} "
                         f"{s['n_above_half']}/{s['songs']} songs > 0.5")
            if args.permutation and s:
                print(f"  [{tag} {dist}] permutation test ({args.permutation} shuffles)...", flush=True)
                pt = permutation_test(computed, args.permutation)
                res[f"{dist}_perm"] = pt
                line += (f" | shuffled-label median={pt['null_median']:.3f} "
                         f"(95%: {pt['null_95']:.3f}) p={pt['p_value']:.4f}")
        print(line, flush=True)
    results[tag] = res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["real", "synthetic"])
    ap.add_argument("--ot_backend", default="sinkhorn", choices=["exact", "sinkhorn"])
    ap.add_argument("--no_debias", action="store_true",
                    help="plain Sinkhorn cost, to reproduce pre-fix numbers")
    ap.add_argument("--center", default="none", choices=["none", "mean", "zscore"])
    ap.add_argument("--metric", default="cosine", choices=["cosine", "euclidean"])
    ap.add_argument("--per_song_z", action="store_true")
    ap.add_argument("--auc", action="store_true", help="per-song AUC (recommended)")
    ap.add_argument("--permutation", type=int, default=0,
                    help="N label shuffles for a permutation test (e.g. 1000)")
    ap.add_argument("--save_json", default="")
    # real
    ap.add_argument("--manifest", default="data/msa/songformbench_manifest.json",
                    help="one or more, comma-separated; reported per dataset")
    ap.add_argument("--limit", type=int, default=40, help="per manifest; 0 = all")
    ap.add_argument("--rms_audio_dir", default="",
                    help="audio dir(s), comma-separated, for the loudness baseline")
    ap.add_argument("--frame_rate", type=float, default=2.0,
                    help="manifest frame rate (for the loudness baseline)")
    # synthetic
    ap.add_argument("--forms", default="ABABCB,ABACABA,AABBA")
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--seg_len", type=int, default=120)
    ap.add_argument("--variation_scale", type=float, default=0.4)
    ap.add_argument("--copy_paste", action="store_true")
    ap.add_argument("--n_per_form", type=int, default=10)
    ap.add_argument("--shared_component", type=float, default=0.0)
    args = ap.parse_args()

    cfg = FormAdherenceConfig(ot_backend=args.ot_backend,
                              sinkhorn_debias=not args.no_debias)
    settings = {k: v for k, v in vars(args).items()}
    print("[settings] " + json.dumps(settings))
    results = {}

    if args.mode == "real":
        items = load_manifests(args.manifest, limit=args.limit)
        extractors = sorted({it.get("extractor", "unknown (old manifest)") for it in items})
        print(f"[features] {extractors}")
        settings["features"] = extractors
        for name, grp in by_dataset(items).items():
            songs = [(np.load(it["emb"]), it["boundaries"], it["labels"]) for it in grp]
            run_block(name, songs, args, cfg, results)
        if args.rms_audio_dir:
            audio = {}
            for d in args.rms_audio_dir.split(","):
                audio.update(index_audio_dir(d.strip()))
            for name, grp in by_dataset(items).items():
                songs = []
                print(f"  [{name} LOUDNESS-ONLY] reading {len(grp)} audio files...", flush=True)
                for it in grp:
                    p = audio.get(track_id(it) or "")
                    if p:
                        songs.append((rms_frames(p, args.frame_rate),
                                      it["boundaries"], it["labels"]))
                if songs:
                    run_block(f"{name} LOUDNESS-ONLY", songs, args, cfg, results,
                              metric="euclidean", distances=("ot",))
                else:
                    print(f"[{name} LOUDNESS-ONLY] no audio found for these ids")
    else:
        songs = load_synthetic(args.forms.split(","), args.dim, args.seg_len,
                               args.variation_scale, args.copy_paste,
                               args.n_per_form, args.shared_component)
        print(f"SYNTHETIC: {len(songs)} songs, forms={args.forms}, "
              f"shared_component={args.shared_component}")
        run_block("synthetic", songs, args, cfg, results)

    if args.save_json:
        os.makedirs(os.path.dirname(args.save_json) or ".", exist_ok=True)
        with open(args.save_json, "w") as f:
            json.dump({"settings": settings, "results": results}, f, indent=2)
        print(f"[saved] {args.save_json}")


if __name__ == "__main__":
    main()
