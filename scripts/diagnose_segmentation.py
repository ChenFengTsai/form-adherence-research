#!/usr/bin/env python
"""Is the segmentation right? Compare discovered boundaries with the
ANNOTATED ones, using the standard boundary hit rate of music structure
analysis.

For each song the segmenter is told how many boundaries to find (the true
count, as the metric is told by the target form), then:

  hit rate @ tolerance  a found boundary counts as a hit if it lies within
                        tolerance seconds of a true boundary not already
                        matched (one-to-one). Reported as precision, recall,
                        F. With the count given, P = R = F unless the
                        segmenter fails and falls back to equal cuts.
                        Standard tolerances: 0.5 s (strict) and 3 s (lenient).
  median deviation      seconds from each true boundary to the nearest found one

Methods compared:
  foote k=..        Foote novelty peaks (told the section count), several kernels
  form-aware w=..   cuts fitted to the song's form (told the letter sequence,
                    e.g. ABACAB): repeats share a template, plus w x Foote
                    novelty at each cut. w=0 = template fit only
  equal cuts        what the metric falls back to when discovery fails
  random cuts       chance level (average of 20 random draws)
If Foote is not clearly above equal and random cuts, it is not finding
section changes.

    # on the manifest's CLAP+EnCodec embeddings (no audio needed)
    python scripts/diagnose_segmentation.py --manifest data/msa/songformbench_manifest.json \
        --limit 0 --save_json results/sfb_segmentation.json

    # also try Foote on chroma (harmony) features from the audio
    python scripts/diagnose_segmentation.py --manifest data/msa/songformbench_manifest.json \
        --features embedding,chroma --audio_dir $DATA/songformbench/audio --limit 100
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from formadherence.segmentation import form_aware_boundaries, segment_boundaries
from formadherence.manifest import load_manifests
from formadherence import audio_io


def hit_rate(est, ref, tol):
    """One-to-one matching within tol (seconds), closest pairs first.
    Returns (precision, recall, f)."""
    est, ref = list(est), list(ref)
    if not est and not ref:
        return 1.0, 1.0, 1.0
    if not est or not ref:
        return 0.0, 0.0, 0.0
    pairs = sorted((abs(e - r), i, j) for i, e in enumerate(est)
                   for j, r in enumerate(ref) if abs(e - r) <= tol)
    used_e, used_r, hits = set(), set(), 0
    for _, i, j in pairs:
        if i not in used_e and j not in used_r:
            used_e.add(i)
            used_r.add(j)
            hits += 1
    p, r = hits / len(est), hits / len(ref)
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def median_dev(est, ref):
    if not est or not ref:
        return float("nan")
    return float(np.median([min(abs(r - e) for e in est) for r in ref]))


def chroma_frames(path, fr):
    """Chroma pooled to the manifest's frame grid (fr frames/s)."""
    from formadherence.variation import chroma_features
    wav, sr = audio_io.read_audio(path)
    C, fps = chroma_features(wav, sr, fps=10.0)
    k = max(1, int(round(fps / fr)))
    n = len(C) // k
    return C[:n * k].reshape(n, k, 12).mean(axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--limit", type=int, default=100, help="0 = all")
    ap.add_argument("--frame_rate", type=float, default=2.0)
    ap.add_argument("--kernels", default="8,16,24,32", help="Foote kernel sizes, frames")
    ap.add_argument("--tolerances", default="0.5,3")
    ap.add_argument("--features", default="embedding", help="embedding,chroma")
    ap.add_argument("--audio_dir", default="", help="needed for chroma features")
    ap.add_argument("--form_aware_weights", default="0,2,5",
                    help="novelty weights for form-aware segmentation; '' = skip")
    ap.add_argument("--form_aware_kernel", type=int, default=24)
    ap.add_argument("--min_len_s", type=float, default=3.0)
    ap.add_argument("--save_json", default="")
    args = ap.parse_args()

    fr = args.frame_rate
    kernels = [int(k) for k in args.kernels.split(",")]
    tols = [float(t) for t in args.tolerances.split(",")]
    feats = [f.strip() for f in args.features.split(",") if f.strip()]
    audio = audio_io.index_audio_dir(args.audio_dir) if args.audio_dir else {}
    if "chroma" in feats and not audio:
        raise SystemExit("--features chroma needs --audio_dir")
    items = load_manifests(args.manifest, limit=args.limit)
    rng = np.random.default_rng(0)

    fa_w = [float(w) for w in args.form_aware_weights.split(",") if w.strip()]
    methods = ([f"foote k={k} [{f}]" for f in feats for k in kernels]
               + [f"form-aware w={w:g} [{f}]" for f in feats for w in fa_w]
               + ["equal cuts", "random cuts"])
    res = {m: {"f": {t: [] for t in tols}, "dev": [], "fallback": 0} for m in methods}
    n_songs, t0 = 0, time.time()
    for it in items:
        E = np.load(it["emb"])
        T = len(E)
        ref_frames = [b for b in it["boundaries"] if 0 < b < T]
        if not ref_frames:
            continue
        ref = [b / fr for b in ref_frames]
        n_b = len(ref_frames)
        feat_arrays = {"embedding": E}
        if "chroma" in feats:
            p = audio.get(audio_io.track_id(it) or "")
            if p is None:
                continue
            Cf = chroma_frames(p, fr)
            feat_arrays["chroma"] = Cf[:T] if len(Cf) >= T else np.vstack(
                [Cf, np.zeros((T - len(Cf), 12))])
        n_songs += 1

        def record(name, est_frames, fell_back=False):
            est = [b / fr for b in est_frames]
            for t in tols:
                res[name]["f"][t].append(hit_rate(est, ref, t)[2])
            res[name]["dev"].append(median_dev(est, ref))
            res[name]["fallback"] += int(fell_back)

        for f in feats:
            X = feat_arrays[f]
            for k in kernels:
                est = segment_boundaries(X, n_boundaries=n_b, window=k)
                fell = len(est) != n_b
                if fell:          # what the metric does: equal cuts
                    est = [round(T * j / (n_b + 1)) for j in range(1, n_b + 1)]
                record(f"foote k={k} [{f}]", est, fell)
            letters = [str(l) for l in it["labels"]]
            if len(letters) == n_b + 1:
                for w in fa_w:
                    est = form_aware_boundaries(
                        X, letters, kernel_size=args.form_aware_kernel, novelty_weight=w,
                        min_len=max(1, int(round(args.min_len_s * fr))))
                    record(f"form-aware w={w:g} [{f}]", est, len(est) != n_b)
        record("equal cuts", [round(T * j / (n_b + 1)) for j in range(1, n_b + 1)])
        # random: average over 20 draws, recorded as one song-level value
        draws = {t: [] for t in tols}
        devs = []
        for _ in range(20):
            est = sorted(rng.choice(np.arange(1, T), size=min(n_b, T - 1), replace=False))
            e_s = [b / fr for b in est]
            for t in tols:
                draws[t].append(hit_rate(e_s, ref, t)[2])
            devs.append(median_dev(e_s, ref))
        for t in tols:
            res["random cuts"]["f"][t].append(float(np.mean(draws[t])))
        res["random cuts"]["dev"].append(float(np.mean(devs)))
        if n_songs % 25 == 0:
            print(f"[prog] {n_songs} songs  {time.time() - t0:.0f}s", flush=True)

    print(f"\n=== {n_songs} songs: discovered vs annotated boundaries "
          f"(segmenter told the true count) ===")
    head = "  method                      " + "  ".join(f"F@{t:g}s" for t in tols) + \
        "   median dev (s)   fell back"
    print(head)
    out = {}
    for m in methods:
        r = res[m]
        fs = [float(np.mean(r["f"][t])) for t in tols]
        dev = float(np.nanmedian(r["dev"]))
        fb = r["fallback"] / max(n_songs, 1)
        out[m] = {"f": dict(zip([str(t) for t in tols], fs)), "median_dev_s": dev,
                  "fallback_rate": fb}
        print(f"  {m:27} " + "  ".join(f"{v:6.3f}" for v in fs) +
              f"      {dev:7.2f}        {fb:5.0%}")
    print("\nHow to read it:")
    print("  F@3s is the usual lenient score (1 = every boundary within 3 s).")
    print("  Foote must beat 'equal cuts' and 'random cuts' clearly to be useful.")
    print("  'fell back' = Foote found too few peaks, so equal cuts were used.")
    print("  form-aware is told the letter sequence (as the metric is by the target")
    print("  form); it should beat Foote, especially at 0.5 s.")
    if args.save_json:
        os.makedirs(os.path.dirname(args.save_json) or ".", exist_ok=True)
        with open(args.save_json, "w") as f:
            json.dump({"settings": vars(args), "n_songs": n_songs, "results": out}, f, indent=2)
        print(f"[saved] {args.save_json}")


if __name__ == "__main__":
    main()
