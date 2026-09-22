#!/usr/bin/env python
"""Calibrate the metric's distance-scale parameters against human-annotated
structure (SALAMI / RWC-Pop / Harmonix). THIS IS NOT OPTIONAL: the shipped
defaults are fit on synthetic embeddings and will be wrong for real CLAP/EnCodec
distances.

What it fits:
  * similarity_temperature -- so same-letter pairs score high and different-letter
    pairs score low (maximizes the distinctness/similarity gap).
  * variation_floor, tau  -- the acceptable-variation band, set from the observed
    distribution of same-letter (repeat) distances on real annotated repeats.

Input: a manifest JSON describing annotated songs:
    [
      {"emb": "cache/embeddings/<key>.npy",
       "boundaries": [frame indices...],
       "labels": ["A","B","A","C","A"]},
      ...
    ]

Output: configs/eval_baseline.json (a RunConfig with the fitted metric).

    python scripts/calibrate.py --manifest data/msa/manifest.json \
        --out configs/eval_baseline.json

NOTE on speed: --ot_backend exact solves a transportation LP per segment pair,
which is slow on songs with many segments. --ot_backend sinkhorn is much faster
(entropic approximation). Whichever you calibrate with, evaluate with the same
backend so the fitted tau/floor/temperature match your eval distances.
"""

import argparse
import json
import os
import time
import numpy as np
from itertools import combinations


def collect_pair_distances(manifest, ot_backend="exact", progress_every=1,
                           max_frames=None):
    """Return (same_letter_distances, diff_letter_distances) pooled over the
    corpus, using the metric's own OT on the annotated segments."""
    from formadherence.ot import wasserstein_distance, sinkhorn_distance
    from formadherence.config import FormAdherenceConfig
    from formadherence.metric import _segment_distribution
    from formadherence.segmentation import boundaries_to_segments

    cfg = FormAdherenceConfig(ot_backend=ot_backend)
    cap = max_frames if max_frames is not None else cfg.max_frames_per_segment
    rng = np.random.default_rng(0)
    ot = (wasserstein_distance if ot_backend == "exact"
          else lambda a, b, **k: sinkhorn_distance(a, b, eps=cfg.sinkhorn_eps))

    # Pre-count total pairs so we can show a real ETA.
    total_pairs = 0
    for item in manifest:
        n = len(item["labels"])
        total_pairs += n * (n - 1) // 2
    print(f"[plan] {len(manifest)} songs, ~{total_pairs} segment pairs to "
          f"compute with ot_backend={ot_backend}, max_frames={cap}")

    same, diff = [], []
    done_pairs = 0
    t0 = time.time()
    n_songs = len(manifest)
    for si, item in enumerate(manifest, 1):
        emb_path = item["emb"]
        if not os.path.exists(emb_path):
            print(f"[skip] {si}/{n_songs} missing embedding {emb_path}")
            continue
        try:
            emb = np.load(emb_path)
            segs = boundaries_to_segments(item["boundaries"], len(emb),
                                          labels=item["labels"])
            dists = [_segment_distribution(s, emb, cap, rng) for s in segs]
            for i, j in combinations(range(len(segs)), 2):
                d = ot(dists[i], dists[j], metric="cosine")
                if segs[i].label == segs[j].label:
                    same.append(d)
                else:
                    diff.append(d)
                done_pairs += 1
        except Exception as e:
            print(f"[warn] {si}/{n_songs} {item.get('salami_id', emb_path)}: "
                  f"{type(e).__name__}: {e}")
            continue

        if si % progress_every == 0 or si == n_songs:
            elapsed = time.time() - t0
            frac = done_pairs / max(total_pairs, 1)
            eta = (elapsed / frac - elapsed) if frac > 0 else float("nan")
            print(f"[prog] song {si}/{n_songs}  pairs {done_pairs}/{total_pairs}"
                  f"  same={len(same)} diff={len(diff)}"
                  f"  elapsed={elapsed:.0f}s  eta={eta:.0f}s", flush=True)

    return np.array(same), np.array(diff)


def fit_parameters(same, diff):
    """Pick temperature so the two populations separate, and the band from the
    same-letter distance quantiles."""
    m_same = np.median(same) if len(same) else 0.02
    m_diff = np.median(diff) if len(diff) else 1.0
    midpoint = 0.5 * (m_same + m_diff)
    temperature = float(max(midpoint / np.log(2), 1e-3))

    floor = float(np.quantile(same, 0.05)) if len(same) else 0.005
    tau = float(np.quantile(same, 0.95)) if len(same) else 0.12
    floor = max(floor, 1e-3)
    return temperature, floor, tau


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", default="configs/eval_baseline.json")
    ap.add_argument("--ot_backend", default="exact",
                    choices=["exact", "sinkhorn"])
    ap.add_argument("--progress_every", type=int, default=1,
                    help="print progress every N songs")
    ap.add_argument("--max_frames", type=int, default=None,
                    help="override max_frames_per_segment (smaller = faster "
                         "exact OT; default uses the config value, 256)")
    args = ap.parse_args()

    from formadherence import RunConfig
    from formadherence.config import FormAdherenceConfig

    with open(args.manifest) as f:
        manifest = json.load(f)

    same, diff = collect_pair_distances(
        manifest, ot_backend=args.ot_backend,
        progress_every=args.progress_every, max_frames=args.max_frames)

    print(f"[data] same-letter pairs: {len(same)}  diff-letter pairs: {len(diff)}")
    if len(same):
        print(f"       same median={np.median(same):.4f} "
              f"5%={np.quantile(same,0.05):.4f} 95%={np.quantile(same,0.95):.4f}")
    if len(diff):
        print(f"       diff median={np.median(diff):.4f}")
    # The key sanity check: same should be clearly BELOW diff.
    if len(same) and len(diff):
        gap = np.median(diff) - np.median(same)
        verdict = "GOOD separation" if gap > 0 else "WARNING: no separation!"
        print(f"       median gap (diff - same) = {gap:.4f}  -> {verdict}")

    temperature, floor, tau = fit_parameters(same, diff)
    print(f"[fit]  similarity_temperature={temperature:.4f}  "
          f"variation_floor={floor:.4f}  tau={tau:.4f}")

    metric = FormAdherenceConfig(
        ot_backend=args.ot_backend,
        similarity_temperature=temperature,
        variation_floor=floor,
        tau=tau,
    )
    cfg = RunConfig(mode="evaluation", metric=metric)
    cfg.to_json(args.out)
    print(f"[out]  wrote {args.out}")
    print("       Use this file with run_evaluation.py and as the metric core "
          "for train_rl.py so eval and RL stay calibrated together.")


if __name__ == "__main__":
    main()