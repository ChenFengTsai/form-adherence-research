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
You produce this manifest from the dataset annotations + your extracted
embeddings (align annotated second-boundaries to frame indices via frame_rate).

Output: configs/eval_baseline.json (a RunConfig with the fitted metric).

    python scripts/calibrate.py --manifest data/msa/manifest.json \
        --out configs/eval_baseline.json
"""

import argparse
import json
import numpy as np
from itertools import combinations


def collect_pair_distances(manifest, ot_backend="exact"):
    """Return (same_letter_distances, diff_letter_distances) pooled over the
    corpus, using the metric's own OT on the annotated segments."""
    from formadherence.ot import wasserstein_distance, sinkhorn_distance
    from formadherence.config import FormAdherenceConfig
    from formadherence.metric import _segment_distribution
    from formadherence.segmentation import boundaries_to_segments

    cfg = FormAdherenceConfig(ot_backend=ot_backend)
    rng = np.random.default_rng(0)
    ot = (wasserstein_distance if ot_backend == "exact"
          else lambda a, b, **k: sinkhorn_distance(a, b, eps=cfg.sinkhorn_eps))

    same, diff = [], []
    for item in manifest:
        emb = np.load(item["emb"])
        segs = boundaries_to_segments(item["boundaries"], len(emb),
                                      labels=item["labels"])
        dists = [_segment_distribution(s, emb, cfg.max_frames_per_segment, rng)
                 for s in segs]
        for i, j in combinations(range(len(segs)), 2):
            d = ot(dists[i], dists[j], metric="cosine")
            if segs[i].label == segs[j].label:
                same.append(d)
            else:
                diff.append(d)
    return np.array(same), np.array(diff)


def fit_parameters(same, diff):
    """Pick temperature so the two populations separate, and the band from the
    same-letter distance quantiles."""
    # Temperature: set so the midpoint between the medians maps near sim=0.5.
    m_same = np.median(same) if len(same) else 0.02
    m_diff = np.median(diff) if len(diff) else 1.0
    midpoint = 0.5 * (m_same + m_diff)
    # exp(-midpoint / T) = 0.5  ->  T = midpoint / ln(2)
    temperature = float(max(midpoint / np.log(2), 1e-3))

    # Variation band: keep the central mass of real repeats inside (floor, tau).
    floor = float(np.quantile(same, 0.05)) if len(same) else 0.005
    tau = float(np.quantile(same, 0.95)) if len(same) else 0.12
    # A tiny margin so exact copies (near 0) still fall below the floor.
    floor = max(floor, 1e-3)
    return temperature, floor, tau


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", default="configs/eval_baseline.json")
    ap.add_argument("--ot_backend", default="exact",
                    choices=["exact", "sinkhorn"])
    args = ap.parse_args()

    from formadherence import RunConfig
    from formadherence.config import FormAdherenceConfig

    with open(args.manifest) as f:
        manifest = json.load(f)

    same, diff = collect_pair_distances(manifest, ot_backend=args.ot_backend)
    print(f"[data] same-letter pairs: {len(same)}  diff-letter pairs: {len(diff)}")
    if len(same):
        print(f"       same median={np.median(same):.4f} "
              f"5%={np.quantile(same,0.05):.4f} 95%={np.quantile(same,0.95):.4f}")
    if len(diff):
        print(f"       diff median={np.median(diff):.4f}")

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
