#!/usr/bin/env python
"""Pin down WHY same/diff separation is capped (~0.17-0.23) across all features.

Four features all landed in a narrow gap/spread band, which suggests the limit is
not the features but the measurement formulation (segmentation and/or the OT
distance). This script isolates the cause with two controlled knobs:

  distance:   'ot'   = Wasserstein over each segment's frame cloud (current)
              'mean' = cosine between mean-pooled segment vectors (cruder)
  data:       'real'      = your manifest embeddings + annotated boundaries
              'synthetic' = make_song_embeddings with PERFECT boundaries/labels

Interpretation matrix (gap/spread ratio, higher = better separation):
  * synthetic + ot separates well, real + ot does not
        -> features/OT are fine; REAL boundaries/labels are the problem.
  * synthetic + ot ALSO fails
        -> the OT-over-frame-clouds formulation itself caps separation.
  * real + mean >> real + ot
        -> OT is washing out structure; switch to a segment-summary distance.

Examples:
  # real data, both distances, using precomputed manifest embeddings
  python scripts/diagnose_formulation.py --mode real \
      --manifest data/msa/manifest.json

  # synthetic oracle (perfect boundaries), both distances, several forms
  python scripts/diagnose_formulation.py --mode synthetic \
      --forms ABABCB,ABACABA,AABBA --variation_scale 0.4
"""

import argparse, json, os
import numpy as np
from itertools import combinations


# ---- the two distance backends ------------------------------------------

def seg_frames(emb, start, end, cap, rng):
    fr = emb[start:end]
    if len(fr) == 0:
        return None
    if len(fr) > cap:
        idx = rng.choice(len(fr), size=cap, replace=False); idx.sort()
        fr = fr[idx]
    return fr


def dist_ot(a, b, ot_backend, cfg):
    from formadherence.ot import wasserstein_distance, sinkhorn_distance
    if ot_backend == "exact":
        return wasserstein_distance(a, b, metric="cosine")
    return sinkhorn_distance(a, b, eps=cfg.sinkhorn_eps, metric="cosine")


def dist_mean(a, b, ot_backend, cfg):
    # cosine distance between mean-pooled (then L2-normalized) segment vectors
    ma = a.mean(axis=0); mb = b.mean(axis=0)
    ma = ma / (np.linalg.norm(ma) + 1e-12)
    mb = mb / (np.linalg.norm(mb) + 1e-12)
    return float(1.0 - ma @ mb)


DISTS = {"ot": dist_ot, "mean": dist_mean}


# ---- separation over a list of (emb, boundaries, labels) ----------------

def separation(items, distance, ot_backend):
    from formadherence.config import FormAdherenceConfig
    from formadherence.segmentation import boundaries_to_segments
    cfg = FormAdherenceConfig(ot_backend=ot_backend)
    dfun = DISTS[distance]
    rng = np.random.default_rng(0)

    same, diff = [], []
    for emb, boundaries, labels in items:
        T = len(emb)
        b = [x for x in boundaries if 0 < x < T]
        segs = boundaries_to_segments(b, T, labels=None)
        labs = list(labels)[:len(segs)]
        while len(labs) < len(segs):
            labs.append(labs[-1] if labs else "A")
        for s, lab in zip(segs, labs):
            s.label = lab
        clouds = [seg_frames(emb, s.start, s.end, cfg.max_frames_per_segment, rng)
                  for s in segs]
        for i, j in combinations(range(len(segs)), 2):
            if clouds[i] is None or clouds[j] is None:
                continue
            d = dfun(clouds[i], clouds[j], ot_backend, cfg)
            (same if segs[i].label == segs[j].label else diff).append(d)

    same, diff = np.array(same), np.array(diff)
    if not len(same) or not len(diff):
        return None
    ms, md = float(np.median(same)), float(np.median(diff))
    spread = float(np.quantile(same, 0.95) - np.quantile(same, 0.05))
    return dict(same=len(same), diff=len(diff), same_med=ms, diff_med=md,
                gap=md - ms, spread=spread, ratio=(md - ms) / max(spread, 1e-9))


def report(tag, res):
    if res is None:
        print(f"[{tag}] not enough pairs"); return
    print(f"[{tag}] same={res['same']} diff={res['diff']}  "
          f"same_med={res['same_med']:.4f} diff_med={res['diff_med']:.4f} "
          f"gap={res['gap']:.4f} spread={res['spread']:.4f} "
          f"gap/spread={res['ratio']:.2f}")


# ---- data loaders --------------------------------------------------------

def load_real(manifest_path, limit):
    items = []
    with open(manifest_path) as f:
        manifest = json.load(f)
    for item in manifest:
        if not os.path.exists(item["emb"]):
            continue
        emb = np.load(item["emb"])
        items.append((emb, item["boundaries"], item["labels"]))
        if limit and len(items) >= limit:
            break
    return items


def load_synthetic(forms, dim, seg_len, variation_scale, copy_paste, n_per_form):
    from formadherence.synthetic import make_song_embeddings
    # NOTE: adjust import if synthetic.py lives elsewhere in your tree.
    items = []
    for fi, form in enumerate(forms):
        for k in range(n_per_form):
            emb, bounds = make_song_embeddings(
                form, dim=dim, seg_len=seg_len,
                variation_scale=variation_scale, copy_paste=copy_paste,
                seed=fi * 100 + k)
            items.append((emb, bounds, list(form)))
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["real", "synthetic"])
    ap.add_argument("--ot_backend", default="sinkhorn",
                    choices=["exact", "sinkhorn"])
    # real
    ap.add_argument("--manifest", default="data/msa/manifest.json")
    ap.add_argument("--limit", type=int, default=40)
    # synthetic
    ap.add_argument("--forms", default="ABABCB,ABACABA,AABBA")
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--seg_len", type=int, default=120)
    ap.add_argument("--variation_scale", type=float, default=0.4)
    ap.add_argument("--copy_paste", action="store_true")
    ap.add_argument("--n_per_form", type=int, default=10)
    args = ap.parse_args()

    if args.mode == "real":
        items = load_real(args.manifest, args.limit)
        print(f"REAL data: {len(items)} songs from {args.manifest}\n")
    else:
        forms = args.forms.split(",")
        items = load_synthetic(forms, args.dim, args.seg_len,
                               args.variation_scale, args.copy_paste,
                               args.n_per_form)
        print(f"SYNTHETIC oracle: {len(items)} songs, forms={forms}, "
              f"variation_scale={args.variation_scale}, "
              f"copy_paste={args.copy_paste}\n")

    print("Comparing distance formulations (gap/spread, higher better):\n")
    report(f"{args.mode} + OT",   separation(items, "ot",   args.ot_backend))
    report(f"{args.mode} + MEAN", separation(items, "mean", args.ot_backend))

    print("\nReading the result:")
    print("  synthetic+OT good but real+OT bad  -> boundaries/labels are the problem")
    print("  synthetic+OT also bad              -> OT formulation caps separation")
    print("  real+MEAN >> real+OT               -> OT is washing out structure")


if __name__ == "__main__":
    main()