#!/usr/bin/env python
"""Axis check 1: do DISTINCTNESS and SIMILARITY each catch their own error?

Uses ANNOTATED boundaries (so cutting errors play no part) and cached
embeddings (no GPU). For each annotated song, the true form is edited two ways:

  SPLIT  one occurrence of a repeated letter gets a NEW letter
         (ABACA -> ABADA). The form now asks two sections that ARE the same
         to be different -> should lower DISTINCTNESS.
  MERGE  one section is relabelled to another existing letter
         (ABACA -> ABABA). The form now asks two sections that ARE different
         to be the same -> should lower SIMILARITY.

Reported per edit type: the median change of each axis (edited - true form),
how often each axis went down, and how often each axis took the bigger hit.
If each edit type mainly hits "its" axis, the two axes measure different
things. If both always move together, they are two views of one contrast.

Also reported: correlation of distinctness and similarity across the true
forms of all songs.

    python scripts/check_axes.py --manifest data/msa/songformbench_manifest.json \
        --config configs/eval_sfb_mean.json --limit 150 \
        --save_json results/sfb_axes.json
"""

import argparse
import json
import os
import sys
import time
from dataclasses import replace

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import formadherence.metric as M
from formadherence.config import FormAdherenceConfig
from formadherence.manifest import load_manifests

AXES = ("distinctness", "similarity", "variation", "overall")


def new_letter(form):
    for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        if c not in form:
            return c
    return None


def split_edits(form, k, rng):
    """Relabel one occurrence (not the first) of a repeated letter as new."""
    nl = new_letter(form)
    cands = []
    if nl:
        for i, c in enumerate(form):
            if form.count(c) >= 2 and form.index(c) != i:
                cands.append(form[:i] + nl + form[i + 1:])
    rng.shuffle(cands)
    return cands[:k]


def merge_edits(form, k, rng):
    """Relabel one section to a different letter that already exists."""
    letters = sorted(set(form))
    cands = []
    for i, c in enumerate(form):
        for x in letters:
            if x != c:
                cands.append(form[:i] + x + form[i + 1:])
    cands = sorted(set(cands))
    rng.shuffle(cands)
    return cands[:k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--config", default="configs/eval_sfb_mean.json")
    ap.add_argument("--limit", type=int, default=150, help="0 = all")
    ap.add_argument("--n_edits", type=int, default=3, help="per edit type per song")
    ap.add_argument("--aggregate", default="mean", choices=["mean", "geometric"])
    ap.add_argument("--chance_margin", type=float, default=0.05)
    ap.add_argument("--backend", choices=["exact", "sinkhorn", "mean"], default=None)
    ap.add_argument("--save_json", default="")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = FormAdherenceConfig.from_dict(json.load(f)["metric"])
    cfg = replace(cfg, aggregate=args.aggregate, chance_margin=args.chance_margin)
    if args.backend:
        cfg = replace(cfg, ot_backend=args.backend)
    print(f"[cfg] distance={cfg.ot_backend} aggregate={cfg.aggregate} "
          f"margin={cfg.chance_margin} kappa={cfg.contrast_temperature:.3f}")

    M.enable_ot_cache()
    rng = np.random.default_rng(0)
    items = [it for it in load_manifests(args.manifest, limit=args.limit)
             if all(len(str(l)) == 1 for l in it["labels"])]
    own_axes, deltas = [], {"split": [], "merge": []}
    t0 = time.time()
    for k, it in enumerate(items, 1):
        M.clear_ot_cache()
        E, b = np.load(it["emb"]), it["boundaries"]
        form = "".join(it["labels"])
        own = M.score_form_adherence(E, form, boundaries=b, config=cfg)
        own_axes.append([getattr(own, a) for a in AXES] + [own.auc if own.auc is not None else np.nan])
        for kind, gen in (("split", split_edits), ("merge", merge_edits)):
            for f2 in gen(form, args.n_edits, rng):
                r = M.score_form_adherence(E, f2, boundaries=b, config=cfg)
                deltas[kind].append([getattr(r, a) - getattr(own, a) for a in AXES])
        if k % 10 == 0 or k == len(items):
            print(f"[prog] {k}/{len(items)}  {time.time() - t0:.0f}s", flush=True)

    out = {"settings": vars(args), "n_songs": len(items)}
    print(f"\n=== {len(items)} songs, annotated boundaries ===")
    print("Change when the form is edited (edited - true). Negative = axis noticed.\n")
    print("  edit    expected to hit   n     d_distinct  d_similar  d_variation  d_overall")
    for kind, expect in (("split", "distinctness"), ("merge", "similarity")):
        D = np.asarray(deltas[kind])
        if not len(D):
            print(f"  {kind}: no edits possible")
            continue
        med = np.median(D, axis=0)
        print(f"  {kind:6}  {expect:15} {len(D):5}    {med[0]:+.3f}      {med[1]:+.3f}     "
              f"{med[2]:+.3f}       {med[3]:+.3f}")
        down = (D < -1e-9).mean(axis=0)
        bigger = float((D[:, 0] < D[:, 1]).mean())   # distinctness took the bigger hit
        print(f"          went down:              {down[0]:6.0%}     {down[1]:6.0%}      "
              f"{down[2]:6.0%}       {down[3]:6.0%}")
        print(f"          distinctness hit harder than similarity in {bigger:.0%} of edits")
        out[kind] = {"n": int(len(D)), "median_delta": dict(zip(AXES, med.round(4).tolist())),
                     "frac_down": dict(zip(AXES, down.round(4).tolist())),
                     "distinct_hit_harder": bigger}

    A = np.asarray(own_axes, dtype=float)

    def rho(x, y):
        ok = ~(np.isnan(x) | np.isnan(y))
        if ok.sum() < 3 or np.ptp(x[ok]) == 0 or np.ptp(y[ok]) == 0:
            return float("nan")
        return float(spearmanr(x[ok], y[ok]).correlation)

    rho_ds, rho_da, rho_sa = rho(A[:, 0], A[:, 1]), rho(A[:, 0], A[:, 4]), rho(A[:, 1], A[:, 4])
    fmt = lambda v: "n/a" if np.isnan(v) else f"{v:.2f}"
    print(f"\nAcross true forms: Spearman(distinctness, similarity) = {fmt(rho_ds)};  "
          f"with per-song AUC: distinctness {fmt(rho_da)}, similarity {fmt(rho_sa)}")
    print(f"Median true-form axes: distinct {np.median(A[:, 0]):.3f}  "
          f"similar {np.median(A[:, 1]):.3f}  variation {np.median(A[:, 2]):.3f}  "
          f"overall {np.median(A[:, 3]):.3f}")
    out["correlation"] = {"distinct_vs_similar": float(rho_ds),
                          "distinct_vs_auc": float(rho_da), "similar_vs_auc": float(rho_sa)}

    print("\nHow to read it:")
    print("  split mostly lowers distinctness AND merge mostly lowers similarity")
    print("    -> the two axes catch different errors: keep them as separate axes")
    print("  both edit types lower both axes about equally, and correlation is high")
    print("    -> they are two views of one contrast: say so in the proposal")
    if args.save_json:
        os.makedirs(os.path.dirname(args.save_json) or ".", exist_ok=True)
        with open(args.save_json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[saved] {args.save_json}")


if __name__ == "__main__":
    main()
