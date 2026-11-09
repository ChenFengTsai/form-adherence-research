#!/usr/bin/env python
"""Validity test on annotated songs: does a song score higher on ITS OWN
annotated form than on wrong forms -- and which scale setting discriminates
best? Uses the ANNOTATED boundaries (so segmentation errors don't confound it)
and, optionally, discovered boundaries as a second condition.

    python scripts/validate_human.py --manifest data/msa/songformbench_manifest.json \
        --config configs/eval_sfb.json --limit 150 --n_wrong 5 \
        --save_json results/sfb_validity.json

For every (aggregate, chance_margin) setting it reports:
  pairwise   P(own form scores higher than a given wrong form)   chance 50%
  beats_all  P(own form beats every wrong form)                  chance 1/(k+1)
  own / wrong median overall
Wrong forms: single-letter edits of the true form (a section relabelled to
another letter or a new one) plus other songs' forms of the same length.

OT distances are cached across forms and settings, so the cost is about one
scoring pass per song.
"""

import argparse
import json
import os
import sys
import time
from dataclasses import replace

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import formadherence.metric as M
from formadherence.config import FormAdherenceConfig
from formadherence.manifest import load_manifests
from score_corpus import wrong_forms

M.enable_ot_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--config", default="configs/eval_sfb.json")
    ap.add_argument("--limit", type=int, default=150, help="0 = all")
    ap.add_argument("--n_wrong", type=int, default=5)
    ap.add_argument("--margins", default="0,0.05,0.1,0.15")
    ap.add_argument("--also_discovered", action="store_true",
                    help="also test with discovered (not annotated) boundaries")
    ap.add_argument("--also_form_aware", action="store_true",
                    help="also test with form-aware cuts, fitted SEPARATELY to the own "
                         "form and to every wrong form (fair: each gets its best cuts)")
    ap.add_argument("--foote_kernel", type=int, default=None,
                    help="override the config's Foote kernel (mp_window) for 'discovered'")
    ap.add_argument("--backend", choices=["exact", "sinkhorn", "mean"], default=None,
                    help="override the config's distance (e.g. mean = mean-pooled baseline)")
    ap.add_argument("--save_json", default="")
    args = ap.parse_args()

    with open(args.config) as f:
        base = FormAdherenceConfig.from_dict(json.load(f)["metric"])
    if args.backend:
        base = replace(base, ot_backend=args.backend)
    if args.foote_kernel:
        base = replace(base, mp_window=args.foote_kernel)
    fa_cfg = replace(base, segmentation_method="form_aware")
    print(f"[cfg] distance={base.ot_backend}  kappa={base.contrast_temperature:.3f}")
    margins = [float(x) for x in args.margins.split(",")]
    settings = [(agg, mg) for agg in ("geometric", "mean") for mg in margins]
    items = [it for it in load_manifests(args.manifest, limit=args.limit)
             if all(len(str(l)) == 1 for l in it["labels"])]
    pool = ["".join(it["labels"]) for it in items]
    rng = np.random.default_rng(0)
    conditions = (["annotated"] + (["discovered"] if args.also_discovered else [])
                  + (["form_aware"] if args.also_form_aware else []))

    res = {c: {s: {"own": [], "wrong": [], "pair": [], "all": []} for s in settings}
           for c in conditions}
    t0 = time.time()
    for k, it in enumerate(items, 1):
        E = np.load(it["emb"])
        form = "".join(it["labels"])
        wrongs = wrong_forms(form, args.n_wrong, rng, pool)
        if not wrongs:
            continue
        M.clear_ot_cache()
        for cond in conditions:
            if cond == "form_aware":
                # each form gets its own fitted cuts
                cuts = {x: [s.start for s in M.score_form_adherence(
                            E, x, config=fa_cfg).segments[1:]] for x in [form] + wrongs}
            else:
                b = it["boundaries"] if cond == "annotated" else None
                if b is None:   # discover once, then reuse the same cuts for wrong forms
                    r0 = M.score_form_adherence(E, form, config=base)
                    b = [s.start for s in r0.segments[1:]]
                cuts = {x: b for x in [form] + wrongs}
            for agg, mg in settings:
                cfg = replace(base, aggregate=agg, chance_margin=mg)
                own = M.score_form_adherence(E, form, boundaries=cuts[form], config=cfg).overall
                w = [M.score_form_adherence(E, x, boundaries=cuts[x], config=cfg).overall
                     for x in wrongs]
                d = res[cond][(agg, mg)]
                d["own"].append(own)
                d["wrong"].extend(w)
                d["pair"].extend([float(own > x) + 0.5 * float(own == x) for x in w])
                d["all"].append(float(own > max(w)))
        if k % 10 == 0 or k == len(items):
            print(f"[prog] {k}/{len(items)}  {time.time() - t0:.0f}s", flush=True)

    out = {}
    chance_all = 1.0 / (args.n_wrong + 1)
    for cond in conditions:
        print(f"\n=== {cond} boundaries: {len(res[cond][settings[0]]['own'])} songs, "
              f"{args.n_wrong} wrong forms each ===")
        print(f"  setting                 pairwise  beats_all   own_med  wrong_med")
        print(f"  (chance)                  50.0%     {100 * chance_all:5.1f}%")
        out[cond] = {}
        for s in settings:
            d = res[cond][s]
            row = dict(pairwise=float(np.mean(d["pair"])), beats_all=float(np.mean(d["all"])),
                       own_median=float(np.median(d["own"])),
                       wrong_median=float(np.median(d["wrong"])))
            out[cond][f"{s[0]} margin={s[1]}"] = row
            print(f"  {s[0]:9} margin={s[1]:.2f}    {100 * row['pairwise']:5.1f}%     "
                  f"{100 * row['beats_all']:5.1f}%     {row['own_median']:.3f}     "
                  f"{row['wrong_median']:.3f}")
    if args.save_json:
        os.makedirs(os.path.dirname(args.save_json) or ".", exist_ok=True)
        with open(args.save_json, "w") as f:
            json.dump({"settings": vars(args), "results": out}, f, indent=2)
        print(f"[saved] {args.save_json}")


if __name__ == "__main__":
    main()
