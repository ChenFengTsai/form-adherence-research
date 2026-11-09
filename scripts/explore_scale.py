#!/usr/bin/env python
"""Why do human songs score low? Score once, then re-summarize under different
scale settings (chance_margin x aggregate) without recomputing OT.

    python scripts/explore_scale.py --manifest data/msa/songformbench_manifest.json \
        --config configs/eval_sfb.json --limit 100

Prints:
  * median distinctness / similarity / overall for each (margin, aggregate)
  * which annotated section types (labels_raw) have the weakest repeats and
    which letter pairs are most confused -- e.g. intro vs outro
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from formadherence.config import FormAdherenceConfig
from formadherence.metric import score_form_adherence
from formadherence.manifest import load_manifests


def rescale(x, margin):
    lo = 0.5 + margin
    return float(np.clip((x - lo) / (1 - lo), 0, 1))


def agg(vals, how, floor):
    v = np.array([max(x, floor) for x in vals]) if vals else np.array([1.0])
    return float(v.mean()) if how == "mean" else float(np.exp(np.log(v).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--config", default="configs/eval_sfb.json")
    ap.add_argument("--limit", type=int, default=100, help="0 = all")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = FormAdherenceConfig.from_dict(json.load(f)["metric"])
    cfg.chance_to_zero = False                  # keep raw 0.5-chance values
    items = load_manifests(args.manifest, limit=args.limit)
    print(f"[cfg] {args.config}: kappa={cfg.contrast_temperature:.3f} "
          f"margin={cfg.chance_margin} aggregate={cfg.aggregate}")

    songs = []
    t0 = time.time()
    for k, it in enumerate(items, 1):
        r = score_form_adherence(np.load(it["emb"]), it["labels"],
                                 boundaries=it["boundaries"], config=cfg)
        # letter -> most common raw label, for readable diagnostics
        fam = defaultdict(list)
        for L, raw in zip(it["labels"], it.get("labels_raw", it["labels"])):
            fam[L].append(str(raw).lower().rstrip("0123456789"))
        name = {L: max(set(v), key=v.count) for L, v in fam.items()}
        songs.append((r, name))
        if k % 10 == 0 or k == len(items):
            print(f"[prog] {k}/{len(items)}  {time.time() - t0:.0f}s", flush=True)

    nl = [len(r.per_letter_similarity) for r, _ in songs]
    npair = [len(r.per_letter_pair_distinctness) for r, _ in songs]
    print(f"\nrepeated letters per song: median {np.median(nl):.0f}; "
          f"letter pairs per song: median {np.median(npair):.0f}")
    allsim = [v for r, _ in songs for v in r.per_letter_similarity.values()]
    allpair = [v for r, _ in songs for v in r.per_letter_pair_distinctness.values()]
    print(f"raw per-letter similarity:   median {np.median(allsim):.3f}  "
          f"(25% {np.quantile(allsim, .25):.3f}, 75% {np.quantile(allsim, .75):.3f})")
    print(f"raw per-letter-pair distinct: median {np.median(allpair):.3f}  "
          f"(25% {np.quantile(allpair, .25):.3f}, 75% {np.quantile(allpair, .75):.3f})")
    print("  (raw: 0.5 = chance, 1 = perfect)")

    print("\nmedian over songs        distinct  similar  overall")
    for how in ("geometric", "mean"):
        for margin in (0.0, 0.05, 0.1, 0.15):
            D, S, O = [], [], []
            for r, _ in songs:
                d = agg([rescale(v, margin) for v in r.per_letter_pair_distinctness.values()],
                        how, cfg.comparison_floor) if r.per_letter_pair_distinctness else 1.0
                s = agg([rescale(v, margin) for v in r.per_letter_similarity.values()],
                        how, cfg.comparison_floor) if r.per_letter_similarity else 1.0
                o = float(np.exp(np.mean(np.log(np.array([d, s, r.variation]) + 1e-12))))
                D.append(d); S.append(s); O.append(o)
            print(f"  {how:9} margin={margin:.2f}   {np.median(D):.3f}    "
                  f"{np.median(S):.3f}    {np.median(O):.3f}")

    sim_by = defaultdict(list)
    pair_by = defaultdict(list)
    for r, name in songs:
        for L, v in r.per_letter_similarity.items():
            sim_by[name.get(L, L)].append(v)
        for k, v in r.per_letter_pair_distinctness.items():
            a, b = k.split("-")
            pair_by[" vs ".join(sorted((name.get(a, a), name.get(b, b))))].append(v)
    print("\nsection types by repeat coherence (raw similarity, n>=10):")
    for k, v in sorted(sim_by.items(), key=lambda kv: np.median(kv[1])):
        if len(v) >= 10:
            print(f"  {k:14} median {np.median(v):.3f}  n={len(v)}")
    print("\nmost confused section-type pairs (raw distinctness, n>=10):")
    rows = [(k, v) for k, v in pair_by.items() if len(v) >= 10]
    for k, v in sorted(rows, key=lambda kv: np.median(kv[1]))[:8]:
        print(f"  {k:28} median {np.median(v):.3f}  n={len(v)}")


if __name__ == "__main__":
    main()
