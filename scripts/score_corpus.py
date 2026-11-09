#!/usr/bin/env python
"""Score a corpus of generated songs against their target forms, plus the
first validity check for the metric on generated audio.

corpus.json: [{"audio": path | "emb": path.npy, "target_form": "ABACA",
               "model": "YuE", "boundaries": [...] (optional), ...}, ...]
  If an entry has "boundaries" they are used (pass --discover to ignore them);
  generated songs have none, so their cuts are discovered.

    python scripts/score_corpus.py --corpus outputs/corpus/corpus.json \
        --config configs/eval_sfb.json --out_csv results/corpus_scores.csv

MISMATCH CHECK (--n_wrong K, default 3): each song is also scored against K
WRONG forms of the same length (same boundaries, different same/different
pattern). If the metric measures adherence, a song should score higher on its
own form than on wrong ones. Reported: own vs wrong mean, and the share of
songs where own > every wrong form. Near 50% / no gap = the metric isn't
seeing the requested form in this audio.
"""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from formadherence.config import FormAdherenceConfig
from formadherence.metric import score_form_adherence
from formadherence.audio_io import EmbeddingCache, add_extractor_args, spec_from_args


def pattern(form):
    m = {}
    return "".join(m.setdefault(c, chr(ord("A") + len(m))) for c in form)


def wrong_forms(form, k, rng, pool=()):
    """Up to k forms of the same length with a different repetition pattern:
    other corpus forms first, then single-letter edits."""
    own = pattern(form)
    cands = {pattern(f) for f in pool if len(f) == len(form)} - {own}
    letters = sorted(set(form)) + [chr(ord("A") + len(set(form)))]
    for i in range(len(form)):
        for c in letters:
            f = form[:i] + c + form[i + 1:]
            if pattern(f) != own:
                cands.add(pattern(f))
    cands = sorted(cands)
    rng.shuffle(cands)
    return cands[:k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--config", default="configs/eval_sfb.json")
    ap.add_argument("--cache_dir", default="cache/embeddings")
    ap.add_argument("--out_csv", default="results/corpus_scores.csv")
    ap.add_argument("--n_wrong", type=int, default=3)
    ap.add_argument("--discover", action="store_true",
                    help="ignore entries' annotated boundaries and discover cuts")
    add_extractor_args(ap)
    args = ap.parse_args()

    if os.path.exists(args.config):
        with open(args.config) as f:
            cfg = FormAdherenceConfig.from_dict(json.load(f)["metric"])
        print(f"[cfg] {args.config} (score_mode={cfg.score_mode})")
    else:
        cfg = FormAdherenceConfig()
        print("[cfg] defaults (run calibrate.py first for real numbers)")

    with open(args.corpus) as f:
        corpus = json.load(f)
    cache = None
    rng = np.random.default_rng(0)
    pool = [c["target_form"] for c in corpus]
    rows, own_vs_wrong = [], []
    for c in corpus:
        try:
            if c.get("emb"):
                emb = np.load(c["emb"])
            else:
                if cache is None:
                    cache = EmbeddingCache(args.cache_dir, spec_from_args(args),
                                           reuse_legacy=args.reuse_legacy_cache)
                emb, _ = cache.get(c["audio"])
            b0 = None if args.discover else c.get("boundaries")
            chroma = None
            if c.get("audio") and cfg.variation_method == "chroma_dtw":
                from formadherence.audio_io import read_audio
                from formadherence.variation import chroma_features
                wav, sr = read_audio(c["audio"])
                chroma = chroma_features(wav, sr, fps=cfg.chroma_fps, method=cfg.chroma_method)
            r = score_form_adherence(emb, c["target_form"], boundaries=b0, config=cfg,
                                     chroma=chroma)
        except Exception as e:
            print(f"[fail] {c.get('audio') or c.get('emb')}: {type(e).__name__}: {e}")
            continue
        row = {"file": c.get("audio") or c.get("emb"), "model": c.get("model", ""),
               "form": c["target_form"], "overall": r.overall,
               "distinctness": r.distinctness, "similarity": r.similarity,
               "variation": r.variation, "auc": r.auc, "contrast": r.contrast,
               "segmentation_fallback": r.segmentation_fallback,
               "variation_method": r.variation_method}
        if args.n_wrong:
            # reuse the same cuts so only the requested pattern changes
            b = [s.start for s in r.segments[1:]]
            wrong = [score_form_adherence(emb, w, boundaries=b, config=cfg, chroma=chroma).overall
                     for w in wrong_forms(c["target_form"], args.n_wrong, rng, pool)]
            if wrong:
                row["wrong_mean"] = float(np.mean(wrong))
                row["own_beats_all_wrong"] = bool(r.overall > max(wrong))
                own_vs_wrong.append((r.overall, float(np.mean(wrong)), r.overall > max(wrong)))
        rows.append(row)
        print(f"[ok]   {row['form']:10} overall={r.overall:.3f} D={r.distinctness:.3f} "
              f"S={r.similarity:.3f} V={r.variation:.3f}"
              + (f"  wrong={row['wrong_mean']:.3f}" if "wrong_mean" in row else "")
              + ("  [fallback cuts]" if r.segmentation_fallback else ""))

    if not rows:
        raise SystemExit("nothing scored")
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    keys = sorted({k for r in rows for k in r})
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    print(f"\n== {len(rows)} songs -> {args.out_csv} ==")
    groups = defaultdict(list)
    for r in rows:
        groups[(r["model"], r["form"])].append(r)
    for (model, form), g in sorted(groups.items()):
        print(f"   {model or '-':8} {form:10} n={len(g):3}  overall median="
              f"{np.median([x['overall'] for x in g]):.3f}  fallback="
              f"{np.mean([x['segmentation_fallback'] for x in g]):.0%}")
    if own_vs_wrong:
        o = np.array(own_vs_wrong, dtype=float)
        print(f"\nMISMATCH CHECK: own-form mean={o[:, 0].mean():.3f}  "
              f"wrong-form mean={o[:, 1].mean():.3f}  own beats every wrong form "
              f"in {o[:, 2].mean():.0%} of songs (chance {1 / (args.n_wrong + 1):.0%}); "
              f"own > mean of wrong in {(o[:, 0] > o[:, 1]).mean():.0%} (chance 50%)")


if __name__ == "__main__":
    main()
