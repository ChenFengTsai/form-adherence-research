#!/usr/bin/env python
"""Which FEATURES separate same-label from different-label sections best?

Replaces diagnose_separation.py and diagnose_chroma.py (same experiments, one
script). For each feature set it extracts embeddings (through the cache, so
re-runs are free), cuts them at the manifest's annotated boundaries, and
reports per-song AUC + pooled gap/spread, per dataset.

    # CLAP+EnCodec vs CLAP vs EnCodec (old "sources" mode)
    python scripts/diagnose_features.py \
        --manifest data/msa/songformbench_manifest.json \
        --audio_dir /storage/ssd1/richtsai1103/form-adherence-research/songformbench/audio \
        --sets "clap,encodec;clap;encodec"

    # CLAP window sweep (old "clap_window_sweep" mode)
    python scripts/diagnose_features.py --manifest ... --audio_dir ... \
        --sets clap --clap_windows 1,4,8

    # chroma+MFCC vs CLAP 4 s (old diagnose_chroma.py)
    python scripts/diagnose_features.py --manifest ... --audio_dir ... \
        --sets "chroma_mfcc;clap" --clap_windows 4

Manifest boundaries are in frames at --frame_rate; every feature set is
extracted at that same frame rate so the boundaries stay valid.
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from formadherence.config import FormAdherenceConfig
from formadherence.audio_io import ExtractorSpec, EmbeddingCache, index_audio_dir, track_id
from formadherence.manifest import load_manifests, by_dataset
from diagnose_formulation import (compute_songs, pooled_separation, song_aucs,
                                  auc_summary)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="comma-separated ok")
    ap.add_argument("--audio_dir", required=True, help="comma-separated ok")
    ap.add_argument("--cache_dir", default="cache/embeddings")
    ap.add_argument("--sets", default="clap,encodec;clap;encodec",
                    help="feature sets separated by ';', sources by ','")
    ap.add_argument("--clap_windows", default="1", help="e.g. 1,4,8 (hop = w/2)")
    ap.add_argument("--encodec_pool", type=int, default=8)
    ap.add_argument("--frame_rate", type=float, default=2.0)
    ap.add_argument("--ot_backend", default="sinkhorn", choices=["exact", "sinkhorn"])
    ap.add_argument("--limit", type=int, default=40, help="per manifest; 0 = all")
    ap.add_argument("--save_json", default="")
    args = ap.parse_args()

    cfg = FormAdherenceConfig(ot_backend=args.ot_backend)
    items = load_manifests(args.manifest, limit=0, require_emb=False)
    audio = {}
    for d in args.audio_dir.split(","):
        audio.update(index_audio_dir(d.strip()))

    specs = []
    for s in args.sets.split(";"):
        sources = tuple(x.strip() for x in s.split(",") if x.strip())
        wins = [float(w) for w in args.clap_windows.split(",")] if "clap" in sources else [1.0]
        for w in wins:
            specs.append(ExtractorSpec(sources=sources, clap_window_s=w, clap_hop_s=w / 2,
                                       encodec_pool=args.encodec_pool,
                                       frame_rate=args.frame_rate))

    results = {}
    for spec in specs:
        cache = EmbeddingCache(args.cache_dir, spec)
        print(f"\n=== {spec.describe()} ===")
        for name, grp in by_dataset(items).items():
            songs, missing = [], 0
            for it in grp:
                p = audio.get(track_id(it) or "")
                if p is None:
                    missing += 1
                    continue
                try:
                    emb, _ = cache.get(p)
                except Exception as e:
                    print(f"  [warn] {track_id(it)}: {type(e).__name__}: {e}")
                    continue
                songs.append((emb, it["boundaries"], it["labels"]))
                if args.limit and len(songs) >= args.limit:
                    break
            if missing:
                print(f"  [{name}] {missing} manifest ids had no audio file")
            computed = compute_songs(songs, "ot", cfg, "none", "cosine")
            sep, s = pooled_separation(computed), auc_summary(song_aucs(computed))
            results[f"{spec.describe()} | {name}"] = {"pooled": sep, "auc": s}
            print(f"  [{name}] songs={len(songs)}  "
                  + (f"AUC median={s['median']:.3f} ({s['n_above_half']}/{s['songs']} > 0.5)  "
                     if s else "AUC n/a  ")
                  + (f"gap/spread={sep['gap_over_spread']:.2f}" if sep else ""))

    if args.save_json:
        os.makedirs(os.path.dirname(args.save_json) or ".", exist_ok=True)
        with open(args.save_json, "w") as f:
            json.dump({"settings": vars(args), "results": results}, f, indent=2)
        print(f"[saved] {args.save_json}")


if __name__ == "__main__":
    main()
