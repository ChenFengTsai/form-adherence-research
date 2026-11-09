#!/usr/bin/env python
"""Extract per-frame embeddings for every audio file in a directory and cache
them as .npy, keyed by audio content AND extractor settings.

Run on the GPU box in the `fa` env (torch, transformers, encodec, soundfile).

    python scripts/extract_embeddings.py --audio_dir outputs/wavs \
        --cache_dir cache/embeddings

Writes cache/embeddings/<audiohash>_<spectag>.npy plus a .json sidecar saying
which audio file and settings produced it, and an index.csv mapping file name
-> .npy so you can find a song's embeddings later (run_evaluation.py --emb).

FIRST-RUN SMOKE TEST (do this before trusting anything downstream):
  * shape: T ~= duration_seconds * frame_rate, d = CLAP_dim + EnCodec_dim
  * block balance: if CLAP and EnCodec parts have very different sizes, cosine
    distance only "sees" the bigger block. Fix in ConcatExtractor (L2-normalize
    each block before concatenating), then bump --extractor_version.
"""

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from formadherence.audio_io import (EmbeddingCache, add_extractor_args,
                                    spec_from_args, index_audio_dir,
                                    block_norm_report)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio_dir", required=True)
    ap.add_argument("--cache_dir", default="cache/embeddings")
    ap.add_argument("--clap_dim", type=int, default=512,
                    help="width of the CLAP block, for the balance check")
    add_extractor_args(ap)
    args = ap.parse_args()

    spec = spec_from_args(args)
    cache = EmbeddingCache(args.cache_dir, spec, reuse_legacy=args.reuse_legacy_cache)
    print(f"[extractor] {spec.describe()}  (cache tag {spec.tag()})")

    paths = sorted(index_audio_dir(args.audio_dir).values())
    if not paths:
        raise SystemExit(f"no audio files in {args.audio_dir}")

    index_rows = []
    for i, path in enumerate(paths):
        existed = os.path.exists(cache.path_for(path))
        emb, out = cache.get(path)
        index_rows.append((os.path.basename(path), out))
        print(f"[{'skip' if existed else 'ok  '}] {os.path.basename(path)} -> "
              f"{os.path.basename(out)}  shape={emb.shape}")
        if i == 0:
            print(f"       SMOKE TEST: T={emb.shape[0]} (expect ~duration*"
                  f"{spec.frame_rate}), d={emb.shape[1]}")
            dims = getattr(cache._extractor, "block_dims", None) if cache._extractor else None
            if dims and len(dims) > 1:
                print(f"       blocks: {dict(zip(spec.sources, dims))}")
                print("       " + block_norm_report(emb, dims[0]))
            elif set(spec.sources) == {"clap", "encodec"} and emb.shape[1] > args.clap_dim:
                print("       " + block_norm_report(emb, args.clap_dim))

    idx_path = os.path.join(args.cache_dir, f"index_{spec.tag()}.csv")
    with open(idx_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["audio_file", "emb"])
        w.writerows(index_rows)
    print(f"[index] {idx_path}")


if __name__ == "__main__":
    main()
