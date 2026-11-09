#!/usr/bin/env python
"""Build a manifest from downloaded SALAMI audio + annotations.

For each SALAMI track with BOTH audio and a sections annotation:
  1. get per-frame embeddings (cached by audio content AND extractor settings),
  2. convert second-based section starts to frame boundaries,
  3. map labels to letters (uppercase A/B/A' kept as letters, primes recorded
     as `variants`; functional labels folded into families),
  4. write {emb, boundaries, labels, labels_raw, variants, salami_id, ...}.

    python scripts/build_salami_manifest.py \
        --data_home data/msa/salami \
        --audio_dir data/msa/audio \
        --cache_dir cache/embeddings \
        --out data/msa/salami_manifest.json

NOTE: SALAMI's Internet-Archive audio is the Live Music Archive -- live concert
recordings (applause, long jams), not studio pop. Report it as such, and compare
with SongFormBench (studio pop) rather than pooling the two.

Extract these embeddings with the SAME extractor settings you evaluate with.
The cache key now includes those settings, so a mismatch can't sneak in.
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from formadherence.audio_io import (EmbeddingCache, add_extractor_args,
                                    spec_from_args, index_audio_dir)
from formadherence.labels import annotation_to_manifest_fields, DEFAULT_SPECIAL_LETTERS

SECTION_ATTRS = ("sections_uppercase", "sections_annotator_1_uppercase",
                 "sections_annotator_2_uppercase", "sections_lowercase",
                 "sections_annotator_1_lowercase", "sections_annotator_2_lowercase",
                 "sections")


def flatten_sections(cand):
    """mirdata returns either a flat SectionData (.intervals) or a
    MultiAnnotator wrapper (.annotations). Return the first usable one."""
    if cand is None:
        return None
    if getattr(cand, "intervals", None) is not None:
        return cand
    for a in getattr(cand, "annotations", None) or []:
        if a is not None and getattr(a, "intervals", None) is not None:
            return a
    return None


def get_sections(track, prefer):
    attrs = SECTION_ATTRS if prefer == "uppercase" else \
        tuple(a for a in SECTION_ATTRS if "lowercase" in a) + SECTION_ATTRS
    for attr in attrs:
        s = flatten_sections(getattr(track, attr, None))
        if s is not None:
            return s, attr
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_home", default="data/msa/salami")
    ap.add_argument("--audio_dir", default="data/msa/audio")
    ap.add_argument("--cache_dir", default="cache/embeddings")
    ap.add_argument("--out", default="data/msa/salami_manifest.json")
    ap.add_argument("--labels", choices=["uppercase", "lowercase"], default="uppercase",
                    help="uppercase = structural letters (A/B/A'); "
                         "lowercase = functional (verse/chorus)")
    ap.add_argument("--special", choices=["unique", "keep", "drop"], default="unique",
                    help="Silence/End/applause segments: own letter each (unique, "
                         "default), grouped (keep, old behaviour), or merged away")
    ap.add_argument("--z_is_section", action="store_true",
                    help="treat SALAMI letter Z as a real section (default: special)")
    ap.add_argument("--min_segments", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None)
    add_extractor_args(ap)
    args = ap.parse_args()

    import mirdata
    salami = mirdata.initialize("salami", data_home=args.data_home)
    tracks = salami.load_tracks()

    spec = spec_from_args(args)
    cache = EmbeddingCache(args.cache_dir, spec, reuse_legacy=args.reuse_legacy_cache)
    special_letters = () if args.z_is_section else DEFAULT_SPECIAL_LETTERS
    print(f"[extractor] {spec.describe()}  (cache tag {spec.tag()})")

    audio_index = index_audio_dir(args.audio_dir)
    print(f"[audio] indexed {len(audio_index)} files in {args.audio_dir}")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    manifest, n_seen, n_ok = [], 0, 0
    for tid, track in tracks.items():
        if args.limit and n_seen >= args.limit:
            break
        audio_path = audio_index.get(str(tid))
        if audio_path is None:
            continue
        sections, attr = get_sections(track, args.labels)
        if sections is None:
            continue
        n_seen += 1
        intervals = np.asarray(sections.intervals, dtype=float)
        labels = list(sections.labels)
        if len(intervals) < args.min_segments:
            continue
        try:
            emb, emb_path = cache.get(audio_path)
            fields = annotation_to_manifest_fields(
                intervals[:, 0].tolist(), float(intervals[-1, 1]), labels,
                n_frames=len(emb), frame_rate=spec.frame_rate,
                special=args.special, special_letters=special_letters)
            if fields is None or len(fields["labels"]) < args.min_segments:
                continue
            manifest.append({
                "emb": os.path.abspath(emb_path),
                **fields,
                "salami_id": str(tid),
                "dataset": "salami",
                "annotation": attr,
                "extractor": spec.describe(),
                "extractor_tag": spec.tag(),
            })
            n_ok += 1
            print(f"[ok]   {tid}: {len(fields['labels'])} segs -> "
                  f"{''.join(l if len(l) == 1 else '?' for l in fields['labels'])}")
        except Exception as e:
            print(f"[fail] {tid}: {type(e).__name__}: {e}")

    with open(args.out, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n[done] wrote {n_ok} entries (of {n_seen} with audio+annot) to {args.out}")


if __name__ == "__main__":
    main()
