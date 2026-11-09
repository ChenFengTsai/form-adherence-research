#!/usr/bin/env python
"""Filter a manifest by genre (SALAMI via mirdata, or an entry's own `genre` field).

    # list what's there
    python scripts/filter_manifest_by_genre.py --manifest data/msa/salami_manifest.json --list
    # keep one genre
    python scripts/filter_manifest_by_genre.py --manifest data/msa/salami_manifest.json \
        --genre Popular --out data/msa/salami_popular.json

Fix: the old version fell back to SALAMI's `source` field (which archive the
track came from: IA, Codaich, RWC, Isophonics) and treated it as a genre. Use
--by source if you really want to filter by source.
"""
import argparse
import json
from collections import Counter


def salami_field(track, by):
    attrs = ("genre", "class") if by == "genre" else ("source",)
    for attr in attrs:
        v = getattr(track, attr, None)
        if v:
            return str(v)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--data_home", default="data/msa/salami")
    ap.add_argument("--by", choices=["genre", "source"], default="genre")
    ap.add_argument("--genre", default=None, help="value to keep (case-insensitive substring)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    with open(args.manifest) as f:
        manifest = json.load(f)

    salami = None
    tagged, counts = [], Counter()
    for e in manifest:
        if args.by == "genre" and e.get("genre"):
            g = e["genre"]                                  # entries that carry a genre field
        elif e.get("salami_id") is not None:
            if salami is None:
                import mirdata
                salami = mirdata.initialize("salami", data_home=args.data_home)
            try:
                g = salami_field(salami.track(str(e["salami_id"])), args.by)
            except Exception:
                g = None
        else:
            g = None
        counts[g] += 1
        tagged.append((e, g))

    if args.list or not args.genre:
        print(f"{args.by} values in manifest (count):")
        for g, c in counts.most_common():
            print(f"  {g!r}: {c}")
        if not args.genre:
            return

    want = args.genre.lower()
    kept = [e for e, g in tagged if g and want in g.lower()]
    out = args.out or args.manifest.replace(".json", f"_{args.genre}.json")
    with open(out, "w") as f:
        json.dump(kept, f, indent=2)
    print(f"kept {len(kept)}/{len(manifest)} with {args.by}~{args.genre!r} -> {out}")


if __name__ == "__main__":
    main()
