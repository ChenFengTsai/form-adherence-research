#!/usr/bin/env python
"""Filter an existing manifest to a single SALAMI genre/class.

SALAMI tags each track with a broad class (e.g. Classical, Jazz, Popular, World,
Live...). This reads that field from mirdata for each manifest entry's salami_id
and keeps only the tracks matching --genre, writing a new manifest.

    # first, find the exact genre strings present:
    python scripts/filter_manifest_by_genre.py \
        --manifest data/msa/manifest.json --data_home data/msa/salami --list

    # then filter to one:
    python scripts/filter_manifest_by_genre.py \
        --manifest data/msa/manifest.json --data_home data/msa/salami \
        --genre Popular --out data/msa/manifest_popular.json
"""
import argparse, json
from collections import Counter


def get_genre(track):
    # SALAMI's class/genre field name varies; try the common ones.
    for attr in ("genre", "class", "source", "annotator_1_genre"):
        v = getattr(track, attr, None)
        if v:
            return str(v)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--data_home", default="data/msa/salami")
    ap.add_argument("--genre", default=None,
                    help="genre/class to keep (case-insensitive substring)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--list", action="store_true",
                    help="just list the genres present and their counts")
    args = ap.parse_args()

    import mirdata
    s = mirdata.initialize("salami", data_home=args.data_home)

    with open(args.manifest) as f:
        manifest = json.load(f)

    # attach genre to each entry
    tagged = []
    counts = Counter()
    for e in manifest:
        sid = e.get("salami_id")
        if sid is None:
            continue
        try:
            g = get_genre(s.track(str(sid)))
        except Exception:
            g = None
        counts[g] += 1
        tagged.append((e, g))

    if args.list or not args.genre:
        print("genres present in manifest (count):")
        for g, c in counts.most_common():
            print(f"  {g!r}: {c}")
        if not args.genre:
            print("\nPass --genre <name> --out <path> to write a filtered manifest.")
            return

    want = args.genre.lower()
    kept = [e for (e, g) in tagged if g and want in g.lower()]
    out = args.out or args.manifest.replace(".json", f"_{args.genre}.json")
    with open(out, "w") as f:
        json.dump(kept, f, indent=2)
    print(f"kept {len(kept)}/{len(manifest)} tracks with genre~{args.genre!r} "
          f"-> {out}")


if __name__ == "__main__":
    main()