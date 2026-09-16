#!/usr/bin/env python
"""Download SALAMI's Internet-Archive audio using the official `internetarchive`
library, which handles redirects, retries, and IA's current API far better than
raw urllib (the old http://www.archive.org/download/... links 403/redirect now).

The SALAMI id_index_internetarchive.csv URL column is already a direct media
link, e.g.
    http://www.archive.org/download/<IDENTIFIER>/<FILE>
so we split it into (identifier, file) and let `internetarchive` fetch that one
file from that item. Files are saved as <SONG_ID>.<ext> to match the manifest
builder.

    pip install internetarchive
    python scripts/download_salami_ia_ialib.py \
        --map data/msa/salami-data-public/metadata/id_index_internetarchive.csv \
        --out_dir data/msa/audio \
        --limit 5

Notes:
  * Many LMA items still exist; some were removed (404) or are gated. Misses are
    logged, not fatal -- you keep whatever you can get.
  * Respectful defaults: one file at a time, a short sleep between items.
"""

import argparse
import csv
import os
import time
from urllib.parse import urlparse, unquote


def parse_identifier_and_file(url: str):
    """From .../download/<identifier>/<file...> return (identifier, file).
    Returns (None, None) if the URL isn't an IA download link."""
    path = urlparse(url).path            # /download/<id>/<file...>
    parts = path.split("/download/", 1)
    if len(parts) != 2 or not parts[1]:
        return None, None
    rest = parts[1]
    ident, _, fname = rest.partition("/")
    if not ident or not fname:
        return None, None
    return unquote(ident), unquote(fname)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", required=True)
    ap.add_argument("--out_dir", default="data/msa/audio")
    ap.add_argument("--id_col", default="SONG_ID")
    ap.add_argument("--url_col", default="URL")
    ap.add_argument("--sleep", type=float, default=1.0)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    from internetarchive import download as ia_download, get_item

    os.makedirs(args.out_dir, exist_ok=True)
    with open(args.map, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        header = reader.fieldnames or []
    print(f"[map]  {len(rows)} rows   [cols] {header}")
    if args.id_col not in header or args.url_col not in header:
        raise SystemExit(f"need --id_col/--url_col present in {header}")

    def resolve_filename(ident, wanted):
        """The CSV filename (e.g. ..d1t03_vbr.mp3) may no longer exist; IA has
        re-derived files (..d1t03.mp3). Match on the stem, ignoring a trailing
        _vbr and the extension, and prefer an mp3."""
        item = get_item(ident)
        if not item.exists:
            return None, "item does not exist"
        names = [f.name for f in item.get_files()]
        if wanted in names:
            return wanted, None
        base = os.path.basename(wanted)
        stem = base.rsplit(".", 1)[0]
        # strip common derivative markers
        for suf in ("_vbr", "_64kb", "_128kbps", "_flac"):
            if stem.endswith(suf):
                stem = stem[: -len(suf)]
        # candidates whose (suffix-stripped) stem matches, mp3 preferred
        def norm(n):
            s = os.path.basename(n).rsplit(".", 1)[0]
            for suf in ("_vbr", "_64kb", "_128kbps", "_flac"):
                if s.endswith(suf):
                    s = s[: -len(suf)]
            return s
        cands = [n for n in names if norm(n) == stem]
        mp3s = [n for n in cands if n.lower().endswith(".mp3")]
        if mp3s:
            return mp3s[0], None
        if cands:
            return cands[0], None
        return None, f"no file matching stem {stem!r} in {len(names)} files"

    ok, skip, miss = 0, 0, 0
    for row in rows:
        sid = row.get(args.id_col, "").strip()
        url = row.get(args.url_col, "").strip()
        if not sid or not url:
            continue

        ident, fname = parse_identifier_and_file(url)
        if not ident:
            miss += 1
            print(f"[miss] {sid}: cannot parse identifier from {url}")
            continue

        # Resolve the CSV filename to a file that actually exists in the item.
        try:
            real_fname, err = resolve_filename(ident, fname)
        except Exception as e:
            real_fname, err = None, f"{type(e).__name__}: {e}"
        if real_fname is None:
            miss += 1
            print(f"[miss] {sid}: {ident}: {err}")
            time.sleep(args.sleep)
            continue
        fname = real_fname

        final = os.path.join(args.out_dir, f"{sid}{os.path.splitext(fname)[1]}")
        if os.path.exists(final) and os.path.getsize(final) > 0:
            skip += 1
            print(f"[skip] {sid} (already have it)")
            continue

        try:
            try:
                r = ia_download(ident, files=[fname], destdir=args.out_dir,
                                verbose=False, ignore_existing=True, retries=3)
            except TypeError:
                r = ia_download(ident, files=[fname], destdir=args.out_dir)
            # IA preserves the item path: out_dir/<ident>/<fname> (fname may
            # itself contain a subdir). Find whatever actually landed.
            expected = os.path.join(args.out_dir, ident, fname)
            found = expected if os.path.exists(expected) else None
            if found is None:
                # fall back: scan the item dir for the basename
                item_dir = os.path.join(args.out_dir, ident)
                base = os.path.basename(fname)
                for root, _dirs, files in os.walk(item_dir):
                    if base in files:
                        found = os.path.join(root, base)
                        break
            if found and os.path.getsize(found) > 0:
                os.replace(found, final)
                ok += 1
                print(f"[ok]   {sid} <- {ident}/{fname}")
            else:
                miss += 1
                print(f"[miss] {sid}: {ident}/{fname} not retrieved "
                      f"(looked for {expected})")
        except Exception as e:
            miss += 1
            print(f"[miss] {sid}: {type(e).__name__}: {e}")
        finally:
            # Always remove the IA item folder, whether the download succeeded,
            # failed, or left a partial -- so no <identifier>/ dirs pile up.
            item_dir = os.path.join(args.out_dir, ident)
            if os.path.isdir(item_dir):
                import shutil
                shutil.rmtree(item_dir, ignore_errors=True)

        time.sleep(args.sleep)
        if args.limit and ok >= args.limit:
            print(f"[stop] hit --limit {args.limit}")
            break

    print(f"\n[done] ok={ok} skip={skip} miss={miss} -> {args.out_dir}")


if __name__ == "__main__":
    main()