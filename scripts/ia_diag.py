#!/usr/bin/env python
"""Diagnostic: try to download ONE known SALAMI IA file with full output and no
wrapper logic, and report whether internetarchive is authenticated. Run this to
see the real reason downloads aren't landing.

    python scripts/ia_diag.py
"""

import os

IDENT = "brdnhand2005-07-27.shnf"
FNAME = "brdnhand2005-07-27d1t03_vbr.mp3"
OUT = "data/msa/audio_diag"


def main():
    import internetarchive as ia
    print("internetarchive version:", getattr(ia, "__version__", "unknown"))

    # 1) Is there a configured session (credentials)?
    try:
        s = ia.get_session()
        cfg = getattr(s, "config", {}) or {}
        has_cookies = bool(cfg.get("cookies"))
        has_s3 = bool(cfg.get("s3"))
        print(f"[auth] session config present: cookies={has_cookies} s3={has_s3}")
    except Exception as e:
        print(f"[auth] could not build session: {e}")

    # 2) Does the item exist and list this file?
    try:
        item = ia.get_item(IDENT)
        print(f"[item] {IDENT} exists={item.exists}")
        names = [f.name for f in item.get_files()]
        print(f"[item] {len(names)} files; target present: {FNAME in names}")
        # show a few mp3s actually in the item (derivative names may differ!)
        mp3s = [n for n in names if n.lower().endswith('.mp3')][:10]
        print("[item] sample mp3 files in item:")
        for n in mp3s:
            print("       ", n)
    except Exception as e:
        print(f"[item] get_item failed: {type(e).__name__}: {e}")
        return

    # 3) Actually download, fully verbose, no ignore_existing.
    os.makedirs(OUT, exist_ok=True)
    print("[dl]   attempting download (verbose)...")
    try:
        r = ia.download(IDENT, files=[FNAME], destdir=OUT, verbose=True,
                        ignore_existing=False)
        print("[dl]   download() returned:", r)
    except Exception as e:
        print(f"[dl]   download failed: {type(e).__name__}: {e}")

    # 4) What actually landed on disk?
    print("[disk] tree under", OUT)
    for root, _d, files in os.walk(OUT):
        for f in files:
            p = os.path.join(root, f)
            print("       ", p, os.path.getsize(p), "bytes")


if __name__ == "__main__":
    main()