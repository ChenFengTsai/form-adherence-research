#!/usr/bin/env python
"""Download Harmonix Set audio from YouTube with yt-dlp, styled like the SALAMI
IA downloader.

Reads dataset/youtube_urls.csv (columns: File, URL), downloads each track's
audio, and saves it as <File>.<ext> in --out_dir so build_harmonix_manifest.py's
audio_index picks it up by stem. Optionally filters by the DTW alignment scores
in youtube_alignment_scores.csv so you skip YouTube versions that don't match the
annotated audio (misaligned audio silently corrupts your segment boundaries).

    pip install yt-dlp          # and have ffmpeg on PATH for -x extraction
    python download_harmonix_youtube.py \
        --urls    data/msa/harmonixset/dataset/youtube_urls.csv \
        --out_dir data/msa/harmonix/audio \
        --align   data/msa/harmonixset/dataset/youtube_alignment_scores.csv \
        --min_alignment 0.9 \
        --limit 5

Notes:
  * Some YouTube videos are removed/region-locked/age-gated -> those are logged
    as misses, not fatal; you keep whatever you can get (same as the SALAMI IA
    downloader).
  * Alignment polarity is NOT documented: it may be a similarity (higher =
    closer) or a DTW cost (lower = closer). Inspect the CSV, then use
    --min_alignment (keep score >= X) and/or --max_alignment (keep score <= X)
    to match whichever direction means "well aligned" in your copy.
  * Downloading from YouTube may conflict with YouTube's Terms of Service; make
    sure your use (e.g. non-distributed academic research) is actually permitted
    where you are before running this.
"""

import argparse
import csv
import os
import shutil
import subprocess
import time


def load_alignment_scores(path):
    """Map File -> float score, defensively (column names/polarity vary)."""
    scores = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        if not cols:
            return scores
        file_col = next((c for c in cols if c.strip().lower() in
                         ("file", "filename", "id", "track", "track_id")), cols[0])
        score_col = next((c for c in cols if "align" in c.lower()
                          or "score" in c.lower() or "dtw" in c.lower()), None)
        for row in reader:
            fid = (row.get(file_col) or "").strip()
            if not fid:
                continue
            val = None
            if score_col is not None:
                try:
                    val = float(row[score_col])
                except (TypeError, ValueError):
                    val = None
            scores[fid] = val
    return scores


def already_have(out_dir, sid):
    """Return an existing non-empty <sid>.<audio-ext> path, else None."""
    for ext in (".mp3", ".m4a", ".opus", ".wav", ".flac", ".webm"):
        p = os.path.join(out_dir, sid + ext)
        if os.path.exists(p) and os.path.getsize(p) > 0:
            return p
    return None


def find_output(out_dir, sid):
    """After yt-dlp runs, find whatever <sid>.* audio file landed."""
    for f in os.listdir(out_dir):
        stem, ext = os.path.splitext(f)
        if stem == sid and ext.lower() in (
                ".mp3", ".m4a", ".opus", ".wav", ".flac", ".webm"):
            p = os.path.join(out_dir, f)
            if os.path.getsize(p) > 0:
                return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urls", required=True, help="youtube_urls.csv (File, URL)")
    # Audio is heavy -> lives on the SSD. out_dir defaults to None and is derived
    # from --data_root (matches build_harmonix_manifest.py's default audio_dir).
    ap.add_argument("--data_root",
                    default="/storage/ssd1/richtsai1103/form-adherence-research",
                    help="SSD root for heavy files")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--id_col", default="File")
    ap.add_argument("--url_col", default="URL")
    ap.add_argument("--audio_format", default="mp3",
                    help="yt-dlp -x target format (mp3, wav, m4a, ...)")
    ap.add_argument("--audio_quality", default="128K")
    ap.add_argument("--sleep", type=float, default=2.0)
    ap.add_argument("--limit", type=int, default=0)
    # optional alignment gate
    ap.add_argument("--align", default="", help="youtube_alignment_scores.csv")
    ap.add_argument("--min_alignment", type=float, default=None,
                    help="keep tracks with score >= this")
    ap.add_argument("--max_alignment", type=float, default=None,
                    help="keep tracks with score <= this")
    args = ap.parse_args()

    if args.out_dir is None:
        args.out_dir = os.path.join(args.data_root, "harmonix", "audio")
    print(f"[paths] out_dir={args.out_dir}")

    if shutil.which("yt-dlp") is None:
        raise SystemExit("yt-dlp not found on PATH. Install with: pip install yt-dlp")
    if shutil.which("ffmpeg") is None:
        print("[warn] ffmpeg not found on PATH; -x audio extraction will fail.")

    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.urls, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        header = reader.fieldnames or []
    print(f"[urls] {len(rows)} rows   [cols] {header}")
    if args.id_col not in header or args.url_col not in header:
        raise SystemExit(f"need --id_col/--url_col present in {header}")

    align = {}
    if args.align:
        align = load_alignment_scores(args.align)
        print(f"[align] loaded {len(align)} scores "
              f"(min={args.min_alignment}, max={args.max_alignment})")

    def alignment_ok(sid):
        if args.min_alignment is None and args.max_alignment is None:
            return True
        sc = align.get(sid)
        if sc is None:
            return False  # no score -> can't vouch for it, skip when gating
        if args.min_alignment is not None and sc < args.min_alignment:
            return False
        if args.max_alignment is not None and sc > args.max_alignment:
            return False
        return True

    ok, skip, miss, gated = 0, 0, 0, 0
    for row in rows:
        sid = (row.get(args.id_col) or "").strip()
        url = (row.get(args.url_col) or "").strip()
        if not sid or not url:
            continue

        if not alignment_ok(sid):
            gated += 1
            continue

        have = already_have(args.out_dir, sid)
        if have:
            skip += 1
            print(f"[skip] {sid} (already have it)")
            continue

        out_tmpl = os.path.join(args.out_dir, sid + ".%(ext)s")
        cmd = [
            "yt-dlp", url,
            "-x", "--audio-format", args.audio_format,
            "--audio-quality", args.audio_quality,
            "-o", out_tmpl,
            "--no-playlist", "--no-progress", "--quiet", "--no-warnings",
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                miss += 1
                err = (proc.stderr or proc.stdout or "").strip().splitlines()
                reason = err[-1] if err else f"yt-dlp exit {proc.returncode}"
                print(f"[miss] {sid}: {reason}")
            else:
                found = find_output(args.out_dir, sid)
                if found:
                    ok += 1
                    print(f"[ok]   {sid} <- {url}")
                else:
                    miss += 1
                    print(f"[miss] {sid}: yt-dlp ok but no audio file found")
        except Exception as e:
            miss += 1
            print(f"[miss] {sid}: {type(e).__name__}: {e}")

        time.sleep(args.sleep)
        if args.limit and ok >= args.limit:
            print(f"[stop] hit --limit {args.limit}")
            break

    print(f"\n[done] ok={ok} skip={skip} miss={miss} gated={gated} -> {args.out_dir}")


if __name__ == "__main__":
    main()