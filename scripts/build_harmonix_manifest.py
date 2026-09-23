#!/usr/bin/env python
"""Build the calibrate.py / diagnose_formulation.py manifest from the Harmonix Set.

Sibling of build_salami_manifest.py. For each Harmonix track that has BOTH an
embedding source (audio you fetched, or the released mel-spectrograms) AND a
segments annotation, this:
  1. gets per-frame features (embeddings from audio via your extractor, OR the
     released mel-specs),
  2. converts the segments file's second-based boundaries to frame indices at the
     target frame_rate,
  3. maps Harmonix's functional labels (intro/verse/chorus/...) to letters
     (A/B/C) so two segments share a letter iff they share a function,
  4. writes one manifest entry {emb, boundaries, labels, harmonix_id, genre}.

Harmonix segments format (dataset/segments/<File>.txt), tab-separated:
    <boundary_time_stamp>\t<label>
The label is the segment that STARTS on that boundary; the last line is the
end-of-track marker (label usually "end") and is NOT a segment.

Usage (embeddings from audio -- KEEPS THE SAME FEATURE SPACE AS YOUR SALAMI RUN):
    python build_harmonix_manifest.py \
        --emb_source audio \
        --meta      data/msa/harmonixset/dataset/metadata.csv \
        --seg_dir   data/msa/harmonixset/dataset/segments \
        --audio_dir data/msa/harmonix/audio \
        --cache_dir cache/embeddings \
        --out       data/msa/harmonix_manifest.json \
        --frame_rate 2.0

Usage (embeddings from the released mel-specs -- STANDALONE ONLY, see WARNING):
    python build_harmonix_manifest.py \
        --emb_source melspec \
        --meta        data/msa/harmonixset/dataset/metadata.csv \
        --seg_dir     data/msa/harmonixset/dataset/segments \
        --melspec_dir data/msa/harmonix/melspecs \
        --out         data/msa/harmonix_melspec_manifest.json

WARNING (feature space): the released mel-specs are NOT CLAP/Encodec embeddings.
If you calibrated tau/floor/temperature on SALAMI CLAP+Encodec embeddings, those
values are meaningless against mel-spec distances. Use --emb_source melspec only
for a *standalone* Harmonix diagnosis where you re-fit calibration on mel-specs.
For any SALAMI-vs-Harmonix comparison, use --emb_source audio with the SAME
patched extractor / transformers version you used for SALAMI.
"""

import argparse
import hashlib
import json
import os

import numpy as np


# --------------------------------------------------------------------------
# label -> letter (functional grouping)
# --------------------------------------------------------------------------

def canonical_label(raw: str) -> str:
    """Collapse a Harmonix functional label to a coarse family.

    Kept deliberately close to the SALAMI builder's canonical_label so that
    letters mean the same thing across datasets. Harmonix labels are single
    lowercase tokens, sometimes numbered (verse2) or compound (instchorus).
    Two segments get the SAME letter iff they map to the same family here.
    """
    s = raw.strip().lower().replace("_", " ").replace("-", " ")
    s = s.split("(")[0].strip()
    # strip trailing section numbers: "verse 2" / "verse2" -> "verse"
    s = s.rstrip("0123456789").strip()
    families = {
        # core pop functions
        "verse": "verse", "chorus": "chorus", "bridge": "bridge",
        "intro": "intro", "outro": "outro",
        "prechorus": "prechorus", "pre chorus": "prechorus",
        "postchorus": "postchorus", "post chorus": "postchorus",
        # instrumental-ish
        "inst": "instrumental", "instrumental": "instrumental",
        "solo": "solo", "break": "interlude", "breakdown": "interlude",
        "interlude": "interlude", "transition": "interlude",
        "gtr": "instrumental", "guitarsolo": "solo",
        # synonyms folded to match SALAMI mapping
        "refrain": "chorus", "hook": "chorus", "theme": "theme",
        "head": "head", "coda": "outro", "fadeout": "outro",
        "fade out": "outro", "silence": "silence", "end": "outro",
    }
    # compound instrumental variants: instchorus/instverse -> keep the base
    for base in ("chorus", "verse", "intro", "outro", "bridge"):
        if s == "inst" + base:
            return families.get(base, base)
    for key, fam in families.items():
        if s.startswith(key):
            return fam
    return s.split()[0] if s else "unknown"


def families_to_letters(families: list[str]) -> list[str]:
    """A, B, C, ... assigned in order of first appearance; repeats share a letter."""
    mapping: dict[str, str] = {}
    letters = []
    for fam in families:
        if fam not in mapping:
            mapping[fam] = chr(ord("A") + len(mapping))
        letters.append(mapping[fam])
    return letters


# --------------------------------------------------------------------------
# segments file -> interior frame boundaries + per-segment letters
# --------------------------------------------------------------------------

def read_harmonix_segments(path):
    """Return sorted list of (time_sec, label)."""
    rows = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                parts = line.split()  # tolerate space-separated
            if len(parts) < 2:
                continue
            try:
                t = float(parts[0])
            except ValueError:
                continue  # skip a header line if present
            rows.append((t, parts[1].strip()))
    rows.sort(key=lambda r: r[0])
    return rows


def segments_to_boundaries_labels(rows, n_frames, frame_rate, canon=True):
    """rows = [(t_sec, label), ...] with a trailing end marker.

    Segment i spans rows[i].t -> rows[i+1].t with label rows[i].label; the last
    row is the end-of-track marker (dropped). Returns (interior_frame_boundaries,
    per_segment_letters) sized so that len(letters) == len(boundaries)+1.
    """
    if len(rows) < 2:
        return [], []
    times = [r[0] for r in rows]
    labs = [r[1] for r in rows]

    seg_labels_raw = labs[:-1]          # drop the end marker's label
    interior_secs = times[1:-1]         # drop start (0) and end

    interior = []
    for t in interior_secs:
        fr = int(round(t * frame_rate))
        if 0 < fr < n_frames:
            interior.append(fr)
    interior = sorted(set(interior))

    fams = [canonical_label(l) for l in seg_labels_raw] if canon \
        else [l.strip().lower() for l in seg_labels_raw]
    letters = families_to_letters(fams)

    # dedup/clamp of boundaries can change the segment count -> reconcile
    n_segs = len(interior) + 1
    if len(letters) > n_segs:
        letters = letters[:n_segs]
    while len(letters) < n_segs:
        letters.append(letters[-1] if letters else "A")
    return interior, letters


# --------------------------------------------------------------------------
# embedding sources
# --------------------------------------------------------------------------

def file_key(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()[:16]


def read_audio(path):
    import soundfile as sf
    try:
        wav, sr = sf.read(path)
        return np.asarray(wav), sr
    except Exception:
        import librosa
        wav, sr = librosa.load(path, sr=None, mono=False)
        if wav.ndim == 2:
            wav = wav.T
        return np.asarray(wav), sr


def load_melspec(path, n_mels_hint=None):
    """Load a released mel-spec .npy and orient it to (T, n_mels)."""
    arr = np.load(path)
    if arr.ndim != 2:
        raise ValueError(f"expected 2D mel-spec, got shape {arr.shape}")
    if n_mels_hint is not None:
        if arr.shape[1] == n_mels_hint:
            return arr
        if arr.shape[0] == n_mels_hint:
            return arr.T
    # fallback heuristic: time is the longer axis
    return arr if arr.shape[0] >= arr.shape[1] else arr.T


def melspec_frame_rate(melspec_dir, fallback):
    """Derive frames-per-second from info.json if present.

    NOTE: verify these key names against the info.json inside Harmonix_melspecs.tgz;
    they are not documented field-by-field. Common possibilities handled below.
    """
    info_path = os.path.join(melspec_dir, "info.json")
    if not os.path.exists(info_path):
        print(f"[melspec] no info.json in {melspec_dir}; using --frame_rate={fallback}")
        return fallback, None
    with open(info_path) as f:
        info = json.load(f)
    for k in ("frame_rate", "fps", "frames_per_second"):
        if k in info:
            fr = float(info[k])
            print(f"[melspec] info.json {k}={fr}")
            return fr, info.get("n_mels")
    sr = info.get("sr") or info.get("sample_rate")
    hop = info.get("hop_length") or info.get("hop")
    if sr and hop:
        fr = float(sr) / float(hop)
        print(f"[melspec] info.json sr={sr} hop={hop} -> {fr:.4f} fps")
        return fr, info.get("n_mels")
    print(f"[melspec] info.json present but no rate keys; using --frame_rate={fallback}")
    return fallback, info.get("n_mels")


# --------------------------------------------------------------------------
# optional filters
# --------------------------------------------------------------------------

def load_alignment_scores(path):
    """Map File -> float alignment score, defensively (column names vary)."""
    import csv
    scores = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
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


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb_source", choices=["audio", "melspec"], default="audio")
    ap.add_argument("--meta", default="data/msa/harmonixset/dataset/metadata.csv")
    ap.add_argument("--seg_dir", default="data/msa/harmonixset/dataset/segments")
    # Manifest stays in the repo (small, code-adjacent, version-controlled).
    ap.add_argument("--out", default="data/msa/harmonix_manifest.json")
    # Heavy data (audio + embedding cache) lives on the SSD. audio_dir/cache_dir/
    # melspec_dir default to None and are derived from --data_root when not given.
    ap.add_argument("--data_root",
                    default="/storage/ssd1/richtsai1103/form-adherence-research",
                    help="SSD root for heavy files (audio, embedding cache, melspecs)")
    ap.add_argument("--frame_rate", type=float, default=2.0)
    ap.add_argument("--min_segments", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--file_col", default="File")
    ap.add_argument("--genre_col", default="Genre")
    ap.add_argument("--genres", default="",
                    help="comma-separated genre substrings to keep (case-insensitive); empty = all")
    # audio mode (default None -> derived from --data_root)
    ap.add_argument("--audio_dir", default=None)
    ap.add_argument("--cache_dir", default=None)
    ap.add_argument("--encodec_pool", type=int, default=8)
    ap.add_argument("--youtube_alignment", default="",
                    help="path to youtube_alignment_scores.csv to drop poorly-aligned YT audio")
    ap.add_argument("--min_alignment", type=float, default=None,
                    help="keep tracks with alignment score >= this (needs --youtube_alignment)")
    # melspec mode (default None -> derived from --data_root)
    ap.add_argument("--melspec_dir", default=None)
    args = ap.parse_args()

    # Derive heavy-data dirs from the SSD root unless explicitly overridden.
    if args.audio_dir is None:
        args.audio_dir = os.path.join(args.data_root, "harmonix", "audio")
    if args.cache_dir is None:
        args.cache_dir = os.path.join(args.data_root, "cache", "embeddings")
    if args.melspec_dir is None:
        args.melspec_dir = os.path.join(args.data_root, "harmonix", "melspecs")
    print(f"[paths] data_root={args.data_root}")
    print(f"[paths] audio_dir={args.audio_dir}")
    print(f"[paths] cache_dir={args.cache_dir}")
    print(f"[paths] melspec_dir={args.melspec_dir}")
    print(f"[paths] manifest out={args.out}")

    import csv

    # ---- read Harmonix metadata (comma-separated) ------------------------
    with open(args.meta, newline="") as f:
        reader = csv.DictReader(f)
        meta_rows = list(reader)
        cols = reader.fieldnames or []
    if args.file_col not in cols:
        raise SystemExit(f"--file_col {args.file_col!r} not in metadata cols {cols}")
    print(f"[meta] {len(meta_rows)} tracks; cols={cols}")

    keep_genres = {g.strip().lower() for g in args.genres.split(",") if g.strip()}
    if keep_genres:
        print(f"[meta] keeping genres matching {sorted(keep_genres)}")

    align = {}
    if args.youtube_alignment:
        align = load_alignment_scores(args.youtube_alignment)
        print(f"[align] loaded {len(align)} alignment scores")

    # ---- set up embedding source ----------------------------------------
    extractor = None
    melspec_index = {}
    n_mels_hint = None
    fr = args.frame_rate

    if args.emb_source == "audio":
        from formadherence.integration import (
            ClapExtractor, EncodecExtractor, ConcatExtractor,
        )
        extractor = ConcatExtractor(
            [ClapExtractor(window_s=1.0, hop_s=0.5),
             EncodecExtractor(bandwidth=6.0, pool=args.encodec_pool)],
            target_frame_rate=args.frame_rate,
        )
        os.makedirs(args.cache_dir, exist_ok=True)
        audio_index = {}
        for f in os.listdir(args.audio_dir):
            stem, ext = os.path.splitext(f)
            if ext.lower() in (".wav", ".mp3", ".flac", ".m4a"):
                audio_index[stem] = os.path.join(args.audio_dir, f)
        print(f"[audio] indexed {len(audio_index)} files in {args.audio_dir}")
    else:
        fr, n_mels_hint = melspec_frame_rate(args.melspec_dir, args.frame_rate)
        for f in os.listdir(args.melspec_dir):
            stem, ext = os.path.splitext(f)
            if ext.lower() == ".npy":
                melspec_index[stem] = os.path.join(args.melspec_dir, f)
        print(f"[melspec] indexed {len(melspec_index)} .npy files; frame_rate={fr:.4f}")
        print("[melspec] WARNING: mel-spec feature space != CLAP/Encodec. Do NOT reuse "
              "SALAMI calibration; re-fit on these features.")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    manifest = []
    n_seen = n_ok = 0
    for row in meta_rows:
        if args.limit and n_seen >= args.limit:
            break
        fid = (row.get(args.file_col) or "").strip()
        if not fid:
            continue

        if keep_genres:
            g = (row.get(args.genre_col) or "").strip().lower()
            if not any(k in g for k in keep_genres):
                continue

        seg_path = os.path.join(args.seg_dir, f"{fid}.txt")
        if not os.path.exists(seg_path):
            continue
        rows_seg = read_harmonix_segments(seg_path)
        if len(rows_seg) - 1 < args.min_segments:   # -1 for the end marker
            continue

        # alignment filter (audio-from-YouTube quality gate)
        if args.min_alignment is not None and align:
            sc = align.get(fid)
            if sc is None or sc < args.min_alignment:
                continue

        n_seen += 1
        try:
            # ---- features + n_frames ------------------------------------
            if args.emb_source == "audio":
                audio_path = audio_index.get(fid)
                if audio_path is None:
                    n_seen -= 1
                    continue
                key = file_key(audio_path)
                emb_path = os.path.join(args.cache_dir, f"{key}.npy")
                if os.path.exists(emb_path):
                    emb = np.load(emb_path)
                else:
                    wav, sr = read_audio(audio_path)
                    emb = extractor.extract(np.asarray(wav), sr)
                    np.save(emb_path, emb)
            else:
                mel_path = melspec_index.get(fid)
                if mel_path is None:
                    n_seen -= 1
                    continue
                emb = load_melspec(mel_path, n_mels_hint)
                # cache path just points at the released file (already on disk)
                emb_path = mel_path

            n_frames = len(emb)
            boundaries, letters = segments_to_boundaries_labels(
                rows_seg, n_frames, fr)
            if len(letters) < args.min_segments:
                continue

            manifest.append({
                # absolute so diagnosis/calibration resolve SSD embeddings no
                # matter which directory they're run from (manifest is in repo).
                "emb": os.path.abspath(emb_path),
                "boundaries": boundaries,
                "labels": letters,
                "harmonix_id": fid,
                "genre": (row.get(args.genre_col) or "").strip(),
            })
            n_ok += 1
            print(f"[ok]   {fid}: {len(letters)} segs, "
                  f"{len(set(letters))} letters -> {''.join(letters)}")
        except Exception as e:
            print(f"[fail] {fid}: {type(e).__name__}: {e}")

    with open(args.out, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n[done] wrote {n_ok} entries (of {n_seen} with features+segments) "
          f"to {args.out}")
    print(f"Next: python scripts/diagnose_formulation.py --mode real --manifest {args.out}")
    print(f"  or: python scripts/calibrate.py --manifest {args.out} "
          f"--out configs/eval_harmonix.json")


if __name__ == "__main__":
    main()