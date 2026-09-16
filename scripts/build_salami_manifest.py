#!/usr/bin/env python
"""Build the calibrate.py manifest from downloaded SALAMI audio + annotations.

For each SALAMI track that has BOTH audio and a sections annotation, this:
  1. extracts per-frame embeddings (same extractor as everything else),
  2. converts the annotation's second-based section boundaries to frame indices
     using the target frame_rate,
  3. maps the messy functional labels (Verse/Chorus/Solo/...) to letters (A/B/C),
  4. writes one manifest entry {emb, boundaries, labels}.

The resulting manifest.json is exactly what scripts/calibrate.py --manifest reads.

Usage (reads annotations via mirdata, audio from your download dir):
    python build_salami_manifest.py \
        --data_home data/msa/salami \
        --audio_dir data/msa/salami/audio \
        --cache_dir cache/embeddings \
        --out data/msa/manifest.json \
        --frame_rate 2.0

Prereqs: mirdata, plus the extraction extras (torch/transformers/encodec/
soundfile) in your `fa` env -- the same env you ran extract_embeddings.py in.

IMPORTANT: extract these calibration embeddings with the SAME patched extractor
and SAME transformers version you use at evaluation time. If the CLAP output
space differs, the fitted tau/floor/temperature won't match your eval distances
and calibration is worthless.
"""

import argparse
import hashlib
import json
import os

import numpy as np


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


def canonical_label(raw: str) -> str:
    """Collapse SALAMI's many functional labels to a coarse family.

    SALAMI free-text labels include Verse, Chorus, Bridge, Intro, Outro, Solo,
    Head, Coda, Pre-Chorus, Interlude, Instrumental, Silence, etc. (and mixed
    case). We fold them to a small set of families; two sections get the SAME
    letter iff they map to the same family. Tune this to taste -- the calibration
    only needs a consistent notion of 'same section type'.
    """
    s = raw.strip().lower()
    # strip trailing annotation cruft like "verse a", "chorus 2"
    s = s.replace("_", " ").split("(")[0].strip()
    families = {
        "verse": "verse", "chorus": "chorus", "bridge": "bridge",
        "intro": "intro", "outro": "outro", "pre-chorus": "prechorus",
        "prechorus": "prechorus", "pre chorus": "prechorus",
        "interlude": "interlude", "instrumental": "instrumental",
        "solo": "solo", "head": "head", "coda": "outro", "theme": "theme",
        "refrain": "chorus", "hook": "chorus", "break": "interlude",
        "transition": "interlude", "fade-out": "outro", "fadeout": "outro",
        "silence": "silence", "end": "outro", "main theme": "theme",
    }
    for key, fam in families.items():
        if s.startswith(key):
            return fam
    # fall back to the first token so unknowns still group consistently
    return s.split()[0] if s else "unknown"


def families_to_letters(families: list[str]) -> list[str]:
    """Assign A, B, C, ... to families in order of first appearance, so repeats
    of a family share a letter."""
    mapping: dict[str, str] = {}
    letters = []
    for fam in families:
        if fam not in mapping:
            mapping[fam] = chr(ord("A") + len(mapping))
        letters.append(mapping[fam])
    return letters


def sections_to_boundaries_labels(intervals, labels, n_frames, frame_rate):
    """Convert (intervals[:,2], labels) to interior frame boundaries + per-segment
    letters. intervals is an (N,2) array of [start_sec, end_sec] per section."""
    # Interior boundaries = the start of every section except the first,
    # converted to frames and clamped inside (0, n_frames).
    starts = [iv[0] for iv in intervals]
    interior = []
    for t in starts[1:]:
        f = int(round(t * frame_rate))
        if 0 < f < n_frames:
            interior.append(f)
    interior = sorted(set(interior))

    # Labels: SALAMI uppercase annotations are ALREADY structural letters
    # (A, B, A', C, ...). Normalize those directly (strip prime marks, upcase)
    # and only fall back to the functional-label->family mapping for lowercase
    # functional labels (verse/chorus/...). Number of segments after
    # boundaries_to_segments is len(interior)+1, so produce exactly that many.
    def is_letter_label(l):
        s = str(l).strip().rstrip("'").rstrip("’")
        return len(s) == 1 and s.isalpha()

    if labels and all(is_letter_label(l) for l in labels):
        # already a letter form: A, B, A', C -> A, B, A, C (repeats keep letter)
        letters = [str(l).strip().rstrip("'").rstrip("’").upper() for l in labels]
    else:
        fams = [canonical_label(l) for l in labels]
        letters = families_to_letters(fams)
    # If boundary dedup/clamping dropped some, trim labels to match segment count.
    n_segs = len(interior) + 1
    if len(letters) > n_segs:
        letters = letters[:n_segs]
    while len(letters) < n_segs:
        letters.append(letters[-1] if letters else "A")
    return interior, letters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_home", default="data/msa/salami",
                    help="mirdata SALAMI data_home (annotations)")
    ap.add_argument("--audio_dir", default="data/msa/audio",
                    help="dir of <salami_id>.wav|.mp3 you downloaded")
    ap.add_argument("--cache_dir", default="cache/embeddings")
    ap.add_argument("--out", default="data/msa/manifest.json")
    ap.add_argument("--frame_rate", type=float, default=2.0)
    ap.add_argument("--encodec_pool", type=int, default=8)
    ap.add_argument("--min_segments", type=int, default=3,
                    help="skip tracks with fewer sections than this")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    import mirdata
    from formadherence.integration import (
        ClapExtractor, EncodecExtractor, ConcatExtractor,
    )

    salami = mirdata.initialize("salami", data_home=args.data_home)
    tracks = salami.load_tracks()

    extractor = ConcatExtractor(
        [ClapExtractor(window_s=1.0, hop_s=0.5),
         EncodecExtractor(bandwidth=6.0, pool=args.encodec_pool)],
        target_frame_rate=args.frame_rate,
    )

    os.makedirs(args.cache_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    # Index downloaded audio by SALAMI id (stem), case-insensitive extension.
    audio_index = {}
    for f in os.listdir(args.audio_dir):
        stem, ext = os.path.splitext(f)
        if ext.lower() in (".wav", ".mp3", ".flac", ".m4a"):
            audio_index[stem] = os.path.join(args.audio_dir, f)
    print(f"[audio] indexed {len(audio_index)} files in {args.audio_dir}")

    manifest = []
    n_seen = n_ok = 0
    for tid, track in tracks.items():
        if args.limit and n_seen >= args.limit:
            break

        # find the audio file for this SALAMI id (case-insensitive extension)
        audio_path = audio_index.get(str(tid))
        if audio_path is None:
            continue  # no audio downloaded for this track

        # get sections annotation. SALAMI in mirdata is split by annotator and
        # by label case: *_uppercase are coarse structural letters (A/B/C) --
        # exactly the form we want; *_lowercase are fine functional labels
        # (verse/chorus/...). Newer mirdata returns a MultiAnnotator wrapper
        # (has .annotations, a list of per-annotator SectionData); older returns
        # a flat SectionData (has .intervals). Handle both.
        def flatten_sections(cand):
            if cand is None:
                return None
            if getattr(cand, "intervals", None) is not None:
                return cand  # already a flat SectionData
            anns = getattr(cand, "annotations", None)
            if anns:
                for a in anns:  # first annotator with usable data
                    if a is not None and getattr(a, "intervals", None) is not None:
                        return a
            return None

        sections = None
        for attr in ("sections_uppercase",
                     "sections_lowercase",
                     "sections_annotator_1_uppercase",
                     "sections_annotator_2_uppercase",
                     "sections_annotator_1_lowercase",
                     "sections_annotator_2_lowercase",
                     "sections"):
            sections = flatten_sections(getattr(track, attr, None))
            if sections is not None:
                break
        if sections is None:
            continue

        n_seen += 1
        intervals = np.asarray(sections.intervals)
        labels = list(sections.labels)
        if len(intervals) < args.min_segments:
            continue

        try:
            wav, sr = read_audio(audio_path)
            key = file_key(audio_path)
            emb_path = os.path.join(args.cache_dir, f"{key}.npy")
            if os.path.exists(emb_path):
                emb = np.load(emb_path)
            else:
                emb = extractor.extract(np.asarray(wav), sr)
                np.save(emb_path, emb)
            n_frames = len(emb)

            boundaries, letters = sections_to_boundaries_labels(
                intervals, labels, n_frames, args.frame_rate)
            if len(letters) < args.min_segments:
                continue
            # need at least one repeated letter for same-letter calibration
            if len(set(letters)) == len(letters):
                # no repeats -> contributes only diff-letter pairs; still useful
                pass

            manifest.append({
                "emb": emb_path,
                "boundaries": boundaries,
                "labels": letters,
                "salami_id": tid,
            })
            n_ok += 1
            print(f"[ok]   {tid}: {len(letters)} segs, "
                  f"{len(set(letters))} letters -> {''.join(letters)}")
        except Exception as e:
            print(f"[fail] {tid}: {e}")

    with open(args.out, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n[done] wrote {n_ok} entries (of {n_seen} with audio+annot) "
          f"to {args.out}")
    print("Next: python scripts/calibrate.py --manifest "
          f"{args.out} --out configs/eval_baseline.json")


if __name__ == "__main__":
    main()