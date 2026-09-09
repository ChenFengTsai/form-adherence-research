#!/usr/bin/env python
"""Extract per-frame embeddings for every audio file in a directory and cache
them as .npy, keyed by a hash of the file so nothing is ever re-extracted.

Run on your GPU box in the `fa` env (needs the `extract` extras: torch,
transformers, encodec, soundfile).

    python scripts/extract_embeddings.py \
        --audio_dir outputs/wavs \
        --cache_dir cache/embeddings \
        --frame_rate 2.0

By default this picks up both .wav and .mp3 files. MP3 is lossy, so its
embeddings won't be bit-identical to the WAV of the same song -- keep a song's
pipeline on one format for comparable numbers.

FIRST-RUN SMOKE TEST (do this before trusting anything downstream):
    the script prints the (T, d) shape of the first file. Confirm
    T ~= duration_seconds * frame_rate and d = CLAP_dim + EnCodec_dim.
    If sections of a song you know are different look identical in the
    embeddings, the metric cannot work -- fix the extractor before proceeding.
"""

import argparse
import hashlib
import os
import glob
import numpy as np


def file_key(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()[:16]


def read_audio(path):
    """Read wav or mp3 as (samples, sample_rate).

    Tries soundfile first (fast, handles wav and -- with libsndfile >= 1.1.0 --
    mp3). Falls back to librosa/audioread (ffmpeg backend) for mp3 when the
    installed libsndfile lacks MP3 support.
    """
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio_dir", required=True)
    ap.add_argument("--cache_dir", default="cache/embeddings")
    ap.add_argument("--frame_rate", type=float, default=2.0,
                    help="common frames/sec grid for the fused embedding")
    ap.add_argument("--encodec_pool", type=int, default=8,
                    help="time-pool EnCodec frames to lower T")
    ap.add_argument("--glob", default=None,
                    help="override file glob; by default matches *.wav and "
                         "*.mp3 (case-insensitive)")
    args = ap.parse_args()

    # Imported here so the script fails loudly only when actually run without
    # the extraction extras installed.
    from formadherence.integration import (
        ClapExtractor, EncodecExtractor, ConcatExtractor,
    )

    extractor = ConcatExtractor(
        [ClapExtractor(window_s=1.0, hop_s=0.5),
         EncodecExtractor(bandwidth=6.0, pool=args.encodec_pool)],
        target_frame_rate=args.frame_rate,
    )

    os.makedirs(args.cache_dir, exist_ok=True)

    if args.glob:
        patterns = [args.glob]
    else:
        patterns = ["*.wav", "*.WAV", "*.mp3", "*.MP3"]
    paths = sorted({p for pat in patterns
                    for p in glob.glob(os.path.join(args.audio_dir, pat))})
    if not paths:
        want = args.glob or "*.wav / *.mp3"
        raise SystemExit(f"no files matching {want} in {args.audio_dir}")

    for i, path in enumerate(paths):
        key = file_key(path)
        out = os.path.join(args.cache_dir, f"{key}.npy")
        if os.path.exists(out):
            print(f"[skip] {os.path.basename(path)} -> {key}.npy (cached)")
            continue
        wav, sr = read_audio(path)
        emb = extractor.extract(np.asarray(wav), sr)
        np.save(out, emb)
        print(f"[ok]   {os.path.basename(path)} -> {key}.npy  shape={emb.shape}")
        if i == 0:
            print(f"       SMOKE TEST: first file embedding shape = {emb.shape} "
                  f"(expect T ~= duration*{args.frame_rate}, d = clap+encodec)")


if __name__ == "__main__":
    main()