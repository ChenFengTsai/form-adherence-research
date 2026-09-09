#!/usr/bin/env python
"""Evaluation mode (metrics only): score how well a generated song realizes a
target form. No RL, no reward shaping.

Two ways to use it:

  A) Score an audio file you already have (wav or mp3), or one generated
     separately with YuE's infer.py:
       python scripts/run_evaluation.py \
           --audio outputs/wavs/song.wav --form ABACA
       python scripts/run_evaluation.py \
           --audio outputs/wavs/song.mp3 --form ABACA

  B) Score a cached embedding array directly (fastest for the AI test corpus):
       python scripts/run_evaluation.py \
           --emb cache/embeddings/<key>.npy --form ABACA

Loads calibration from configs/eval_baseline.json if present, so the metric uses
your fitted tau/floor/temperature rather than the synthetic-data defaults.

NOTE: --wav is kept as an alias of --audio for backwards compatibility; both
accept wav or mp3. MP3 is lossy, so its embeddings won't be bit-identical to
the WAV of the same song -- keep a song's pipeline on one format for comparable
numbers.
"""

import argparse
import os
import json
import numpy as np


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
    ap.add_argument("--form", required=True,
                    help="target letter form, e.g. ABACA")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--audio", "--wav", dest="audio",
                     help="path to an audio file to score (wav or mp3)")
    src.add_argument("--emb", help="path to a cached (T,d) .npy embedding array")
    ap.add_argument("--config", default="configs/eval_baseline.json")
    ap.add_argument("--boundaries", default=None,
                    help="comma-separated interior frame indices (optional)")
    args = ap.parse_args()

    from formadherence import RunConfig, build_runner

    if os.path.exists(args.config):
        cfg = RunConfig.from_json(args.config)
        print(f"[cfg] loaded {args.config}")
    else:
        cfg = RunConfig(mode="evaluation")
        print("[cfg] no config file -> DEFAULTS (synthetic-calibrated; run "
              "scripts/calibrate.py before trusting real-audio numbers)")
    assert cfg.mode == "evaluation", "use mode=evaluation for this script"

    boundaries = ([int(x) for x in args.boundaries.split(",")]
                  if args.boundaries else None)

    if args.emb:
        # Score precomputed embeddings; no extractor needed.
        from formadherence.run import EvaluationRunner
        runner = EvaluationRunner(cfg, extractor=None)  # extractor unused here
        emb = np.load(args.emb)
        res = runner.evaluate_from_embeddings(emb, args.form,
                                              boundaries=boundaries)
    else:
        from formadherence.integration import (
            GeneratedSong, ClapExtractor, EncodecExtractor, ConcatExtractor)
        extractor = ConcatExtractor(
            [ClapExtractor(window_s=1.0, hop_s=0.5),
             EncodecExtractor(bandwidth=6.0, pool=8)],
            target_frame_rate=2.0)
        runner = build_runner(cfg, extractor)
        wav, sr = read_audio(args.audio)
        song = GeneratedSong(audio=np.asarray(wav), sample_rate=sr,
                             target_form=args.form)
        res = runner.evaluate(song, boundaries=boundaries)

    print(json.dumps({
        "form": res.target_form,
        "overall": round(res.overall, 4),
        "distinctness": round(res.distinctness, 4),
        "similarity": round(res.similarity, 4),
        "variation": round(res.variation, 4),
        "n_segments": len(res.form_result.segments),
    }, indent=2))


if __name__ == "__main__":
    main()