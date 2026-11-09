#!/usr/bin/env python
"""Score how well ONE song realizes a target form (metrics only, no RL).

  A) from audio (wav/mp3):
       python scripts/run_evaluation.py --audio outputs/wavs/song.wav --form ABACA
  B) from cached embeddings (fast):
       python scripts/run_evaluation.py --emb cache/embeddings/<key>.npy --form ABACA

Loads configs/eval_sfb.json (from scripts/calibrate.py) if present.
Use the SAME extractor flags (and --ot_backend in calibration) that the config
was calibrated with. --wav is an alias of --audio.

Output: overall + the three axes, the per-song AUC/contrast, and whether
boundary discovery failed (segmentation_fallback=true means the song was cut
into equal pieces -- treat that score with suspicion).
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from formadherence.config import FormAdherenceConfig
from formadherence.metric import score_form_adherence
from formadherence.audio_io import (EmbeddingCache, add_extractor_args, spec_from_args,
                                    read_audio)


def load_metric_config(path):
    if os.path.exists(path):
        with open(path) as f:
            d = json.load(f)
        if d.get("mode", "evaluation") != "evaluation":
            print(f"[cfg] note: {path} has mode={d.get('mode')}; using its metric part")
        print(f"[cfg] loaded {path}")
        return FormAdherenceConfig.from_dict(d["metric"])
    print("[cfg] no config file -> defaults (relative mode, uncalibrated copy "
          "threshold; run scripts/calibrate.py)")
    return FormAdherenceConfig()


def r4(x):
    return None if x is None else round(float(x), 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--form", required=True, help="target letter form, e.g. ABACA")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--audio", "--wav", dest="audio", help="wav or mp3 to score")
    src.add_argument("--emb", help="cached (T, d) .npy embedding array")
    ap.add_argument("--config", default="configs/eval_sfb.json")
    ap.add_argument("--cache_dir", default="cache/embeddings")
    ap.add_argument("--boundaries", default=None,
                    help="comma-separated interior frame indices (optional)")
    ap.add_argument("--audio_for_variation", default="",
                    help="with --emb: the song's audio, so Axis 3 uses note changes")
    ap.add_argument("--verbose", action="store_true", help="print per-pair details")
    ap.add_argument("--json_out", default="")
    add_extractor_args(ap)
    args = ap.parse_args()

    cfg = load_metric_config(args.config)
    boundaries = [int(x) for x in args.boundaries.split(",")] if args.boundaries else None

    wav = sr = None
    if args.emb:
        emb = np.load(args.emb)
        if args.audio_for_variation:
            wav, sr = read_audio(args.audio_for_variation)
    else:
        emb, path = EmbeddingCache(args.cache_dir, spec_from_args(args),
                                   reuse_legacy=args.reuse_legacy_cache).get(args.audio)
        print(f"[emb] {path} shape={emb.shape}")
        wav, sr = read_audio(args.audio)

    # audio -> Axis 3 measured on notes (chroma + DTW); without it, embedding fallback
    r = score_form_adherence(emb, args.form, boundaries=boundaries, config=cfg,
                             audio=wav, sample_rate=sr)
    out = {
        "form": r.target_form,
        "score_mode": r.score_mode,
        "overall": r4(r.overall),
        "distinctness": r4(r.distinctness),
        "similarity": r4(r.similarity),
        "variation": r4(r.variation),
        "auc": r4(r.auc),
        "contrast": r4(r.contrast),
        "n_segments": len(r.segments),
        "segment_starts_frames": [s.start for s in r.segments],
        "segmentation_fallback": r.segmentation_fallback,
        "variation_method": r.variation_method,
    }
    if args.verbose:
        out.update(per_letter_similarity=r.per_letter_similarity,
                   per_letter_variation=r.per_letter_variation,
                   cross_pair_distances=r.cross_pair_distances,
                   same_pair_distances=r.same_pair_distances,
                   within_segment_distances=r.within_segment_distances,
                   copy_ratios=r.copy_ratios, note_change=r.note_change)
    print(json.dumps(out, indent=2))
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
