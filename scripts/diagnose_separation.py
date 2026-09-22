#!/usr/bin/env python
"""Measure how well each embedding source separates same-letter from
different-letter sections, and sweep CLAP window length.

Reports, per source, the honest separation metric:
    gap/spread ratio = (diff_median - same_median) / same_spread(5-95%)
Higher is better; ~>1 is good separation. The fused CLAP+EnCodec baseline sat
around 0.17.

Examples:
    # compare CLAP-only vs EnCodec-only vs fused at the default 1s window
    python scripts/diagnose_separation.py \
        --manifest data/msa/manifest.json --audio_dir data/msa/audio \
        --limit 40 --ot_backend sinkhorn

    # sweep CLAP window lengths (CLAP-only), to see if longer context separates
    python scripts/diagnose_separation.py \
        --manifest data/msa/manifest.json --audio_dir data/msa/audio \
        --limit 40 --ot_backend sinkhorn \
        --mode clap_window_sweep --windows 1,4,8
"""

import argparse
import json
import os
import numpy as np
from itertools import combinations


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


def measure(name, extractor, manifest, audio_index, frame_rate,
            ot_backend, limit):
    from formadherence.ot import wasserstein_distance, sinkhorn_distance
    from formadherence.config import FormAdherenceConfig
    from formadherence.metric import _segment_distribution
    from formadherence.segmentation import boundaries_to_segments

    cfg = FormAdherenceConfig(ot_backend=ot_backend)
    rng = np.random.default_rng(0)
    ot = (wasserstein_distance if ot_backend == "exact"
          else lambda a, b, **k: sinkhorn_distance(a, b, eps=cfg.sinkhorn_eps))

    same, diff = [], []
    n = 0
    for item in manifest:
        sid = item.get("salami_id")
        p = audio_index.get(str(sid)) if sid else None
        if p is None:
            continue
        try:
            wav, sr = read_audio(p)
            emb = extractor.extract(np.asarray(wav), sr)
            T = len(emb)
            b = [x for x in item["boundaries"] if 0 < x < T]
            segs = boundaries_to_segments(b, T, labels=None)
            labels = item["labels"][:len(segs)]
            while len(labels) < len(segs):
                labels.append(labels[-1] if labels else "A")
            for s, lab in zip(segs, labels):
                s.label = lab
            dists = [_segment_distribution(s, emb, cfg.max_frames_per_segment, rng)
                     for s in segs]
            for i, j in combinations(range(len(segs)), 2):
                d = ot(dists[i], dists[j], metric="cosine")
                (same if segs[i].label == segs[j].label else diff).append(d)
            n += 1
            if limit and n >= limit:
                break
        except Exception as e:
            print(f"  [warn] {sid}: {type(e).__name__}: {e}")
            continue

    same, diff = np.array(same), np.array(diff)
    if len(same) and len(diff):
        ms, md = float(np.median(same)), float(np.median(diff))
        spread = float(np.quantile(same, 0.95) - np.quantile(same, 0.05))
        ratio = (md - ms) / max(spread, 1e-9)
        print(f"[{name}] songs={n} same={len(same)} diff={len(diff)}")
        print(f"    same median={ms:.4f}  diff median={md:.4f}  gap={md-ms:.4f}")
        print(f"    same-spread(5-95%)={spread:.4f}  gap/spread={ratio:.2f} "
              f"(higher better; >~1 good)")
        return ratio
    print(f"[{name}] not enough pairs (same={len(same)} diff={len(diff)})")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--audio_dir", default="data/msa/audio")
    ap.add_argument("--frame_rate", type=float, default=2.0)
    ap.add_argument("--encodec_pool", type=int, default=8)
    ap.add_argument("--ot_backend", default="sinkhorn",
                    choices=["exact", "sinkhorn"])
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--mode", default="sources",
                    choices=["sources", "clap_window_sweep"],
                    help="'sources' = CLAP vs EnCodec vs fused; "
                         "'clap_window_sweep' = CLAP-only at several windows")
    ap.add_argument("--window_s", type=float, default=1.0,
                    help="CLAP window length (seconds) for 'sources' mode")
    ap.add_argument("--hop_s", type=float, default=0.5,
                    help="CLAP hop (seconds); if omitted in sweep, uses window/2")
    ap.add_argument("--windows", default="1,4,8",
                    help="comma-sep window lengths for clap_window_sweep")
    args = ap.parse_args()

    from formadherence.integration import (
        ClapExtractor, EncodecExtractor, ConcatExtractor)

    with open(args.manifest) as f:
        manifest = json.load(f)

    audio_index = {}
    for fn in os.listdir(args.audio_dir):
        stem, ext = os.path.splitext(fn)
        if ext.lower() in (".wav", ".mp3", ".flac", ".m4a"):
            audio_index[stem] = os.path.join(args.audio_dir, fn)

    if args.mode == "sources":
        print(f"Sources @ CLAP window={args.window_s}s hop={args.hop_s}s "
              f"(limit={args.limit}, {args.ot_backend} OT)\n")
        clap = ClapExtractor(window_s=args.window_s, hop_s=args.hop_s)
        enc = EncodecExtractor(bandwidth=6.0, pool=args.encodec_pool)
        measure("CLAP-only",
                ConcatExtractor([clap], target_frame_rate=args.frame_rate),
                manifest, audio_index, args.frame_rate, args.ot_backend, args.limit)
        measure("EnCodec-only",
                ConcatExtractor([enc], target_frame_rate=args.frame_rate),
                manifest, audio_index, args.frame_rate, args.ot_backend, args.limit)
        measure("Fused",
                ConcatExtractor([clap, enc], target_frame_rate=args.frame_rate),
                manifest, audio_index, args.frame_rate, args.ot_backend, args.limit)
    else:  # clap_window_sweep
        wins = [float(x) for x in args.windows.split(",")]
        print(f"CLAP-only window sweep {wins} "
              f"(limit={args.limit}, {args.ot_backend} OT)\n")
        results = {}
        for w in wins:
            hop = w / 2.0
            clap = ClapExtractor(window_s=w, hop_s=hop)
            r = measure(f"CLAP w={w}s hop={hop}s",
                        ConcatExtractor([clap], target_frame_rate=args.frame_rate),
                        manifest, audio_index, args.frame_rate,
                        args.ot_backend, args.limit)
            results[w] = r
            print()
        print("=== window sweep summary (gap/spread, higher better) ===")
        for w in wins:
            print(f"  window {w}s: {results[w]}")
        best = max((w for w in wins if results[w] is not None),
                   key=lambda w: results[w], default=None)
        if best is not None:
            print(f"  -> best window: {best}s (ratio {results[best]:.2f})")


if __name__ == "__main__":
    main()