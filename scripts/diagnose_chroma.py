#!/usr/bin/env python
"""Measure same/diff separation for the chroma+MFCC feature and compare it to
CLAP-only, using the SAME manifest, boundaries, labels, and OT distance as
diagnose_separation.py. This is the apples-to-apples test of whether a
structure-native feature beats CLAP's ~0.17-0.19 gap/spread ratio.

Requires ChromaMfccExtractor to be importable -- either pasted into
formadherence/integration/embeddings.py, or run from a dir where
chroma_mfcc_extractor.py is on the path.

    python scripts/diagnose_chroma.py \
        --manifest data/msa/manifest.json --audio_dir data/msa/audio \
        --limit 40 --ot_backend sinkhorn
"""

import argparse, json, os
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


def measure(name, extractor, manifest, audio_index, frame_rate, ot_backend, limit):
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
        print(f"    same-spread(5-95%)={spread:.4f}  gap/spread={ratio:.2f}")
        return ratio
    print(f"[{name}] not enough pairs")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--audio_dir", default="data/msa/audio")
    ap.add_argument("--frame_rate", type=float, default=2.0)
    ap.add_argument("--ot_backend", default="sinkhorn",
                    choices=["exact", "sinkhorn"])
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--n_mfcc", type=int, default=20)
    ap.add_argument("--also_clap", action="store_true",
                    help="also measure CLAP-only for side-by-side comparison")
    args = ap.parse_args()

    from formadherence.integration import ConcatExtractor
    try:
        from formadherence.integration.embeddings import ChromaMfccExtractor
    except ImportError:
        from chroma_mfcc_extractor import ChromaMfccExtractor  # standalone

    with open(args.manifest) as f:
        manifest = json.load(f)
    audio_index = {}
    for fn in os.listdir(args.audio_dir):
        stem, ext = os.path.splitext(fn)
        if ext.lower() in (".wav", ".mp3", ".flac", ".m4a"):
            audio_index[stem] = os.path.join(args.audio_dir, fn)

    print(f"Chroma+MFCC vs CLAP (limit={args.limit}, {args.ot_backend} OT)\n")

    cm = ChromaMfccExtractor(frame_rate=args.frame_rate, n_mfcc=args.n_mfcc)
    r_cm = measure("Chroma+MFCC",
                   ConcatExtractor([cm], target_frame_rate=args.frame_rate),
                   manifest, audio_index, args.frame_rate, args.ot_backend, args.limit)

    if args.also_clap:
        from formadherence.integration import ClapExtractor
        clap = ClapExtractor(window_s=4.0, hop_s=2.0)
        r_clap = measure("CLAP w=4s (best prior)",
                         ConcatExtractor([clap], target_frame_rate=args.frame_rate),
                         manifest, audio_index, args.frame_rate,
                         args.ot_backend, args.limit)
        print("\n=== comparison (gap/spread, higher better) ===")
        print(f"  Chroma+MFCC: {r_cm}")
        print(f"  CLAP(4s):    {r_clap}")


if __name__ == "__main__":
    main()