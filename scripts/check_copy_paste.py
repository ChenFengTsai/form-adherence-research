#!/usr/bin/env python
"""Axis check 2: does VARIATION catch a real copy-paste?

For each annotated song, take its most repeated letter (usually the chorus),
pick two occurrences, and overwrite the later one's AUDIO with a copy of the
earlier one. Conditions:

  original     unchanged audio
  copy         exact sample copy, starting at the section boundary
  copy_offset  copy taken 0.25 s later in the source (frames no longer line up)
  copy_noise   exact copy + faint noise (-30 dB), like a re-rendered copy

Embeddings are re-extracted for every condition (the original too, so all
four go through the identical path), then scored with the ANNOTATED
boundaries. Expected if Axis 3 works:
  * the copied pair's copy ratio q collapses and its variation score -> 0
  * distinctness and similarity barely change (similarity may rise a little:
    copies cohere perfectly)

GPU (real extractor, same settings as your manifest):
    python scripts/check_copy_paste.py --manifest data/msa/songformbench_manifest.json \
        --audio_dir /storage/ssd1/richtsai1103/form-adherence-research/songformbench/audio \
        --config configs/eval_sfb_mean.json --limit 30 \
        --save_json results/sfb_copy_paste.json

Dry run without GPU (simple spectral features, just to check it runs):
    ... same command ... --extractor spectral --limit 3
"""

import argparse
import json
import os
import sys
import time
from dataclasses import replace

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from formadherence import audio_io
from formadherence.config import FormAdherenceConfig
from formadherence.metric import score_form_adherence, copy_score
from formadherence.manifest import load_manifests

CONDITIONS = ("original", "copy", "copy_offset", "copy_noise")


class SpectralExtractor:
    """Tiny numpy stand-in for dry runs: log band energies per frame."""
    def __init__(self, frame_rate=2.0, n_bands=32):
        self.frame_rate, self.n_bands = frame_rate, n_bands

    def extract(self, wav, sr):
        x = np.asarray(wav, float)
        x = x.mean(axis=1) if x.ndim == 2 else x
        hop = int(round(sr / self.frame_rate))
        T = max(1, int(round(len(x) / sr * self.frame_rate)))
        win = 2 * hop
        xp = np.pad(x, (hop // 2, win))
        edges = np.unique(np.logspace(0, np.log10(win // 2), self.n_bands + 1).astype(int))
        rows = []
        for k in range(T):
            seg = xp[k * hop:k * hop + win] * np.hanning(win)
            mag = np.abs(np.fft.rfft(seg)) + 1e-8
            rows.append([np.log(mag[a:b].mean()) for a, b in zip(edges[:-1], edges[1:])])
        E = np.asarray(rows)
        E = E - E.mean(0)
        return E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)


def pick_pair(labels, boundaries, n_frames, min_frames):
    """Most repeated letter; (source, target) = two of its occurrences, both
    long enough, source at least as long as target if possible."""
    edges = [0] + list(boundaries) + [n_frames]
    spans = [(edges[i], edges[i + 1]) for i in range(len(labels))]
    counts = {c: labels.count(c) for c in set(labels)}
    for letter in sorted(counts, key=lambda c: -counts[c]):
        if counts[letter] < 2:
            break
        occ = [i for i, c in enumerate(labels) if c == letter
               and spans[i][1] - spans[i][0] >= min_frames]
        pairs = [(i, j) for i in occ for j in occ if i < j]
        if not pairs:
            continue
        pairs.sort(key=lambda p: (spans[p[0]][1] - spans[p[0]][0]) <
                   (spans[p[1]][1] - spans[p[1]][0]))     # prefer src >= tgt
        return letter, pairs[0][0], pairs[0][1], spans
    return None


def make_condition(x, sr, src, tgt, cond, rng):
    """x: mono audio. src/tgt: (start_s, end_s). Returns edited copy."""
    y = x.copy()
    if cond == "original":
        return y
    s0 = int(round(src[0] * sr)) + (int(round(0.25 * sr)) if cond == "copy_offset" else 0)
    s1 = int(round(src[1] * sr))
    t0, t1 = int(round(tgt[0] * sr)), int(round(tgt[1] * sr))
    L = max(0, min(s1 - s0, t1 - t0, len(x) - s0))
    y[t0:t0 + L] = x[s0:s0 + L]
    if cond == "copy_noise" and L > 0:
        rms = np.sqrt(np.mean(x[s0:s0 + L] ** 2)) + 1e-9
        y[t0:t0 + L] += rng.normal(0, rms * 10 ** (-30 / 20), L)
    return y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--audio_dir", required=True)
    ap.add_argument("--config", default="configs/eval_sfb_mean.json")
    ap.add_argument("--limit", type=int, default=30, help="songs to test")
    ap.add_argument("--min_seconds", type=float, default=6.0,
                    help="both sections must be at least this long")
    ap.add_argument("--extractor", choices=["real", "spectral"], default="real")
    ap.add_argument("--aggregate", default="mean", choices=["mean", "geometric"])
    ap.add_argument("--chance_margin", type=float, default=0.05)
    ap.add_argument("--save_json", default="")
    audio_io.add_extractor_args(ap)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = FormAdherenceConfig.from_dict(json.load(f)["metric"])
    cfg = replace(cfg, aggregate=args.aggregate, chance_margin=args.chance_margin)
    spec = audio_io.spec_from_args(args)
    fr = spec.frame_rate
    if args.extractor == "real":
        print(f"[extractor] {spec.describe()} -- should match the manifest's")
        extractor = audio_io.build_extractor(spec)
    else:
        print("[extractor] spectral DRY RUN (numbers are not meaningful for CLAP)")
        extractor = SpectralExtractor(frame_rate=fr)
    print(f"[cfg] distance={cfg.ot_backend} copy_floor={cfg.copy_floor:.3f} "
          f"copy_ramp={cfg.copy_ramp:.3f}")

    audio = audio_io.index_audio_dir(args.audio_dir)
    items = [it for it in load_manifests(args.manifest, limit=0)
             if audio.get(audio_io.track_id(it) or "")]
    rng = np.random.default_rng(0)
    rows, t0 = [], time.time()
    for it in items:
        if args.limit and len(rows) >= args.limit:
            break
        labels = [str(l) for l in it["labels"]]
        n_frames = len(np.load(it["emb"], mmap_mode="r"))
        pick = pick_pair(labels, it["boundaries"], n_frames, int(args.min_seconds * fr))
        if pick is None:
            continue
        letter, si, ti, spans = pick
        wav, sr = audio_io.read_audio(audio[audio_io.track_id(it)])
        x = np.asarray(wav, float)
        x = x.mean(axis=1) if x.ndim == 2 else x
        src = (spans[si][0] / fr, spans[si][1] / fr)
        tgt = (spans[ti][0] / fr, spans[ti][1] / fr)
        row = {"id": audio_io.track_id(it), "letter": letter, "pair": [si, ti],
               "raw_label": (it.get("labels_raw") or labels)[si]}
        for cond in CONDITIONS:
            E = extractor.extract(make_condition(x, sr, src, tgt, cond, rng), sr)
            b = [v for v in it["boundaries"] if 0 < v < len(E)]
            if len(b) != len(labels) - 1:
                row = None
                break
            r = score_form_adherence(E, labels, boundaries=b, config=cfg)
            q = r.copy_ratios.get(f"{letter}{si}-{letter}{ti}", float("nan"))
            row[cond] = {"q": q, "pair_variation": copy_score(q, cfg.copy_floor, cfg.copy_ramp),
                         "letter_variation": r.per_letter_variation.get(letter),
                         "distinctness": r.distinctness, "similarity": r.similarity,
                         "variation": r.variation, "overall": r.overall}
        if row is None:
            continue
        rows.append(row)
        o, c = row["original"], row["copy"]
        print(f"[{len(rows):3d}] {row['id']:28} {row['raw_label']:10} q: "
              + "  ".join(f"{k}={row[k]['q']:.2f}" for k in CONDITIONS)
              + f"   ({time.time() - t0:.0f}s)", flush=True)

    if not rows:
        raise SystemExit("no song had a long-enough repeated section")
    print(f"\n=== {len(rows)} songs: copied pair of the most repeated letter ===")
    print("  condition      median q   pair variation   frac pair var = 0   "
          "d_distinct   d_similar   d_overall")
    out = {"settings": vars(args), "n_songs": len(rows), "conditions": {}}
    for cond in CONDITIONS:
        q = np.array([r[cond]["q"] for r in rows])
        pv = np.array([r[cond]["pair_variation"] for r in rows])
        dd = np.array([r[cond]["distinctness"] - r["original"]["distinctness"] for r in rows])
        ds = np.array([r[cond]["similarity"] - r["original"]["similarity"] for r in rows])
        do = np.array([r[cond]["overall"] - r["original"]["overall"] for r in rows])
        print(f"  {cond:12}   {np.nanmedian(q):7.3f}     {np.median(pv):7.3f}          "
              f"{(pv == 0).mean():6.0%}           {np.median(dd):+.3f}      "
              f"{np.median(ds):+.3f}      {np.median(do):+.3f}")
        out["conditions"][cond] = {"median_q": float(np.nanmedian(q)),
                                   "median_pair_variation": float(np.median(pv)),
                                   "frac_pair_variation_zero": float((pv == 0).mean()),
                                   "median_d_distinct": float(np.median(dd)),
                                   "median_d_similar": float(np.median(ds)),
                                   "median_d_overall": float(np.median(do))}
    q_orig = np.array([r["original"]["q"] for r in rows])
    print(f"\n  original repeats: q 5%={np.nanquantile(q_orig, .05):.3f}  "
          f"median={np.nanmedian(q_orig):.3f}  (copy_floor={cfg.copy_floor:.3f})")
    print("\nHow to read it:")
    print("  copy rows: q far below the original's, pair variation ~0  -> Axis 3 works")
    print("  copy_offset / copy_noise still caught                       -> robust to")
    print("    frame misalignment / re-rendering; if NOT caught, copy_floor is too low")
    print("  d_distinct and d_similar near 0                              -> no side effects")
    if args.save_json:
        os.makedirs(os.path.dirname(args.save_json) or ".", exist_ok=True)
        with open(args.save_json, "w") as f:
            json.dump({**out, "songs": rows}, f, indent=2)
        print(f"[saved] {args.save_json}")


if __name__ == "__main__":
    main()
