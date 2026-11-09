#!/usr/bin/env python
"""Calibrate the metric on human-annotated songs and write a config file.

    python scripts/calibrate.py --manifest data/msa/songformbench_manifest.json \
        --out configs/eval_sfb.json --ot_backend sinkhorn

score_mode relative (default)
  The axes compare distances within each song, so there is no global distance
  scale to fit. What IS fitted is the copy-paste threshold of Axis 3, from the
  copy ratios q of real annotated repeats:
      copy_floor = 0.5 * (5th percentile of human q)
      copy_ramp  = (5th percentile) - copy_floor
  so ~all human repeats get full variation credit and only repeats far more
  identical than any human repeat are penalized.
  Also fitted: contrast_temperature (from the typical per-song gap between
  different- and same-letter distances, in units of the song's median
  distance) and repeat_ratio, the anchor used when a form gives no in-song
  reference (no repeated letter, or a single letter): the geometric midpoint
  of the human repeat and different-letter ratios (distance / inside-change).
  With --audio_dir it also reports how much human repeats change their notes
  (Axis 3) and uses note change in the final report. Axis 3's thresholds are
  set by scripts/check_variation.py --update_config, not fitted here.
  It also REPORTS what human songs score on every axis. That is your sanity
  check: annotated human songs should score well above 0 on distinctness
  and similarity (0 = chance, 1 = perfect; auc/contrast: 0.5 = chance). If they don't, the features can't support the metric yet.

score_mode absolute (legacy)
  The original fit: similarity_temperature from the same/diff medians, and
  (variation_floor, tau) from the 5th/95th percentile of same-letter distances.

--mode rl_reward writes a config that scripts/train_rl.py accepts.
Calibrate and evaluate with the SAME --ot_backend and extractor settings.
"""

import argparse
import json
import os
import sys
import time
from dataclasses import asdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from formadherence.config import FormAdherenceConfig
from formadherence.metric import score_form_adherence, song_scale
from formadherence.manifest import load_manifests, by_dataset


def score_songs(items, cfg, progress_every=10, chromas=None):
    rows = []
    t0 = time.time()
    for k, it in enumerate(items, 1):
        try:
            emb = np.load(it["emb"])
            r = score_form_adherence(emb, it["labels"], boundaries=it["boundaries"],
                                     config=cfg, chroma=(chromas or {}).get(it["emb"]))
            rows.append({"dataset": it.get("dataset"), "r": r})
        except Exception as e:
            print(f"[warn] {it.get('salami_id') or it.get('id') or it['emb']}: "
                  f"{type(e).__name__}: {e}")
        if k % progress_every == 0 or k == len(items):
            print(f"[prog] {k}/{len(items)} songs  {time.time() - t0:.0f}s", flush=True)
    return rows


def summarize(rows, title):
    def med(xs):
        xs = [x for x in xs if x is not None]
        return (f"{np.median(xs):.3f}", len(xs)) if xs else ("n/a", 0)
    print(f"\n== {title}: {len(rows)} songs ==")
    for name in ("distinctness", "similarity", "variation", "overall", "auc", "contrast"):
        m, n = med([getattr(x["r"], name) for x in rows])
        print(f"   median {name:12} = {m}   (n={n})")
    aucs = [x["r"].auc for x in rows if x["r"].auc is not None]
    if aucs:
        print(f"   songs with AUC > 0.5: {sum(a > 0.5 for a in aucs)}/{len(aucs)}")


def fit_absolute(items, cfg):
    same, diff = [], []
    for x in score_songs(items, cfg):
        same += list(x["r"].same_pair_distances.values())
        diff += list(x["r"].cross_pair_distances.values())
    same, diff = np.array(same), np.array(diff)
    print(f"[data] same pairs={len(same)} diff pairs={len(diff)}")
    if len(same) and len(diff):
        print(f"       same median={np.median(same):.4f}  diff median={np.median(diff):.4f}")
    m_same = np.median(same) if len(same) else 0.02
    m_diff = np.median(diff) if len(diff) else 1.0
    cfg.similarity_temperature = float(max(0.5 * (m_same + m_diff) / np.log(2), 1e-3))
    cfg.variation_floor = float(max(np.quantile(same, 0.05), 1e-3)) if len(same) else 0.005
    cfg.tau = float(np.quantile(same, 0.95)) if len(same) else 0.12
    print(f"[fit]  temperature={cfg.similarity_temperature:.4f} "
          f"floor={cfg.variation_floor:.4f} tau={cfg.tau:.4f}")
    return cfg


def fit_note_change(items, cfg, audio_index):
    """Axis 3 on notes: chroma per song (cached for the final report), note
    change f for every annotated repeat pair, and the copy threshold fitted
    from it the same way as the embedding copy threshold."""
    from formadherence.audio_io import read_audio, track_id
    from formadherence.variation import chroma_features, note_change, trim_edges
    chromas, fs = {}, []
    t0 = time.time()
    for k, it in enumerate(items, 1):
        p = audio_index.get(track_id(it) or "")
        if p is None:
            continue
        wav, sr = read_audio(p)
        C, fps = chroma_features(wav, sr, fps=cfg.chroma_fps, method=cfg.chroma_method)
        chromas[it["emb"]] = (C, fps)
        labels = [str(l) for l in it["labels"]]
        edges = [0] + list(it["boundaries"]) + [None]
        secs = []
        for a, b in zip(edges[:-1], edges[1:]):
            s = int(round(a / cfg.frame_rate * fps))
            e = len(C) if b is None else int(round(b / cfg.frame_rate * fps))
            secs.append(trim_edges(C[s:e], fps, cfg.edge_trim_s))
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                if labels[i] == labels[j] and len(secs[i]) > 1 and len(secs[j]) > 1:
                    fs.append(note_change(secs[i], secs[j], cfg.note_change_threshold,
                                          cfg.dtw_band)[0])
        if k % 25 == 0:
            print(f"[chroma] {k}/{len(items)}  {time.time() - t0:.0f}s", flush=True)
    if not fs:
        print("[warn] no audio matched the manifest ids; Axis 3 thresholds unchanged")
        return chromas
    fs = np.asarray(fs)
    q05 = float(np.quantile(fs, 0.05))
    print(f"\n[data] note change f of human repeats: n={len(fs)}  5%={q05:.3f}  "
          f"median={np.median(fs):.3f}  95%={np.quantile(fs, .95):.3f}  "
          f"share near-copy (f<0.01)={float((fs < 0.01).mean()):.0%}")
    # Thresholds are NOT fitted from human repeats any more: some human repeats
    # are production copies (f ~ 0), which dragged a percentile floor to ~0.
    # Set them from measurement with scripts/check_variation.py --update_config.
    print(f"[info] Axis 3 thresholds kept: note_change_floor={cfg.note_change_floor:.4f}  "
          f"ramp={cfg.note_change_ramp:.4f}  (set them with check_variation.py "
          f"--update_config)")
    return chromas


def fit_relative(items, cfg, audio_index=None):
    rows = score_songs(items, cfg)
    q = np.array([v for x in rows for v in x["r"].copy_ratios.values()
                  if np.isfinite(v)])
    if len(q):
        q05 = float(np.quantile(q, 0.05))
        print(f"\n[data] human repeat copy-ratios q: n={len(q)}  5%={q05:.3f}  "
              f"median={np.median(q):.3f}  95%={np.quantile(q, .95):.3f}")
        if q05 > 1e-3:
            cfg.copy_floor = 0.5 * q05
            cfg.copy_ramp = q05 - cfg.copy_floor
        else:
            print("[warn] some human repeats are near-exact copies (q~0); keeping "
                  "default copy_floor/copy_ramp. Check those songs.")
    else:
        print("[warn] no same-letter pairs found; keeping default copy thresholds")
    print(f"[fit]  copy_floor={cfg.copy_floor:.4f}  copy_ramp={cfg.copy_ramp:.4f}")

    # anchor for forms without an in-song reference (no repeats / one letter)
    from formadherence.metric import song_ratios
    rs, rd = [], []
    for it in items:
        try:
            s_, d_ = song_ratios(np.load(it["emb"]), it["boundaries"], it["labels"], cfg)
            rs += s_
            rd += d_
        except Exception as e:
            print(f"[warn] ratios: {type(e).__name__}: {e}")
    rs = np.array([x for x in rs if np.isfinite(x) and x > 0])
    rd = np.array([x for x in rd if np.isfinite(x) and x > 0])
    if len(rs) and len(rd):
        ms, md = float(np.median(rs)), float(np.median(rd))
        print(f"[data] ratio to inside-change: repeats median={ms:.3f}  "
              f"different letters median={md:.3f}")
        if md > ms:
            cfg.repeat_ratio = float(np.sqrt(ms * md))
            # temperature: about a quarter of the log-gap, so the two medians
            # land near 0.1 / 0.9 of the sigmoid
            cfg.ratio_temperature = float(max(np.log(md / ms) / 4.4, 0.05))
        else:
            print("[warn] repeats are NOT closer than different letters in "
                  "these features; keeping default repeat_ratio")
    print(f"[fit]  repeat_ratio={cfg.repeat_ratio:.4f}  "
          f"ratio_temperature={cfg.ratio_temperature:.4f}")

    # contrast temperature: typical per-song gap (median diff - median same)
    # in units of the song's median pair distance, mapped to sigmoid 0.95 --
    # so a typically-contrasted annotated song reads ~0.9 on that comparison
    gaps = []
    for x in rows:
        r = x["r"]
        s_, d_ = list(r.same_pair_distances.values()), list(r.cross_pair_distances.values())
        if s_ and d_:
            mm = song_scale(s_, d_, s_ + d_)
            if mm > 0:
                gaps.append((np.median(d_) - np.median(s_)) / mm)
    if gaps and np.median(gaps) > 0:
        cfg.contrast_temperature = float(np.clip(np.median(gaps) / np.log(19), 0.02, 2.0))
        print(f"[data] per-song normalized gap: median={np.median(gaps):.3f}")
    else:
        print("[warn] no positive same/diff gap in these songs; keeping default "
              "contrast_temperature")
    print(f"[fit]  contrast_temperature={cfg.contrast_temperature:.4f}")
    chromas = fit_note_change(items, cfg, audio_index) if audio_index else None
    # re-score with the fitted thresholds so the report reflects the final config
    rows = score_songs(items, cfg, progress_every=10 ** 9, chromas=chromas)
    for name, grp in by_dataset([dict(dataset=x["dataset"], r=x["r"]) for x in rows]).items():
        summarize(grp, f"human songs, {name}")
    return cfg


def write_config(cfg, mode, out):
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    try:
        from formadherence.run import RunConfig
        RunConfig(mode=mode, metric=cfg).to_json(out)
    except ImportError as e:
        if mode != "evaluation":
            raise SystemExit(f"--mode {mode} needs the integration package: {e}")
        with open(out, "w") as f:   # same layout RunConfig.to_json writes
            json.dump({"mode": "evaluation", "metric": asdict(cfg),
                       "evaluation": {"discover_boundaries": True},
                       "reward": None}, f, indent=2)
    print(f"\n[out] wrote {out} (mode={mode}, score_mode={cfg.score_mode}, "
          f"ot_backend={cfg.ot_backend})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="one or more, comma-separated")
    ap.add_argument("--out", default="configs/eval_sfb.json")
    ap.add_argument("--score_mode", choices=["relative", "absolute"], default="relative")
    ap.add_argument("--ot_backend", choices=["exact", "sinkhorn", "mean"], default="sinkhorn")
    ap.add_argument("--no_debias", action="store_true",
                    help="plain Sinkhorn cost (only to reproduce old numbers)")
    ap.add_argument("--mode", choices=["evaluation", "rl_reward"], default="evaluation")
    ap.add_argument("--max_frames", type=int, default=None)
    ap.add_argument("--limit", type=int, default=0, help="per manifest; 0 = all")
    ap.add_argument("--audio_dir", default="",
                    help="audio folder(s), comma-separated: fit and report Axis 3 on "
                         "note changes (chroma + DTW) from the human repeats")
    args = ap.parse_args()

    cfg = FormAdherenceConfig(score_mode=args.score_mode, ot_backend=args.ot_backend,
                              sinkhorn_debias=not args.no_debias)
    if args.max_frames:
        cfg.max_frames_per_segment = args.max_frames
    items = load_manifests(args.manifest, limit=args.limit)
    if not items:
        raise SystemExit("no usable manifest entries")
    audio_index = None
    if args.audio_dir:
        from formadherence.audio_io import index_audio_dir
        audio_index = {}
        for d in args.audio_dir.split(","):
            audio_index.update(index_audio_dir(d.strip()))
    cfg = fit_relative(items, cfg, audio_index) if args.score_mode == "relative" \
        else fit_absolute(items, cfg)
    write_config(cfg, args.mode, args.out)


if __name__ == "__main__":
    main()
