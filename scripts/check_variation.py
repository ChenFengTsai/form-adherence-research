#!/usr/bin/env python
"""Axis 3 check on real songs: does the note-change variation score

  * call copies copies, even with noise / EQ / volume changes on top, and
  * call a repeat with a few changed notes a variation?

For each annotated song: take the most repeated letter (usually the chorus),
pick two occurrences, and build versions of the later one from the earlier:

  original       untouched song (the human repeat as it is)
  copy           exact copy
  copy_noise     copy + faint noise (-30 dB)          -> must still be a COPY
  copy_lowpass   copy through a 2 kHz low-pass (EQ)   -> must still be a COPY
  notes_1s       copy, 1 s of it shifted up a semitone -> should be VARIATION
  notes_3s       copy, 3 s shifted up a semitone       -> more variation

Prints, per condition, the note-change fraction f of the pair, its variation
score, and the share scored as a copy (score 0). CPU only (chroma), no GPU.
--with_embedding also runs the OLD embedding-based Axis 3 (copy ratio q,
needs the GPU extractor) side by side.

    python scripts/check_variation.py --manifest data/msa/songformbench_manifest.json \
        --audio_dir /storage/ssd1/richtsai1103/form-adherence-research/songformbench/audio \
        --config configs/eval_sfb_mean.json --limit 30 \
        --save_json results/sfb_variation.json

SALAMI A vs A' (annotated variations), once you have SALAMI audio + manifest:
    python scripts/check_variation.py --mode variants \
        --manifest data/msa/salami_manifest.json --audio_dir data/msa/audio
  -> is f larger for A-vs-A' pairs than for A-vs-A pairs?
"""

import argparse
import json
import os
import sys
import time
from dataclasses import replace
from itertools import combinations

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from formadherence import audio_io
from formadherence.config import FormAdherenceConfig
from formadherence.manifest import load_manifests
from formadherence.variation import (chroma_features, note_change, note_change_score,
                                     trim_edges)
from check_copy_paste import pick_pair

CONDITIONS = ("original", "copy", "copy_noise", "copy_lowpass", "notes_1s", "notes_3s")


def build_condition(x, sr, src, tgt, cond, rng):
    """x mono audio; src/tgt (start_s, end_s), src before tgt.

    Copies are SPLICED: the whole target section is replaced by the whole
    source section (the song gets longer or shorter by their length
    difference). Overwriting only min(len) samples, as the first version did,
    left the rest of a longer target original -- real changed material, so
    those "copies" were partly varied. Returns (audio, new target span)."""
    if cond == "original":
        return x.copy(), tgt
    s0, s1 = int(round(src[0] * sr)), int(round(src[1] * sr))
    t0, t1 = int(round(tgt[0] * sr)), int(round(tgt[1] * sr))
    y = np.concatenate([x[:t0], x[s0:s1], x[t1:]])
    L = s1 - s0
    new_tgt = (t0 / sr, (t0 + L) / sr)
    seg = slice(t0, t0 + L)
    if cond == "copy_noise":
        rms = np.sqrt(np.mean(x[s0:s0 + L] ** 2)) + 1e-9
        y[seg] += rng.normal(0, rms * 10 ** (-30 / 20), L)
    elif cond == "copy_lowpass":
        from scipy.signal import butter, sosfilt
        y[seg] = sosfilt(butter(4, 2000, "low", fs=sr, output="sos"), y[seg])
    elif cond.startswith("notes_"):
        from scipy.signal import resample
        secs = float(cond.split("_")[1].rstrip("s"))
        a = t0 + int(min(2.0, 0.2 * L / sr) * sr)          # start 2 s into the repeat
        n = int(secs * sr)
        r = 2 ** (1 / 12)                                   # one semitone up
        chunk = y[a:a + int(n * r)]
        if len(chunk) > 10 and a + n <= t0 + L:
            y[a:a + n] = resample(chunk, n)                 # same duration, higher pitch
    return y, new_tgt


def pair_f(C, fps, src, tgt, cfg):
    def cut(se):
        a, b = int(round(se[0] * fps)), int(round(se[1] * fps))
        return trim_edges(C[a:max(b, a + 2)], fps, cfg.edge_trim_s)
    f, _ = note_change(cut(src), cut(tgt), cfg.note_change_threshold, cfg.dtw_band)
    return f


def run_conditions(args, cfg, items, audio, fr):
    extractor = None
    if args.with_embedding:
        extractor = audio_io.build_extractor(audio_io.spec_from_args(args))
        from formadherence.metric import score_form_adherence
    rng = np.random.default_rng(0)
    rows, t_start = [], time.time()
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
        row = {"id": audio_io.track_id(it), "dataset": it.get("dataset", "all"),
               "letter": letter, "pair": [si, ti],
               "raw_label": (it.get("labels_raw") or labels)[si]}
        for cond in CONDITIONS:
            y, tgt_c = build_condition(x, sr, src, tgt, cond, rng)
            C, fps = chroma_features(y, sr, fps=cfg.chroma_fps, method=cfg.chroma_method)
            f = pair_f(C, fps, src, tgt_c, cfg)
            row[cond] = {"f": f, "score": note_change_score(f, cfg.note_change_floor,
                                                              cfg.note_change_ramp)}
            if extractor is not None:
                E = extractor.extract(y, sr)
                b = [v for v in it["boundaries"] if 0 < v < len(E)]
                if len(b) == len(labels) - 1:
                    r = score_form_adherence(E, labels, boundaries=b, config=replace(
                        cfg, variation_method="embedding"))
                    row[cond]["q_embedding"] = r.copy_ratios.get(f"{letter}{si}-{letter}{ti}")
        row["seconds"] = [round(src[1] - src[0], 1), round(tgt[1] - tgt[0], 1)]
        rows.append(row)
        print(f"[{len(rows):3d}] {row['id']:28} {row['raw_label']:10} f: "
              + "  ".join(f"{c}={row[c]['f']:.3f}" for c in CONDITIONS)
              + f"  ({time.time() - t_start:.0f}s)", flush=True)
    return rows


def summarize_conditions(rows, cfg, args):
    print(f"\n=== {len(rows)} songs: note-change Axis 3 on the copied pair ===")
    print(f"(copy below f = {cfg.note_change_floor:.3f}; full variation at "
          f"f >= {cfg.note_change_floor + cfg.note_change_ramp:.3f})\n")
    head = "  condition      should be   median f   scored as copy   median score"
    if args.with_embedding:
        head += "   | old Axis 3: median q"
    print(head)
    expect = {"original": "variation?", "copy": "copy", "copy_noise": "copy",
              "copy_lowpass": "copy", "notes_1s": "variation", "notes_3s": "variation"}
    out = {}
    for c in CONDITIONS:
        f = np.array([r[c]["f"] for r in rows])
        s = np.array([r[c]["score"] for r in rows])
        line = (f"  {c:13}  {expect[c]:10}  {np.median(f):8.3f}     {(s == 0).mean():6.0%}"
                f"          {np.median(s):.2f}")
        out[c] = {"median_f": float(np.median(f)), "frac_copy": float((s == 0).mean()),
                  "median_score": float(np.median(s))}
        q = [r[c].get("q_embedding") for r in rows if r[c].get("q_embedding") is not None]
        if args.with_embedding and q:
            line += f"          | {np.median(q):.3f}"
            out[c]["median_q_embedding"] = float(np.median(q))
        print(line)
    dsets = sorted({r.get("dataset", "all") for r in rows})
    if len(dsets) > 1:
        print("\n  per dataset: share scored as copy (median f)")
        print("  dataset     songs   " + "  ".join(f"{c:>15}" for c in CONDITIONS))
        out["per_dataset"] = {}
        for d in dsets:
            rs = [r for r in rows if r.get("dataset", "all") == d]
            cells, od = [], {"songs": len(rs)}
            for c in CONDITIONS:
                f = np.array([r[c]["f"] for r in rs])
                sc = np.array([r[c]["score"] for r in rs])
                cells.append(f"{(sc == 0).mean():4.0%} ({np.median(f):.3f})")
                od[c] = {"median_f": float(np.median(f)), "frac_copy": float((sc == 0).mean())}
            out["per_dataset"][d] = od
            print(f"  {d:10}  {len(rs):5d}   " + "  ".join(f"{x:>15}" for x in cells))
    # thresholds from MEASUREMENT: copy = what true copies measure, full score
    # = what ~1 s of changed notes measures
    copies = np.array([r[c]["f"] for r in rows for c in ("copy", "copy_noise", "copy_lowpass")])
    notes1 = np.array([r["notes_1s"]["f"] for r in rows])
    floor = float(max(0.002, np.quantile(copies, 0.95)))
    full = float(np.median(notes1))
    out["suggested"] = {"note_change_floor": floor,
                        "note_change_ramp": float(max(full - floor, 0.002)),
                        "copies_p95_f": float(np.quantile(copies, 0.95)),
                        "notes_1s_median_f": full}
    print(f"\n  true copies (all three kinds): 95% have f <= {np.quantile(copies, .95):.4f}; "
          f"1 s of changed notes: median f = {full:.4f}")
    if full <= floor:
        print("  [warn] 1 s of changed notes is not above copy noise: can't resolve "
              "changes that small with these settings")
    print(f"  suggested: note_change_floor={floor:.4f}  note_change_ramp="
          f"{out['suggested']['note_change_ramp']:.4f}  (--update_config writes them)")
    print("\nHow to read it:")
    print("  copy / copy_noise / copy_lowpass all scored as copy (~100%) -> noise and EQ")
    print("    no longer fake variation (the old axis failed copy_noise)")
    print("  notes_1s / notes_3s mostly NOT copy, f growing with the change -> real")
    print("    note changes count as variation")
    print("  original: how human repeats look; a share near 0 = production copies")
    return out


def run_variants(args, cfg, items, audio, fr):
    same, diff = [], []
    for it in items:
        var = it.get("variants")
        if not var:
            continue
        labels = [str(l) for l in it["labels"]]
        wav, sr = audio_io.read_audio(audio[audio_io.track_id(it)])
        C, fps = chroma_features(wav, sr, fps=cfg.chroma_fps, method=cfg.chroma_method)
        edges = [0] + list(it["boundaries"]) + [len(np.load(it["emb"], mmap_mode="r"))]
        secs = [(edges[k] / fr, edges[k + 1] / fr) for k in range(len(labels))]
        for i, j in combinations(range(len(labels)), 2):
            if labels[i] != labels[j] or secs[i][1] - secs[i][0] < args.min_seconds \
                    or secs[j][1] - secs[j][0] < args.min_seconds:
                continue
            f = pair_f(C, fps, secs[i], secs[j], cfg)
            (same if var[i] == var[j] else diff).append(f)
    print(f"\n=== annotated variants: {len(same)} A-vs-A pairs, {len(diff)} A-vs-A' pairs ===")
    if not same or not diff:
        print("need both kinds of pairs (SALAMI uppercase labels with primes, e.g. A and A')")
        return {}
    s, d = np.sort(same), np.asarray(diff)
    auc = float((np.searchsorted(s, d, "left").sum() +
                 0.5 * (np.searchsorted(s, d, "right") - np.searchsorted(s, d, "left")).sum())
                / (len(s) * len(d)))
    print(f"  median f: A-vs-A {np.median(same):.3f}   A-vs-A' {np.median(diff):.3f}")
    print(f"  P(f(A,A') > f(A,A)) = {auc:.3f}   (0.5 = no difference)")
    return {"n_same": len(same), "n_diff": len(diff), "median_f_same": float(np.median(same)),
            "median_f_diff": float(np.median(diff)), "auc": auc}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["conditions", "variants"], default="conditions")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--audio_dir", required=True)
    ap.add_argument("--config", default="configs/eval_sfb_mean.json")
    ap.add_argument("--limit", type=int, default=30, help="songs to test; 0 = all")
    ap.add_argument("--min_seconds", type=float, default=6.0)
    ap.add_argument("--with_embedding", action="store_true",
                    help="also run the old embedding Axis 3 (GPU)")
    ap.add_argument("--update_config", action="store_true",
                    help="write the suggested note_change_floor/ramp into --config")
    ap.add_argument("--also_update", default="",
                    help="more config files (comma-separated) to receive the same "
                         "thresholds, e.g. configs/eval_sfb.json")
    ap.add_argument("--save_json", default="")
    audio_io.add_extractor_args(ap)
    args = ap.parse_args()

    cfg = FormAdherenceConfig()
    if os.path.exists(args.config):
        with open(args.config) as f:
            cfg = FormAdherenceConfig.from_dict(json.load(f)["metric"])
    fr = cfg.frame_rate
    print(f"[cfg] chroma {cfg.chroma_method} @ {cfg.chroma_fps} fps, note threshold "
          f"{cfg.note_change_threshold}, floor {cfg.note_change_floor}, ramp {cfg.note_change_ramp}")
    audio = audio_io.index_audio_dir(args.audio_dir)
    items = [it for it in load_manifests(args.manifest, limit=0)
             if audio.get(audio_io.track_id(it) or "")]
    if args.mode == "conditions":
        rows = run_conditions(args, cfg, items, audio, fr)
        if not rows:
            raise SystemExit("no song had a long-enough repeated section")
        out = {"summary": summarize_conditions(rows, cfg, args), "songs": rows}
        sug = out["summary"].get("suggested")
        if args.update_config and sug and os.path.exists(args.config):
            targets = [args.config] + [p.strip() for p in args.also_update.split(",") if p.strip()]
            for path in targets:
                if not os.path.exists(path):
                    print(f"[config] {path} not found, skipped")
                    continue
                with open(path) as f:
                    d = json.load(f)
                d["metric"]["note_change_floor"] = sug["note_change_floor"]
                d["metric"]["note_change_ramp"] = sug["note_change_ramp"]
                d["metric"]["edge_trim_s"] = cfg.edge_trim_s
                with open(path, "w") as f:
                    json.dump(d, f, indent=2)
                print(f"[config] wrote note_change_floor/ramp into {path}")
    else:
        out = run_variants(args, cfg, items, audio, fr)
    if args.save_json:
        os.makedirs(os.path.dirname(args.save_json) or ".", exist_ok=True)
        with open(args.save_json, "w") as f:
            json.dump({"settings": vars(args), **out}, f, indent=2)
        print(f"[saved] {args.save_json}")


if __name__ == "__main__":
    main()
