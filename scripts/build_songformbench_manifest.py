#!/usr/bin/env python
"""Build a CLEAN manifest from SongFormBench (huggingface: ASLP-lab/SongFormBench).

Why this dataset: its 200 HarmonixSet songs have expert-revised annotations and
audio made from Harmonix's own mel-spectrograms, i.e. the same version that was
annotated -- no YouTube version mismatch. Plus 100 Chinese pop songs (CN).

Why not load_dataset(): on this repo it picks the plain-text reader and returns
the label .txt files as loose lines (3824 rows of "0.0 silence", no song ids,
no audio). So this script reads the repo's FILES directly: it lists them, pairs
each annotation file with its audio file by name, and downloads only those.

What it does
  1. lists the repo, pairs <subset>/<id>.txt annotations with <id> audio files
     (files under utils/ are ignored), downloads them, copies the audio to
     --out_dir/audio/ and the normalised annotation to --out_dir/segments/;
  2. only the 300 expert-verified songs: stops unless exactly 200 HarmonixSet +
     100 CN annotation files with distinct ids were found (the clean check may
     then drop some -- the manifest is a subset of those 300, never more).
     The dataset commit is recorded in the report;
  3. checks EVERY song and drops the ones that fail (report in --report):
       - no audio file, labels outside the 8-class set, times not increasing,
         no end time
       - audio shorter than the annotation (> --dur_tol), or > --max_tail s of
         unannotated MUSIC after the annotation end (a quiet fade/silence tail
         is fine), or > --max_head s before the first section
       - < 3 musical sections, audio < 30 s,
         near-silent audio, duplicate audio or duplicate id
     warnings (kept): very short sections, adjacent identical labels;
     kept songs are CROPPED to exactly the annotated span (original file in
     --out_dir/original_audio/), so audio and labels cover the same time;
  4. optional --harmonix_seg_dir: compares the revised HarmonixSet boundaries
     with the ORIGINAL Harmonix annotation (how much the experts changed);
  5. extracts embeddings (same cache/extractor flags as every other script)
     and writes the manifest. dataset = sfb_hx or sfb_cn per entry.

    # 0) see how the repo is laid out and how files pair up (downloads 1 label file)
    python scripts/build_songformbench_manifest.py --inspect

    # 1) download + clean check only (no GPU)
    python scripts/build_songformbench_manifest.py --check_only

    # 2) full build (extracts embeddings)
    python scripts/build_songformbench_manifest.py

Label -> letter: verse/chorus/... become letters (same family table as the
other builders). pre-chorus keeps its own letter by default (--prechorus verse
folds it into verse, the paper's 7-class setting). Silence is merged into the
neighbouring section by default (--special drop): it is not a form section.
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from formadherence.audio_io import (EmbeddingCache, add_extractor_args,  # noqa: E402
                                    read_audio, spec_from_args, write_wav)
from formadherence.labels import annotation_to_manifest_fields  # noqa: E402
from formadherence.songformbench import (boundary_hit_rate, check_song,  # noqa: E402
                                         edge_is_quiet,
                                         parse_annotation, parse_msa_text,
                                         write_msa_txt)

AUDIO_EXTS = (".wav", ".flac", ".mp3", ".ogg", ".m4a", ".opus", ".aac")
MEL_EXTS = (".npy", ".pt", ".pth")
LABEL_EXTS = (".txt", ".lab", ".msa")
SKIP_TOP = ("utils",)            # helper code / vocoder assets, not benchmark data


# --------------------------------------------------------------------------
# repo layout
# --------------------------------------------------------------------------

def subset_from_path(path):
    """HarmonixSet / CN from the folder names, else None."""
    parts = [p.lower() for p in path.replace("\\", "/").split("/")[:-1]]
    for p in parts:
        if "harmonix" in p or p in ("hx", "bhx"):
            return "HarmonixSet"
    for p in parts:
        toks = p.replace("-", "_").split("_")
        if p in ("cn", "bc", "chinese", "zh") or "cn" in toks or "chinese" in p:
            return "CN"
    return None


def subset_from_id(sid):
    """Fallback: Harmonix ids look like 0001_12step."""
    return "HarmonixSet" if sid[:4].isdigit() and sid[4:5] == "_" else None


def classify_files(files):
    """Repo file list -> dict kind -> {(subset, stem): path} plus skipped files."""
    out = {"label": {}, "audio": {}, "mel": {}}
    dups = defaultdict(list)
    other = []
    for f in files:
        top = f.split("/")[0]
        if "/" not in f or top in SKIP_TOP or top.startswith("."):
            continue                    # README, dataset.py, .gitattributes, utils/
        stem, ext = os.path.splitext(os.path.basename(f))
        ext = ext.lower()
        kind = ("label" if ext in LABEL_EXTS else "audio" if ext in AUDIO_EXTS
                else "mel" if ext in MEL_EXTS else None)
        if kind is None or stem.lower().startswith("readme"):
            other.append(f)
            continue
        sub = subset_from_path(f) or subset_from_id(stem)
        key = (sub, stem)
        if key in out[kind]:
            dups[kind].append(f)
            continue
        out[kind][key] = f
    return out, dups, other


def find_match(index, sub, stem):
    if (sub, stem) in index:
        return index[(sub, stem)]
    hits = [p for (s, st), p in index.items() if st == stem]
    return hits[0] if len(hits) == 1 else None


def layout_report(files, cls, dups, other):
    tops = Counter("/".join(f.split("/")[:3]) for f in files if "/" in f)
    exts = Counter(os.path.splitext(f)[1].lower() or "(none)" for f in files)
    print(f"[repo] {len(files)} files")
    print("  extensions:", dict(exts.most_common()))
    print("  folders (first 3 levels):")
    for k, v in sorted(tops.items()):
        print(f"    {v:5d}  {k}/")
    for kind in ("label", "audio", "mel"):
        per = Counter(s for s, _ in cls[kind])
        ex = list(cls[kind].values())[:2]
        print(f"  {kind:5s}: {dict(per)}  e.g. {ex}")
    if any(dups.values()):
        print("  duplicates:", {k: v[:3] for k, v in dups.items() if v})
    if other:
        print(f"  other files ignored: {len(other)} e.g. {other[:3]}")
    paired = sum(1 for (s, st) in cls["label"] if find_match(cls["audio"], s, st))
    print(f"  annotation files with a matching audio file: {paired}/{len(cls['label'])}")
    if cls["mel"] and paired < len(cls["label"]):
        print("  NOTE: some songs ship as mel-spectrograms only; the repo's utils/ has the "
              "BigVGAN script to turn them into audio (see the dataset README).")


# --------------------------------------------------------------------------

REASONS = (("unannotated audio after", "long unannotated music after the end"),
           ("audio shorter", "audio shorter than the annotation"),
           ("no audio", "no audio file in the repo"),
           ("unknown labels", "labels outside the 8-class set"),
           ("not strictly increasing", "section times not increasing"),
           ("end time", "missing/invalid end time"),
           ("annotation starts", "annotation starts late"),
           ("musical sections", "too few musical sections"),
           ("audio only", "audio too short"),
           ("near-silent", "near-silent audio"),
           ("duplicate", "duplicate audio/id"),
           ("unreadable", "audio unreadable"),
           ("annotation", "annotation missing/unparsable"))


WARNS = (("trailing", "unannotated tail (fade/silence or short) -> cropped"),
         ("leading", "unannotated start -> cropped"),
         ("section(s) <", "very short musical section(s)"),
         ("adjacent", "adjacent sections with the same label (e.g. chorus, chorus)"),
         ("end marker", "no end marker"))


def warn_key(w):
    for needle, name in WARNS:
        if needle in w:
            return name
    return w[:50]


def reason_key(problem):
    for needle, name in REASONS:
        if needle in problem:
            return name
    return problem.split(":")[0][:50]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="ASLP-lab/SongFormBench")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--data_root", default="/storage/ssd1/richtsai1103/form-adherence-research")
    ap.add_argument("--out_dir", default=None, help="default <data_root>/songformbench")
    ap.add_argument("--cache_dir", default=None, help="default <data_root>/cache/embeddings")
    ap.add_argument("--out", default="data/msa/songformbench_manifest.json")
    ap.add_argument("--report", default="results/songformbench_clean.json")
    ap.add_argument("--subsets", default="HarmonixSet,CN")
    ap.add_argument("--prechorus", choices=["keep", "verse"], default="keep")
    ap.add_argument("--special", choices=["drop", "unique", "keep"], default="drop")
    ap.add_argument("--dur_tol", type=float, default=1.0,
                    help="audio may be this much shorter than the annotation end (s)")
    ap.add_argument("--max_tail", type=float, default=10.0,
                    help="unannotated MUSIC allowed after the annotation end (s); "
                         "a quiet tail (fade/silence, < -20 dB) is always allowed")
    ap.add_argument("--max_head", type=float, default=10.0,
                    help="unannotated audio allowed before the first section (s)")
    ap.add_argument("--min_sections", type=int, default=3)
    ap.add_argument("--harmonix_seg_dir", default=None,
                    help="original Harmonix segments dir, to measure how much was revised")
    ap.add_argument("--inspect", action="store_true", help="print repo layout, exit")
    ap.add_argument("--check_only", action="store_true", help="no embeddings / manifest")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--allow_count_mismatch", action="store_true",
                    help="don't stop if the dataset is not exactly 200 HX + 100 CN")
    add_extractor_args(ap)
    args = ap.parse_args()

    from huggingface_hub import HfApi, hf_hub_download

    out_dir = args.out_dir or os.path.join(args.data_root, "songformbench")
    audio_dir, seg_dir = os.path.join(out_dir, "audio"), os.path.join(out_dir, "segments")
    api = HfApi()
    print(f"[repo] listing {args.repo}")
    files = api.list_repo_files(args.repo, repo_type="dataset", revision=args.revision)
    cls, dups, other = classify_files(files)

    if args.inspect:
        layout_report(files, cls, dups, other)
        if cls["label"]:
            (sub, stem), rel = next(iter(sorted(cls["label"].items(), key=lambda kv: kv[1])))
            p = hf_hub_download(args.repo, rel, repo_type="dataset", revision=args.revision)
            with open(p, encoding="utf-8") as f:
                text = f.read()
            print(f"\n[example annotation] {rel}\n{text.strip()[:600]}")
            try:
                st, lb, end = parse_annotation(text)
                print(f"  -> parsed {len(st)} sections, end {end}, labels {sorted(set(lb))}")
            except Exception as e:
                print(f"  -> could not parse: {e}")
        return

    keep_subsets = {s.strip() for s in args.subsets.split(",") if s.strip()}
    labels = {k: v for k, v in sorted(cls["label"].items(), key=lambda kv: kv[1])
              if k[0] in keep_subsets}
    unknown_sub = [v for (s, _), v in cls["label"].items() if s is None]
    if unknown_sub:
        print(f"[warn] {len(unknown_sub)} annotation files with no subset in their path, "
              f"e.g. {unknown_sub[:3]} -- not used; run --inspect")
    if args.limit:
        labels = dict(list(labels.items())[:args.limit])

    # ---- is this exactly the 300-song expert-verified benchmark?
    loaded = Counter(s for s, _ in labels)
    ids = {(s, st) for s, st in labels}
    expected = {k: v for k, v in {"HarmonixSet": 200, "CN": 100}.items() if k in keep_subsets}
    count_ok = dict(loaded) == expected and len(ids) == sum(expected.values()) \
        and not dups.get("label")
    print(f"[benchmark] annotation files {dict(loaded)} ({len(ids)} distinct ids); "
          f"expected {expected} -> {'OK' if count_ok else 'MISMATCH'}")
    if not count_ok and not args.limit:
        msg = ("SongFormBench should be exactly 200 HarmonixSet + 100 CN expert-verified "
               "songs. Run --inspect and send the output (wrong repo/revision, or the "
               "folder layout is not what the script expects).")
        if not args.allow_count_mismatch:
            raise SystemExit("[stop] " + msg)
        print("[warn] " + msg)

    # ---- download only the files we need (annotations + their audio)
    pairs = []
    for (sub, stem), lrel in labels.items():
        arel = find_match(cls["audio"], sub, stem)
        mrel = find_match(cls["mel"], sub, stem)
        pairs.append((sub, stem, lrel, arel, mrel))
    need = sorted({p for _, _, l, a, _ in pairs for p in (l, a) if p})
    print(f"[download] {len(need)} files (annotations + audio); utils/ and mels are skipped")
    local = {}
    for n, rel in enumerate(need, 1):   # exact paths (names may contain [ ] etc.)
        local[rel] = hf_hub_download(args.repo, rel, repo_type="dataset",
                                     revision=args.revision)
        if n % 100 == 0 or n == len(need):
            print(f"  {n}/{len(need)}")
    try:
        commit = api.dataset_info(args.repo, revision=args.revision).sha
    except Exception:
        commit = None

    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(seg_dir, exist_ok=True)
    songs, report, seen_hash = [], [], {}
    orig_dir = os.path.join(out_dir, "original_audio")
    os.makedirs(orig_dir, exist_ok=True)
    for sub, sid, lrel, arel, mrel in pairs:
        stem = sid if sub == "HarmonixSet" else f"cn_{sid}"
        rec = {"id": sid, "stem": stem, "subset": sub, "annotation_file": lrel,
               "audio_file": arel, "problems": [], "warnings": []}

        # ---- audio (the repo file, untouched, goes to original_audio/)
        audio_dur = rms = mono = sr = None
        if arel is None:
            rec["problems"].append("no audio file in the repo" +
                                   (" (mel-spectrogram only)" if mrel else ""))
        else:
            src = os.path.join(orig_dir, stem + os.path.splitext(arel)[1].lower())
            try:
                if not os.path.exists(src):
                    shutil.copyfile(local[arel], src)
                with open(src, "rb") as f:
                    h = hashlib.sha1(f.read()).hexdigest()
                if h in seen_hash:
                    rec["problems"].append(f"duplicate audio of {seen_hash[h]}")
                seen_hash.setdefault(h, stem)
                wav, sr = read_audio(src)
                wav = np.asarray(wav, dtype=np.float64)
                mono = wav.mean(axis=1) if wav.ndim == 2 else wav
                audio_dur = len(mono) / float(sr)
                rms = float(np.sqrt(np.mean(mono ** 2))) if len(mono) else 0.0
                rec.update(audio_dur=round(audio_dur, 3), sample_rate=int(sr))
            except Exception as e:
                rec["problems"].append(f"audio unreadable: {type(e).__name__}: {e}")
                audio_dur = rms = mono = None

        # ---- annotation
        try:
            with open(local[lrel], encoding="utf-8") as f:
                starts, raw_labels, end_time = parse_annotation(f.read())
        except Exception as e:
            rec["problems"].append(f"annotation unparsable: {e}")
            report.append(rec)
            continue
        order = sorted(range(len(starts)), key=lambda k: starts[k])
        starts = [starts[k] for k in order]
        raw_labels = [raw_labels[k] for k in order]
        if end_time is None and audio_dur is not None:
            rec["warnings"].append("no end marker; using audio length")
            end_time = audio_dur

        # is the unannotated tail just fade-out / silence?
        tail_quiet = tail_db = None
        if mono is not None and end_time is not None and audio_dur - end_time > 0.05:
            tail_quiet, tail_db = edge_is_quiet(mono, sr, end_time, audio_dur,
                                                starts[0], end_time)
        probs, warns, norm = check_song(
            starts, raw_labels, end_time, audio_dur, rms, prechorus=args.prechorus,
            tail_quiet=tail_quiet, dur_tol=args.dur_tol, max_tail=args.max_tail,
            max_first_start=args.max_head, min_sections=args.min_sections)
        rec["problems"] += probs
        rec["warnings"] += warns
        rec.update(n_sections=len(starts), end_time=end_time, labels8=norm,
                   tail_rel_db=tail_db,
                   len_gap=(round(audio_dur - end_time, 3)
                            if audio_dur is not None and end_time is not None else None))

        # ---- how much did the experts change the original Harmonix?
        if args.harmonix_seg_dir and sub == "HarmonixSet":
            op = os.path.join(args.harmonix_seg_dir, f"{sid}.txt")
            if os.path.exists(op):
                with open(op) as f:
                    o_st, o_lb, o_end = parse_msa_text(f.read())
                rec["vs_original"] = {
                    "f_0.5s": round(boundary_hit_rate(o_st[1:], starts[1:], 0.5), 3),
                    "f_3s": round(boundary_hit_rate(o_st[1:], starts[1:], 3.0), 3),
                    "n_sections_original": len(o_st),
                    "end_diff_s": (round(end_time - o_end, 2)
                                   if o_end is not None and end_time is not None else None),
                }

        # ---- kept songs: audio = exactly the annotated span [first start, end]
        if not rec["problems"]:
            t0 = starts[0] if starts[0] > 0.05 else 0.0
            t1 = min(end_time, audio_dur)
            for fn in os.listdir(audio_dir):            # stale files of this song
                if os.path.splitext(fn)[0] == stem:
                    os.remove(os.path.join(audio_dir, fn))
            if t0 > 0 or audio_dur - t1 > 0.05:
                dst = os.path.join(audio_dir, stem + ".wav")
                write_wav(dst, wav[int(round(t0 * sr)):int(round(t1 * sr))], sr)
                rec["crop_s"] = [round(t0, 3), round(t1, 3)]
            else:
                dst = os.path.join(audio_dir, os.path.basename(src))
                shutil.copyfile(src, dst)
            starts = [round(x - t0, 3) for x in starts]
            end_time = round(t1 - t0, 3)
            rec["audio"] = os.path.abspath(dst)
            write_msa_txt(os.path.join(seg_dir, stem + ".txt"), starts, norm, end_time)
            songs.append((rec, starts, norm, end_time))
        report.append(rec)

    # ---------------- summary
    os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
    reasons = Counter(reason_key(p) for r in report for p in r["problems"])
    print(f"\n[clean] {len(songs)} of {len(report)} songs pass every check")
    for sub in sorted({r["subset"] for r in report}):
        rs = [r for r in report if r["subset"] == sub]
        ok = [r for r in rs if not r["problems"]]
        gaps = [abs(r["len_gap"]) for r in rs if r.get("len_gap") is not None]
        print(f"  {sub:12s} kept {len(ok):3d}/{len(rs):3d}   |audio - annotation| (all songs) "
              f"median {np.median(gaps) if gaps else float('nan'):.2f}s, "
              f"max {max(gaps) if gaps else float('nan'):.2f}s")
    if reasons:
        print("  drop reasons:")
        for k, v in reasons.most_common():
            print(f"    {v:3d}  {k}")
    for r in [r for r in report if r["problems"]][:15]:
        print(f"    - {r['stem']}: {'; '.join(r['problems'])}")
    nwarn = sum(1 for r in report if r["warnings"] and not r["problems"])
    wk = Counter(warn_key(w) for r in report if not r["problems"] for w in r["warnings"])
    ncrop = sum(1 for r in report if "crop_s" in r)
    print(f"  {nwarn} kept songs have warnings; {ncrop} cropped to the annotated span")
    for k, v in wk.most_common():
        print(f"    {v:3d}  {k}")
    vs = [r["vs_original"] for r in report if "vs_original" in r]
    if vs:
        print(f"\n[revised vs original Harmonix] {len(vs)} songs: boundary agreement "
              f"F@0.5s median {np.median([v['f_0.5s'] for v in vs]):.3f}, "
              f"F@3s median {np.median([v['f_3s'] for v in vs]):.3f}; "
              f"songs with F@3s < 0.8: {sum(v['f_3s'] < 0.8 for v in vs)}")
    labs = Counter(l for r, _, n, _ in songs for l in n)
    print(f"\n[labels in clean set] {dict(labs.most_common())}")

    summary = {"n_total": len(report), "n_clean": len(songs),
               "benchmark": {"repo": args.repo, "commit": commit, "loaded": dict(loaded),
                             "expected": expected, "distinct_ids": len(ids),
                             "exact_300": count_ok},
               "settings": {k: getattr(args, k) for k in
                            ("repo", "revision", "subsets", "prechorus", "special",
                             "dur_tol", "max_tail", "max_head", "min_sections")},
               "songs": report}
    with open(args.report, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[report] {args.report}")
    if args.check_only:
        print("[check_only] skipping embeddings and manifest")
        return

    # ---------------- embeddings + manifest
    spec = spec_from_args(args)
    cache_dir = args.cache_dir or os.path.join(args.data_root, "cache", "embeddings")
    cache = EmbeddingCache(cache_dir, spec)
    fr = spec.frame_rate
    print(f"\n[extractor] {spec.describe()}  (cache tag {spec.tag()})")
    manifest = []
    for k, (rec, starts, norm, end_time) in enumerate(songs, 1):
        try:
            emb, emb_path = cache.get(rec["audio"])
            fields = annotation_to_manifest_fields(starts, end_time, norm,
                                                   n_frames=len(emb), frame_rate=fr,
                                                   special=args.special)
            if fields is None or len(fields["labels"]) < args.min_sections:
                print(f"[skip] {rec['stem']}: too few segments after conversion")
                continue
            manifest.append({
                "emb": os.path.abspath(emb_path),
                **fields,
                "id": rec["stem"],
                "dataset": "sfb_hx" if rec["subset"] == "HarmonixSet" else "sfb_cn",
                "subset": rec["subset"],
                "audio": rec["audio"],
                "extractor": spec.describe(),
                "extractor_tag": spec.tag(),
            })
            print(f"[ok {k}/{len(songs)}] {rec['stem']}: {len(fields['labels'])} segs "
                  f"{''.join(fields['labels'])}")
        except Exception as e:
            print(f"[fail] {rec['stem']}: {type(e).__name__}: {e}")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"\n[done] wrote {len(manifest)} entries to {args.out}")
    print(f"       audio for chroma / loudness checks: --audio_dir {audio_dir}")


if __name__ == "__main__":
    main()
