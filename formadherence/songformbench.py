"""SongFormBench (ASLP-lab) helpers: parse annotations, normalise labels,
and check that every song is clean before it enters a manifest.

SongFormBench = 200 HarmonixSet songs (expert-revised annotations, audio made
from Harmonix's own mel-spectrograms, so it is the SAME version that was
annotated) + 100 Chinese pop songs. Labels use one 8-class scheme:
intro, verse, pre-chorus, chorus, bridge, inst, outro, silence.

Annotation format (MSA txt): one "<start_sec> <label>" per line, the label is
the section that STARTS there; the last line is "<end_sec> end".

Nothing here downloads anything; scripts/build_songformbench_manifest.py does.
"""

from __future__ import annotations

import json
import re

LABELS8 = ("intro", "verse", "pre-chorus", "chorus", "bridge", "inst",
           "outro", "silence")

# spelling variants only -- anything else is reported as an unknown label
# rather than silently remapped (the benchmark should use exactly LABELS8)
_ALIASES = {
    "prechorus": "pre-chorus", "instrumental": "inst", "silent": "silence",
}


def normalize_label(raw, prechorus: str = "keep"):
    """Raw label -> one of LABELS8, or None if unknown.
    prechorus='verse' folds pre-chorus into verse (the paper's 7-class eval)."""
    c = re.sub(r"[\s_\-]+", "", str(raw).strip().lower()).rstrip("0123456789")
    lab = _ALIASES.get(c, c)
    if lab == "prechorus":
        lab = "pre-chorus"
    if lab not in LABELS8:
        return None
    if lab == "pre-chorus" and prechorus == "verse":
        return "verse"
    return lab


# --------------------------------------------------------------------------
# parsing: accept every shape the dataset might use
# --------------------------------------------------------------------------

def _is_num(s) -> bool:
    try:
        float(s)
        return True
    except (TypeError, ValueError):
        return False


def parse_msa_text(text: str):
    """MSA txt -> (starts, labels, end_time).

    Accepts "<t> <label>" lines ending with "<t> end", and also
    "<start> <end> <label>" lines (end time = last segment end)."""
    starts, labels, end_time, last_seg_end = [], [], None, None
    for line in str(text).splitlines():
        parts = line.strip().split()
        if not parts or line.strip().startswith("#"):
            continue
        if len(parts) >= 3 and _is_num(parts[0]) and _is_num(parts[1]):
            starts.append(float(parts[0]))
            labels.append(" ".join(parts[2:]))
            last_seg_end = float(parts[1])
            continue
        if len(parts) >= 2 and _is_num(parts[0]):
            lab = " ".join(parts[1:])
            if lab.strip().lower() == "end":
                end_time = float(parts[0])
            else:
                starts.append(float(parts[0]))
                labels.append(lab)
    if end_time is None:
        end_time = last_seg_end
    return starts, labels, end_time


def parse_annotation(obj):
    """Any annotation object -> (starts, labels, end_time) or raise ValueError.

    Handles: MSA text, a JSON string, a list of dicts
    ({start,end,label} / {time,label} / {start_time,end_time,label}),
    a list of [start, end, label] or [time, label], or a dict holding one of
    those under segments/labels/annotations."""
    if obj is None:
        raise ValueError("no annotation")
    if isinstance(obj, bytes):
        obj = obj.decode("utf-8", "replace")
    if isinstance(obj, str):
        s = obj.strip()
        if s[:1] in "[{":
            try:
                return parse_annotation(json.loads(s))
            except json.JSONDecodeError:
                pass
        st, lb, end = parse_msa_text(s)
        if not st:
            raise ValueError("could not parse annotation text")
        return st, lb, end
    if isinstance(obj, dict):
        for k in ("segments", "labels", "label", "annotations", "annotation",
                  "structure", "msa"):
            if k in obj:
                return parse_annotation(obj[k])
        # column-wise dict {start:[...], end:[...], label:[...]}
        if "start" in obj and "label" in obj:
            rows = [dict(start=s, label=l, end=e) for s, l, e in
                    zip(obj["start"], obj["label"], obj.get("end", [None] * len(obj["start"])))]
            return parse_annotation(rows)
        raise ValueError(f"unknown annotation dict keys {list(obj)[:8]}")
    if isinstance(obj, (list, tuple)):
        starts, labels, end_time = [], [], None
        for it in obj:
            if isinstance(it, dict):
                st = it.get("start", it.get("start_time", it.get("time", it.get("onset"))))
                en = it.get("end", it.get("end_time", it.get("offset")))
                lab = it.get("label", it.get("labels", it.get("name")))
                if isinstance(lab, (list, tuple)):
                    lab = lab[0] if lab else None
            elif isinstance(it, (list, tuple)) and len(it) >= 2:
                if len(it) >= 3 and _is_num(it[1]):
                    st, en, lab = it[0], it[1], it[2]
                else:
                    st, en, lab = it[0], None, it[1]
            else:
                raise ValueError(f"unknown segment item {it!r}")
            if lab is None or st is None:
                raise ValueError(f"segment without start/label: {it!r}")
            if str(lab).strip().lower() == "end":
                end_time = float(st)
                continue
            starts.append(float(st))
            labels.append(str(lab))
            if en is not None:
                end_time = max(end_time or 0.0, float(en))
        if not starts:
            raise ValueError("empty annotation")
        return starts, labels, end_time
    raise ValueError(f"unsupported annotation type {type(obj).__name__}")


# --------------------------------------------------------------------------
# cleanliness checks
# --------------------------------------------------------------------------

DEFAULT_CHECKS = dict(
    dur_tol=1.0,          # audio may be this much SHORTER than the annotation end (s)
    max_tail=10.0,        # unannotated MUSIC allowed after the annotation end (s);
                          # a quiet tail (fade/silence) is allowed at any length.
                          # Kept songs are cropped to the annotated span.
    max_first_start=10.0, # unannotated audio allowed before the first section (s);
                          # cropped away
    min_sections=3,       # musical (non-silence) sections needed
    min_duration=30.0,    # shortest usable song (s)
    min_rms=1e-4,         # audio must not be (near) silent
    short_section=1.0,    # musical sections shorter than this -> warning
)


def check_song(starts, raw_labels, end_time, audio_dur=None, rms=None,
               prechorus="keep", tail_quiet=None, **kw):
    """Return (problems, warnings, norm_labels).

    problems -> the song is dropped; warnings -> kept, but reported."""
    c = {**DEFAULT_CHECKS, **kw}
    problems, warnings = [], []
    norm = [normalize_label(l, prechorus) for l in raw_labels]
    unknown = sorted({r for r, n in zip(raw_labels, norm) if n is None})
    if unknown:
        problems.append(f"unknown labels {unknown}")
    if not starts:
        problems.append("no sections")
        return problems, warnings, norm
    if any(b <= a for a, b in zip(starts, starts[1:])):
        problems.append("section times not strictly increasing")
    if end_time is None:
        problems.append("no end time")
    elif end_time <= starts[-1]:
        problems.append("end time before last section start")
    if starts[0] > c["max_first_start"]:
        problems.append(f"annotation starts at {starts[0]:.1f}s (> {c['max_first_start']}s)")
    music = [n for n in norm if n not in (None, "silence")]
    if len(music) < c["min_sections"]:
        problems.append(f"only {len(music)} musical sections")
    if audio_dur is not None:
        if audio_dur < c["min_duration"]:
            problems.append(f"audio only {audio_dur:.1f}s")
        if end_time is not None:
            gap = audio_dur - end_time
            if gap < -c["dur_tol"]:
                problems.append(f"audio shorter than annotation: {audio_dur:.1f}s vs "
                                f"end {end_time:.1f}s ({gap:+.1f}s)")
            elif gap > c["max_tail"] and not tail_quiet:
                problems.append(f"unannotated audio after the annotation end: {gap:.1f}s "
                                f"of music ({audio_dur:.1f}s audio, end {end_time:.1f}s)")
            elif gap > c["dur_tol"]:
                warnings.append(f"trailing {gap:.1f}s not annotated "
                                f"({'quiet' if tail_quiet else 'music'}) -> cropped")
    if rms is not None and rms < c["min_rms"]:
        problems.append(f"audio near-silent (rms {rms:.2e})")
    # warnings
    if 0.05 < starts[0] <= c["max_first_start"]:
        warnings.append(f"leading {starts[0]:.1f}s not annotated -> cropped")
    if end_time is not None and starts:
        bounds = list(starts) + [end_time]
        short = [n for n, a, b in zip(norm, bounds, bounds[1:])
                 if n not in (None, "silence") and b - a < c["short_section"]]
        if short:
            warnings.append(f"{len(short)} musical section(s) < {c['short_section']}s")
    if any(a == b and a is not None for a, b in zip(norm, norm[1:])):
        warnings.append("adjacent sections with the same label")
    return problems, warnings, norm


def edge_is_quiet(mono, sr, t0, t1, song_t0, song_t1, rel_db=-20.0):
    """Is audio in [t0, t1) quiet (fade/silence) relative to the song body
    [song_t0, song_t1)? rel_db: RMS threshold relative to the body."""
    import numpy as np
    seg = mono[int(t0 * sr):int(t1 * sr)]
    body = mono[int(song_t0 * sr):int(song_t1 * sr)]
    if len(seg) == 0:
        return True, float("-inf")
    rs = float(np.sqrt(np.mean(np.square(seg))))
    rb = float(np.sqrt(np.mean(np.square(body)))) if len(body) else 0.0
    if rb <= 0:
        return False, float("nan")
    db = 20 * np.log10(max(rs, 1e-12) / rb)
    return bool(db < rel_db), round(float(db), 1)


def boundary_hit_rate(est, ref, tol):
    """One-to-one boundary matching within tol seconds -> F-measure."""
    est, ref = list(est), list(ref)
    if not est and not ref:
        return 1.0
    if not est or not ref:
        return 0.0
    pairs = sorted((abs(e - r), i, j) for i, e in enumerate(est)
                   for j, r in enumerate(ref) if abs(e - r) <= tol)
    ue, ur, hits = set(), set(), 0
    for _, i, j in pairs:
        if i not in ue and j not in ur:
            ue.add(i)
            ur.add(j)
            hits += 1
    p, r = hits / len(est), hits / len(ref)
    return 2 * p * r / (p + r) if p + r else 0.0


def write_msa_txt(path, starts, labels, end_time):
    with open(path, "w") as f:
        for t, l in zip(starts, labels):
            f.write(f"{t:.3f}\t{l}\n")
        f.write(f"{end_time:.3f}\tend\n")
