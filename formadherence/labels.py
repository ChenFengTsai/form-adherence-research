"""Annotation -> (frame boundaries, letters), shared by every manifest builder.

Fixes vs the old per-builder copies:
  * Label alignment. Converting second-based section starts to frames can make
    a short section collapse to zero frames. The old code then dropped the LAST
    label, shifting every later label onto the wrong segment. sections_to_frames
    drops exactly the collapsed section, so labels stay attached to their frames.
  * Special labels (Silence, End, applause, ...) get a UNIQUE letter each by
    default, so two silences never count as a "same-letter" pair (they are
    trivially similar and inflated the AUC). Options: unique | keep | drop.
  * One code path for SALAMI letter labels whether or not "Silence" appears
    (before, one Silence label switched the whole song to a different path and
    A / A' became different letters). The prime count is kept as `variants`
    (A=0, A'=1, A''=2) so annotated variations can be used for Axis 3 later.
  * One functional-label family table shared by SALAMI and SongFormBench, so
    letters mean the same thing in both datasets.
"""

from __future__ import annotations

import re

PRIMES = "'’′\""

# Labels that are not musical sections. Matched on the compact form (lowercase,
# no spaces/underscores/hyphens). "z" is SALAMI's non-music/silence letter --
# check one of your annotation files and pass special_letters=() if your copy
# uses Z as a real section letter.
SPECIAL = {"silence", "end", "nofunction", "applause", "crowd", "noise",
           "talking", "speech", "tuning", "count", "countin"}
DEFAULT_SPECIAL_LETTERS = ("Z",)

# compact key -> family; checked longest-key-first so prefixes don't steal.
FAMILIES = {
    "verse": "verse", "chorus": "chorus", "bridge": "bridge",
    "intro": "intro", "outro": "outro",
    "prechorus": "prechorus", "postchorus": "postchorus",
    "inst": "instrumental", "instrumental": "instrumental",
    "solo": "solo", "guitarsolo": "solo", "gtr": "instrumental",
    "break": "interlude", "breakdown": "interlude", "interlude": "interlude",
    "transition": "interlude",
    "refrain": "chorus", "hook": "chorus",
    "theme": "theme", "maintheme": "theme", "head": "head",
    "coda": "outro", "fadeout": "outro", "fade": "outro", "ending": "outro",
}
_KEYS = sorted(FAMILIES, key=len, reverse=True)


def _compact(raw: str) -> str:
    s = str(raw).strip().lower().split("(")[0]
    return re.sub(r"[\s_\-]+", "", s)


def is_special(raw: str, special_letters=DEFAULT_SPECIAL_LETTERS) -> bool:
    c = _compact(raw).rstrip("0123456789")
    if c in SPECIAL:
        return True
    letter = str(raw).strip().rstrip(PRIMES)
    return letter in special_letters


def canonical_family(raw: str) -> str:
    """Functional label -> coarse family (verse/chorus/...)."""
    c = _compact(raw).rstrip("0123456789")
    for base in ("chorus", "verse", "intro", "outro", "bridge"):
        if c == "inst" + base:        # instchorus -> chorus
            return FAMILIES[base]
    for k in _KEYS:
        if c.startswith(k):
            return FAMILIES[k]
    return c or "unknown"


def is_letter_label(raw: str) -> bool:
    s = str(raw).strip().rstrip(PRIMES)
    return len(s) == 1 and s.isalpha()


def label_to_token(raw: str, special_letters=DEFAULT_SPECIAL_LETTERS):
    """Return (token, variant, special). token: letter for A/B/A' labels,
    family for functional labels."""
    if is_special(raw, special_letters):
        return _compact(raw) or "special", 0, True
    if is_letter_label(raw):
        s = str(raw).strip()
        stripped = s.rstrip(PRIMES)
        variant = len(s) - len(stripped)
        return "L:" + stripped.upper(), variant, False
    return "F:" + canonical_family(raw), 0, False


def tokens_to_letters(tokens, specials, special="unique"):
    """Assign A, B, C ... in order of first appearance. special='unique' gives
    each special segment its own letter; 'keep' groups them like any token."""
    mapping: dict[str, str] = {}
    out = []
    n_unique = 0

    def next_letter():
        k = len(mapping) + n_unique
        return chr(ord("A") + k) if k < 26 else f"X{k}"

    for tok, sp in zip(tokens, specials):
        if sp and special == "unique":
            out.append(next_letter())
            n_unique += 1
            continue
        if tok not in mapping:
            mapping[tok] = next_letter()
        out.append(mapping[tok])
    return out


def sections_to_frames(starts, end_time, n_frames, frame_rate):
    """Section start times (sorted, seconds) + end of last section -> list of
    (index_into_sections, start_frame) for the sections that keep >= 1 frame.

    Segment k spans [f_k, f_{k+1}). Sections that round to zero frames, or that
    start past the end of the audio, are dropped -- and only those. The first
    kept section is stretched to frame 0 and the last to n_frames (the audio
    may be a bit longer/shorter than the annotation).
    """
    # end_time is accepted for clarity of call sites, but the last section is
    # always stretched to the end of the audio (same as the old behaviour).
    f = [min(n_frames, max(0, int(round(t * frame_rate)))) for t in starts]
    f.append(n_frames)
    kept = [(k, f[k]) for k in range(len(starts)) if f[k + 1] > f[k]]
    if kept:
        kept[0] = (kept[0][0], 0)
    return kept


def annotation_to_manifest_fields(starts, end_time, raw_labels, n_frames,
                                  frame_rate, special="unique",
                                  special_letters=DEFAULT_SPECIAL_LETTERS):
    """Full conversion used by both builders.

    Returns dict(boundaries, labels, labels_raw, variants) with
    len(labels) == len(boundaries) + 1, or None if nothing survives.
    """
    order = sorted(range(len(starts)), key=lambda k: starts[k])
    starts = [float(starts[k]) for k in order]
    raw_labels = [str(raw_labels[k]) for k in order]

    kept = sections_to_frames(starts, end_time, n_frames, frame_rate)
    if not kept:
        return None
    toks = [label_to_token(raw_labels[k], special_letters) for k, _ in kept]
    if special == "drop":
        # merge each special section into its predecessor (or successor at the
        # start) so frames stay covered and labels stay aligned
        merged = []
        for (k, fr), (tok, var, sp) in zip(kept, toks):
            if sp and merged:
                continue
            merged.append(((k, fr), (tok, var, sp)))
        while merged and merged[0][1][2] and len(merged) > 1:
            merged.pop(0)
            merged[0] = ((merged[0][0][0], 0), merged[0][1])
        kept = [m[0] for m in merged]
        toks = [m[1] for m in merged]
    letters = tokens_to_letters([t for t, _, _ in toks], [s for _, _, s in toks],
                                special=special)
    return {
        "boundaries": [fr for _, fr in kept[1:]],
        "labels": letters,
        "labels_raw": [raw_labels[k] for k, _ in kept],
        "variants": [v for _, v, _ in toks],
    }
