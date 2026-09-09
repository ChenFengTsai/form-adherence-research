"""Synthetic per-frame embeddings with a known form, for testing/calibration.

Each letter gets a distinct latent "theme" vector; a segment for that letter is a
cloud of frames around the theme. Knobs let us construct the failure modes the
metric must detect:

  * variation_scale  : within-segment jitter (natural variation vs copy-paste)
  * copy_paste       : if True, all repeats of a letter are byte-identical
  * merge_letters    : map letters together to simulate "verse == chorus"
"""

from __future__ import annotations

import numpy as np


def make_song_embeddings(
    form: str,
    dim: int = 32,
    seg_len: int = 120,
    theme_sep: float = 3.0,
    variation_scale: float = 0.4,
    frame_noise: float = 0.15,
    copy_paste: bool = False,
    merge_letters: dict[str, str] | None = None,
    seed: int = 0,
):
    """Return (embeddings (T,d), boundaries list). Frames are L2-normalized so
    the cosine ground metric in the OT step is well behaved."""
    rng = np.random.default_rng(seed)
    merge_letters = merge_letters or {}

    unique = sorted(set(form))
    themes = {}
    for lab in unique:
        src = merge_letters.get(lab, lab)
        if src not in themes:
            themes[src] = rng.normal(0, theme_sep, size=dim)
        themes[lab] = themes[src]

    # A fixed per-letter "instance offset" so repeats differ by a controlled
    # amount unless copy_paste is requested.
    per_instance = {}

    segments = []
    boundaries = []
    cursor = 0
    counts: dict[str, int] = {}
    for lab in form:
        counts[lab] = counts.get(lab, 0) + 1
        inst = counts[lab] - 1
        base = themes[lab].copy()
        if not copy_paste and inst > 0:
            key = (lab, inst)
            if key not in per_instance:
                per_instance[key] = rng.normal(0, variation_scale, size=dim)
            base = base + per_instance[key]
        frames = base[None, :] + rng.normal(0, frame_noise, size=(seg_len, dim))
        segments.append(frames)
        cursor += seg_len
        boundaries.append(cursor)
    boundaries = boundaries[:-1]  # drop the final edge (== T)

    E = np.vstack(segments)
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
    return E, boundaries
