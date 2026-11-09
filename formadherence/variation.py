"""Axis 3 (within-letter variation) measured on MUSICAL content.

Goal: the second A should be recognisably A, but with small musical changes
(a few different notes, an embellished melody) -- not a copy. Added noise, a
different mix or EQ, or re-encoding must NOT count as variation.

Why not the embedding version: CLAP/EnCodec distances grow for any change in
sound, so faint noise (-30 dB) made exact copies look like genuine repeats on
real songs (caught 13% vs 80% without noise). And OT ignores order, while a
melody is notes in order.

Method
------
1. Chroma: per frame (default 10 per second), the energy of each of the 12
   pitch classes, log-compressed and L2-normalized. Notes change chroma;
   noise, EQ and timbre change it little.
2. DTW: align the two repeats in time (tempo drift allowed within a band), so
   bar 3 of A1 is compared with bar 3 of A2.
3. note_change f = share of aligned moments whose chroma differs by more than
   note_change_threshold (cosine distance). Copy -> f ~ 0. A few changed notes
   -> a small f. Completely different material -> large f (that is Axis 2's
   problem, not this axis's: no penalty here for large f).
4. Score: 0 at f <= note_change_floor, rising linearly to 1 at
   note_change_floor + note_change_ramp. Set them from MEASUREMENT, with
   scripts/check_variation.py --update_config: floor = what true copies
   measure (measurement noise), full score = what ~1 s of changed notes
   measures. Not from human repeats: some are production copies (f ~ 0),
   which drags a percentile-based floor to ~0 (seen in early tests: 0.0033).
   0.3 s at each section edge is ignored (edge_trim_s).
"""

from __future__ import annotations

from itertools import combinations
import numpy as np


# ---------------------------------------------------------------- chroma ----

def chroma_features(audio, sample_rate, fps: float = 10.0, method: str = "stft",
                    fmin: float = 55.0, fmax: float = 4200.0, n_fft: int | None = None):
    """(T, 12) chroma at about `fps` frames per second, and the true fps.

    method "stft": dependency-free (numpy), log-magnitude STFT folded into pitch
    classes. method "cqt": librosa.feature.chroma_cqt (better pitch resolution
    for low notes; needs librosa).
    """
    x = np.asarray(audio, dtype=np.float64)
    if x.ndim == 2:
        x = x.mean(axis=0 if x.shape[0] < x.shape[1] else 1)
    sr = int(sample_rate)
    if method == "cqt":
        import librosa
        hop = max(128, int(round(sr / fps / 128)) * 128)
        C = librosa.feature.chroma_cqt(y=x.astype(np.float32), sr=sr, hop_length=hop).T
        return _normalize_chroma(C, None), sr / hop

    hop = max(1, int(round(sr / fps)))
    if n_fft is None:
        n_fft = 1 << int(np.ceil(np.log2(max(2048, 0.1 * sr))))     # ~>= 0.1 s
    if len(x) < n_fft:
        x = np.pad(x, (0, n_fft - len(x)))
    n = 1 + (len(x) - n_fft) // hop
    win = np.hanning(n_fft)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    keep = (freqs >= fmin) & (freqs <= fmax)
    pc = (np.round(69 + 12 * np.log2(freqs[keep] / 440.0)).astype(int)) % 12
    fold = np.zeros((keep.sum(), 12))
    fold[np.arange(keep.sum()), pc] = 1.0
    C = np.empty((n, 12))
    energy = np.empty(n)
    for k in range(n):
        seg = x[k * hop:k * hop + n_fft] * win
        pw = np.abs(np.fft.rfft(seg))[keep] ** 2
        energy[k] = float(pw.sum())
        C[k] = pw @ fold
    # Power chroma, normalized per frame, then LIGHT log compression. Strong
    # compression (log1p(100*mag), tried first) lifts the noise floor so much
    # that faint noise looked like changed notes; this keeps sound-only
    # changes (gain, -30 dB noise, low-pass EQ) at ~0 change on synthetic music.
    C = np.log1p(10.0 * C / (C.max(axis=1, keepdims=True) + 1e-12))
    return _normalize_chroma(C, energy), sr / hop


def _normalize_chroma(C, energy):
    C = np.asarray(C, dtype=np.float64)
    norms = np.linalg.norm(C, axis=1, keepdims=True)
    out = C / (norms + 1e-12)
    if energy is not None:
        silent = energy < 1e-6 * (np.median(energy) + 1e-12)
        out[silent] = 0.0
    return out


# ------------------------------------------------------------------- DTW ----

def _cos_dist(A, B):
    """Cosine distance matrix; two silent (zero) frames match (0), silence vs
    sound is a full mismatch (1)."""
    na = np.linalg.norm(A, axis=1) > 0
    nb = np.linalg.norm(B, axis=1) > 0
    D = 1.0 - np.clip(A @ B.T, -1.0, 1.0)
    D[~na[:, None] & ~nb[None, :]] = 0.0
    D[na[:, None] ^ nb[None, :]] = 1.0
    return np.clip(D, 0.0, 2.0)


def dtw_align(A, B, band: float = 0.2):
    """Align sequences A (n,12), B (m,12). Sakoe-Chiba band as a fraction of
    the length. Returns the per-step costs along the optimal path."""
    n, m = len(A), len(B)
    if n == 0 or m == 0:
        return np.array([1.0])
    C = _cos_dist(A, B)
    w = max(int(np.ceil(band * max(n, m))), abs(n - m) + 1)
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        jc = int(round(i * m / n))
        lo, hi = max(1, jc - w), min(m, jc + w)
        for j in range(lo, hi + 1):
            D[i, j] = C[i - 1, j - 1] + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    # backtrack
    i, j, steps = n, m, []
    while i > 0 and j > 0:
        steps.append(C[i - 1, j - 1])
        moves = (D[i - 1, j - 1], D[i - 1, j], D[i, j - 1])
        k = int(np.argmin(moves))
        if k == 0:
            i, j = i - 1, j - 1
        elif k == 1:
            i -= 1
        else:
            j -= 1
    return np.asarray(steps[::-1])


def note_change(A, B, threshold: float = 0.2, band: float = 0.2):
    """Share of aligned moments whose notes differ, and the mean aligned
    distance."""
    steps = dtw_align(A, B, band)
    return float((steps > threshold).mean()), float(steps.mean())


# ------------------------------------------------------- the axis itself ----

def note_change_score(f: float, floor: float, ramp: float) -> float:
    if f <= floor:
        return 0.0
    return float(np.clip((f - floor) / max(ramp, 1e-9), 0.0, 1.0))


def trim_edges(sec, fps, trim_s):
    """Drop trim_s seconds at both ends of a section (if it stays >= 1 s):
    frames there straddle the boundary and pick up the neighbouring section,
    and annotated boundaries are only accurate to a fraction of a second."""
    k = int(round(trim_s * fps))
    if k > 0 and len(sec) - 2 * k >= max(2, int(fps)):
        return sec[k:len(sec) - k]
    return sec


def variation_from_chroma(chroma, chroma_fps, segment_seconds, letters, cfg):
    """Axis 3 for one song.

    segment_seconds: list of (start_s, end_s) per section; letters: per section.
    Returns dict(per_letter, per_letter_soft, pairs) where pairs maps
    "A0-A2" -> note-change fraction f.
    """
    T = len(chroma)
    secs = []
    for s, e in segment_seconds:
        a = int(np.clip(round(s * chroma_fps), 0, T))
        b = int(np.clip(round(e * chroma_fps), a, T))
        secs.append(trim_edges(chroma[a:b], chroma_fps, getattr(cfg, "edge_trim_s", 0.3)))
    groups: dict[str, list[int]] = {}
    for k, lab in enumerate(letters):
        groups.setdefault(lab, []).append(k)
    per_letter, per_letter_soft, pairs = {}, {}, {}
    top = cfg.note_change_floor + cfg.note_change_ramp
    for lab, members in groups.items():
        if len(members) < 2:
            continue
        sc, soft = [], []
        for i, j in combinations(members, 2):
            if len(secs[i]) < 2 or len(secs[j]) < 2:
                continue
            f, _ = note_change(secs[i], secs[j], cfg.note_change_threshold, cfg.dtw_band)
            pairs[f"{lab}{i}-{lab}{j}"] = f
            sc.append(note_change_score(f, cfg.note_change_floor, cfg.note_change_ramp))
            soft.append(float(np.clip(f / max(top, 1e-9), 0.01, 1.0)))
        if sc:
            per_letter[lab] = float(np.mean(sc))
            per_letter_soft[lab] = float(np.mean(soft))
    return {"per_letter": per_letter, "per_letter_soft": per_letter_soft, "pairs": pairs}
