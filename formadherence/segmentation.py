"""Matrix-Profile segmentation in embedding space.

The Matrix Profile (Yeh et al. 2016) stores, for every length-w subsequence of a
series, the z-normalized Euclidean distance to its nearest non-trivial
neighbour. Low values mark motifs (repeated material); high values mark discords
(novel material). We compute an MP over the per-frame embedding sequence and
derive boundaries from a corrected arc-curve (the FLUSS/regime-change idea of
Gharghabi et al. 2017): counting how many nearest-neighbour arcs cross each
frame index, section boundaries appear as valleys where few arcs cross.

Two modes, per the design decision for this build:
  * supply boundaries explicitly (segment_boundaries with boundaries=...)
  * discover them from the MP arc-curve (boundaries=None)
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class Segment:
    """A contiguous span of frames [start, end) with an optional form label."""
    start: int
    end: int
    label: str | None = None

    @property
    def length(self) -> int:
        return self.end - self.start

    def frames(self, embeddings: np.ndarray) -> np.ndarray:
        return embeddings[self.start:self.end]


def _znorm(x: np.ndarray) -> np.ndarray:
    mu = x.mean()
    sd = x.std()
    if sd < 1e-8:
        return x - mu
    return (x - mu) / sd


def matrix_profile(
    embeddings: np.ndarray,
    window: int = 32,
    exclusion_zone: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute a (multi-dimensional) matrix profile over a frame sequence.

    embeddings : (T, d) per-frame vectors.
    Returns (profile, index) where profile[i] is the distance from subsequence i
    to its nearest non-trivial neighbour, and index[i] is that neighbour's start.

    Distance is the mean over dimensions of the per-dimension z-normalized
    Euclidean distance -- a simple, dependency-free multivariate MP suitable for
    the moderate T of a song at a few Hz frame rate. (stumpy.mstump is the
    production path; this is the auditable reference.)
    """
    E = np.asarray(embeddings, dtype=np.float64)
    T, d = E.shape
    L = T - window + 1
    if L <= 1:
        raise ValueError(f"series too short (T={T}) for window={window}")

    # Pre-z-normalize every window per dimension: shape (L, window, d).
    subs = np.empty((L, window, d))
    for i in range(L):
        for c in range(d):
            subs[i, :, c] = _znorm(E[i:i + window, c])

    excl = max(1, int(window * exclusion_zone))
    profile = np.full(L, np.inf)
    index = np.full(L, -1, dtype=int)

    for i in range(L):
        # Euclidean distance from window i to all windows j, meaned over dims.
        diff = subs - subs[i][None, :, :]           # (L, window, d)
        dist = np.sqrt((diff ** 2).sum(axis=1)).mean(axis=1)  # (L,)
        lo, hi = max(0, i - excl), min(L, i + excl + 1)
        dist[lo:hi] = np.inf                          # exclusion zone
        j = int(np.argmin(dist))
        profile[i] = dist[j]
        index[i] = j
    return profile, index


def _foote_novelty(embeddings: np.ndarray, kernel_size: int = 32) -> np.ndarray:
    """Foote (2000) checkerboard-kernel novelty curve over the SSM.

    Boundaries are peaks: points where the self-similarity matrix transitions
    from one homogeneous block to another. This is robust to recurring sections
    (unlike the arc-curve, whose long recurrence arcs wash out valleys on forms
    like ABACA), which is why it is the boundary-discovery signal here. The
    Matrix Profile is retained for motif/discord characterization.
    """
    E = np.asarray(embeddings, dtype=np.float64)
    T = len(E)
    En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
    S = En @ En.T  # cosine self-similarity in [-1, 1]

    L = kernel_size
    if L % 2 == 1:
        L += 1
    half = L // 2
    # Radial Gaussian-tapered checkerboard kernel: +++/--- quadrant signs.
    ax = np.arange(-half, half)
    gx, gy = np.meshgrid(ax, ax)
    taper = np.exp(-(gx ** 2 + gy ** 2) / (2 * (half / 2.0) ** 2))
    sign = np.sign(gx) * np.sign(gy)
    kernel = sign * taper

    nov = np.zeros(T)
    for i in range(T):
        lo, hi = i - half, i + half
        if lo < 0 or hi > T:
            continue
        block = S[lo:hi, lo:hi]
        nov[i] = float(np.sum(block * kernel))
    nov[nov < 0] = 0.0
    if nov.max() > 0:
        nov /= nov.max()
    return nov


def _peaks(curve: np.ndarray, n: int, min_gap: int) -> list[int]:
    """Top-n local maxima under a minimum-separation constraint."""
    order = np.argsort(curve)[::-1]
    chosen: list[int] = []
    for i in order:
        i = int(i)
        if curve[i] <= 0:
            break
        if all(abs(i - c) >= min_gap for c in chosen):
            chosen.append(i)
        if len(chosen) >= n:
            break
    return sorted(chosen)


def _corrected_arc_curve(index: np.ndarray) -> np.ndarray:
    """FLUSS arc-crossing curve, corrected for the parabolic edge bias.

    Each subsequence points to its nearest neighbour; the arc between them
    crosses every index in between. Boundaries are valleys (few crossings).
    """
    L = len(index)
    crossings = np.zeros(L + 1)
    for i in range(L):
        j = index[i]
        if j < 0:
            continue
        lo, hi = min(i, j), max(i, j)
        crossings[lo] += 1
        crossings[hi] -= 1
    arc = np.cumsum(crossings)[:L]

    # Expected arc count under a uniform-random model is an inverted parabola;
    # divide it out so edges are not spuriously favoured as boundaries.
    idx = np.arange(L)
    ideal = 2.0 * idx * (L - idx) / max(L, 1)
    ideal[ideal < 1e-8] = 1e-8
    corrected = np.minimum(arc / ideal, 1.0)
    return corrected


def segment_boundaries(
    embeddings: np.ndarray,
    boundaries: list[int] | None = None,
    n_boundaries: int | None = None,
    window: int = 32,
    exclusion_zone: float = 0.5,
    min_gap: int | None = None,
) -> list[int]:
    """Return interior boundary frame indices (not including 0 or T).

    If `boundaries` is given, it is validated and returned (known-boundary mode).
    Otherwise boundaries are discovered from the corrected arc-curve by picking
    the `n_boundaries` deepest, sufficiently separated valleys.
    """
    T = len(embeddings)
    if boundaries is not None:
        b = sorted(int(x) for x in boundaries if 0 < x < T)
        return b

    if n_boundaries is None or n_boundaries <= 0:
        return []

    if min_gap is None:
        min_gap = max(window, T // (n_boundaries + 1) // 2)
    nov = _foote_novelty(embeddings, kernel_size=window)
    return _peaks(nov, n_boundaries, min_gap)


def boundaries_to_segments(
    boundaries: list[int],
    T: int,
    labels: list[str] | None = None,
) -> list[Segment]:
    """Turn interior boundaries into contiguous Segments spanning [0, T)."""
    edges = [0] + list(boundaries) + [T]
    segs = [Segment(edges[k], edges[k + 1]) for k in range(len(edges) - 1)]
    if labels is not None:
        if len(labels) != len(segs):
            raise ValueError(
                f"{len(labels)} labels for {len(segs)} segments"
            )
        for s, lab in zip(segs, labels):
            s.label = lab
    return segs
