"""Segmentation in embedding space.

WHAT THE METRIC ACTUALLY USES: Foote (2000) checkerboard novelty on the cosine
self-similarity matrix. Boundaries are the n highest novelty peaks, where n is
len(target_form) - 1 unless cfg.n_boundaries is set. So the segmenter is told
HOW MANY sections to find (from the target form) but not where they are.

Why not the Matrix Profile arc-curve (FLUSS)? Its long recurrence arcs wash out
the valleys on forms with returning sections (ABACA), so boundaries were missed.
matrix_profile() and _corrected_arc_curve() are kept only for motif/discord
diagnostics; segment_boundaries() does not call them.

Two modes:
  * supply boundaries explicitly (segment_boundaries with boundaries=...)
  * discover them from Foote novelty (boundaries=None)
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

    # KNOWN LIMITATION: novelty is 0 within kernel_size/2 frames of either end
    # (16 frames = 8 s at the default 32-frame kernel and 2 fps), so a shorter
    # intro/outro cannot get its own boundary. Cropping the kernel at the edges
    # was tried and benchmarked: it found some short intros but cut overall
    # boundary accuracy from ~0.83 to 0.39-0.62 on synthetic songs, so it was
    # NOT adopted. If short intros matter, lower cfg.mp_window (e.g. 16).
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


# ---------------------------------------------------------------------------
# form-aware segmentation
# ---------------------------------------------------------------------------

def _align_to_templates(En, letters, templates, bonus, min_len):
    """Exact DP: cut [0, T) into len(letters) segments, in order, each at least
    min_len frames, maximising  sum_t cos(frame_t, template of its letter)
    + sum over boundaries of bonus[b].  O(K*T) with a running max."""
    T, K = len(En), len(letters)
    if K * min_len > T:
        return None, -np.inf
    P = {L: np.concatenate([[0.0], np.cumsum(En @ templates[L])]) for L in set(letters)}
    NEG = -np.inf
    best = np.full((K + 1, T + 1), NEG)
    arg = np.zeros((K + 1, T + 1), dtype=int)
    best[0, 0] = 0.0
    for k in range(1, K + 1):
        Pk = P[letters[k - 1]]
        # cand[s] = best[k-1][s] + bonus at s (if s is an interior cut) - Pk[s]
        b = bonus if k > 1 else np.zeros(T + 1)
        cand = best[k - 1] + b - Pk
        pm = np.maximum.accumulate(cand)               # best start s <= e - min_len
        am = np.maximum.accumulate(np.where(cand == pm, np.arange(T + 1), 0))
        e = np.arange(min_len, T + 1)
        ok = np.isfinite(pm[e - min_len])
        best[k, e[ok]] = pm[e[ok] - min_len] + Pk[e[ok]]
        arg[k, e[ok]] = am[e[ok] - min_len]
    if not np.isfinite(best[K, T]):
        return None, -np.inf
    cuts, e = [], T
    for k in range(K, 0, -1):
        s = arg[k, e]
        cuts.append(s)
        e = s
    cuts = sorted(cuts)[1:]                       # drop the 0 start
    return cuts, float(best[K, T])


def form_aware_boundaries(
    embeddings: np.ndarray,
    letters,
    kernel_size: int = 24,
    novelty_weight: float = 2.0,
    min_len: int = 6,
    n_iter: int = 15,
    inits: list[list[int]] | None = None,
) -> list[int]:
    """Cut a song so that it best realises the given letter sequence.

    Alternates two steps until the cuts stop moving (like k-means with the
    section order fixed by the form):
      1. a template per LETTER = mean of the (L2-normalised) frames currently
         assigned to it -- repeats of a letter share one template;
      2. an exact dynamic-programming alignment of the letter sequence to the
         frames: every frame is scored by its cosine similarity to its letter's
         template, plus a bonus of novelty_weight x Foote novelty at each cut
         (novelty_weight is in frames of similarity, so 2 = "worth two
         well-matched frames").
    Started from Foote cuts and from equal cuts; the higher objective wins.
    Same-letter sections are pulled together and different letters apart, so
    the cuts follow the repeat pattern the form asks for, not novelty alone.

    Caveat: the cuts are fitted to the form, so they are the song's BEST case
    for that form. Compare forms fairly by fitting each form separately
    (validate_human.py --also_form_aware does this).
    """
    E = np.asarray(embeddings, dtype=np.float64)
    T, K = len(E), len(letters)
    letters = [str(x) for x in letters]
    if K <= 1:
        return []
    min_len = max(1, min(min_len, T // K))
    En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
    nov = _foote_novelty(E, kernel_size=kernel_size) if novelty_weight > 0 else np.zeros(T)
    bonus = np.concatenate([novelty_weight * nov, [0.0]])     # index = cut position
    if inits is None:
        inits = []
        f = _peaks(nov, K - 1, max(min_len, T // K // 2)) if novelty_weight > 0 else []
        if len(f) == K - 1:
            inits.append(f)
        inits.append([round(T * k / K) for k in range(1, K)])
    best_cuts, best_obj = None, -np.inf
    for b in inits:
        cuts = sorted(int(x) for x in b)
        obj = -np.inf
        for _ in range(n_iter):
            edges = [0] + cuts + [T]
            templates = {}
            for L in set(letters):
                fr = [En[edges[k]:edges[k + 1]] for k in range(K) if letters[k] == L
                      and edges[k + 1] > edges[k]]
                m = np.vstack(fr).mean(0) if fr else En.mean(0)
                templates[L] = m / (np.linalg.norm(m) + 1e-12)
            new, obj = _align_to_templates(En, letters, templates, bonus, min_len)
            if new is None or new == cuts:
                break
            cuts = new
        if new is not None and obj > best_obj:
            best_cuts, best_obj = cuts, obj
    return best_cuts if best_cuts is not None else [round(T * k / K) for k in range(1, K)]


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
