"""The three-axis Form-Adherence metric and its hierarchical aggregation.

Inputs
------
embeddings : (T, d) per-frame embeddings for one generated song (already
             extracted; the CLAP/EnCodec stage is external to this package).
target_form : str  e.g. "ABACA"  -- the requested letter sequence.
boundaries  : optional interior frame indices. If omitted, discovered via
              Matrix Profile with as many boundaries as the form implies.

Axes
----
Axis 1  Cross-Letter Distinctness : mean OT distance between segments of
        DIFFERENT letters -> higher is better, mapped to [0, 1].
Axis 2  Same-Letter Similarity    : closeness of segments sharing a letter
        -> higher is better.
Axis 3  Within-Letter Variation   : same-letter pairs should sit inside the
        band (variation_floor, tau): not copy-paste, not drifted.

Aggregation is hierarchical: score each letter group, then combine groups, then
combine axes. This yields per-axis and per-letter diagnostics plus one overall
number -- the shapeable reward vector Part II consumes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
import numpy as np

from .config import FormAdherenceConfig
from .segmentation import (
    Segment,
    segment_boundaries,
    boundaries_to_segments,
)
from .ot import wasserstein_distance, sinkhorn_distance


@dataclass
class FormAdherenceResult:
    overall: float
    distinctness: float          # Axis 1
    similarity: float            # Axis 2
    variation: float             # Axis 3
    target_form: str
    segments: list[Segment]
    # Diagnostics for interpretability / reward-hacking audits:
    per_letter_similarity: dict[str, float] = field(default_factory=dict)
    per_letter_variation: dict[str, float] = field(default_factory=dict)
    cross_pair_distances: dict[str, float] = field(default_factory=dict)
    same_pair_distances: dict[str, float] = field(default_factory=dict)

    def as_reward_vector(self) -> np.ndarray:
        """The (distinctness, similarity, variation) vector for RL shaping."""
        return np.array([self.distinctness, self.similarity, self.variation],
                        dtype=np.float64)


def _segment_distribution(seg: Segment, embeddings: np.ndarray,
                          max_frames: int, rng: np.random.Generator) -> np.ndarray:
    """Empirical frame distribution for a segment, subsampled to a cap.

    Modeling a segment as a *distribution* of frames (not a pooled vector) is
    the point of the OT formulation -- it preserves internal temporal texture,
    which is what lets Axis 3 distinguish real variation from copy-paste.
    """
    frames = seg.frames(embeddings)
    if len(frames) == 0:
        raise ValueError(f"empty segment {seg}")
    if len(frames) > max_frames:
        idx = rng.choice(len(frames), size=max_frames, replace=False)
        idx.sort()
        frames = frames[idx]
    return frames


def _ot(a: np.ndarray, b: np.ndarray, cfg: FormAdherenceConfig) -> float:
    if cfg.ot_backend == "exact":
        return wasserstein_distance(a, b, metric="cosine")
    return sinkhorn_distance(a, b, eps=cfg.sinkhorn_eps,
                             max_iter=cfg.sinkhorn_max_iter,
                             tol=cfg.sinkhorn_tol, metric="cosine")


def _sim_from_dist(dist: float, cfg: FormAdherenceConfig) -> float:
    """Map an OT distance (cosine ground metric, >=0) to a [0,1] similarity."""
    return float(np.exp(-dist / max(cfg.similarity_temperature, 1e-8)))


def _variation_score(dist: float, cfg: FormAdherenceConfig) -> float:
    """Score a same-letter pair distance against the variation band.

    1.0 inside (variation_floor, tau); ramps smoothly to 0 outside on both
    sides. Smoothness matters because this is the most game-able axis and is
    used as a reward -- a hard step function would create exploitable cliffs.
    """
    floor = cfg.variation_floor
    tau = cfg.tau
    ramp = max(cfg.variation_softness * tau, 1e-6)

    if dist <= floor:
        # copy-paste region: 0 at/below floor. (A distance at or under the floor
        # means the repeats are effectively identical -> no real variation.)
        return 0.0
    if dist < floor + ramp:
        # ramp up out of the copy-paste region into the acceptable band
        return float(np.clip((dist - floor) / ramp, 0.0, 1.0))
    if dist >= tau:
        # drifted region: 1 at tau, falling to 0 by tau+ramp
        return float(np.clip((tau + ramp - dist) / ramp, 0.0, 1.0))
    return 1.0


def score_form_adherence(
    embeddings: np.ndarray,
    target_form: str,
    boundaries: list[int] | None = None,
    config: FormAdherenceConfig | None = None,
    seed: int = 0,
) -> FormAdherenceResult:
    cfg = config or FormAdherenceConfig()
    rng = np.random.default_rng(seed)
    E = np.asarray(embeddings, dtype=np.float64)
    T = len(E)

    letters = list(target_form)
    n_segments = len(letters)
    if n_segments < 1:
        raise ValueError("target_form must contain at least one letter")

    # ---- Segmentation ---------------------------------------------------
    n_b = cfg.n_boundaries if cfg.n_boundaries is not None else n_segments - 1
    b = segment_boundaries(
        E, boundaries=boundaries, n_boundaries=n_b,
        window=cfg.mp_window, exclusion_zone=cfg.mp_exclusion_zone,
    )
    # If discovery returned the wrong count, fall back to uniform division so
    # the metric still returns a labeled result (and flags the mismatch by the
    # scores it produces rather than crashing).
    if len(b) != n_segments - 1:
        b = [round(T * k / n_segments) for k in range(1, n_segments)]
    segs = boundaries_to_segments(b, T, labels=letters)

    # Precompute each segment's frame distribution once.
    dists = [_segment_distribution(s, E, cfg.max_frames_per_segment, rng)
             for s in segs]

    # ---- Axis 1: Cross-Letter Distinctness ------------------------------
    cross_pairs: dict[str, float] = {}
    for i, j in combinations(range(n_segments), 2):
        if letters[i] == letters[j]:
            continue
        d = _ot(dists[i], dists[j], cfg)
        cross_pairs[f"{letters[i]}{i}-{letters[j]}{j}"] = d
    if cross_pairs:
        # Distinct = far apart. Map mean cross distance to [0,1]: 1 - similarity.
        mean_cross = float(np.mean(list(cross_pairs.values())))
        distinctness = 1.0 - _sim_from_dist(mean_cross, cfg)
    else:
        # Single-letter form (e.g. "AAAA"): distinctness is vacuously satisfied.
        distinctness = 1.0

    # ---- Axes 2 & 3: same-letter similarity and variation ---------------
    groups: dict[str, list[int]] = {}
    for k, lab in enumerate(letters):
        groups.setdefault(lab, []).append(k)

    per_letter_sim: dict[str, float] = {}
    per_letter_var: dict[str, float] = {}
    same_pairs: dict[str, float] = {}

    for lab, members in groups.items():
        if len(members) < 2:
            # A letter appearing once imposes no similarity/variation constraint.
            continue
        sims, vars_ = [], []
        for i, j in combinations(members, 2):
            d = _ot(dists[i], dists[j], cfg)
            same_pairs[f"{lab}{i}-{lab}{j}"] = d
            sims.append(_sim_from_dist(d, cfg))
            vars_.append(_variation_score(d, cfg))
        per_letter_sim[lab] = float(np.mean(sims))
        per_letter_var[lab] = float(np.mean(vars_))

    similarity = float(np.mean(list(per_letter_sim.values()))) \
        if per_letter_sim else 1.0
    variation = float(np.mean(list(per_letter_var.values()))) \
        if per_letter_var else 1.0

    # ---- Aggregate axes -------------------------------------------------
    w = cfg.normalized_weights()
    axes = np.array([distinctness, similarity, variation])
    if cfg.combine == "geometric":
        # Weighted geometric mean: exp(sum w_i log a_i). A zero on any axis
        # drags the overall toward zero -> no axis can be traded away.
        eps = 1e-12
        overall = float(np.exp(np.sum(np.array(w) * np.log(axes + eps))))
    else:
        overall = float(np.dot(w, axes))

    return FormAdherenceResult(
        overall=overall,
        distinctness=distinctness,
        similarity=similarity,
        variation=variation,
        target_form=target_form,
        segments=segs,
        per_letter_similarity=per_letter_sim,
        per_letter_variation=per_letter_var,
        cross_pair_distances=cross_pairs,
        same_pair_distances=same_pairs,
    )
