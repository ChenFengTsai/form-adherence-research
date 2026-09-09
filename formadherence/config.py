"""Configuration for the Form-Adherence metric.

Every tunable lives here so a run is fully described by one object. This matters
for Part II of the proposal: reward shaping anneals the axis weights and
re-calibrates the variation band tau during training, so those must be first
-class, serializable knobs rather than magic numbers buried in the code.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Literal
import json


@dataclass
class FormAdherenceConfig:
    # ---- Optimal-transport backend --------------------------------------
    ot_backend: Literal["exact", "sinkhorn"] = "exact"
    # exact  -> exact Wasserstein-1 via LP (non-differentiable; eval / RL reward)
    # sinkhorn -> entropic OT (differentiable surrogate for Section 6)
    sinkhorn_eps: float = 0.05        # entropic regularization strength
    sinkhorn_max_iter: int = 200
    sinkhorn_tol: float = 1e-6

    # Cap on frames sampled per segment when forming its empirical
    # distribution. OT cost is roughly O(n*m) (exact) so this bounds runtime
    # on 3-minute songs. Frames are subsampled uniformly if a segment is longer.
    max_frames_per_segment: int = 256

    # ---- Segmentation (Matrix Profile) ----------------------------------
    mp_window: int = 32               # subsequence length in frames
    mp_exclusion_zone: float = 0.5    # trivial-match exclusion, fraction of window
    # Number of boundaries to keep when discovering them; if None, derived
    # from the number of segments implied by the target form.
    n_boundaries: int | None = None

    # ---- Variation band (Axis 3) ----------------------------------------
    # A same-letter pair should differ but stay on-theme:
    #     0 < distance < tau            -> ideal (natural variation)
    #     distance ~ 0                  -> copy-paste (penalized)
    #     distance > tau                -> drifted off-theme (penalized)
    # Calibrated against cosine-ground Wasserstein on normalized embeddings,
    # where copy-paste repeats measure ~0.002, natural thematic variation
    # ~0.01-0.05, and off-theme drift >0.1. These defaults are a starting point;
    # tau is meant to be re-fit against human-annotated repeats (Section 4.2)
    # and re-calibrated during RL (Section 6.2).
    tau: float = 0.12                 # upper edge of the acceptable band
    variation_floor: float = 0.005    # below this, treat as copy-paste
    # Width over which penalties ramp, as a fraction of tau. The lower ramp out
    # of the copy-paste region is deliberately short so that genuine thematic
    # variation (~0.01-0.05) sits in the flat top of the band, while distances
    # near zero are still penalized.
    variation_softness: float = 0.1

    # ---- Aggregation ----------------------------------------------------
    # Overall score = weighted geometric-ish combination of the three axes.
    # Geometric mean (product) is used by default so a near-zero on any axis
    # cannot be masked by the other two -- the decomposition the proposal
    # relies on. Set combine="weighted_mean" for an additive alternative.
    combine: Literal["geometric", "weighted_mean"] = "geometric"
    axis_weights: tuple[float, float, float] = (1.0, 1.0, 1.0)
    # (distinctness, similarity, variation)

    # ---- Distance normalization -----------------------------------------
    # Cosine-based frame ground metric maps to [0, 2]; we rescale similarity
    # scores into [0, 1] with this temperature (higher = more forgiving).
    # At temperature ~0.3, a same-letter distance of ~0.02 maps to sim ~0.94
    # while a cross-letter distance of ~1.1 maps to sim ~0.02 (distinctness
    # ~0.98) -- i.e. the two regimes land near the ends of [0,1].
    similarity_temperature: float = 0.3

    def normalized_weights(self) -> tuple[float, float, float]:
        w = self.axis_weights
        s = sum(w)
        if s <= 0:
            raise ValueError("axis_weights must sum to a positive value")
        return (w[0] / s, w[1] / s, w[2] / s)

    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "FormAdherenceConfig":
        with open(path) as f:
            d = json.load(f)
        d["axis_weights"] = tuple(d["axis_weights"])
        return cls(**d)
