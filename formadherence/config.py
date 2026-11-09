"""Configuration for the Form-Adherence metric.

Every tunable lives here so a run is fully described by one object. This matters
for Part II of the proposal: reward shaping anneals the axis weights and
re-calibrates the variation band during training, so those must be first-class,
serializable knobs rather than magic numbers buried in the code.

Two scoring modes
-----------------
score_mode = "relative"  (default, recommended)
    Every axis compares distances WITHIN one song, so a song whose sections are
    all far apart and a song whose sections are all close are judged on their
    own contrast, not on an absolute scale. This is what the per-song AUC
    experiments showed works on real audio. See metric.py for the definitions.

score_mode = "absolute"  (legacy)
    The original formulation: absolute OT distances mapped through one global
    temperature and a fixed (floor, tau) band. Kept so old numbers can be
    reproduced. Old config JSON files without a score_mode key load in this
    mode automatically.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, fields
from typing import Literal
import json
import warnings


@dataclass
class FormAdherenceConfig:
    # ---- Scoring mode ----------------------------------------------------
    score_mode: Literal["relative", "absolute"] = "relative"

    # ---- Optimal-transport backend --------------------------------------
    ot_backend: Literal["exact", "sinkhorn", "mean"] = "exact"
    # exact    -> exact Wasserstein-1 via LP (non-differentiable; eval / reward)
    # sinkhorn -> entropic OT (differentiable surrogate for Section 7)
    # mean     -> NOT OT: cosine distance between segment mean vectors (baseline)
    sinkhorn_eps: float = 0.05        # entropic regularization strength
    sinkhorn_max_iter: int = 200
    sinkhorn_tol: float = 1e-6
    # Plain Sinkhorn cost is NOT zero for identical inputs (S(x,x) > 0), so a
    # copy-paste repeat does not measure ~0. The debiased Sinkhorn divergence
    #     S(x,y) - S(x,x)/2 - S(y,y)/2
    # fixes that. Only used when ot_backend == "sinkhorn".
    sinkhorn_debias: bool = True

    # Cap on frames sampled per segment when forming its empirical
    # distribution. OT cost is roughly O(n*m) (exact) so this bounds runtime.
    max_frames_per_segment: int = 256

    # ---- Segmentation (boundary discovery) ------------------------------
    # Boundaries are discovered with Foote checkerboard novelty on the cosine
    # self-similarity matrix (see segmentation.py). The *_window name is kept
    # for config-file compatibility; it is the novelty kernel size in frames.
    mp_window: int = 32
    mp_exclusion_zone: float = 0.5    # used only by matrix_profile() diagnostics
    # Number of boundaries to find; if None, len(target_form) - 1.
    n_boundaries: int | None = None
    # "foote": novelty peaks only (told the section count).
    # "form_aware": cuts fitted to the target form's repeat pattern
    #   (segmentation.form_aware_boundaries), started from Foote cuts.
    segmentation_method: str = "foote"
    form_aware_kernel: int = 24          # Foote kernel for its novelty bonus/init
    form_aware_novelty_weight: float = 2.0
    form_aware_min_len_s: float = 3.0    # shortest section it may cut

    # ---- Relative mode (score_mode="relative") ---------------------------
    # Soft comparisons use sigmoid((farther - closer) / (kappa * m)), where m is
    # the song's own median distance. kappa is therefore unit-free. Fit it with
    # scripts/calibrate.py: it is set from the typical per-song gap between
    # different-letter and same-letter distances in annotated songs (in units
    # of m), so it matches how much contrast real music in your features has.
    contrast_temperature: float = 0.5
    # Lowest value a single letter-pair / letter score can contribute to the
    # geometric aggregate (see metric._score_relative).
    comparison_floor: float = 0.001
    # Distinctness and similarity are 0.5 at chance; map chance -> 0, perfect
    # -> 1 so a failed axis is near 0 and the geometric mean punishes it.
    chance_to_zero: bool = True
    # A comparison must beat chance by this much before it counts (0.5 + margin
    # -> 0). With only a few section pairs per song, chance-level scores are
    # noisy; without a margin a merged letter pair read ~0.2 instead of ~0.
    # 0.15 was chosen on synthetic songs (correct 0.90 vs failures 0.34-0.40,
    # separation 5.6 SD vs 2.9 at 0). The RL reward uses unclipped raw_axes.
    chance_margin: float = 0.15
    # How per-letter / per-letter-pair scores combine into an axis.
    # geometric: every requested contrast/sameness must hold (a dropped bridge
    # pulls distinctness to ~0). mean: lenient. NOTE for human data: annotations
    # sometimes give similar music different labels (intro vs outro); geometric
    # treats that as a failure of the labelled form.
    aggregate: Literal["geometric", "mean"] = "geometric"
    # Anchor used only when the song itself has no reference: forms with no
    # repeated letter (distinctness) or a single letter (similarity). Unit-free:
    # matching-halves distance / inside-change. Above repeat_ratio looks like
    # different material, below like a repeat. Fit with scripts/calibrate.py
    # (midpoint of human repeat and different-letter ratios).
    repeat_ratio: float = 2.0
    ratio_temperature: float = 0.35
    # Copy-paste detection for Axis 3. q = (distance between matching halves of
    # two repeats) / (their mean inside-change); exact copy ~0, genuine repeat
    # ~1+. Score is 0 at q <= copy_floor and ramps to 1 at q >= copy_floor +
    # copy_ramp. Fit these with scripts/calibrate.py on human repeats.
    copy_floor: float = 0.25
    copy_ramp: float = 0.25

    # ---- Axis 3 on musical content (formadherence/variation.py) ----------
    # chroma_dtw: share of DTW-aligned moments whose notes (chroma) differ
    # between two repeats. Used when the audio (or chroma) is passed to
    # score_form_adherence; otherwise Axis 3 falls back to the embedding copy
    # ratio above, and result.variation_method says which one ran.
    variation_method: Literal["chroma_dtw", "embedding"] = "chroma_dtw"
    frame_rate: float = 2.0            # embedding frames/s (section frames -> seconds)
    chroma_fps: float = 10.0
    chroma_method: Literal["stft", "cqt"] = "stft"
    note_change_threshold: float = 0.2   # cosine distance at which notes "differ"
    note_change_floor: float = 0.01      # f at/below this = copy (score 0)
    note_change_ramp: float = 0.02       # full score at floor + ramp
    dtw_band: float = 0.2                # allowed tempo drift, fraction of length
    edge_trim_s: float = 0.3             # ignored at each section edge
    # floor/ramp: set by scripts/check_variation.py --update_config (measured
    # on true copies and ~1 s note changes), not fitted from human repeats.

    # ---- Absolute mode (legacy, score_mode="absolute") -------------------
    tau: float = 0.12                 # upper edge of the acceptable band
    variation_floor: float = 0.005    # below this, treat as copy-paste
    variation_softness: float = 0.1   # ramp width as a fraction of tau
    similarity_temperature: float = 0.3

    # ---- Aggregation ----------------------------------------------------
    # Geometric mean so a near-zero on any axis cannot be masked by the others.
    combine: Literal["geometric", "weighted_mean"] = "geometric"
    axis_weights: tuple[float, float, float] = (1.0, 1.0, 1.0)
    # (distinctness, similarity, variation)

    def normalized_weights(self) -> tuple[float, float, float]:
        w = self.axis_weights
        s = sum(w)
        if s <= 0:
            raise ValueError("axis_weights must sum to a positive value")
        return (w[0] / s, w[1] / s, w[2] / s)

    # ---- serialization --------------------------------------------------

    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "FormAdherenceConfig":
        d = dict(d)
        if "score_mode" not in d:
            warnings.warn(
                "config has no 'score_mode' key: it was written by the old code, "
                "so it is loaded in score_mode='absolute' (legacy). Re-run "
                "scripts/calibrate.py to get a relative-mode config.")
            d["score_mode"] = "absolute"
        if "axis_weights" in d:
            d["axis_weights"] = tuple(d["axis_weights"])
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            warnings.warn(f"ignoring unknown config keys: {sorted(unknown)}")
        return cls(**{k: v for k, v in d.items() if k in known})

    @classmethod
    def from_json(cls, path: str) -> "FormAdherenceConfig":
        with open(path) as f:
            return cls.from_dict(json.load(f))
