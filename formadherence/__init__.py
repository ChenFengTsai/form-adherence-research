"""Form-Adherence: conditional structural-form adherence metric for full-song audio.

Given a target form (a letter sequence such as "ABACA") and per-frame audio
embeddings for a generated song, score how well the song realizes that form
along three orthogonal axes:

    Axis 1  Cross-Letter Distinctness   different letters sound different
    Axis 2  Same-Letter Similarity      repeats of a letter cohere
    Axis 3  Within-Letter Variation     repeats vary, without copy-paste

See README for the full specification.
"""

from .config import FormAdherenceConfig
from .segmentation import matrix_profile, segment_boundaries, Segment
from .ot import wasserstein_distance, sinkhorn_distance
from .metric import (
    FormAdherenceResult,
    score_form_adherence,
)
from .run import (
    RunConfig,
    EvaluationConfig,
    EvaluationResult,
    EvaluationRunner,
    RewardRunner,
    build_runner,
)

__all__ = [
    "FormAdherenceConfig",
    "matrix_profile",
    "segment_boundaries",
    "Segment",
    "wasserstein_distance",
    "sinkhorn_distance",
    "FormAdherenceResult",
    "score_form_adherence",
    "RunConfig",
    "EvaluationConfig",
    "EvaluationResult",
    "EvaluationRunner",
    "RewardRunner",
    "build_runner",
]

__version__ = "0.1.0"
