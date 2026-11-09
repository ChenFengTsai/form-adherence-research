"""Form-Adherence: conditional structural-form adherence metric for full-song audio.

Given a target form (a letter sequence such as "ABACA") and per-frame audio
embeddings for a song, score how well the song realizes that form:

    Axis 1  Cross-Letter Distinctness   different letters sound different
    Axis 2  Same-Letter Similarity      repeats of a letter cohere
    Axis 3  Within-Letter Variation     repeats are not copy-paste

The core (config, ot, segmentation, metric, synthetic, labels, audio_io) needs
only numpy/scipy. The run/integration layer (RunConfig, runners, extractors,
YuE, RL) needs torch etc. and is imported lazily, so `import formadherence`
works on a machine without the extraction extras.
"""

from .config import FormAdherenceConfig
from .segmentation import matrix_profile, segment_boundaries, Segment
from .ot import (wasserstein_distance, sinkhorn_distance, sinkhorn_divergence,
                 segment_distance)
from .metric import FormAdherenceResult, score_form_adherence, hard_auc, soft_auc

_LAZY_RUN = {"RunConfig", "EvaluationConfig", "EvaluationResult",
             "EvaluationRunner", "RewardRunner", "build_runner"}


def __getattr__(name):
    if name in _LAZY_RUN:
        from . import run
        return getattr(run, name)
    raise AttributeError(f"module 'formadherence' has no attribute {name!r}")


__all__ = [
    "FormAdherenceConfig", "matrix_profile", "segment_boundaries", "Segment",
    "wasserstein_distance", "sinkhorn_distance", "sinkhorn_divergence",
    "segment_distance", "FormAdherenceResult", "score_form_adherence",
    "hard_auc", "soft_auc", *sorted(_LAZY_RUN),
]

__version__ = "0.2.0"
