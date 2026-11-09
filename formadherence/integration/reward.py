"""From a generated song to an RL reward.

GeneratedSong -> embeddings -> score_form_adherence -> shaped scalar reward.

Defences against reward hacking:
  1. Axis weights the curriculum can anneal (set_phase).
  2. Penalties computed on the raw audio:
       silence       whole song near-silent, or clipping
       noise         white-noise-like audio (mean spectral flatness); also gates
                     variation credit, since noise inflates repeat distances
       quiet_section NEW: a section far quieter than the song's typical
                     section. Cheapest way to fake "different sections" is to
                     make one letter near-silent; this catches it.
  3. Metric settings can be changed between phases (set_phase(**overrides)).

Fixes in v0.2: the reward uses the metric's graded raw_axes and multiplies
in the penalties, so weak rollouts don't all tie at 0 (no GRPO signal);
penalties work on stereo input (the noise score used to crash
on 2-D audio); set_phase accepts any metric setting (tau alone did nothing in
the new relative mode); result meta reports segmentation_fallback, auc and
contrast for logging.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from ..config import FormAdherenceConfig
from ..metric import score_form_adherence, FormAdherenceResult
from .embeddings import EmbeddingExtractor, _to_mono
from .generators import GeneratedSong


@dataclass
class RewardOutput:
    scalar: float                      # final reward the RL loop optimizes
    axes: np.ndarray                   # (distinctness, similarity, variation)
    penalties: dict[str, float] = field(default_factory=dict)
    form_result: FormAdherenceResult | None = None
    meta: dict = field(default_factory=dict)


@dataclass
class RewardConfig:
    metric: FormAdherenceConfig = field(default_factory=FormAdherenceConfig)
    # The reward's own axis weights (annealed by the curriculum). The metric's
    # axis_weights only affect result.overall, which the reward does not use.
    axis_weights: tuple[float, float, float] = (1.0, 1.0, 1.0)
    silence_penalty: float = 1.0
    noise_penalty: float = 1.0
    quiet_section_penalty: float = 0.5
    quiet_section_db: float = 20.0     # this far below the median section ...
    quiet_section_ramp_db: float = 10.0  # ... starts the penalty, full after +ramp
    gate_variation_by_noise: bool = True
    combine: str = "geometric"
    # Use the metric's graded raw_axes (relative mode) instead of the reported
    # axes. Reported axes clip everything below chance to 0, so early-training
    # rollouts would tie and GRPO would get no advantage signal.
    use_raw_axes: bool = True
    # multiplicative: scalar = shaped * prod(1 - coeff * penalty). Keeps the
    # ordering among weak songs. subtractive (old): shaped - sum(coeff *
    # penalty), clipped at 0 -> weak, slightly penalized songs all tie at 0.
    penalty_mode: str = "multiplicative"


class FormAdherenceReward:
    def __init__(self, extractor: EmbeddingExtractor, config: RewardConfig | None = None):
        self.extractor = extractor
        self.cfg = config or RewardConfig()

    def __call__(self, song: GeneratedSong) -> RewardOutput:
        emb = self.extractor.extract(song.audio, song.sample_rate)
        return self.score_embeddings(emb, song)

    def score_embeddings(self, emb: np.ndarray, song: GeneratedSong) -> RewardOutput:
        """Reward from precomputed embeddings (also handy for testing)."""
        # audio -> Axis 3 on note changes (noise/EQ can't fake variation)
        result = score_form_adherence(emb, song.target_form, boundaries=None,
                                      config=self.cfg.metric, audio=song.audio,
                                      sample_rate=song.sample_rate)
        if self.cfg.use_raw_axes and result.raw_axes is not None:
            axes = np.asarray(result.raw_axes, dtype=np.float64)
        else:
            axes = result.as_reward_vector().copy()   # D, S, V
        audio = _to_mono(song.audio)

        sil = _silence_score(audio)
        noise = _noise_score(audio, song.sample_rate)
        quiet = _quiet_section_score(audio, song.sample_rate, result.segments,
                                     len(emb), self.cfg.quiet_section_db,
                                     self.cfg.quiet_section_ramp_db)
        penalties = {"silence": sil, "noise": noise, "quiet_section": quiet}

        if self.cfg.gate_variation_by_noise:
            axes[2] = axes[2] * (1.0 - noise)

        w = np.asarray(self.cfg.axis_weights, dtype=np.float64)
        w = w / w.sum()
        if self.cfg.combine == "geometric":
            shaped = float(np.exp(np.sum(w * np.log(axes + 1e-12))))
        else:
            shaped = float(np.dot(w, axes))

        terms = [(self.cfg.silence_penalty, sil), (self.cfg.noise_penalty, noise),
                 (self.cfg.quiet_section_penalty, quiet)]
        if self.cfg.penalty_mode == "multiplicative":
            scalar = shaped
            for coeff, pen in terms:
                scalar *= float(np.clip(1.0 - coeff * pen, 0.0, 1.0))
        else:
            scalar = shaped - sum(c * p for c, p in terms)
        return RewardOutput(
            scalar=float(np.clip(scalar, 0.0, 1.0)), axes=axes, penalties=penalties,
            form_result=result,
            meta={"n_segments": len(result.segments), "shaped": shaped,
                  "segmentation_fallback": result.segmentation_fallback,
                  "auc": result.auc, "contrast": result.contrast})

    def set_phase(self, axis_weights: tuple[float, float, float] | None = None,
                  tau: float | None = None, **metric_overrides) -> None:
        """Curriculum control between phases: reward axis weights, and any
        metric setting (e.g. copy_floor=0.3; tau only matters in absolute mode)."""
        if axis_weights is not None:
            self.cfg.axis_weights = tuple(axis_weights)
        if tau is not None:
            metric_overrides["tau"] = tau
        for k, v in metric_overrides.items():
            if not hasattr(self.cfg.metric, k):
                raise AttributeError(f"FormAdherenceConfig has no setting {k!r}")
            setattr(self.cfg.metric, k, v)


# ---- cheap degeneracy detectors (mono float audio) ----------------------

def _silence_score(audio: np.ndarray) -> float:
    """1.0 if effectively silent; otherwise grows with the clipping fraction."""
    a = np.asarray(audio, dtype=np.float64)
    if a.size == 0:
        return 1.0
    if float(np.sqrt(np.mean(a ** 2))) < 1e-3:
        return 1.0
    return float(np.clip(np.mean(np.abs(a) > 0.999) * 2.0, 0.0, 1.0))


def _noise_score(audio: np.ndarray, sample_rate: int,
                 frame: int = 2048, hop: int = 1024) -> float:
    """Mean spectral flatness in [0, 1]: ~1 white noise, low for music."""
    a = np.asarray(audio, dtype=np.float64).ravel()
    if len(a) < frame:
        return 0.0
    n = 1 + (len(a) - frame) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(n)[:, None]
    spec = np.abs(np.fft.rfft(a[idx] * np.hanning(frame)[None, :], axis=1)) ** 2 + 1e-12
    flat = np.exp(np.mean(np.log(spec), axis=1)) / np.mean(spec, axis=1)
    return float(np.clip(flat.mean(), 0.0, 1.0))


def _quiet_section_score(audio, sample_rate, segments, n_frames,
                         thresh_db=20.0, ramp_db=10.0) -> float:
    """Fraction-weighted penalty for sections much quieter than the median
    section: 0 until thresh_db below the median, 1 at thresh_db + ramp_db."""
    a = np.asarray(audio, dtype=np.float64)
    if len(segments) < 2 or len(a) == 0 or n_frames == 0:
        return 0.0
    spf = len(a) / n_frames                       # samples per embedding frame
    levels = []
    for s in segments:
        x = a[int(s.start * spf):max(int(s.end * spf), int(s.start * spf) + 1)]
        levels.append(20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-8))
    levels = np.asarray(levels)
    drop = np.median(levels) - levels
    per_seg = np.clip((drop - thresh_db) / max(ramp_db, 1e-6), 0.0, 1.0)
    return float(per_seg.mean())
