"""From a generated song to a reinforcement-learning reward.

Pipeline:  GeneratedSong -> embeddings -> score_form_adherence -> shaped reward.

This is the object the RL loop calls. It bundles the three defences the proposal
names in Section 6.2:

  1. Reward-shaping over axes. The three axes are weighted independently and the
     weights can be annealed across training (curriculum: reward distinctness &
     similarity first, phase in variation once structure is stable).
  2. Anti-hacking ensemble. Axis 3 (variation) is the most game-able -- a policy
     can inject meaningless noise that lands in the band. We subtract two cheap
     penalties that noise-injection triggers but real variation does not:
        - a silence/degeneracy penalty (near-silent or clipping audio)
        - a spectral-flatness penalty (white-noise-like sections)
     and expose hooks to add OA / SSM ensemble terms.
  3. Recalibration hook. `tau` and friends live in the metric config and can be
     re-set between training phases from held-out human annotations.

The reward is returned both as a scalar (for vanilla policy gradient) and as the
raw axis vector + penalty breakdown (for shaped/multi-objective training and for
the reward-hacking audits in the evaluation plan).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from ..config import FormAdherenceConfig
from ..metric import score_form_adherence, FormAdherenceResult
from .embeddings import EmbeddingExtractor
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
    axis_weights: tuple[float, float, float] = (1.0, 1.0, 1.0)
    # Penalty coefficients (subtracted from the shaped axis reward).
    silence_penalty: float = 1.0
    noise_penalty: float = 1.0
    # A song landing in the variation band ONLY because it is noisy should not
    # be rewarded: gate variation credit by (1 - noise_score).
    gate_variation_by_noise: bool = True
    combine: str = "geometric"         # mirror metric.combine by default


class FormAdherenceReward:
    def __init__(self, extractor: EmbeddingExtractor,
                 config: RewardConfig | None = None):
        self.extractor = extractor
        self.cfg = config or RewardConfig()

    # -- public API the RL loop uses --------------------------------------

    def __call__(self, song: GeneratedSong) -> RewardOutput:
        emb = self.extractor.extract(song.audio, song.sample_rate)
        result = score_form_adherence(
            emb, song.target_form, boundaries=None, config=self.cfg.metric)

        axes = result.as_reward_vector().copy()   # D, S, V

        # ----- anti-hacking penalties (computed on raw audio) -----
        sil = _silence_score(song.audio)
        noise = _noise_score(song.audio, song.sample_rate)
        penalties = {"silence": sil, "noise": noise}

        if self.cfg.gate_variation_by_noise:
            # Noise inflates apparent variation; discount it.
            axes[2] = axes[2] * (1.0 - noise)

        w = np.asarray(self.cfg.axis_weights, dtype=np.float64)
        w = w / w.sum()
        if self.cfg.combine == "geometric":
            shaped = float(np.exp(np.sum(w * np.log(axes + 1e-12))))
        else:
            shaped = float(np.dot(w, axes))

        scalar = shaped \
            - self.cfg.silence_penalty * sil \
            - self.cfg.noise_penalty * noise
        scalar = float(np.clip(scalar, 0.0, 1.0))

        return RewardOutput(scalar=scalar, axes=axes, penalties=penalties,
                            form_result=result,
                            meta={"n_segments": len(result.segments)})

    def set_phase(self, axis_weights: tuple[float, float, float],
                  tau: float | None = None) -> None:
        """Curriculum control: call between training phases to anneal axis
        weights and (optionally) recalibrate the variation band."""
        self.cfg.axis_weights = axis_weights
        if tau is not None:
            self.cfg.metric.tau = tau


# ---- cheap degeneracy detectors -----------------------------------------

def _silence_score(audio: np.ndarray) -> float:
    """1.0 if effectively silent or clipping-degenerate, 0.0 if healthy."""
    a = np.asarray(audio, dtype=np.float64)
    if a.size == 0:
        return 1.0
    rms = float(np.sqrt(np.mean(a ** 2)))
    if rms < 1e-3:
        return 1.0
    clip_frac = float(np.mean(np.abs(a) > 0.999))
    return float(np.clip(clip_frac * 2.0, 0.0, 1.0))


def _noise_score(audio: np.ndarray, sample_rate: int,
                 frame: int = 2048, hop: int = 1024) -> float:
    """Mean spectral flatness in [0, 1]; ~1 for white noise, low for tonal/
    musical content. High flatness is the signature of the 'inject noise to hit
    the variation band' exploit, so we surface it as a penalty."""
    a = np.asarray(audio, dtype=np.float64)
    if len(a) < frame:
        return 0.0
    flats = []
    for s in range(0, len(a) - frame + 1, hop):
        w = a[s:s + frame] * np.hanning(frame)
        spec = np.abs(np.fft.rfft(w)) ** 2 + 1e-12
        gmean = np.exp(np.mean(np.log(spec)))
        amean = np.mean(spec)
        flats.append(gmean / amean)
    return float(np.clip(np.mean(flats), 0.0, 1.0)) if flats else 0.0
