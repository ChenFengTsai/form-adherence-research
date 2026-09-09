"""Top-level run configuration: one switch selects what the pipeline does with a
generated song.

Two modes, sharing the exact same metric core (the point of the proposal -- the
measurement used for benchmarking is the measurement the optimizer consumes):

  mode = "evaluation"
      Generate a song, compute the three-axis Form-Adherence metric directly on
      it, report per-axis + overall scores. No reward shaping, no anti-hacking
      penalties, no token log-probs required. This is the benchmarking path
      (proposal Sections 5 and 7): use it to score any model's outputs, build the
      AI test corpus, and correlate against human MOS.

  mode = "rl_reward"
      The full training path: embeddings -> metric -> shaped reward vector with
      §6.2 anti-hacking guards and curriculum, producing a scalar reward for
      policy-gradient updates. Requires an RL-capable generator (token log-probs).

Usage:
    cfg = RunConfig(mode="evaluation")          # or "rl_reward"
    runner = build_runner(cfg, extractor)
    result = runner.evaluate(song)              # EvaluationResult or RewardOutput

Both branches call score_form_adherence with the SAME FormAdherenceConfig, so a
song scored in evaluation mode and the metric-portion of its RL reward agree by
construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal
import json

from .config import FormAdherenceConfig
from .integration.embeddings import EmbeddingExtractor
from .integration.reward import FormAdherenceReward, RewardConfig, RewardOutput
from .integration.generators import GeneratedSong
from .metric import score_form_adherence, FormAdherenceResult


Mode = Literal["evaluation", "rl_reward"]


@dataclass
class EvaluationConfig:
    """Knobs specific to the metrics-only path. Deliberately minimal: evaluation
    should not shape or penalize -- it just measures."""
    # Discover boundaries from the audio (True) or expect them supplied (False).
    discover_boundaries: bool = True
    # If you have reference/annotated boundaries per song, pass them at call time
    # to evaluate(); this flag only sets the default when none are given.


@dataclass
class RunConfig:
    mode: Mode = "evaluation"

    # Shared metric core -- identical object used by BOTH modes.
    metric: FormAdherenceConfig = field(default_factory=FormAdherenceConfig)

    # Mode-specific sub-configs. Only the one matching `mode` is used.
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    reward: RewardConfig | None = None   # built from `metric` if left None

    def __post_init__(self):
        if self.mode not in ("evaluation", "rl_reward"):
            raise ValueError(f"unknown mode {self.mode!r}")
        if self.mode == "rl_reward" and self.reward is None:
            # Keep the reward's metric config pointed at the shared core so the
            # two modes cannot silently diverge.
            self.reward = RewardConfig(metric=self.metric)
        if self.mode == "rl_reward":
            # Enforce the invariant even if a reward config was passed in.
            self.reward.metric = self.metric

    # ---- serialization --------------------------------------------------

    def to_json(self, path: str) -> None:
        from dataclasses import asdict
        d = asdict(self)
        with open(path, "w") as f:
            json.dump(d, f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "RunConfig":
        with open(path) as f:
            d = json.load(f)
        metric = FormAdherenceConfig(**{**d["metric"],
                                        "axis_weights": tuple(d["metric"]["axis_weights"])})
        evaluation = EvaluationConfig(**d.get("evaluation", {}))
        reward = None
        if d.get("reward") is not None:
            rd = d["reward"]
            reward = RewardConfig(
                metric=metric,
                axis_weights=tuple(rd["axis_weights"]),
                silence_penalty=rd["silence_penalty"],
                noise_penalty=rd["noise_penalty"],
                gate_variation_by_noise=rd["gate_variation_by_noise"],
                combine=rd["combine"],
            )
        return cls(mode=d["mode"], metric=metric, evaluation=evaluation,
                   reward=reward)


# ---- results ------------------------------------------------------------

@dataclass
class EvaluationResult:
    """What the metrics-only path returns: the raw metric, nothing shaped."""
    overall: float
    distinctness: float
    similarity: float
    variation: float
    target_form: str
    form_result: FormAdherenceResult

    @classmethod
    def from_form_result(cls, fr: FormAdherenceResult) -> "EvaluationResult":
        return cls(overall=fr.overall, distinctness=fr.distinctness,
                   similarity=fr.similarity, variation=fr.variation,
                   target_form=fr.target_form, form_result=fr)


# ---- runners ------------------------------------------------------------

class EvaluationRunner:
    """Metrics-only. Generate (or accept) a song, measure it, report. No RL."""

    def __init__(self, cfg: RunConfig, extractor: EmbeddingExtractor):
        assert cfg.mode == "evaluation"
        self.cfg = cfg
        self.extractor = extractor

    def evaluate(self, song: GeneratedSong,
                 boundaries: list[int] | None = None) -> EvaluationResult:
        emb = self.extractor.extract(song.audio, song.sample_rate)
        if boundaries is None and not self.cfg.evaluation.discover_boundaries:
            raise ValueError(
                "discover_boundaries=False but no boundaries supplied")
        fr = score_form_adherence(emb, song.target_form,
                                  boundaries=boundaries, config=self.cfg.metric)
        return EvaluationResult.from_form_result(fr)

    def evaluate_from_embeddings(self, embeddings, target_form: str,
                                 boundaries: list[int] | None = None
                                 ) -> EvaluationResult:
        """Skip generation and extraction; score a precomputed (T,d) array. Handy
        for the calibration corpus where embeddings are cached."""
        fr = score_form_adherence(embeddings, target_form,
                                  boundaries=boundaries, config=self.cfg.metric)
        return EvaluationResult.from_form_result(fr)


class RewardRunner:
    """RL path: the shaped, penalized reward from integration.reward."""

    def __init__(self, cfg: RunConfig, extractor: EmbeddingExtractor):
        assert cfg.mode == "rl_reward"
        self.cfg = cfg
        self.reward_fn = FormAdherenceReward(extractor, cfg.reward)

    def evaluate(self, song: GeneratedSong) -> RewardOutput:
        return self.reward_fn(song)


def build_runner(cfg: RunConfig, extractor: EmbeddingExtractor):
    """Return the runner matching cfg.mode. Both expose .evaluate(song)."""
    if cfg.mode == "evaluation":
        return EvaluationRunner(cfg, extractor)
    return RewardRunner(cfg, extractor)
