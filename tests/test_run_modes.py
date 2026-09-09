"""Verify the two-mode RunConfig: evaluation vs rl_reward. Both must dispatch to
the right runner, share the same metric core, and agree on the metric portion of
the score."""

import numpy as np
from formadherence import (
    RunConfig, EvaluationConfig, build_runner,
    EvaluationRunner, RewardRunner, FormAdherenceConfig,
)
from formadherence.integration import GeneratedSong
from formadherence.integration.embeddings import EmbeddingExtractor


class PassThroughExtractor(EmbeddingExtractor):
    """Deterministic extractor for testing: reshapes a stored embedding array.
    We bypass audio entirely by stashing embeddings in song.meta."""
    frame_rate = 1.0
    def extract(self, waveform, sample_rate):
        # `waveform` here is actually a (T,d) embedding array we pass through.
        return self._finalize(np.asarray(waveform, np.float64))


def _song_with_embeddings(form):
    from formadherence.synthetic import make_song_embeddings
    E, b = make_song_embeddings(form)
    # Store embeddings AS the audio so PassThroughExtractor returns them.
    return GeneratedSong(audio=E, sample_rate=1, target_form=form,
                         token_ids=np.arange(10),
                         token_logprobs=np.full(10, -1.0)), b


def test_evaluation_mode_returns_metric_only():
    cfg = RunConfig(mode="evaluation")
    runner = build_runner(cfg, PassThroughExtractor())
    assert isinstance(runner, EvaluationRunner)
    song, _ = _song_with_embeddings("ABACA")
    res = runner.evaluate(song)
    # Has raw axes, no shaped scalar / penalties.
    assert hasattr(res, "overall") and not hasattr(res, "penalties")
    assert 0.0 <= res.overall <= 1.0


def test_rl_mode_returns_shaped_reward():
    cfg = RunConfig(mode="rl_reward")
    runner = build_runner(cfg, PassThroughExtractor())
    assert isinstance(runner, RewardRunner)
    song, _ = _song_with_embeddings("ABACA")
    out = runner.evaluate(song)
    assert hasattr(out, "scalar") and hasattr(out, "penalties")
    assert 0.0 <= out.scalar <= 1.0


def test_both_modes_share_metric_core():
    """The metric axes computed in evaluation mode must match the (pre-shaping)
    axes inside the RL reward, since both use the same FormAdherenceConfig."""
    metric_cfg = FormAdherenceConfig(ot_backend="exact")
    ev = build_runner(RunConfig(mode="evaluation", metric=metric_cfg),
                      PassThroughExtractor())
    rl = build_runner(RunConfig(mode="rl_reward", metric=metric_cfg),
                      PassThroughExtractor())
    song, _ = _song_with_embeddings("ABACA")

    ev_res = ev.evaluate(song)
    rl_out = rl.evaluate(song)
    # RL may gate variation by noise; distinctness & similarity are untouched by
    # shaping, so those must match exactly.
    assert abs(ev_res.distinctness - rl_out.axes[0]) < 1e-9
    assert abs(ev_res.similarity - rl_out.axes[1]) < 1e-9


def test_rl_config_metric_pinned_to_shared_core():
    """Even if a RewardConfig with a different metric is passed, __post_init__
    must repoint it at the shared core so the modes cannot diverge."""
    from formadherence.integration.reward import RewardConfig
    shared = FormAdherenceConfig(tau=0.09)
    stray = RewardConfig(metric=FormAdherenceConfig(tau=0.99))
    cfg = RunConfig(mode="rl_reward", metric=shared, reward=stray)
    assert cfg.reward.metric.tau == 0.09


def test_roundtrip_json(tmp_path=None):
    import tempfile, os
    cfg = RunConfig(mode="rl_reward", metric=FormAdherenceConfig(tau=0.11))
    d = tempfile.mkdtemp()
    p = os.path.join(d, "run.json")
    cfg.to_json(p)
    loaded = RunConfig.from_json(p)
    assert loaded.mode == "rl_reward"
    assert loaded.metric.tau == 0.11
    assert loaded.reward is not None and loaded.reward.metric.tau == 0.11


def test_evaluation_boundaries_required_when_not_discovering():
    cfg = RunConfig(mode="evaluation",
                    evaluation=EvaluationConfig(discover_boundaries=False))
    runner = build_runner(cfg, PassThroughExtractor())
    song, b = _song_with_embeddings("ABACA")
    # No boundaries -> error; with boundaries -> ok.
    try:
        runner.evaluate(song)
        raise AssertionError("expected ValueError without boundaries")
    except ValueError:
        pass
    res = runner.evaluate(song, boundaries=b)
    assert 0.0 <= res.overall <= 1.0


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    p = 0
    for fn in fns:
        try:
            fn(); print("PASS", fn.__name__); p += 1
        except AssertionError as e:
            print("FAIL", fn.__name__, e)
        except Exception as e:
            print("ERROR", fn.__name__, type(e).__name__, e)
    print(f"\n{p}/{len(fns)} passed")
