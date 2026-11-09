"""Tests for formadherence.integration (no torch / CLAP / YuE needed)."""

import json
import os
import subprocess
import sys
import warnings

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from formadherence.integration.embeddings import (
    EmbeddingExtractor, ConcatExtractor, _resample_sequence)
from formadherence.integration.generators import (
    lyrics_by_occurrence, build_structured_lyrics, CommandLineGenerator, GeneratedSong)
from formadherence.integration.reward import (
    FormAdherenceReward, RewardConfig, _noise_score, _quiet_section_score)
from formadherence.integration.rl_loop import (
    grpo_token_loss, GRPOTrainer, GRPOConfig, PromptSpec, audit_axis_growth)
from formadherence.segmentation import Segment
from formadherence.synthetic import make_song_embeddings


# --------------------------------------------------------- time alignment ----

class FakeWindowed(EmbeddingExtractor):
    """CLAP-like: window `win` s, hop win/2, frame time = window centre.
    Feature = [1,0] before the change at t=`change` s, [0,1] after."""
    def __init__(self, win, change=14.0):
        self.win, self.frame_rate, self.change = win, 2.0 / win, change
    def extract_with_times(self, w, sr):
        dur = len(w) / sr
        starts = np.arange(0, dur - self.win + 1e-9, self.win / 2)
        t = starts + self.win / 2
        return np.stack([(t < self.change).astype(float), (t >= self.change).astype(float)], 1), t
    def extract(self, w, sr): return self.extract_with_times(w, sr)[0]


def _change_frame(seq):
    return int(np.argmax(seq[:, 1] > seq[:, 0]))


def test_time_aligned_concat_puts_change_in_the_right_place():
    sr, dur = 100, 60
    wav = np.zeros(sr * dur)
    ce = ConcatExtractor([FakeWindowed(8.0)], target_frame_rate=2.0)
    new = _change_frame(ce.extract(wav, sr))
    seq, _ = FakeWindowed(8.0).extract_with_times(wav, sr)
    old = _change_frame(_resample_sequence(seq, 120))
    # change at 14 s = frame 28. (At the exact midpoint the old stretch happens
    # to be right; its error grows toward the start and end of the song.)
    assert abs(new - 28) <= 1
    assert abs(old - 28) >= 3
    assert ce.block_dims == [2]


def test_concat_blocks_equal_weight():
    class Big(FakeWindowed):
        def extract_with_times(self, w, sr):
            s, t = super().extract_with_times(w, sr)
            return np.hstack([s, s, s]) * 100, t
    ce = ConcatExtractor([FakeWindowed(1.0), Big(1.0)], target_frame_rate=2.0)
    E = ce.extract(np.zeros(100 * 20), 100)
    assert ce.block_dims == [2, 6]
    a, b = np.linalg.norm(E[:, :2], axis=1), np.linalg.norm(E[:, 2:], axis=1)
    assert np.allclose(a, b)


# ---------------------------------------------------------------- lyrics ----

def test_lyrics_dict_and_occurrences():
    lyr = {"A": ["v1", "v2", "v3"], "B": "chorus", "C": "bridge"}
    assert lyrics_by_occurrence("ABACA", lyr) == ["v1", "chorus", "v2", "bridge", "v3"]
    s = build_structured_lyrics("ABACA", lyr)
    assert s.startswith("[verse]\nv1") and "[chorus]\nchorus" in s and "[bridge]" in s


def test_plain_lyrics_no_longer_duplicated():
    with pytest.raises(ValueError):
        lyrics_by_occurrence("ABA", "only one paragraph")
    assert lyrics_by_occurrence("ABA", "verse text\n\nchorus text") == \
        ["verse text", "chorus text", "verse text"]


def test_tagged_lyrics_pattern_checked():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        build_structured_lyrics("ABA", "[verse]\nx\n\n[chorus]\ny\n\n[bridge]\nz")
    assert any("does not match" in str(x.message) for x in w)


def test_cli_generator(tmp_path):
    script = tmp_path / "fake_yue.py"
    script.write_text(
        "import sys, os\n"
        "a = dict(zip(sys.argv[1::2], sys.argv[2::2]))\n"
        "os.makedirs(os.path.join(a['--out'], 'mix'), exist_ok=True)\n"
        "open(os.path.join(a['--out'], 'mix', 'song_mix.mp3'), 'w').write("
        "open(a['--lyrics']).read() + '|' + a['--n'] + '|' + a['--seed'])\n")
    tpl = (f"{sys.executable} {script} --genre {{genre_file}} --lyrics {{lyrics_file}} "
           f"--n {{n_segments}} --seed {{seed}} --out {{out_dir}}")
    seen = {}

    def read(path):
        seen["text"] = open(path).read()
        return np.zeros((1000, 2)), 44100        # stereo on purpose
    gen = CommandLineGenerator(tpl, str(tmp_path / "out"), read_wav=read)
    song = gen.generate("ABA", {"A": "verse words", "B": "chorus words"}, "pop", seed=7)
    assert song.audio.ndim == 1 and song.sample_rate == 44100
    assert "[verse]\nverse words" in seen["text"] and "[chorus]\nchorus words" in seen["text"]
    assert seen["text"].endswith("|3|7")          # n_segments = len(form), seed passed


# ---------------------------------------------------------------- reward ----

def test_noise_score_and_stereo():
    sr = 16000
    t = np.arange(sr * 2) / sr
    tone = np.sin(2 * np.pi * 440 * t)
    noise = np.random.default_rng(0).normal(size=len(t))
    assert _noise_score(noise, sr) > 0.4 > 0.05 > _noise_score(tone, sr)
    assert _noise_score(np.stack([tone, tone], 1).mean(1), sr) < 0.05


def test_quiet_section_penalty():
    sr, fr = 1000, 2.0
    loud = np.random.default_rng(0).normal(0, 0.3, sr * 10)
    audio = np.concatenate([loud, loud * 1e-4, loud, loud * 1e-4])  # ABAB, B silent
    segs = [Segment(0, 20), Segment(20, 40), Segment(40, 60), Segment(60, 80)]
    assert _quiet_section_score(audio, sr, segs, 80) == pytest.approx(0.5)
    even = np.concatenate([loud] * 4)
    assert _quiet_section_score(even, sr, segs, 80) == 0.0


def test_reward_on_embeddings_and_set_phase():
    E, b = make_song_embeddings("ABACA", seg_len=40, seed=1)
    sr = 1000
    audio = np.random.default_rng(0).normal(0, 0.1, (len(E) * sr // 2, 2))  # stereo
    song = GeneratedSong(audio=audio, sample_rate=sr, target_form="ABACA")
    rf = FormAdherenceReward(extractor=None, config=RewardConfig())
    out = rf.score_embeddings(E, song)
    assert set(out.penalties) == {"silence", "noise", "quiet_section"}
    assert "segmentation_fallback" in out.meta and 0 <= out.scalar <= 1
    rf.set_phase((1, 1, 0.5), copy_floor=0.3)
    assert rf.cfg.metric.copy_floor == 0.3 and rf.cfg.axis_weights == (1, 1, 0.5)
    with pytest.raises(AttributeError):
        rf.set_phase(not_a_setting=1)


# ------------------------------------------------------------------- RL -----

def test_grpo_loss_numpy():
    lp = np.log(np.full(5, 0.5))
    assert grpo_token_loss(lp, lp, lp, 2.0) == pytest.approx(-2.0)   # ratio 1, KL 0
    # big ratio is clipped for positive advantage
    assert grpo_token_loss(lp + 1.0, lp, lp + 1.0, 1.0, clip_ratio=0.2) == pytest.approx(-1.2)
    assert grpo_token_loss(lp, lp, lp + 0.5, 0.0) > 0                 # KL > 0


class FakePolicy:
    def __init__(self): self.calls = 0
    def generate(self, form, lyrics, genre, seed=None):
        self.calls += 1
        rng = np.random.default_rng(seed % 1000)
        return GeneratedSong(audio=np.zeros(10), sample_rate=10, target_form=form,
                             token_ids=np.arange(4), token_logprobs=rng.normal(-1, .1, 4))
    def score_tokens(self, song, model=None, with_grad=True):
        return song.token_logprobs


class FakeReward:
    def __init__(self): self.phases = []
    def __call__(self, song):
        from formadherence.integration.reward import RewardOutput
        r = float(np.random.default_rng(len(song.token_logprobs) + int(1e3 * song.token_logprobs[0]) % 97).random())
        return RewardOutput(scalar=r, axes=np.array([r, r, r]),
                            penalties={"noise": 0.0}, meta={"segmentation_fallback": False})
    def set_phase(self, w, tau=None, **kw): self.phases.append((w, tau, kw))


def test_train_step_reports_axis_mean():
    pol, rew = FakePolicy(), FakeReward()
    tr = GRPOTrainer(pol, rew, [PromptSpec("ABA", "a\n\nb")],
                     GRPOConfig(group_size=3, grad_accum_prompts=2))  # 1 prompt < 2
    s = tr.train_step(0)
    assert len(s["axis_mean"]) == 3 and pol.calls == 6 and s["loss_mean"] is not None
    assert rew.phases[-1][0] == (1.0, 1.0, 0.2)
    hist = [np.array([0.1, 0.1, 0.1 + 0.05 * i]) for i in range(20)]
    assert audit_axis_growth(hist)["variation_racing"]


# ------------------------------------------------- config / scripts --------

def test_rl_config_roundtrip(tmp_path):
    from formadherence.run import RunConfig
    from formadherence.config import FormAdherenceConfig
    p = str(tmp_path / "rl.json")
    RunConfig(mode="rl_reward", metric=FormAdherenceConfig(copy_floor=0.3)).to_json(p)
    cfg = RunConfig.from_json(p)
    assert cfg.mode == "rl_reward" and cfg.reward.metric is cfg.metric
    assert cfg.metric.copy_floor == 0.3


def test_score_corpus_mismatch_check(tmp_path):
    corpus = []
    for k, form in enumerate(["ABACA", "ABAB", "AABBA", "ABACA"]):
        E, _ = make_song_embeddings(form, seg_len=40, seed=k, shared_component=2.0)
        np.save(tmp_path / f"s{k}.npy", E)
        corpus.append({"emb": str(tmp_path / f"s{k}.npy"), "target_form": form, "model": "synth"})
    (tmp_path / "corpus.json").write_text(json.dumps(corpus))
    r = subprocess.run([sys.executable, "scripts/score_corpus.py", "--corpus",
                        str(tmp_path / "corpus.json"), "--config", "none.json",
                        "--out_csv", str(tmp_path / "s.csv")],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "MISMATCH CHECK" in r.stdout and (tmp_path / "s.csv").exists()
    line = [l for l in r.stdout.splitlines() if "MISMATCH" in l][0]
    assert "own beats every wrong form in 100% of songs" in line


# ---------------------------------------------------------- axis checks ----

def test_axis_check_edits():
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import check_axes as A
    rng = np.random.default_rng(0)
    for f in A.split_edits("ABACA", 10, rng):        # a repeat gets a new letter
        assert sum(a != b for a, b in zip(f, "ABACA")) == 1 and "D" in f
    for f in A.merge_edits("ABACA", 10, rng):        # a section takes an existing letter
        assert sum(a != b for a, b in zip(f, "ABACA")) == 1 and set(f) <= set("ABC")
    assert A.split_edits("ABC", 5, rng) == []


def test_copy_paste_condition_is_exact():
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import check_copy_paste as C
    sr = 100
    x = np.random.default_rng(1).normal(size=30 * sr)
    y = C.make_condition(x, sr, (0.0, 10.0), (20.0, 30.0), "copy", np.random.default_rng(0))
    assert np.array_equal(y[20 * sr:30 * sr], x[0:10 * sr]) and np.array_equal(y[:20 * sr], x[:20 * sr])
    y = C.make_condition(x, sr, (0.0, 10.0), (20.0, 30.0), "copy_offset", np.random.default_rng(0))
    assert np.array_equal(y[20 * sr:20 * sr + 100], x[25:125])
    letter, si, ti, spans = C.pick_pair(list("ABABCB"), [10, 20, 30, 40, 50], 60, 6)
    assert letter == "B" and si < ti


def test_segmentation_hit_rate():
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import diagnose_segmentation as S
    assert S.hit_rate([10.0, 20.0], [10.4, 20.0], 0.5)[2] == 1.0
    assert S.hit_rate([10.0, 20.0], [12.0, 20.0], 0.5)[2] == 0.5     # one miss
    assert S.hit_rate([10.0, 10.2], [10.1], 3.0)[:2] == (0.5, 1.0)    # one-to-one
    assert S.hit_rate([], [5.0], 3.0)[2] == 0.0
