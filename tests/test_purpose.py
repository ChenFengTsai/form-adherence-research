"""Requirements from the CURRENT proposal, tested directly.

The metric must, within one song and without an absolute distance scale:
  R1  score the correct form above every structural failure (verse == chorus,
      dropped bridge, drifted repeats, copy-paste)
  R2  not depend on the song's shared production component
  R3  blame the right thing (which letter pair is merged, which letter drifts,
      which letter is copy-pasted)
  R4  work for forms with no repeats and single-letter forms (via the
      calibrated anchor), instead of scoring them vacuously
  R5  score a song higher on its own form than on any wrong form
  R6  give the same ordering with exact OT and Sinkhorn
and the RL reward must
  R7  rise with form quality, rank songs like the evaluation score, and keep
      weak rollouts apart (no ties at 0 -> GRPO needs differences).

Songs are synthetic embeddings where we control exactly which sections share a
theme (same material), which are fresh instances (natural variation), and
which are byte-identical copies. Every instance varies -- no "canonical" first
instance -- so a fake repeat is exchangeable with a real one, as in music.
"""

import os
import sys

import numpy as np
import pytest
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from formadherence import FormAdherenceConfig, score_form_adherence
from formadherence.metric import song_ratios, song_scale
from formadherence.integration.reward import FormAdherenceReward, RewardConfig
from formadherence.integration.generators import GeneratedSong


def build(spec, seg_len=30, dim=32, theme_sep=3.0, var=0.6, noise=0.5, shared=3.0, seed=0):
    """spec: list of (theme, instance). Same (theme, instance) = exact copy;
    same theme, new instance = natural variation; new theme = new material."""
    rng = np.random.default_rng(seed)
    sh = rng.normal(0, theme_sep * shared, dim)
    themes, cache, segs, b, cur = {}, {}, [], [], 0
    for th, inst in spec:
        if (th, inst) not in cache:
            if th not in themes:
                themes[th] = rng.normal(0, theme_sep, dim)
            cache[(th, inst)] = (sh + themes[th] + rng.normal(0, var, dim)
                                 + rng.normal(0, noise, (seg_len, dim)))
        segs.append(cache[(th, inst)].copy())
        cur += seg_len
        b.append(cur)
    E = np.vstack(segs)
    return E / np.linalg.norm(E, axis=1, keepdims=True), b[:-1]


def natural(form):
    seen, spec = {}, []
    for c in form:
        spec.append((c.lower(), seen.get(c, 0)))
        seen[c] = seen.get(c, 0) + 1
    return spec


@pytest.fixture(scope="module")
def cal():
    return calibrated_config()


def calibrated_config():
    """Calibrate on correctly-formed songs, like calibrate.py does."""
    rs, rd = [], []
    for k, form in enumerate(["ABAB", "ABACAB", "AABA", "ABCABC", "ABABCB"] * 2):
        E, b = build(natural(form), seed=100 + k)
        s, d = song_ratios(E, b, list(form))
        rs += s
        rd += d
    ms, md = float(np.median(rs)), float(np.median(rd))
    assert md > ms
    cfg = FormAdherenceConfig(repeat_ratio=float(np.sqrt(ms * md)),
                              ratio_temperature=float(np.log(md / ms) / 4.4))
    gaps = []                                   # contrast temperature, as calibrate.py
    for k, form in enumerate(["ABAB", "ABACAB", "AABA", "ABCABC", "ABABCB"] * 2):
        E, b = build(natural(form), seed=100 + k)
        r = score_form_adherence(E, form, boundaries=b, config=cfg)
        s_, d_ = list(r.same_pair_distances.values()), list(r.cross_pair_distances.values())
        mm = song_scale(s_, d_, s_ + d_)
        gaps.append((np.median(d_) - np.median(s_)) / mm)
    cfg.contrast_temperature = float(np.clip(np.median(gaps) / np.log(19), 0.02, 2.0))
    return cfg


def score(spec, form, cfg, seeds=3, **kw):
    rs = []
    for sd in range(seeds):
        E, b = build(spec, seed=sd, **kw)
        rs.append(score_form_adherence(E, form, boundaries=b, config=cfg))
    return rs


def mean(rs, attr="overall"):
    return float(np.mean([getattr(r, attr) for r in rs]))


GOOD = [("a", 0), ("b", 0), ("a", 1), ("c", 0), ("a", 2)]
MERGED = [("a", 0), ("a", 9), ("a", 1), ("c", 0), ("a", 2)]     # B is just another A
DROPPED = [("a", 0), ("b", 0), ("a", 1), ("a", 3), ("a", 2)]    # bridge C is an A
DRIFT = [("a", 0), ("b", 0), ("x", 0), ("c", 0), ("y", 0)]      # A repeats unrelated
COPY = [("a", 0), ("b", 0), ("a", 0), ("c", 0), ("a", 0)]       # A repeats identical


def test_R1_correct_form_beats_failures(cal):
    good = mean(score(GOOD, "ABACA", cal))
    assert good > 0.8
    for bad in (MERGED, DROPPED, DRIFT):
        assert mean(score(bad, "ABACA", cal)) < good - 0.3
    copy = score(COPY, "ABACA", cal)
    assert mean(copy) < 1e-3 and mean(copy, "variation") == 0.0
    assert mean(copy, "similarity") > 0.8            # copies DO cohere


def test_R2_independent_of_shared_component(cal):
    for spec in (GOOD, MERGED):
        vals = [mean(score(spec, "ABACA", cal, shared=s)) for s in (0.0, 3.0, 8.0)]
        assert max(vals) - min(vals) < 0.15, vals


def test_R3_attribution(cal):
    bc = score([("a", 0), ("b", 0), ("b", 5), ("a", 1), ("b", 1), ("b", 6)], "ABCABC", cal)[0]
    lp = bc.per_letter_pair_distinctness
    assert lp["B-C"] < 0.2 and lp["A-B"] > 0.8 and lp["A-C"] > 0.8, lp

    cp = score([("a", 0), ("b", 0), ("c", 0), ("a", 0), ("b", 1), ("c", 1)], "ABCABC", cal)[0]
    assert cp.per_letter_variation == {"A": 0.0, "B": 1.0, "C": 1.0}

    dr = score([("a", 0), ("b", 0), ("c", 0), ("z", 0), ("b", 1), ("c", 1)], "ABCABC", cal)[0]
    assert dr.per_letter_similarity["A"] < 0.2
    assert dr.per_letter_similarity["B"] > 0.8 and dr.per_letter_similarity["C"] > 0.8
    # drift is not blamed on contrast: letter pairs stay far above failure level
    assert min(dr.per_letter_pair_distinctness.values()) > 0.6


def test_R4_no_repeat_and_single_letter_forms(cal):
    assert mean(score([("a", 0), ("b", 0)], "AB", cal), "distinctness") > 0.6
    assert mean(score([("a", 0), ("a", 1)], "AB", cal), "distinctness") < 0.2   # B ~ A
    assert mean(score([("a", 0), ("a", 0)], "AB", cal), "distinctness") < 0.2   # B = A
    assert mean(score([("a", 0), ("a", 1), ("a", 2)], "AAA", cal), "similarity") > 0.5
    assert mean(score([("a", 0), ("b", 0), ("c", 0)], "AAA", cal), "similarity") < 0.2
    assert mean(score([("a", 0), ("a", 1), ("a", 2)], "ABC", cal)) < 0.2


def test_R5_own_form_beats_wrong_forms(cal):
    for sd in range(3):
        E, b = build(GOOD, seed=sd)
        own = score_form_adherence(E, "ABACA", boundaries=b, config=cal).overall
        for wrong in ["ABABA", "AABAA", "ABCDE", "ABCBA", "AAAAA", "ABBBA"]:
            w = score_form_adherence(E, wrong, boundaries=b, config=cal).overall
            assert own > w, (wrong, own, w)


def test_R6_sinkhorn_same_ordering(cal):
    sk = FormAdherenceConfig(ot_backend="sinkhorn", repeat_ratio=cal.repeat_ratio,
                             ratio_temperature=cal.ratio_temperature)
    g, m, c = (mean(score(s, "ABACA", sk, seeds=2)) for s in (GOOD, MERGED, COPY))
    assert g > m > c


def _tonal_song(form, n_frames, sr=400):
    t = np.arange(n_frames * sr // 2) / sr
    return GeneratedSong(audio=0.3 * np.sin(2 * np.pi * 40 * t), sample_rate=sr,
                         target_form=form)


def test_R7_reward_graded_and_consistent(cal):
    rf = FormAdherenceReward(None, RewardConfig(metric=cal))
    # (a) rises as B moves from "= A" to its own theme
    rng = np.random.default_rng(0)
    rewards = []
    for alpha in (0.0, 0.1, 0.2, 0.4):
        vals = []
        for sd in range(3):
            r = np.random.default_rng(sd)
            ta, tb, tc = (r.normal(0, 3, 32) for _ in range(3))
            sh = r.normal(0, 9, 32)
            th = {"A": ta, "B": (1 - alpha) * ta + alpha * tb, "C": tc}
            E = np.vstack([sh + th[c] + r.normal(0, .6, 32) + r.normal(0, .5, (30, 32))
                           for c in "ABACA"])
            E /= np.linalg.norm(E, axis=1, keepdims=True)
            vals.append(rf.score_embeddings(E, _tonal_song("ABACA", len(E))).scalar)
        rewards.append(np.mean(vals))
    assert all(a < b for a, b in zip(rewards, rewards[1:])), rewards

    # (b) ranks random songs like the evaluation score, (c) no ties among bad ones
    themes = ["a", "b", "c", "x", "y"]
    ev, rw = [], []
    for k in range(40):
        spec = [(themes[rng.integers(0, 5)], int(rng.integers(0, 3))) for _ in range(5)]
        E, _ = build(spec, seed=k)
        out = rf.score_embeddings(E, _tonal_song("ABACA", len(E)))
        ev.append(out.form_result.overall)
        rw.append(out.scalar)
    ev, rw = np.array(ev), np.array(rw)
    assert spearmanr(ev, rw).correlation > 0.75
    bad = rw[ev < 0.3]                          # clearly failing songs
    assert len(bad) >= 5 and len(set(np.round(bad, 5))) == len(bad)
