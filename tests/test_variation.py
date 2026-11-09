"""Axis 3 on musical content (chroma + DTW): copies stay copies under noise /
EQ / gain; changed notes count as variation, in proportion to how much."""

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from formadherence import FormAdherenceConfig, score_form_adherence
from formadherence.variation import chroma_features, note_change, variation_from_chroma

SR = 16000


def _note(f, dur, harm=(1, .5, .3, .2)):
    t = np.arange(int(dur * SR)) / SR
    env = np.minimum(1, t / 0.01) * np.exp(-2.5 * t)
    return env * sum(a * np.sin(2 * np.pi * (k + 1) * f * t) for k, a in enumerate(harm))


def section(seed, sec=12.0, melody=None):
    """Melody + chord/bass + hi-hat noise; returns (audio, melody)."""
    r = np.random.default_rng(seed)
    scale = [0, 2, 4, 5, 7, 9, 11, 12]
    mel = melody if melody is not None else [scale[i] for i in r.integers(0, 8, int(sec / .5))]
    out = np.zeros(int(sec * SR))
    for k, m in enumerate(mel):
        a = int(k * .5 * SR)
        n = _note(261.6 * 2 ** (m / 12), .5)
        out[a:a + len(n)] += 0.3 * n[:len(out) - a]
    for k in range(int(sec / 2)):
        a = int(k * 2 * SR)
        root = [0, 5, 7, 0][k % 4]
        n = sum(_note(130.8 * 2 ** ((root + iv) / 12), 2.0, (1, .3)) for iv in (0, 4, 7))
        out[a:a + len(n)] += 0.15 * n[:len(out) - a]
    for k in range(int(sec / .25)):
        a = int(k * .25 * SR)
        L = int(.03 * SR)
        out[a:a + L] += 0.05 * r.normal(size=L) * np.exp(-np.arange(L) / (L / 4))
    return out, mel


@pytest.fixture(scope="module")
def a1():
    return section(1)


def f_between(x, y):
    return note_change(chroma_features(x, SR)[0], chroma_features(y, SR)[0])[0]


def test_sound_only_changes_are_copies(a1):
    from scipy.signal import butter, sosfilt
    x, _ = a1
    rng = np.random.default_rng(0)
    noise = x + rng.normal(0, np.sqrt(np.mean(x ** 2)) * 10 ** (-30 / 20), len(x))
    lowp = sosfilt(butter(4, 2000, "low", fs=SR, output="sos"), x)
    for y in (x.copy(), 0.5 * x, noise, lowp):
        assert f_between(x, y) < 0.01


def test_note_changes_are_variation_and_graded(a1):
    from scipy.signal import resample
    x, mel = a1

    def shift(secs):
        y = x.copy()
        a, n = 4 * SR, int(secs * SR)
        y[a:a + n] = resample(x[a:a + int(n * 2 ** (1 / 12))], n)
        return y
    f1, f3 = f_between(x, shift(1)), f_between(x, shift(3))
    new_mel = section(1, melody=[m + (2 if i % 3 == 0 else 0) for i, m in enumerate(mel)])[0]
    fm, fd = f_between(x, new_mel), f_between(x, section(99)[0])
    assert 0.04 < f1 < f3 < fm < fd                      # more change -> larger f
    assert abs(f1 - 1 / 12) < 0.04 and abs(f3 - 3 / 12) < 0.06   # ~ the changed share


def test_metric_uses_chroma_axis_when_audio_given(a1):
    x, mel = a1
    b_, _ = section(2)
    copy_song = np.concatenate([x, b_, x])               # A B A, second A = exact copy
    a2 = section(3, melody=[m + (2 if i % 5 == 2 else 0) for i, m in enumerate(mel)])[0]
    var_song = np.concatenate([x, b_, a2])               # second A = varied notes
    fr = 2.0
    E = np.random.default_rng(0).normal(size=(int(36 * fr), 8))  # embeddings irrelevant here
    b = [24, 48]
    cfg = FormAdherenceConfig()
    rc = score_form_adherence(E, "ABA", boundaries=b, config=cfg, audio=copy_song, sample_rate=SR)
    rv = score_form_adherence(E, "ABA", boundaries=b, config=cfg, audio=var_song, sample_rate=SR)
    assert rc.variation_method == rv.variation_method == "chroma_dtw"
    assert rc.variation == 0.0 and rv.variation == 1.0
    assert rc.note_change["A0-A2"] < 0.01 < rv.note_change["A0-A2"]
    re = score_form_adherence(E, "ABA", boundaries=b, config=cfg)          # no audio
    assert re.variation_method == "embedding"


def test_variation_from_chroma_vacuous_without_repeats(a1):
    C, fps = chroma_features(a1[0], SR)
    v = variation_from_chroma(C, fps, [(0, 6), (6, 12)], ["A", "B"], FormAdherenceConfig())
    assert v["per_letter"] == {} and v["pairs"] == {}


def test_check_variation_conditions(a1):
    import check_variation as V
    x, _ = a1
    song = np.concatenate([x, section(2)[0], x])
    rng = np.random.default_rng(0)
    cfg = FormAdherenceConfig()
    src, tgt = (0.0, 12.0), (24.0, 36.0)
    fs = {}
    for c in V.CONDITIONS:
        y, tgt_c = V.build_condition(song, SR, src, tgt, c, rng)
        C, fps = chroma_features(y, SR)
        fs[c] = V.pair_f(C, fps, src, tgt_c, cfg)
    assert fs["copy"] < 0.01 and fs["copy_noise"] < 0.01 and fs["copy_lowpass"] < 0.01
    assert 0.03 < fs["notes_1s"] < fs["notes_3s"]


def test_calibrate_fits_note_change(a1, monkeypatch, tmp_path):
    import calibrate as CAL
    import formadherence.audio_io as aio
    x, mel = a1
    songs = {}
    items = []
    for k in range(3):
        a2 = section(10 + k, melody=[m + (2 if i % 4 == k else 0) for i, m in enumerate(mel)])[0]
        songs[f"s{k}"] = np.concatenate([x, section(20 + k)[0], a2])
        p = tmp_path / f"e{k}.npy"
        np.save(p, np.zeros((72, 4)))
        items.append({"emb": str(p), "boundaries": [24, 48], "labels": list("ABA"),
                      "id": f"s{k}"})
    monkeypatch.setattr(aio, "read_audio", lambda p: (songs[p], SR))
    cfg = FormAdherenceConfig()
    chromas = CAL.fit_note_change(items, cfg, {f"s{k}": f"s{k}" for k in range(3)})
    assert len(chromas) == 3 and 0 < cfg.note_change_floor < cfg.note_change_floor + cfg.note_change_ramp


def test_spliced_copy_is_exact_even_with_unequal_lengths(a1):
    """A longer target must be REPLACED by the whole source, not half-overwritten."""
    import check_variation as V
    x, _ = a1
    longer = np.concatenate([section(5, sec=16.0)[0]])
    song = np.concatenate([x, section(2)[0], longer])        # A=12 s, B, A'=16 s
    cfg = FormAdherenceConfig()
    y, tgt = V.build_condition(song, SR, (0.0, 12.0), (24.0, 40.0), "copy", np.random.default_rng(0))
    assert abs((tgt[1] - tgt[0]) - 12.0) < 1e-6 and len(y) == len(song) - 4 * SR
    C, fps = chroma_features(y, SR)
    assert V.pair_f(C, fps, (0.0, 12.0), tgt, cfg) < 0.005
