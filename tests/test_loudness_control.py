"""Loudness-controlled test: on synthetic songs with a known answer.

Case "loud":    embeddings encode ONLY loudness, choruses are louder.
                -> raw AUC high, but matched / residual / normalized ~0.5.
Case "content": embeddings encode the LABEL, loudness is random per section.
                -> matched / residual stay high.
"""

import json
import os
import runpy
import sys

import numpy as np
import pytest

SCRIPT = os.path.join(os.path.dirname(__file__), "..", "scripts", "check_loudness_control.py")
FORM = "ABABCBAB"
SEC_S, SR, FR = 20, 400, 2.0


def _song(kind, seed):
    rng = np.random.default_rng(seed)
    base_level = {"A": -11.0, "B": -6.0, "C": -8.5}          # chorus (B) louder
    angle = {"A": 0.0, "B": 2.0, "C": 4.0}                   # label identity
    audio, emb = [], []
    for lab in FORM:
        if kind == "loud":
            lev = base_level[lab] + rng.normal(0, 2.0)          # overlapping levels
            theta = (lev + 20) / 8.0                          # direction = loudness only
        else:
            lev = rng.uniform(-16, -2)                        # loudness unrelated to label
            theta = angle[lab] + rng.normal(0, 0.15)
        n = SEC_S * SR
        audio.append(10 ** (lev / 20) * np.sqrt(2) * np.sin(2 * np.pi * 50 * np.arange(n) / SR)
                     + rng.normal(0, 10 ** (-40 / 20), n))       # a little noise, like music
        f = int(SEC_S * FR)
        v = np.column_stack([np.cos(theta), np.sin(theta)]).repeat(f, 0)
        emb.append(v + rng.normal(0, 0.4, v.shape))              # frame-to-frame variation
    bounds = [int(SEC_S * FR) * k for k in range(1, len(FORM))]
    return np.concatenate(audio), np.vstack(emb), bounds


def _setup(tmp_path, monkeypatch, kind, n=30):
    import formadherence.audio_io as aio
    adir = tmp_path / "audio"
    adir.mkdir()
    store, man = {}, []
    for k in range(n):
        sid = f"{kind}{k}"
        x, e, b = _song(kind, k)
        store[sid] = x
        (adir / f"{sid}.wav").write_bytes(b"x")
        np.save(tmp_path / f"{sid}.npy", e)
        man.append({"id": sid, "dataset": kind, "emb": str(tmp_path / f"{sid}.npy"),
                    "boundaries": b, "labels": list(FORM)})
    mp = tmp_path / "m.json"
    mp.write_text(json.dumps(man))
    monkeypatch.setattr(aio, "read_audio",
                        lambda p: (store[os.path.splitext(os.path.basename(p))[0]], SR))

    class LevelExtractor:      # "embedding" of audio = its loudness direction
        def extract(self, y, sr):
            hop = int(sr / FR)
            db = np.array([20 * np.log10(np.sqrt(np.mean(y[i:i + hop] ** 2)) + 1e-8)
                           for i in range(0, len(y) - hop + 1, hop)])
            th = (db + 20) / 8.0
            v = np.column_stack([np.cos(th), np.sin(th)])
            return v + np.random.default_rng(int(abs(y[:50]).sum() * 1e6) % 2**32).normal(0, 0.4, v.shape)

    monkeypatch.setattr(aio, "build_extractor", lambda spec: LevelExtractor())
    return mp, adir


def _run(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", [SCRIPT, *argv])
    runpy.run_path(SCRIPT, run_name="__main__")


def test_loudness_only_embeddings_are_exposed(tmp_path, monkeypatch):
    mp, adir = _setup(tmp_path, monkeypatch, "loud")
    out = tmp_path / "r.json"
    _run(monkeypatch, "--manifest", str(mp), "--audio_dir", str(adir), "--distance", "mean",
         "--normalize", "--norm_cache", str(tmp_path / "nc"), "--permutation", "50",
         "--save_json", str(out))
    r = json.load(open(out))["results"]["loud"]
    assert r["loudness_diff_auc"]["median"] > 0.7                # loudness separates
    assert r["mean"]["raw_auc"]["median"] > 0.65                 # so do the embeddings...
    ex = r["mean"]["matched_excess"]                             # ...but not beyond it
    assert ex["sign_test_p"] > 0.05 and ex["median_excess"] < 0.05
    assert r["mean"]["residual_auc"]["median"] < 0.62
    assert abs(r["normalized"]["raw_auc"]["median"] - 0.5) < 0.12
    assert r["normalized"]["loudness_diff_auc_after"]["median"] < 0.65


def test_content_embeddings_survive_control(tmp_path, monkeypatch):
    mp, adir = _setup(tmp_path, monkeypatch, "content")
    out = tmp_path / "r.json"
    _run(monkeypatch, "--manifest", str(mp), "--audio_dir", str(adir), "--distance", "mean",
         "--permutation", "50", "--save_json", str(out))
    r = json.load(open(out))["results"]["content"]
    assert r["mean"]["raw_auc"]["median"] > 0.9
    assert r["mean"]["matched_auc"]["median"] > 0.85
    ex = r["mean"]["matched_excess"]
    assert ex["sign_test_p"] < 0.01 and ex["median_excess"] > 0.3
    assert r["mean"]["residual_auc"]["median"] > 0.85
    assert r["mean"]["matched_perm"]["p_value"] < 0.05
    av = r["mean"]["added_value"]
    assert av["loudness_plus_embedding_median_auc"] > av["loudness_only_median_auc"] + 0.2


def test_matched_auc_unit():
    sys.path.insert(0, os.path.dirname(SCRIPT))
    mod = runpy.run_path(SCRIPT, run_name="lc")
    d = np.array([0.1, 0.2, 0.8, 0.9])
    dl = np.array([1.0, 5.0, 1.2, 5.1])
    same = np.array([True, True, False, False])
    a, share = mod["matched_auc"](d, dl, same, tol=0.5, min_cmp=1)
    assert a == 1.0 and share == 0.5        # only 1.0~1.2 and 5.0~5.1 are compared


def test_sign_test_unit():
    mod = runpy.run_path(SCRIPT, run_name="lc2")
    p, k, n = mod["sign_test"]([0.9] * 10, [0.5] * 10)
    assert (k, n) == (10, 10) and p < 0.001
    p, k, n = mod["sign_test"]([0.4] * 10, [0.5] * 10)
    assert k == 0 and p == 1.0
