"""Tests for the core package and the fixes. Run:  python -m pytest -q tests
Needs only numpy, scipy, pytest (no torch / CLAP)."""

import json
import os
import subprocess
import sys
import warnings

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from formadherence import (FormAdherenceConfig, score_form_adherence, hard_auc,
                           wasserstein_distance, sinkhorn_distance, sinkhorn_divergence)
from formadherence.labels import annotation_to_manifest_fields, canonical_family
from formadherence.segmentation import segment_boundaries
from formadherence.synthetic import make_song_embeddings
from formadherence.audio_io import ExtractorSpec, EmbeddingCache


# ---------------------------------------------------------------- labels ----

def test_short_section_does_not_shift_labels():
    # prechorus at 10.0-10.2 s collapses to 0 frames at 2 fps
    starts = [0.0, 10.0, 10.2, 20.0, 30.0, 40.0]
    raw = ["intro", "verse", "prechorus", "chorus", "verse", "chorus"]
    f = annotation_to_manifest_fields(starts, 50.0, raw, n_frames=100, frame_rate=2.0)
    segs = list(zip([0] + f["boundaries"], f["labels_raw"]))
    # the frames 20-40 (10.2-20 s) belong to the prechorus, 40-60 to the chorus
    assert segs == [(0, "intro"), (20, "prechorus"), (40, "chorus"),
                    (60, "verse"), (80, "chorus")]
    assert f["labels"][2] == f["labels"][4]          # both choruses share a letter
    assert len(f["labels"]) == len(f["boundaries"]) + 1


def test_salami_silence_does_not_change_letter_path():
    a = annotation_to_manifest_fields([0, 10, 20, 30], 40, ["A", "B", "A'", "B"], 80, 2.0)
    b = annotation_to_manifest_fields([0, 10, 20, 30, 40, 50], 60,
                                      ["Silence", "A", "B", "A'", "B", "Silence"], 120, 2.0)
    assert a["labels"] == ["A", "B", "A", "B"]
    inner = b["labels"][1:-1]
    assert inner[0] == inner[2] and inner[1] == inner[3]      # A == A'
    assert b["labels"][0] != b["labels"][-1]                   # silences NOT a pair
    assert b["labels"][0] not in inner and b["labels"][-1] not in inner
    assert b["variants"] == [0, 0, 0, 1, 0, 0]                  # A' recorded


def test_special_drop_and_keep():
    starts, raw = [0, 10, 20, 30, 40], ["silence", "verse", "chorus", "verse", "silence"]
    d = annotation_to_manifest_fields(starts, 50, raw, 100, 2.0, special="drop")
    assert d["labels_raw"] == ["verse", "chorus", "verse"] and d["boundaries"] == [40, 60]
    k = annotation_to_manifest_fields(starts, 50, raw, 100, 2.0, special="keep")
    assert k["labels"][0] == k["labels"][-1]


def test_family_table():
    assert canonical_family("verse2") == "verse"
    assert canonical_family("Pre-Chorus") == "prechorus"
    assert canonical_family("instchorus") == "chorus"
    assert canonical_family("Main Theme") == "theme"
    assert canonical_family("guitarsolo") == "solo"


def test_annotation_longer_than_audio():
    f = annotation_to_manifest_fields([0, 10, 20, 30], 40, ["a", "b", "a", "c"], 50, 2.0)
    assert f["boundaries"] == [20, 40] and f["labels_raw"] == ["a", "b", "a"]


# -------------------------------------------------------------------- OT ----

def test_debiased_sinkhorn_is_zero_for_copy():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 8)) + 3
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    assert sinkhorn_distance(X, X) > 1e-3            # the bias the fix removes
    assert sinkhorn_divergence(X, X) < 1e-6


def test_ot_ignores_order_documented():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(20, 4))
    assert wasserstein_distance(X, X[::-1]) < 1e-9


# ---------------------------------------------------------------- metric ----

@pytest.mark.parametrize("backend", ["exact", "sinkhorn"])
def test_relative_metric_orders_songs(backend):
    cfg = FormAdherenceConfig(ot_backend=backend)

    def score(**kw):
        E, b = make_song_embeddings("ABACA", seg_len=40, seed=3, **kw)
        return score_form_adherence(E, "ABACA", boundaries=b, config=cfg)

    good = score()
    copy = score(copy_paste=True)
    collapsed = score(merge_letters={"B": "A", "C": "A"})
    shared = score(shared_component=3.0)
    assert good.overall > 0.9 and shared.overall > 0.9
    assert copy.variation == 0.0 and copy.similarity > 0.9
    assert collapsed.similarity < 0.5 < good.similarity
    assert good.auc == 1.0


def test_relative_metric_is_scale_free():
    E, b = make_song_embeddings("ABAB", seg_len=30, seed=2)
    cfg = FormAdherenceConfig(score_mode="relative")
    r1 = score_form_adherence(E, "ABAB", boundaries=b, config=cfg)
    # shrinking every distance (pull frames toward a common point) keeps ranks
    E2 = E + 5 * E.mean(0)
    E2 /= np.linalg.norm(E2, axis=1, keepdims=True)
    r2 = score_form_adherence(E2, "ABAB", boundaries=b, config=cfg)
    assert r1.auc == r2.auc == 1.0
    assert r2.similarity > 0.9


def test_forms_without_repeats_and_single_letter():
    E, b = make_song_embeddings("ABC", seg_len=30, seed=4)
    r = score_form_adherence(E, "ABC", boundaries=b)
    assert r.distinctness > 0.9 and r.similarity == 1.0 and r.auc is None
    E, b = make_song_embeddings("AAA", seg_len=30, seed=4)
    r = score_form_adherence(E, "AAA", boundaries=b)
    assert r.distinctness == 1.0


def test_wrong_supplied_boundaries_raise():
    E, b = make_song_embeddings("ABA", seg_len=30, seed=0)
    with pytest.raises(ValueError):                 # one boundary for 3 sections
        score_form_adherence(E, "ABA", boundaries=b[:1])
    with pytest.raises(ValueError):                 # second one is out of range
        score_form_adherence(E, "ABA", boundaries=[b[0], 10_000])


def test_multichar_labels_accepted():
    E, b = make_song_embeddings("ABA", seg_len=30, seed=0)
    r = score_form_adherence(E, ["X26", "B", "X26"], boundaries=b)
    assert r.target_form == "X26|B|X26" and r.auc == 1.0


def test_absolute_mode_still_runs():
    E, b = make_song_embeddings("ABACA", seg_len=30, seed=0)
    r = score_form_adherence(E, "ABACA", boundaries=b,
                             config=FormAdherenceConfig(score_mode="absolute"))
    assert r.score_mode == "absolute" and 0 <= r.overall <= 1


def test_hard_auc_matches_definition():
    same, diff = [0.1, 0.3], [0.2, 0.3, 0.5]
    # pairs: (0.1<0.2,0.3,0.5)=3, (0.3: >0.2 no, ==0.3 tie, <0.5 yes) = 1.5
    assert hard_auc(same, diff) == pytest.approx(4.5 / 6)
    assert hard_auc([], diff) is None


def test_boundary_near_start_is_discoverable():
    rng = np.random.default_rng(0)
    th = rng.normal(size=(3, 16)) * 3
    E = np.vstack([th[0] + 0.1 * rng.normal(size=(10, 16)),
                   th[1] + 0.1 * rng.normal(size=(60, 16)),
                   th[2] + 0.1 * rng.normal(size=(60, 16))])
    # documented limitation: the default 32-frame kernel can't see a 10-frame
    # intro; a 16-frame kernel can.
    b32 = segment_boundaries(E, n_boundaries=2, window=32, min_gap=8)
    assert not any(abs(x - 10) <= 2 for x in b32), b32
    b16 = segment_boundaries(E, n_boundaries=2, window=16, min_gap=8)
    assert any(abs(x - 10) <= 2 for x in b16), b16


# ---------------------------------------------------------------- config ----

def test_old_config_loads_as_absolute(tmp_path):
    old = {"ot_backend": "sinkhorn", "sinkhorn_eps": 0.05, "sinkhorn_max_iter": 200,
           "sinkhorn_tol": 1e-6, "max_frames_per_segment": 256, "mp_window": 32,
           "mp_exclusion_zone": 0.5, "n_boundaries": None, "tau": 0.2,
           "variation_floor": 0.01, "variation_softness": 0.1, "combine": "geometric",
           "axis_weights": [1, 1, 1], "similarity_temperature": 0.15}
    p = tmp_path / "old.json"
    p.write_text(json.dumps(old))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        cfg = FormAdherenceConfig.from_json(str(p))
    assert cfg.score_mode == "absolute" and cfg.tau == 0.2 and w


# ----------------------------------------------------------------- cache ----

def test_cache_key_depends_on_extractor(tmp_path, monkeypatch):
    audio = tmp_path / "song.wav"
    audio.write_bytes(b"not really audio")
    s1, s4 = ExtractorSpec(clap_window_s=1.0), ExtractorSpec(clap_window_s=4.0)
    assert s1.tag() != s4.tag()
    c1, c4 = EmbeddingCache(str(tmp_path), s1), EmbeddingCache(str(tmp_path), s4)
    assert c1.path_for(str(audio)) != c4.path_for(str(audio))

    class Fake:
        def __init__(self, v): self.v = v
        def extract(self, wav, sr): return np.full((5, 3), self.v)
    import formadherence.audio_io as aio
    monkeypatch.setattr(aio, "read_audio", lambda p: (np.zeros(10), 16000))
    monkeypatch.setattr(aio, "build_extractor", lambda spec: Fake(spec.clap_window_s))
    e1, _ = c1.get(str(audio))
    e4, _ = c4.get(str(audio))
    assert e1[0, 0] == 1.0 and e4[0, 0] == 4.0          # no silent reuse
    # legacy reuse only when asked
    np.save(tmp_path / (aio.file_hash(str(audio)) + ".npy"), np.full((5, 3), 9.0))
    c8 = EmbeddingCache(str(tmp_path), ExtractorSpec(clap_window_s=8.0), reuse_legacy=True)
    assert c8.get(str(audio))[0][0, 0] == 9.0


# ---------------------------------------------------- scripts end-to-end ----

def _fake_manifest(tmp_path, name, n=6, seed0=0):
    items = []
    for k in range(n):
        form = ["ABACA", "ABAB", "AABBA"][k % 3]
        E, b = make_song_embeddings(form, seg_len=24, seed=seed0 + k, shared_component=2.0)
        p = tmp_path / f"{name}_{k}.npy"
        np.save(p, E)
        items.append({"emb": str(p), "boundaries": b, "labels": list(form),
                      "dataset": name, "id": f"{name}{k}"})
    m = tmp_path / f"{name}.json"
    m.write_text(json.dumps(items))
    return str(m)


def _run(args):
    r = subprocess.run([sys.executable] + args, capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout


def test_scripts_end_to_end(tmp_path):
    m1 = _fake_manifest(tmp_path, "setA", seed0=0)
    m2 = _fake_manifest(tmp_path, "setB", seed0=50)
    out = _run(["scripts/diagnose_formulation.py", "--mode", "real",
                "--manifest", f"{m1},{m2}", "--limit", "0", "--auc",
                "--permutation", "50", "--save_json", str(tmp_path / "res.json")])
    assert "[setA OT  ]" in out and "[setB OT  ]" in out and "p=" in out
    res = json.load(open(tmp_path / "res.json"))
    assert res["results"]["setA"]["ot_auc"]["median"] == 1.0

    cfg_path = str(tmp_path / "cfg.json")
    out = _run(["scripts/calibrate.py", "--manifest", m1, "--out", cfg_path,
                "--ot_backend", "exact"])
    assert "copy_floor" in out and "human songs, setA" in out
    cfg = json.load(open(cfg_path))
    assert cfg["metric"]["score_mode"] == "relative"

    emb = json.load(open(m1))[0]["emb"]
    out = _run(["scripts/run_evaluation.py", "--emb", emb, "--form", "ABACA",
                "--config", cfg_path])
    r = json.loads(out[out.index("{"):])
    assert r["n_segments"] == 5 and 0 <= r["overall"] <= 1

    out = _run(["scripts/diagnose_formulation.py", "--mode", "synthetic", "--auc",
                "--n_per_form", "2", "--shared_component", "3"])
    assert "AUC median" in out
