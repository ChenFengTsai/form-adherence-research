"""SongFormBench loader: parsing, label normalisation, cleanliness checks, and
an end-to-end run of the build script against a fake `datasets` module."""

import json
import os
import runpy
import sys
import types

import numpy as np
import pytest

from formadherence.songformbench import (boundary_hit_rate, check_song,
                                         normalize_label, parse_annotation,
                                         parse_msa_text)

MSA = "0.0 intro\n12.5 verse\n40.0 pre-chorus\n50.0 chorus\n80.0 outro\n95.0 end\n"


def test_parse_msa_text():
    st, lb, end = parse_msa_text(MSA)
    assert st == [0.0, 12.5, 40.0, 50.0, 80.0]
    assert lb == ["intro", "verse", "pre-chorus", "chorus", "outro"]
    assert end == 95.0


@pytest.mark.parametrize("obj", [
    MSA,
    [{"start": 0, "end": 12.5, "label": "intro"}, {"start": 12.5, "end": 40, "label": "verse"},
     {"start": 40, "end": 50, "label": "pre-chorus"}, {"start": 50, "end": 80, "label": "chorus"},
     {"start": 80, "end": 95, "label": "outro"}],
    [[0, 12.5, "intro"], [12.5, 40, "verse"], [40, 50, "pre-chorus"], [50, 80, "chorus"],
     [80, 95, "outro"]],
    [[0, "intro"], [12.5, "verse"], [40, "pre-chorus"], [50, "chorus"], [80, "outro"], [95, "end"]],
    json.dumps({"segments": [[0, "intro"], [12.5, "verse"], [40, "pre-chorus"],
                             [50, "chorus"], [80, "outro"], [95, "end"]]}),
    "0.0 12.5 intro\n12.5 40 verse\n40 50 pre-chorus\n50 80 chorus\n80 95 outro\n",
])
def test_parse_annotation_shapes(obj):
    st, lb, end = parse_annotation(obj)
    assert st == [0.0, 12.5, 40.0, 50.0, 80.0]
    assert [normalize_label(x) for x in lb] == ["intro", "verse", "pre-chorus", "chorus", "outro"]
    assert end == 95.0


def test_normalize_label():
    assert normalize_label("Pre-Chorus") == "pre-chorus"
    assert normalize_label("prechorus", prechorus="verse") == "verse"
    assert normalize_label("Instrumental") == "inst"
    assert normalize_label("chorus2") == "chorus"
    assert normalize_label("solo") is None          # strict: not in the 8-class set


def test_clean_song_passes():
    st, lb, end = parse_msa_text(MSA)
    probs, warns, norm = check_song(st, lb, end, audio_dur=95.3, rms=0.1)
    assert probs == [] and warns == []


def test_length_mismatch_is_dropped():
    """The Harmonix/YouTube failure: lots of unannotated MUSIC (longer version)."""
    st, lb, end = parse_msa_text(MSA)
    probs, _, _ = check_song(st, lb, end, audio_dur=120.0, rms=0.1, tail_quiet=False)
    assert any("unannotated audio after" in p for p in probs)
    # audio shorter than the annotation
    probs, _, _ = check_song(st, lb, end, audio_dur=90.0, rms=0.1)
    assert any("audio shorter" in p for p in probs)


def test_tails_that_are_fine():
    st, lb, end = parse_msa_text(MSA)
    # a long but QUIET tail (fade-out / silence) is kept and cropped
    probs, warns, _ = check_song(st, lb, end, audio_dur=120.0, rms=0.1, tail_quiet=True)
    assert probs == [] and any("trailing" in w and "quiet" in w for w in warns)
    # a few seconds of unannotated music is kept and cropped
    probs, warns, _ = check_song(st, lb, end, audio_dur=99.0, rms=0.1, tail_quiet=False)
    assert probs == [] and any("trailing 4.0s" in w for w in warns)
    # a 5.6 s unannotated start is kept and cropped
    probs, warns, _ = check_song([5.6] + st[1:], lb, end, audio_dur=95.0, rms=0.1)
    assert probs == [] and any("leading 5.6s" in w for w in warns)


def test_edge_is_quiet():
    from formadherence.songformbench import edge_is_quiet
    sr = 100
    x = np.concatenate([0.5 * np.sin(np.arange(9000)), 0.01 * np.sin(np.arange(1000))])
    assert edge_is_quiet(x, sr, 90, 100, 0, 90)[0] is True
    y = 0.5 * np.sin(np.arange(10000))
    assert edge_is_quiet(y, sr, 90, 100, 0, 90)[0] is False


def test_other_problems():
    st, lb, end = parse_msa_text(MSA)
    assert any("unknown labels" in p for p in check_song(st, lb[:-1] + ["fade"], end, 95, 0.1)[0])
    assert any("increasing" in p for p in check_song([0, 20, 10, 50, 80], lb, end, 95, 0.1)[0])
    assert any("starts at" in p for p in check_song([12.0] + st[1:], lb, end, 95, 0.1)[0])
    assert any("near-silent" in p for p in check_song(st, lb, end, 95, 1e-6)[0])
    assert any("musical" in p for p in check_song([0, 10], ["silence", "verse"], 95, 95, 0.1)[0])
    _, warns, _ = check_song([0, 0.5, 30, 60], ["intro", "verse", "verse", "chorus"], 95, 95, 0.1)
    assert any("< 1.0s" in w for w in warns) and any("adjacent" in w for w in warns)


def test_hit_rate():
    assert boundary_hit_rate([10, 20], [10.2, 20.4], 0.5) == 1.0
    assert boundary_hit_rate([10, 20], [12, 25], 0.5) == 0.0
    assert boundary_hit_rate([10, 20], [12, 25], 3.0) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# end-to-end with a fake Hugging Face repo (files, like the real one)
# ---------------------------------------------------------------------------

def _install_fake_hub(monkeypatch, tmp_path, repo_files):
    """repo_files: {relpath: bytes}. Audio payload "tag|seconds"."""
    store = tmp_path / "hub"
    for rel, data in repo_files.items():
        p = store / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    hub = types.ModuleType("huggingface_hub")

    class HfApi:
        def list_repo_files(self, repo, repo_type=None, revision=None):
            return sorted(repo_files)

        def dataset_info(self, repo, revision=None):
            return types.SimpleNamespace(sha="abc123")

    hub.HfApi = HfApi
    hub.hf_hub_download = lambda repo, rel, repo_type=None, revision=None: str(store / rel)
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)

    import formadherence.audio_io as aio

    def fake_read(path):
        with open(path, "rb") as f:
            secs = float(f.read().decode().split("|")[1])
        return 0.1 * np.sin(np.arange(int(secs * 100))), 100

    class FakeCache:
        def __init__(self, cache_dir, spec, **kw):
            self.dir = cache_dir
            os.makedirs(cache_dir, exist_ok=True)

        def get(self, audio_path):
            emb = np.random.default_rng(0).normal(size=(190, 8))   # 95 s @ 2 fps
            out = os.path.join(self.dir, os.path.basename(audio_path) + ".npy")
            np.save(out, emb)
            return emb, out

    def fake_write(path, wav, sr):
        with open(path, "wb") as f:
            f.write(f"crop|{len(wav) / sr}".encode())

    monkeypatch.setattr(aio, "read_audio", fake_read)
    monkeypatch.setattr(aio, "write_wav", fake_write)
    monkeypatch.setattr(aio, "EmbeddingCache", FakeCache)


SCRIPT = os.path.join(os.path.dirname(__file__), "..", "scripts", "build_songformbench_manifest.py")


def _run(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", [SCRIPT, *argv])
    runpy.run_path(SCRIPT, run_name="__main__")


def test_classify_layout():
    mod = types.SimpleNamespace(**runpy.run_path(SCRIPT, run_name="sfb_build"))
    files = ["README.md", "dataset.py", "utils/HarmonixSet/x.txt",
             "data/labels/HarmonixSet/0001_a.txt", "data/audio/HarmonixSet/0001_a.flac",
             "data/labels/CN/song7.txt", "data/audio/CN/song7.mp3",
             "data/mels/HarmonixSet/0001_a.npy"]
    cls, dups, other = mod.classify_files(files)
    assert set(cls["label"]) == {("HarmonixSet", "0001_a"), ("CN", "song7")}
    assert set(cls["audio"]) == {("HarmonixSet", "0001_a"), ("CN", "song7")}
    assert ("HarmonixSet", "0001_a") in cls["mel"]
    assert mod.find_match(cls["audio"], "CN", "song7") == "data/audio/CN/song7.mp3"


def test_build_script_end_to_end(tmp_path, monkeypatch):
    lab = MSA.encode()
    files = {
        "README.md": b"x", "dataset.py": b"x", "utils/HarmonixSet/readme.txt": b"x",
        "data/labels/HarmonixSet/0001_a.txt": lab,
        "data/audio/HarmonixSet/0001_a.flac": b"a|95.2",
        "data/labels/HarmonixSet/0002_b.txt": lab,
        "data/audio/HarmonixSet/0002_b.flac": b"b|140.0",          # longer version
        "data/labels/CN/song7.txt": lab,
        "data/audio/CN/song7.flac": b"c|95.0",
        "data/labels/CN/song8.txt": MSA.replace("outro", "fadeout").encode(),  # bad label
        "data/audio/CN/song8.flac": b"d|95.0",
        "data/labels/CN/song9.txt": lab,                            # no audio
        "data/labels/CN/song10.txt": lab,
        "data/audio/CN/song10.flac": b"e|99.0",                    # 4 s tail -> crop
    }
    _install_fake_hub(monkeypatch, tmp_path, files)
    out, rep = tmp_path / "m.json", tmp_path / "r.json"
    args = ["--data_root", str(tmp_path), "--out", str(out), "--report", str(rep)]
    with pytest.raises(SystemExit, match="exactly 200 HarmonixSet"):
        _run(monkeypatch, *args)                       # 5 songs != 200 + 100
    _run(monkeypatch, *args, "--allow_count_mismatch")

    report = json.load(open(rep))
    assert report["n_total"] == 6 and report["n_clean"] == 3
    assert report["benchmark"]["loaded"] == {"HarmonixSet": 2, "CN": 4}
    assert report["benchmark"]["commit"] == "abc123"
    bad = {r["stem"]: r["problems"] for r in report["songs"] if r["problems"]}
    assert set(bad) == {"0002_b", "cn_song8", "cn_song9"}
    assert any("unannotated audio after" in p for p in bad["0002_b"])
    crop = [r for r in report["songs"] if r["stem"] == "cn_song10"][0]
    assert crop["crop_s"] == [0.0, 95.0]
    with open(crop["audio"], "rb") as f:
        assert f.read() == b"crop|95.0"                     # audio == annotated span
    assert any("unknown labels" in p for p in bad["cn_song8"])
    assert any("no audio" in p for p in bad["cn_song9"])

    man = json.load(open(out))
    assert sorted(m["id"] for m in man) == ["0001_a", "cn_song10", "cn_song7"]
    assert {m["dataset"] for m in man} == {"sfb_hx", "sfb_cn"}
    hx = [m for m in man if m["id"] == "0001_a"][0]
    assert hx["labels"] == ["A", "B", "C", "D", "E"]       # 5 distinct sections
    assert hx["boundaries"] == [25, 80, 100, 160]
    assert os.path.exists(tmp_path / "songformbench" / "segments" / "0001_a.txt")
    # 0.2 s tail is cropped too, so audio files are .wav; originals kept untouched
    assert os.path.exists(tmp_path / "songformbench" / "audio" / "0001_a.wav")
    assert os.path.exists(tmp_path / "songformbench" / "original_audio" / "0001_a.flac")
    assert sorted(os.listdir(tmp_path / "songformbench" / "audio")) == \
        ["0001_a.wav", "cn_song10.wav", "cn_song7.flac"]


def test_exact_300_passes_guard(tmp_path, monkeypatch):
    files = {}
    for k in range(200):
        files[f"data/labels/HarmonixSet/{k:04d}_s.txt"] = MSA.encode()
        files[f"data/audio/HarmonixSet/{k:04d}_s.flac"] = f"h{k}|95.0".encode()
    for k in range(100):
        files[f"data/labels/CN/c{k}.txt"] = MSA.encode()
        files[f"data/audio/CN/c{k}.flac"] = f"c{k}|95.0".encode()
    _install_fake_hub(monkeypatch, tmp_path, files)
    rep = tmp_path / "r.json"
    _run(monkeypatch, "--data_root", str(tmp_path), "--check_only", "--report", str(rep))
    r = json.load(open(rep))
    assert r["benchmark"]["exact_300"] is True and r["n_clean"] == 300


def test_inspect_prints_layout(tmp_path, monkeypatch, capsys):
    _install_fake_hub(monkeypatch, tmp_path, {
        "data/labels/HarmonixSet/0001_a.txt": MSA.encode(),
        "data/audio/HarmonixSet/0001_a.flac": b"a|95.0"})
    _run(monkeypatch, "--inspect")
    txt = capsys.readouterr().out
    assert "matching audio file: 1/1" in txt and "parsed 5 sections" in txt
