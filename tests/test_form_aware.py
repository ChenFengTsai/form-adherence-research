"""Form-aware segmentation: recovers cuts that Foote novelty misses."""

import numpy as np

from formadherence.config import FormAdherenceConfig
from formadherence.metric import score_form_adherence
from formadherence.segmentation import (_align_to_templates, form_aware_boundaries,
                                        segment_boundaries)


def _song(form, lens, seed=0, dim=16, noise=0.35, bursts=False):
    rng = np.random.default_rng(seed)
    tpl = {L: rng.normal(size=dim) for L in set(form)}
    frames, cuts, t = [], [], 0
    for L, n in zip(form, lens):
        x = tpl[L] + noise * rng.normal(size=(n, dim))
        if bursts and n >= 30:                   # a drum fill: 5 odd frames mid-section
            m = n // 2
            x[m:m + 5] = rng.normal(size=(5, dim)) * 1.5
        frames.append(x)
        t += n
        cuts.append(t)
    return np.vstack(frames), cuts[:-1]


def _err(est, ref):
    return max(min(abs(e - r) for e in est) for r in ref)


def test_dp_exact_alignment():
    form, lens = "ABAB", [10, 20, 15, 12]
    E, ref = _song(form, lens, noise=0.05)
    En = E / np.linalg.norm(E, axis=1, keepdims=True)
    tpl = {}
    edges = [0] + ref + [len(E)]
    for L in "AB":
        m = np.vstack([En[edges[k]:edges[k + 1]] for k in range(4) if form[k] == L]).mean(0)
        tpl[L] = m / np.linalg.norm(m)
    cuts, obj = _align_to_templates(En, list(form), tpl, np.zeros(len(E) + 1), 3)
    assert cuts == ref


def test_recovers_uneven_sections():
    form, lens = "ABACAB", [30, 50, 20, 40, 60, 25]
    E, ref = _song(form, lens)
    est = form_aware_boundaries(E, list(form), min_len=6)
    assert len(est) == 5 and est == sorted(est)
    assert _err(est, ref) <= 2


def test_beats_foote_with_distractors():
    """Mid-section bursts fool novelty; the repeat pattern does not."""
    wins = 0
    for seed in range(10):
        form = "ABABCB"
        lens = list(np.random.default_rng(seed).integers(24, 70, size=6))
        E, ref = _song(form, lens, seed=seed, bursts=True)
        foote = segment_boundaries(E, n_boundaries=5, window=16)
        fa = form_aware_boundaries(E, list(form), kernel_size=16, min_len=6)
        if len(foote) < 5 or _err(fa, ref) < _err(foote, ref):
            wins += 1
        assert _err(fa, ref) <= 3
    assert wins >= 5


def test_min_len_and_degenerate():
    E, _ = _song("AB", [5, 5])
    assert len(form_aware_boundaries(E, ["A", "B"], min_len=3)) == 1
    assert form_aware_boundaries(E, ["A"]) == []
    E, _ = _song("ABCDE", [3, 3, 3, 3, 3])     # min_len shrinks to fit
    assert len(form_aware_boundaries(E, list("ABCDE"), min_len=6)) == 4


def test_metric_uses_form_aware():
    form, lens = "ABACAB", [30, 50, 20, 40, 60, 25]
    E, ref = _song(form, lens)
    cfg = FormAdherenceConfig(segmentation_method="form_aware", form_aware_min_len_s=3.0)
    r = score_form_adherence(E, form, config=cfg)
    got = [s.start for s in r.segments[1:]]
    assert _err(got, ref) <= 2 and not r.segmentation_fallback
    r_wrong = score_form_adherence(E, "ABCDEF", config=cfg)
    assert r.overall > r_wrong.overall


def test_validate_human_form_aware_smoke(tmp_path, monkeypatch):
    import json, os, runpy, sys
    from dataclasses import asdict
    forms = ["ABACAB", "AABBA", "ABABCB", "ABCBCA"]
    man = []
    for k in range(8):
        form = forms[k % 4]
        lens = list(np.random.default_rng(k).integers(24, 60, size=len(form)))
        E, ref = _song(form, lens, seed=k, bursts=True)
        p = tmp_path / f"s{k}.npy"
        np.save(p, E)
        man.append({"id": f"s{k}", "emb": str(p), "boundaries": [int(x) for x in ref],
                    "labels": list(form)})
    (tmp_path / "m.json").write_text(json.dumps(man))
    cfg = asdict(FormAdherenceConfig(ot_backend="mean"))
    cfg["axis_weights"] = list(cfg["axis_weights"])
    (tmp_path / "c.json").write_text(json.dumps({"metric": cfg}))
    script = os.path.join(os.path.dirname(__file__), "..", "scripts", "validate_human.py")
    out = tmp_path / "v.json"
    monkeypatch.setattr(sys, "argv", [script, "--manifest", str(tmp_path / "m.json"),
                                      "--config", str(tmp_path / "c.json"), "--limit", "0",
                                      "--n_wrong", "3", "--margins", "0.05",
                                      "--also_discovered", "--also_form_aware",
                                      "--foote_kernel", "16", "--save_json", str(out)])
    runpy.run_path(script, run_name="__main__")
    r = json.load(open(out))["results"]
    assert set(r) == {"annotated", "discovered", "form_aware"}
    fa = r["form_aware"]["geometric margin=0.05"]["pairwise"]
    assert fa >= 0.75          # clean synthetic songs: own form wins with fitted cuts
