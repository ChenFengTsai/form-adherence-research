"""Behavioural tests: the metric must reward correct form and punish the
specific failure modes the proposal names. Run with: python -m pytest -q
or just execute this file."""

import numpy as np
from formadherence import FormAdherenceConfig, score_form_adherence
from formadherence.synthetic import make_song_embeddings
from formadherence.ot import wasserstein_distance, sinkhorn_distance


def _score(form, backend="exact", known_boundaries=True, **kw):
    E, b = make_song_embeddings(form, **kw)
    cfg = FormAdherenceConfig(ot_backend=backend, mp_window=24)
    boundaries = b if known_boundaries else None
    return score_form_adherence(E, form, boundaries=boundaries, config=cfg)


def test_ot_backends_agree_roughly():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(40, 8)); X /= np.linalg.norm(X, axis=1, keepdims=True)
    Y = rng.normal(size=(40, 8)); Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    w = wasserstein_distance(X, Y)
    s = sinkhorn_distance(X, Y, eps=0.02, max_iter=500)
    assert abs(w - s) < 0.15, (w, s)  # sinkhorn approx exact at small eps


def test_identity_distance_zero():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(30, 8)); X /= np.linalg.norm(X, axis=1, keepdims=True)
    assert wasserstein_distance(X, X) < 1e-6


def test_good_form_scores_high():
    r = _score("ABACA")
    assert r.overall > 0.6, r.overall
    assert r.distinctness > 0.6
    assert r.similarity > 0.6
    assert r.variation > 0.6


def test_copy_paste_tanks_variation_not_similarity():
    r = _score("ABABA", copy_paste=True)
    # Copy-paste => same-letter distance ~ 0 => high similarity, low variation.
    assert r.similarity > 0.8, r.similarity
    assert r.variation < 0.4, r.variation


def test_verse_equals_chorus_tanks_distinctness():
    # Merge B into A: two "different" letters are actually identical.
    r = _score("ABAB", merge_letters={"B": "A"})
    assert r.distinctness < 0.4, r.distinctness


def test_drifted_repeats_tank_similarity():
    # Huge within-letter variation => repeats no longer cohere.
    r = _score("ABABA", variation_scale=3.0)
    assert r.similarity < 0.6, r.similarity


def test_geometric_mean_blocks_trading():
    # A form that is perfect on two axes but zero on one must score low overall.
    r = _score("ABABA", copy_paste=True)  # variation ~ 0
    assert r.overall < 0.6, r.overall


def test_discovered_boundaries_recover_structure():
    # Without given boundaries, MP discovery should still yield a sane score
    # on a clearly-segmented song.
    r = _score("ABACA", known_boundaries=False,
               seg_len=80, theme_sep=4.0, frame_noise=0.08)
    assert r.overall > 0.4, r.overall
    assert len(r.segments) == 5


def test_single_letter_form_vacuous_distinctness():
    r = _score("AAAA")
    assert r.distinctness == 1.0


def test_reward_vector_shape():
    r = _score("ABACA")
    v = r.as_reward_vector()
    assert v.shape == (3,)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL {fn.__name__}: {e}")
        except Exception as e:
            print(f"ERROR {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{len(fns)} passed")
