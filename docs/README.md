# Form-Adherence Metric

A conditional structural-form adherence metric for full-song music generation.
Given a **target form** (a letter sequence such as `ABACA`) and **per-frame
embeddings** of a generated song, it scores how well the song realizes that form
along three orthogonal axes.

This is Part I of the proposal — the measurement the RL optimization (Part II)
will consume. It is written dependency-light (numpy + scipy only) so every
algorithm is auditable; the embedding-extraction stage (CLAP / EnCodec) is
external, and the metric takes already-extracted arrays.

## The three axes

| Axis | Name | Rewards | Fails when |
|------|------|---------|-----------|
| 1 | Cross-Letter Distinctness | different letters sound different | verse == chorus |
| 2 | Same-Letter Similarity | repeats of a letter cohere | repeats drift off-theme |
| 3 | Within-Letter Variation | repeats vary, but stay on-theme | robotic copy-paste |

The overall score is a **weighted geometric mean** of the three axes, so a near
-zero on any one axis drags the whole score down — a model cannot trade a failure
on one axis for a surplus on another. Per-axis and per-letter scores are also
returned for diagnosis and reward-hacking audits.

## Install

```bash
cd formadherence
pip install numpy scipy        # the only dependencies
```

## Usage

```python
import numpy as np
from formadherence import FormAdherenceConfig, score_form_adherence

# embeddings: (T, d) per-frame vectors from CLAP / EnCodec, already extracted.
embeddings = np.load("song_embeddings.npy")

cfg = FormAdherenceConfig(ot_backend="exact")   # or "sinkhorn"
result = score_form_adherence(embeddings, target_form="ABACA", config=cfg)

print(result.overall)          # single Form-Adherence value in [0, 1]
print(result.as_reward_vector())  # (distinctness, similarity, variation) for RL
print(result.per_letter_similarity, result.per_letter_variation)
```

### Boundaries: known or discovered

Both modes are supported (per the build decision):

```python
# Known boundaries (interior frame indices), e.g. from an aligned annotation:
score_form_adherence(embeddings, "ABACA", boundaries=[80, 160, 240, 320])

# Discover boundaries automatically (Foote checkerboard-kernel novelty):
score_form_adherence(embeddings, "ABACA", boundaries=None)
```

## Design notes tied to the proposal

- **Segments as distributions, not pooled vectors.** Each section is modeled as
  its empirical distribution of frame embeddings, compared with Optimal
  Transport. Pooling to one vector would discard the internal temporal texture
  that lets Axis 3 tell natural variation from copy-paste.
- **Two OT backends, selectable via config.**
  `exact` — Wasserstein-1 via the transportation LP; non-differentiable; the
  evaluation metric and episodic RL reward.
  `sinkhorn` — entropic-regularized OT; differentiable surrogate for the
  lower-variance gradients of Section 6. The numpy recurrence ports directly to
  a torch autograd graph.
- **Matrix Profile + Foote novelty.** The Matrix Profile surfaces motifs and
  discords in latent space (proposal §4.1). Boundary *discovery* uses a Foote
  checkerboard-novelty curve over the self-similarity matrix, which is robust to
  recurring sections where the MP arc-curve washes out (a form like `ABACA`
  produces long recurrence arcs that hide boundary valleys).
- **The variation band `(floor, tau)`** is calibrated to cosine-ground
  Wasserstein on normalized embeddings, where copy-paste repeats measure ~0.002,
  natural variation ~0.01–0.05, and off-theme drift >0.1. `tau` is meant to be
  re-fit against human-annotated repeats (§4.2) and re-calibrated during RL
  (§6.2); it lives in the config for exactly that reason.

## Two run modes from one config

The same generator + extractor + metric stack serves two jobs, selected by
`RunConfig.mode`:

```python
from formadherence import RunConfig, build_runner

# Mode A -- metrics only (benchmarking, no RL). Generate a song, measure it.
ev = build_runner(RunConfig(mode="evaluation"), extractor)
result = ev.evaluate(song)              # -> EvaluationResult
print(result.overall, result.distinctness, result.similarity, result.variation)

# Mode B -- RL reward (shaped, penalized, curriculum-ready).
rl = build_runner(RunConfig(mode="rl_reward"), extractor)
out = rl.evaluate(song)                 # -> RewardOutput
print(out.scalar, out.axes, out.penalties)
```

Both branches call `score_form_adherence` with the **same** `FormAdherenceConfig`
(the config pins them together in `__post_init__`), so a song's evaluation score
and the metric portion of its RL reward agree by construction -- distinctness and
similarity are byte-identical across the two modes. Evaluation mode adds no
shaping and no anti-hacking penalties; it just measures. Use it to build the AI
test corpus and correlate against human MOS (proposal §5, §7). Switch to
`rl_reward` only when you start fine-tuning.

Configs serialize, so a whole run is one file:

```python
RunConfig(mode="evaluation").to_json("eval.json")
cfg = RunConfig.from_json("eval.json")
```

## Reference

- `README_integration.md` -- the full YuE → embeddings → reward → RL pipeline.



The current band and temperature defaults are calibrated on **synthetic**
embeddings with a controlled distance scale. Before use on real audio you must
re-fit `tau`, `variation_floor`, and `similarity_temperature` against
human-annotated repeats from SALAMI / RWC-Pop / Harmonix — this is Phase 1 of
the proposal timeline. The synthetic generator (`formadherence.synthetic`) is a
testing tool, not a stand-in for that calibration.

## Tests

```bash
PYTHONPATH=. python tests/test_metric.py
```

The suite checks that the metric rewards correct form and specifically punishes
each named failure mode: copy-paste (variation collapses, similarity does not),
verse==chorus (distinctness collapses), drift (similarity collapses), and that
the geometric mean blocks axis-trading.

## Layout

```
formadherence/
  config.py        FormAdherenceConfig — every tunable, serializable
  segmentation.py  Matrix Profile, Foote novelty, boundary handling
  ot.py            exact Wasserstein (LP) and Sinkhorn
  metric.py        three axes + hierarchical aggregation
  synthetic.py     controllable test-song generator
tests/
  test_metric.py   behavioural tests (10/10 passing)
```

## Status and next steps

Implemented and verified: the full metric pipeline, both OT backends, both
boundary modes, and the diagnostic breakdown.

Not yet done (deliberately, and next up):
1. Real embedding extractors (CLAP semantic + EnCodec acoustic).
2. Calibration on SALAMI / RWC-Pop / Harmonix.
3. Perceptual validation (Pearson/Spearman vs human MOS).
4. Wiring `as_reward_vector()` into PPO/GRPO (YuE) and DDPO/DPOK (DiffRhythm,
   ACE-Step) for Part II.
