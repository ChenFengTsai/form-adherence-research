# Integration: YuE → embeddings → reward → RL

This guide wires the Form-Adherence metric to **YuE** and stands up a
reinforcement-learning loop. It is the bridge between Part I (the metric) and
Part II (optimization).

## Why YuE (for the RL-first path)

YuE (`multimodal-art-projection/YuE`, Apache-2.0 code) is **autoregressive over
audio tokens**. That matters because your proposal (§6.2) names sparse,
long-horizon credit assignment as a core risk and notes token-based models admit
finer credit assignment than diffusion. On an autoregressive policy you get
standard, well-tooled PPO/GRPO; the per-token log-probs you need for the policy
gradient fall out of the sampling loop. DiffRhythm/ACE-Step (diffusion) force you
onto DDPO/DPOK and trajectory-level estimators — do those **second**, once the
reward is proven on the easier substrate.

## The pipeline

```
PromptSpec(target_form="ABACA", lyrics, genre)
      │
      ▼  YuEGenerator.generate()            formadherence.integration.generators
GeneratedSong(audio, sample_rate,
              token_ids, token_logprobs)     ← token_* only from RL-capable path
      │
      ▼  ConcatExtractor([CLAP, EnCodec])   formadherence.integration.embeddings
embeddings (T, d)
      │
      ▼  score_form_adherence()             formadherence.metric
FormAdherenceResult (D, S, V, overall)
      │
      ▼  FormAdherenceReward()              formadherence.integration.reward
RewardOutput(scalar, axes, penalties)
      │
      ▼  GRPOTrainer                        formadherence.integration.rl_loop
policy update
```

## Step 1 — Extract embeddings (run on GPU)

```python
from formadherence.integration import ClapExtractor, EncodecExtractor, ConcatExtractor

extractor = ConcatExtractor([
    ClapExtractor(window_s=1.0, hop_s=0.5),   # semantic
    EncodecExtractor(bandwidth=6.0, pool=8),  # acoustic, pooled to lower T
], target_frame_rate=2.0)                     # common grid, ~2 frames/sec
```

First run is a smoke test: print `extractor.extract(wav, sr).shape` and check it
is `(T, d)` with `T ≈ duration_seconds * target_frame_rate`.

## Step 2 — Build the reward

```python
from formadherence.integration import FormAdherenceReward, RewardConfig
from formadherence import FormAdherenceConfig

reward_fn = FormAdherenceReward(
    extractor,
    RewardConfig(
        metric=FormAdherenceConfig(ot_backend="exact"),  # exact for eval/reward
        axis_weights=(1.0, 1.0, 0.2),                    # start low on variation
        gate_variation_by_noise=True,                    # anti-hacking guard
    ),
)

out = reward_fn(generated_song)
print(out.scalar, out.axes, out.penalties)
```

The reward already includes two §6.2 anti-hacking defences: a silence/clipping
penalty and a spectral-flatness (white-noise) penalty, plus it gates variation
credit by how noisy the audio is — so a policy can't earn variation reward by
injecting noise. Add OA/SSM ensemble terms here if you want the full ensemble.

## Step 3 — Wire YuE (the only model-specific code)

Open `integration/generators.py`, class `YuEGenerator`. There are exactly two
hooks:

- **hook 1** (`_lazy_load`): load YuE's stage-1/stage-2 models + codec, copying
  from your cloned `infer.py`. Kept abstract because YuE's loader signature has
  changed across releases.
- **hook 2** (`generate`): run the two-stage generation. For *evaluation* return
  just the decoded `audio`. For *RL* also return, from the stage-1 sampler, the
  sampled `token_ids` and their per-step `token_logprobs`.

For building the **evaluation corpus** (proposal §5) you don't need RL, so you
can skip the hooks entirely and use `CommandLineGenerator`, which shells out to
`infer.py` and reads the wav. It cannot expose log-probs, so it is eval-only.

## Step 4 — Train (GRPO)

```python
from formadherence.integration.rl_loop import GRPOTrainer, GRPOConfig, PromptSpec

prompts = [
    PromptSpec("AB",   lyrics="...", genre="pop"),      # start SHORT
    PromptSpec("ABA",  lyrics="...", genre="pop"),
    # scale up to ABACA... only once short forms train stably
]
trainer = GRPOTrainer(policy=yue, reward_fn=reward_fn, prompts=prompts,
                      config=GRPOConfig(group_size=8))
trainer.train()
```

GRPO uses the group's mean reward as the baseline (no critic network), which
matters when each rollout is a full-song generation. The loop's grouping,
advantage standardization, and curriculum are complete; four TODO hooks (A–D)
mark where YuE's current-policy log-prob recomputation, the reference-KL, the
clipped loss, and the optimizer step go — all framework-standard once hook 2
gives you tokens + log-probs.

### Curriculum

`GRPOConfig.curriculum` anneals the axis weights and `tau` across training:
reward structure and cohesion first, phase in variation later. This is the
reward-shaping-over-axes mitigation from §6.2, made concrete.

### Reward-hacking audit

Log the mean axis vector each step and call `audit_axis_growth(history)`. If
variation races ahead of the other axes (`variation_racing=True`), inspect
before trusting the gain — that is the tell for the noise-injection exploit, per
your evaluation plan.

## The cost reality — read before designing your run

YuE generates ~30s of audio in ~150s on an H800 (~360s on a 4090). A group of 8
for one prompt is ~20 min of generation at 30s clips. Consequences:

1. **Start with 30s clips and 2–3 section forms** (`AB`, `ABA`), not 3-minute
   `ABACABA`. Prove the reward moves structure before scaling length.
2. **Small groups + gradient accumulation** over prompts, not big batches.
3. **Cache embeddings** keyed by audio hash — never re-extract the same rollout.
4. **Checkpoint aggressively**; these runs are long and interruptions costly.
5. Consider **`ot_backend="sinkhorn"`** during training for speed, keeping
   `exact` for periodic held-out evaluation.

## What is proven vs what you must still do

Proven in-repo (14/14 tests): the metric, both OT backends, both boundary modes,
the reward wrapper, the noise-injection guard, curriculum control, advantage
math, prompt construction, and the audit.

You must still, on your GPU box:
1. Wire YuE hooks 1 & 2, and RL hooks A–D (all marked, all standard).
2. **Calibrate** `tau`, `variation_floor`, `similarity_temperature` against
   human-annotated repeats from SALAMI/RWC-Pop/Harmonix — the current defaults
   are set on synthetic data and WILL be wrong for real CLAP/EnCodec distances.
   This is Phase 1 and is not optional.
3. Validate embedding extraction shapes on real audio (smoke test above).
