# Form-Adherence Research

Measuring and optimizing conditional structural-form adherence in full-song music
generation. Two contributions in one repo:

1. **A three-axis Form-Adherence metric** — given a target form (a letter
   sequence like `ABACA`) and a generated song, score how well the song realizes
   that form along Cross-Letter Distinctness, Same-Letter Similarity, and
   Within-Letter Variation.
2. **The same metric as an RL reward** — fine-tune a music model (YuE) toward
   controllable structure, with anti-reward-hacking guards and a curriculum.

The metric core depends only on numpy + scipy, so it imports and its tests run on
any machine. Audio extraction (CLAP/EnCodec) and generation (YuE) need a GPU.

## Two modes, one config

```python
from formadherence import RunConfig, build_runner
ev = build_runner(RunConfig(mode="evaluation"), extractor)   # benchmark: measure only
rl = build_runner(RunConfig(mode="rl_reward"),  extractor)   # train: shaped reward
```

Both share the same metric core, so an evaluation score and the metric portion of
an RL reward agree by construction. See `docs/README.md` for the metric,
`docs/README_integration.md` for the full pipeline, `docs/PROJECT_LAYOUT.md` for
the directory map.

---

# Step-by-step: build this project from scratch

## Phase 0 — Machine & accounts
1. GPU access: 24 GB VRAM min (short forms); 80 GB (H800/A100) for full songs.
   Note generation cost: ~150 s per 30 s audio on H800, ~360 s on a 4090.
2. System deps: `git`, `git-lfs`, `conda` (or `uv`), CUDA ≥ 11.8.
3. HuggingFace account; accept access for YuE checkpoints
   (`m-a-p/YuE-s1-7B-anneal-en-cot`, `m-a-p/YuE-s2-1B-general`) and CLAP
   (`laion/clap-htsat-unfused`).

## Phase 1 — Project skeleton
4. Put this repo in place and init git:
   ```bash
   cd form-adherence-research && git init && git add . && git commit -m "init"
   ```
5. The scaffold (`configs/`, `scripts/`, `docs/`, `external/`, and the
   git-ignored `data/ cache/ checkpoints/ outputs/`) is already here.

## Phase 2 — Two environments (they have conflicting pins)
YuE wants Python 3.8 + flash-attn; CLAP/EnCodec want a newer stack. Keep them
apart and pass wavs between them via `outputs/wavs/`.

6. **Metric/extraction env:**
   ```bash
   conda create -n fa python=3.11 && conda activate fa
   pip install -e ".[extract,dev]"       # numpy, scipy, torch, transformers, encodec, soundfile, pytest
   python -m pytest -q                    # expect 20 passed
   ```
7. **YuE env:**
   ```bash
   conda create -n yue python=3.8 && conda activate yue
   conda install pytorch torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
   mkdir -p external && cd external
   git clone https://github.com/multimodal-art-projection/YuE.git
   cd YuE && pip install -r requirements.txt
   pip install flash-attn --no-build-isolation
   cd inference && git clone https://huggingface.co/m-a-p/xcodec_mini_infer
   ```
8. Smoke-test YuE with the CoT `infer.py` command from its README; confirm one
   wav lands in `outputs/`.

## Phase 3 — Prove the metric on real audio (no RL)
9. Extract embeddings and **check the shape** (the single most important check):
   ```bash
   conda activate fa
   python scripts/extract_embeddings.py --audio_dir outputs/wavs \
       --cache_dir cache/embeddings --frame_rate 2.0
   ```
   Confirm the printed `(T, d)` has `T ≈ duration × frame_rate`, and that
   sections you know differ don't look identical. If they do, fix the extractor
   before anything else — the metric can only be as good as its embeddings.
10. Score a song:
    ```bash
    python scripts/run_evaluation.py --wav outputs/wavs/<song>.wav --form ABACA
    # or, faster, from a cached embedding:
    python scripts/run_evaluation.py --emb cache/embeddings/<key>.npy --form ABACA
    ```

## Phase 4 — Calibration (NOT optional; this is the real research risk)
The shipped defaults are fit on synthetic data and **will be wrong** for real
CLAP/EnCodec distances.
11. Get SALAMI / RWC-Pop / Harmonix (each has its own access process). Put audio
    + annotations under `data/msa/`.
12. Extract embeddings for the annotated songs (step 9), then build a manifest
    aligning annotated boundaries (seconds → frame indices via `frame_rate`) and
    labels: `data/msa/manifest.json` (schema in `scripts/calibrate.py`).
13. Fit the band and temperature:
    ```bash
    python scripts/calibrate.py --manifest data/msa/manifest.json \
        --out configs/eval_baseline.json
    ```
14. Validate perceptually: gather human form-adherence ratings, correlate with
    metric scores (Pearson/Spearman). This is proposal §7.

## Phase 5 — RL (only after 1–4 are solid)
15. Wire **YuE hooks 1 & 2** in `formadherence/integration/generators.py`
    (model loading; return `token_ids` + `token_logprobs` from stage-1 sampling).
16. Wire **RL hooks A–D** in `formadherence/integration/rl_loop.py` (current-
    policy log-probs, reference-KL, clipped loss, optimizer step; reuse YuE's
    `finetune/` LoRA setup).
17. Make an `rl_reward` config that reuses your calibrated metric, then train —
    starting SHORT:
    ```bash
    python scripts/train_rl.py --config configs/rl_grpo_short.json \
        --yue_repo external/YuE \
        --stage1 m-a-p/YuE-s1-7B-anneal-en-cot \
        --stage2 m-a-p/YuE-s2-1B-general --group_size 4
    ```
    Watch `audit_axis_growth` for the variation axis racing ahead — that is the
    tell for the noise-injection exploit.

## The ordering is load-bearing
Do not jump to Phase 5. If the extractor (Phase 3) can't separate sections the
reward is noise; if calibration (Phase 4) isn't done, RL optimizes a
miscalibrated target and produces confident garbage. Phases 0–2 are ~a couple
days of setup; Phase 4 is where the actual research uncertainty lives.

## Tests
```bash
python -m pytest -q          # 20 tests: metric (10) + reward (4) + run modes (6)
```

## What's done vs what you write
Done and tested in-repo: the metric, both OT backends, both boundary modes, the
reward wrapper + anti-hacking guards, the two run modes, the GRPO scaffolding
(grouping/advantages/curriculum/audit), and runnable script stubs.

You write on your GPU box: YuE hooks 1–2, RL hooks A–D, the calibration manifest,
and the dataset wiring. All hooks are marked in-code.
