# Full project layout

How the Form-Adherence package sits alongside the cloned YuE repo on your GPU
machine. Three zones: **(1)** the YuE clone (external, unchanged), **(2)** this
package (what you have), **(3)** the glue + artifacts you add per experiment.

```
form-adherence-research/                  ← your project root (a git repo)
│
├── README.md                             ← project overview, points into the two packages
├── pyproject.toml                        ← installs `formadherence` (pip install -e .)
├── requirements.txt                      ← formadherence deps: numpy, scipy  (+ torch, transformers, encodec for extraction)
├── .gitignore                            ← ignore YuE/, checkpoints/, outputs/, *.wav, *.npy
│
├── external/
│   └── YuE/                              ← ⓵ git clone https://github.com/multimodal-art-projection/YuE
│       │                                    (Apache-2.0; do NOT edit — treat as read-only dependency)
│       ├── inference/
│       │   ├── infer.py                  ←   the entry point YuEGenerator / CommandLineGenerator call
│       │   ├── xcodec_mini_infer/        ←   git clone https://huggingface.co/m-a-p/xcodec_mini_infer  (the audio codec)
│       │   └── ...                        ←   models/, utils, tokenizer code (hook 1 imports from here)
│       ├── finetune/                     ←   YuE's LoRA finetune code — your RL updates target these model objects
│       ├── prompt_egs/                   ←   genre.txt / lyrics.txt examples (format your prompts match)
│       ├── evals/
│       ├── top_200_tags.json             ←   stable genre-tag vocabulary (use for genre strings)
│       └── requirements.txt              ←   YuE's own deps (python 3.8, torch+cuda, flash-attn)
│
├── formadherence/                        ← ⓶ THIS PACKAGE (the pip-installed library)
│   ├── __init__.py                       ←   public API: score_form_adherence, RunConfig, build_runner, ...
│   ├── config.py                         ←   FormAdherenceConfig  (the metric's tunables; shared by both modes)
│   ├── metric.py                         ←   three-axis metric + hierarchical aggregation
│   ├── ot.py                             ←   exact Wasserstein (LP) + Sinkhorn
│   ├── segmentation.py                   ←   Matrix Profile + Foote-novelty boundaries
│   ├── synthetic.py                      ←   synthetic-embedding generator (testing/calibration)
│   ├── run.py                            ←   RunConfig + EvaluationRunner / RewardRunner  (the two modes)
│   └── integration/
│       ├── __init__.py
│       ├── embeddings.py                 ←   CLAP + EnCodec extractors  (audio → (T,d))
│       ├── generators.py                 ←   YuEGenerator (hooks 1 & 2) + CommandLineGenerator
│       ├── reward.py                     ←   FormAdherenceReward (shaped reward + §6.2 guards)  [rl_reward mode]
│       └── rl_loop.py                    ←   GRPOTrainer skeleton (hooks A–D) + audit_axis_growth
│
├── tests/                                ←   part of ⓶
│   ├── test_metric.py                    ←   10/10
│   ├── test_reward.py                    ←   4/4
│   └── test_run_modes.py                 ←   6/6
│
├── docs/                                 ← ⓶ documentation
│   ├── README.md                         ←   metric overview + two-mode usage
│   ├── README_integration.md            ←   full YuE → embeddings → reward → RL pipeline
│   └── PROJECT_LAYOUT.md                 ←   this file
│
├── configs/                              ← ⓷ YOUR experiment configs (one JSON = one run)
│   ├── eval_baseline.json                ←   RunConfig(mode="evaluation")            → benchmarking
│   ├── rl_grpo_short.json                ←   RunConfig(mode="rl_reward") on AB/ABA    → early training
│   └── rl_grpo_full.json                 ←   RunConfig(mode="rl_reward") on ABACA...  → scaled training
│
├── scripts/                              ← ⓷ thin entry points you write (examples below)
│   ├── extract_embeddings.py            ←   audio dir → cached .npy embeddings
│   ├── run_evaluation.py                ←   generate + score a model (mode A); builds the AI test corpus
│   ├── calibrate.py                     ←   fit tau / floor / temperature on SALAMI/RWC-Pop/Harmonix
│   └── train_rl.py                      ←   wire YuEGenerator + FormAdherenceReward into GRPOTrainer (mode B)
│
├── data/                                 ← ⓷ (git-ignored) datasets & annotations
│   ├── msa/                              ←   SALAMI / RWC-Pop / Harmonix audio + boundary/label annotations
│   └── ai_corpus/                        ←   generated songs with known intended forms (proposal §5)
│
├── cache/                                ← ⓷ (git-ignored) embeddings keyed by audio hash — never re-extract
│   └── embeddings/*.npy
│
├── checkpoints/                          ← ⓷ (git-ignored) RL fine-tuned YuE checkpoints
│   └── grpo_step_*/
│
└── outputs/                              ← ⓷ (git-ignored) generated wavs, metric logs, audit curves
    ├── wavs/
    └── logs/
```

## Where the two packages meet

Only **two files** in `formadherence` import from the YuE clone, and only inside
methods (lazy), so the metric package stays importable without a GPU:

- `integration/generators.py` → `YuEGenerator` adds `external/YuE/inference` to
  `sys.path` and imports YuE's model-loading + sampling code.
  - **hook 1** (`_lazy_load`): load `stage1_model`, `stage2_model`, and the
    xcodec from `inference/xcodec_mini_infer`, following the current `infer.py`.
  - **hook 2** (`generate`): run the two-stage generation. Return `audio` for
    evaluation; also return `token_ids` + `token_logprobs` from the stage-1
    sampler for RL.
- `integration/rl_loop.py` → `GRPOTrainer` hooks A–D compute current-policy
  log-probs, reference-KL, the clipped loss, and the optimizer step against the
  YuE model objects (reuse `external/YuE/finetune/` LoRA setup).

Everything else — the metric, both OT backends, segmentation, the reward
shaping, the two run modes — never touches YuE and is fully tested standalone.

## The two modes map onto two scripts

- `scripts/run_evaluation.py` uses `RunConfig(mode="evaluation")` →
  `EvaluationRunner`. No RL, no token log-probs; works with `CommandLineGenerator`
  (shell out to `infer.py`). This is how you benchmark any model and build the
  AI test corpus.
- `scripts/train_rl.py` uses `RunConfig(mode="rl_reward")` → `RewardRunner` +
  `GRPOTrainer`. Needs `YuEGenerator` with hooks wired (token log-probs).

## Install sketch

```bash
# 1. project + metric package
git clone <your-repo> form-adherence-research && cd form-adherence-research
python -m venv .venv && source .venv/bin/activate
pip install -e .                       # installs `formadherence` (numpy, scipy)
pip install torch transformers encodec soundfile   # for extraction

# 2. YuE clone (its own heavier env; see YuE README — python 3.8, cuda, flash-attn)
mkdir -p external && cd external
git clone https://github.com/multimodal-art-projection/YuE.git
cd YuE/inference && git clone https://huggingface.co/m-a-p/xcodec_mini_infer
```

Note YuE pins **python 3.8** + flash-attn, while modern CLAP/EnCodec want a newer
stack. Two practical options: (a) run YuE generation and embedding extraction in
**separate conda envs**, passing wavs between them via `outputs/wavs/` (simplest,
and `CommandLineGenerator` is built for exactly this); or (b) reconcile versions
in one env if you need in-process RL with token log-probs. Start with (a) for
evaluation, move to (b) only when you begin RL.
```
