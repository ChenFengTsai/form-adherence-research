#!/usr/bin/env python
"""RL mode: fine-tune YuE toward form adherence with GRPO, using the calibrated
metric as reward.

STATUS: skeleton. It depends on hooks in formadherence/integration/ that are not
wired yet (YuEGenerator hooks 1-2 in generators.py, GRPOTrainer hooks A-D in
rl_loop.py). This script is ready for them.

Prerequisites, in order:
  1. scripts/extract_embeddings.py smoke test passes (shape + block balance).
  2. scripts/calibrate.py --mode rl_reward --out configs/rl_grpo.json
  3. YuE and GRPO hooks wired.

    python scripts/train_rl.py --config configs/rl_grpo.json \
        --prompts configs/prompts.json \
        --stage1 m-a-p/YuE-s1-7B-anneal-en-cot --stage2 m-a-p/YuE-s2-1B-general

Reward notes:
  * Use the relative score mode (default). Hard per-song AUC is a poor reward:
    undefined for AB, only 5 possible values for ABA, and it rewards copy-paste.
  * Start with forms that HAVE a repeat (ABA, ABAB) so similarity/variation
    carry signal; AB alone only exercises distinctness.
  * COST: one rollout = one full song (~150 s gen on an H800). group_size 8 x
    1000 steps = 8,000 songs ~ 14 GPU-days of generation alone. Budget for it,
    or generate shorter songs for early experiments.
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from formadherence.audio_io import add_extractor_args, spec_from_args, build_extractor


def load_prompts(path):
    """Prompt manifest: [{"target_form","lyrics","genre"}, ...]."""
    from formadherence.integration.rl_loop import PromptSpec
    with open(path) as f:
        raw = json.load(f)
    return [PromptSpec(target_form=p["target_form"], lyrics=p["lyrics"],
                       genre=p.get("genre", "")) for p in raw]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True,
                    help="RunConfig JSON with mode=rl_reward (calibrate.py --mode rl_reward)")
    ap.add_argument("--prompts", default="configs/prompts.json")
    ap.add_argument("--yue_repo", default="external/YuE")
    ap.add_argument("--stage1", required=True)
    ap.add_argument("--stage2", required=True)
    ap.add_argument("--group_size", type=int, default=8)
    ap.add_argument("--max_steps", type=int, default=1000)
    ap.add_argument("--audit_every", type=int, default=50)
    ap.add_argument("--log", default="runs/rl_log.jsonl")
    add_extractor_args(ap)
    args = ap.parse_args()

    from formadherence import RunConfig
    from formadherence.integration import YuEGenerator, FormAdherenceReward
    from formadherence.integration.rl_loop import GRPOTrainer, GRPOConfig, audit_axis_growth

    cfg = RunConfig.from_json(args.config)
    if cfg.mode != "rl_reward":
        raise SystemExit("train_rl needs a mode=rl_reward config: "
                         "python scripts/calibrate.py --mode rl_reward ...")
    print(f"[cfg] metric score_mode={cfg.metric.score_mode} ot_backend={cfg.metric.ot_backend}")

    spec = spec_from_args(args)
    print(f"[extractor] {spec.describe()} -- must match what calibration used")
    reward_fn = FormAdherenceReward(build_extractor(spec), cfg.reward)
    policy = YuEGenerator(yue_repo_path=args.yue_repo,
                          stage1_model=args.stage1, stage2_model=args.stage2)
    trainer = GRPOTrainer(policy=policy, reward_fn=reward_fn,
                          prompts=load_prompts(args.prompts),
                          config=GRPOConfig(group_size=args.group_size,
                                            max_steps=args.max_steps))

    os.makedirs(os.path.dirname(args.log) or ".", exist_ok=True)
    axis_history = []
    with open(args.log, "a") as logf:
        for step in range(args.max_steps):
            stats = trainer.train_step(step)
            print(f"[step {step:4d}] reward_mean={stats['reward_mean']:.4f} "
                  f"reward_max={stats['reward_max']:.4f}")
            # Reward-hacking audit: needs the mean (distinctness, similarity,
            # variation) of this step's rollouts. Make GRPOTrainer.train_step
            # return it as stats["axis_mean"].
            if "axis_mean" in stats:
                axis_history.append(np.asarray(stats["axis_mean"], dtype=float))
            elif step == 0:
                print("[warn] train_step returned no 'axis_mean'; the "
                      "reward-hacking audit is disabled until it does")
            logf.write(json.dumps({"step": step, **{k: (v.tolist() if hasattr(v, "tolist") else v)
                                                    for k, v in stats.items()}}) + "\n")
            logf.flush()
            if step and step % args.audit_every == 0 and len(axis_history) > 1:
                print("[audit]", audit_axis_growth(axis_history))


if __name__ == "__main__":
    main()
