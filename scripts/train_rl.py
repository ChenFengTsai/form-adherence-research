#!/usr/bin/env python
"""RL mode: fine-tune YuE toward form adherence with GRPO, using the calibrated
Form-Adherence metric as reward.

Prerequisites (in order):
  1. scripts/extract_embeddings.py runs and the SMOKE TEST shape looks right.
  2. scripts/calibrate.py has produced a calibrated config (configs/*.json).
  3. YuEGenerator hooks 1 & 2 are wired (integration/generators.py).
  4. GRPOTrainer hooks A-D are wired (integration/rl_loop.py).

    python scripts/train_rl.py \
        --config configs/rl_grpo_short.json \
        --yue_repo external/YuE \
        --stage1 m-a-p/YuE-s1-7B-anneal-en-cot \
        --stage2 m-a-p/YuE-s2-1B-general

COST WARNING: each rollout is a full song generation (~150s/30s on H800). Start
with SHORT forms (AB, ABA) and a small group size. See docs/README_integration.md.
"""

import argparse
import json
import numpy as np


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
                    help="RunConfig JSON with mode=rl_reward (calibrated metric)")
    ap.add_argument("--prompts", default="configs/prompts.json")
    ap.add_argument("--yue_repo", default="external/YuE")
    ap.add_argument("--stage1", required=True)
    ap.add_argument("--stage2", required=True)
    ap.add_argument("--group_size", type=int, default=8)
    ap.add_argument("--max_steps", type=int, default=1000)
    args = ap.parse_args()

    from formadherence import RunConfig
    from formadherence.integration import (
        YuEGenerator, ClapExtractor, EncodecExtractor, ConcatExtractor,
        FormAdherenceReward)
    from formadherence.integration.rl_loop import GRPOTrainer, GRPOConfig, \
        audit_axis_growth

    cfg = RunConfig.from_json(args.config)
    assert cfg.mode == "rl_reward", "train_rl needs a mode=rl_reward config"

    extractor = ConcatExtractor(
        [ClapExtractor(window_s=1.0, hop_s=0.5),
         EncodecExtractor(bandwidth=6.0, pool=8)],
        target_frame_rate=2.0)

    reward_fn = FormAdherenceReward(extractor, cfg.reward)

    policy = YuEGenerator(
        yue_repo_path=args.yue_repo,
        stage1_model=args.stage1,
        stage2_model=args.stage2,
    )

    prompts = load_prompts(args.prompts)
    trainer = GRPOTrainer(
        policy=policy, reward_fn=reward_fn, prompts=prompts,
        config=GRPOConfig(group_size=args.group_size, max_steps=args.max_steps))

    axis_history = []
    for step in range(args.max_steps):
        stats = trainer.train_step(step)
        print(f"[step {step:4d}] reward_mean={stats['reward_mean']:.4f} "
              f"reward_max={stats['reward_max']:.4f}")
        # Log mean axis vector for the reward-hacking audit (fill from your
        # rollout logging; placeholder shows the call site).
        # axis_history.append(mean_axis_vector_this_step)
        if step and step % 50 == 0 and len(axis_history) > 1:
            print("[audit]", audit_axis_growth(axis_history))


if __name__ == "__main__":
    main()
