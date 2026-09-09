"""A GRPO-style RL training loop skeleton for fine-tuning YuE toward form
adherence, using FormAdherenceReward as the reward.

Why GRPO (Group Relative Policy Optimization) rather than PPO here:
  * No separate value/critic network -- you sample a GROUP of songs per prompt
    and use the group's mean reward as the baseline. For a reward as expensive
    as "generate a 3-minute song then extract embeddings", not training a critic
    is a big practical win.
  * Advantage = (reward - group_mean) / group_std, applied to every token in
    that sample. This is the episodic-reward, sparse-credit setting the proposal
    names -- the whole song gets one reward, spread over its tokens.

This is a SKELETON: the three hooks marked TODO are where YuE's actual sampling
and log-prob computation plug in. Everything else -- grouping, advantage,
the loss, curriculum -- is complete and framework-agnostic (shown with torch).

CRITICAL COST NOTE. YuE generates ~30s of audio in ~150s on an H800. A group of
G=8 samples for one prompt is ~20 minutes of pure generation at 30s clips, more
for full songs. Plan accordingly: small groups, gradient accumulation over
prompts, aggressive checkpointing, and start on SHORT target forms (e.g. "AB",
"ABA") before scaling to "ABACABA". Do not design a loop that assumes cheap
rollouts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from .generators import MusicGenerator, GeneratedSong
from .reward import FormAdherenceReward


@dataclass
class GRPOConfig:
    group_size: int = 8               # samples per prompt (baseline population)
    lr: float = 1e-6
    kl_coeff: float = 0.05            # penalty vs the frozen reference policy
    grad_accum_prompts: int = 4       # prompts per optimizer step
    max_steps: int = 1000
    clip_ratio: float = 0.2
    # Curriculum: (step_threshold, axis_weights, tau). Applied in order.
    curriculum: list[tuple[int, tuple[float, float, float], float]] = field(
        default_factory=lambda: [
            (0,   (1.0, 1.0, 0.2), 0.12),   # first: get structure + cohesion
            (300, (1.0, 1.0, 0.6), 0.12),   # then: phase in variation
            (600, (1.0, 1.0, 1.0), 0.10),   # finally: full weight, tighter band
        ])


@dataclass
class PromptSpec:
    target_form: str
    lyrics: str
    genre: str = ""


class GRPOTrainer:
    def __init__(self, policy: MusicGenerator, reward_fn: FormAdherenceReward,
                 prompts: list[PromptSpec], config: GRPOConfig | None = None):
        self.policy = policy
        self.reward_fn = reward_fn
        self.prompts = prompts
        self.cfg = config or GRPOConfig()
        self._rng = np.random.default_rng(0)

    # -- curriculum -------------------------------------------------------

    def _apply_curriculum(self, step: int) -> None:
        active = None
        for thr, weights, tau in self.cfg.curriculum:
            if step >= thr:
                active = (weights, tau)
        if active:
            self.reward_fn.set_phase(active[0], tau=active[1])

    # -- one group rollout for one prompt ---------------------------------

    def rollout_group(self, prompt: PromptSpec):
        """Generate G songs for one prompt and score them. Returns per-sample
        rewards and the token ids / log-probs needed for the policy-gradient
        update."""
        songs: list[GeneratedSong] = []
        rewards = np.zeros(self.cfg.group_size)
        for g in range(self.cfg.group_size):
            song = self.policy.generate(prompt.target_form, prompt.lyrics,
                                        prompt.genre,
                                        seed=int(self._rng.integers(1 << 30)))
            out = self.reward_fn(song)
            rewards[g] = out.scalar
            song.meta["reward_out"] = out
            songs.append(song)
        return songs, rewards

    @staticmethod
    def group_advantages(rewards: np.ndarray) -> np.ndarray:
        """GRPO advantage: standardize rewards within the group."""
        mu = rewards.mean()
        sd = rewards.std()
        if sd < 1e-8:
            return np.zeros_like(rewards)
        return (rewards - mu) / sd

    # -- the training step (TODO hooks are YuE/torch specifics) -----------

    def train_step(self, step: int):
        self._apply_curriculum(step)
        batch = self._rng.choice(len(self.prompts),
                                 size=self.cfg.grad_accum_prompts, replace=False)

        step_stats = {"reward_mean": [], "reward_max": []}
        for pidx in batch:
            prompt = self.prompts[int(pidx)]
            songs, rewards = self.rollout_group(prompt)
            adv = self.group_advantages(rewards)
            step_stats["reward_mean"].append(float(rewards.mean()))
            step_stats["reward_max"].append(float(rewards.max()))

            for song, a in zip(songs, adv):
                if song.token_logprobs is None:
                    raise RuntimeError(
                        "policy.generate returned no token_logprobs -- use an "
                        "RL-capable generator (YuEGenerator with hook 2 wired), "
                        "not the CLI wrapper."
                    )
                # TODO(hook A): recompute current-policy log-probs of song's
                # tokens WITH gradients (song.token_logprobs were the behaviour
                # log-probs at sampling time; for the ratio you need current
                # ones). ratio = exp(cur_logp - old_logp).
                #
                # TODO(hook B): KL to frozen reference policy on these tokens.
                #
                # TODO(hook C): loss = -min(ratio*A, clip(ratio,1±eps)*A)
                #               + kl_coeff*KL ; backward(); accumulate.
                pass

        # TODO(hook D): optimizer.step(); optimizer.zero_grad()
        return {
            "step": step,
            "reward_mean": float(np.mean(step_stats["reward_mean"])),
            "reward_max": float(np.max(step_stats["reward_max"])),
        }

    def train(self):
        history = []
        for step in range(self.cfg.max_steps):
            history.append(self.train_step(step))
        return history


# ---- reward-hacking audit hook ------------------------------------------

def audit_axis_growth(history_axes: list[np.ndarray]) -> dict:
    """Given per-step mean axis vectors, report which axis is improving fastest.
    Per the eval plan: if one axis (esp. variation) races ahead of the others,
    inspect for reward hacking before trusting the gain."""
    A = np.asarray(history_axes)
    if len(A) < 2:
        return {}
    growth = A[-1] - A[0]
    names = ["distinctness", "similarity", "variation"]
    fastest = int(np.argmax(growth))
    return {
        "growth_per_axis": dict(zip(names, growth.round(4).tolist())),
        "fastest_axis": names[fastest],
        "variation_racing": bool(fastest == 2 and growth[2] > 2 * max(
            growth[0], growth[1], 1e-6)),
    }
