"""GRPO training loop for fine-tuning YuE toward form adherence.

GRPO: sample a GROUP of songs per prompt; advantage = (reward - group mean) /
group std, applied to every stage-1 token of that song. No critic network.

What is complete here: grouping, advantages, the clipped GRPO loss with a KL
penalty (grpo_token_loss, the old hook C), optimizer stepping (old hook D,
when you pass an optimizer), curriculum, logging stats incl. axis_mean for the
reward-hacking audit.

What you still wire (model-specific), all on YuEGenerator:
  hook 1/2   loading + generation (returns token_ids, token_logprobs)
  hook A     score_tokens(song, model, with_grad) -> per-token log-probs.
             Called on the policy (with grads) and on a frozen reference
             model (no grads) -> covers the old hooks A and B.

Fixes in v0.2: train_step returns axis_mean (+ penalties, fallback rate,
how many groups had zero advantage); curriculum entries take metric overrides
(a bare float still means tau); prompt sampling no longer crashes when there
are fewer prompts than grad_accum_prompts; the audit smooths over a window
instead of comparing two single noisy steps.

COST: YuE needs ~150 s per 30 s of audio on an H800. G=8 full songs per prompt
is easily 30+ minutes. Small groups, checkpoint often, start with short forms
that contain a repeat (ABA, ABAB).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from .generators import MusicGenerator, GeneratedSong
from .reward import FormAdherenceReward


@dataclass
class GRPOConfig:
    group_size: int = 8
    lr: float = 1e-6
    kl_coeff: float = 0.05
    grad_accum_prompts: int = 4
    max_steps: int = 1000
    clip_ratio: float = 0.2
    # (step_threshold, reward axis_weights, metric overrides dict). In order.
    # A float in the 3rd slot is treated as tau (legacy absolute mode).
    curriculum: list = field(default_factory=lambda: [
        (0,   (1.0, 1.0, 0.2), {}),   # structure + cohesion first
        (300, (1.0, 1.0, 0.6), {}),   # phase in the copy-paste guard
        (600, (1.0, 1.0, 1.0), {}),   # full weight
    ])


@dataclass
class PromptSpec:
    target_form: str
    lyrics: object          # str (one paragraph per distinct letter) or dict
    genre: str = ""


# ---- the loss (framework-agnostic: torch tensors or numpy arrays) -------

def grpo_token_loss(cur_logp, old_logp, ref_logp, advantage: float,
                    clip_ratio: float = 0.2, kl_coeff: float = 0.05, mask=None):
    """Clipped GRPO objective for ONE sampled song.

    cur_logp : log-probs of the sampled tokens under the current policy (grads)
    old_logp : behaviour log-probs recorded at sampling time
    ref_logp : log-probs under the frozen reference policy
    advantage: this song's group-normalized advantage (scalar)
    Returns the mean per-token loss  -min(r*A, clip(r)*A) + kl_coeff * KL,
    with the k3 KL estimator  exp(ref-cur) - (ref-cur) - 1  (>= 0).
    """
    try:
        import torch
        is_torch = isinstance(cur_logp, torch.Tensor)
    except ImportError:
        is_torch = False
    xp = torch if is_torch else np
    old = old_logp.detach() if is_torch else np.asarray(old_logp, float)
    ref = ref_logp.detach() if is_torch else np.asarray(ref_logp, float)
    cur = cur_logp if is_torch else np.asarray(cur_logp, float)

    ratio = xp.exp(cur - old)
    clipped = (xp.clamp if is_torch else np.clip)(ratio, 1 - clip_ratio, 1 + clip_ratio)
    pg = -xp.minimum(ratio * advantage, clipped * advantage)
    d = ref - cur
    kl = xp.exp(d) - d - 1
    tok = pg + kl_coeff * kl
    if mask is not None:
        m = mask.float() if is_torch else np.asarray(mask, float)
        return (tok * m).sum() / (m.sum() + 1e-8)
    return tok.mean()


class GRPOTrainer:
    def __init__(self, policy: MusicGenerator, reward_fn: FormAdherenceReward,
                 prompts: list[PromptSpec], config: GRPOConfig | None = None,
                 optimizer=None, ref_model=None):
        self.policy = policy
        self.reward_fn = reward_fn
        self.prompts = prompts
        self.cfg = config or GRPOConfig()
        self.optimizer = optimizer        # e.g. torch.optim.AdamW(stage1.parameters())
        self.ref_model = ref_model        # frozen copy of stage 1 for the KL term
        self._rng = np.random.default_rng(0)

    # -- curriculum -------------------------------------------------------

    def _apply_curriculum(self, step: int) -> None:
        active = None
        for thr, weights, extra in self.cfg.curriculum:
            if step >= thr:
                active = (weights, extra)
        if active is None:
            return
        weights, extra = active
        if isinstance(extra, (int, float)):
            self.reward_fn.set_phase(weights, tau=float(extra))
        else:
            self.reward_fn.set_phase(weights, **(extra or {}))

    # -- rollouts ---------------------------------------------------------

    def rollout_group(self, prompt: PromptSpec):
        songs: list[GeneratedSong] = []
        rewards = np.zeros(self.cfg.group_size)
        for g in range(self.cfg.group_size):
            song = self.policy.generate(prompt.target_form, prompt.lyrics, prompt.genre,
                                        seed=int(self._rng.integers(1 << 30)))
            out = self.reward_fn(song)
            rewards[g] = out.scalar
            song.meta["reward_out"] = out
            songs.append(song)
        return songs, rewards

    @staticmethod
    def group_advantages(rewards: np.ndarray) -> np.ndarray:
        mu, sd = rewards.mean(), rewards.std()
        if sd < 1e-8:
            return np.zeros_like(rewards)   # whole group tied: no learning signal
        return (rewards - mu) / sd

    # -- policy update ----------------------------------------------------

    def _update_on_song(self, song: GeneratedSong, advantage: float):
        if song.token_logprobs is None:
            raise RuntimeError("policy.generate returned no token_logprobs -- use an "
                               "RL-capable generator (YuEGenerator, hook 2), not the CLI")
        if advantage == 0.0:
            return None
        cur = self.policy.score_tokens(song, model=None, with_grad=True)       # hook A
        ref = self.policy.score_tokens(song, model=self.ref_model, with_grad=False) \
            if self.ref_model is not None else song.token_logprobs
        loss = grpo_token_loss(cur, song.token_logprobs, ref, float(advantage),
                               self.cfg.clip_ratio, self.cfg.kl_coeff)
        if hasattr(loss, "backward"):
            (loss / self.cfg.grad_accum_prompts).backward()
        return float(loss.detach() if hasattr(loss, "detach") else loss)

    def train_step(self, step: int) -> dict:
        self._apply_curriculum(step)
        n = self.cfg.grad_accum_prompts
        batch = self._rng.choice(len(self.prompts), size=n,
                                 replace=len(self.prompts) < n)
        rmean, rmax, axes, pens, fallbacks, losses = [], [], [], [], [], []
        zero_adv_groups = 0
        for pidx in batch:
            songs, rewards = self.rollout_group(self.prompts[int(pidx)])
            adv = self.group_advantages(rewards)
            zero_adv_groups += int(np.all(adv == 0))
            rmean.append(float(rewards.mean()))
            rmax.append(float(rewards.max()))
            for song, a in zip(songs, adv):
                out = song.meta["reward_out"]
                axes.append(out.axes)
                pens.append(out.penalties)
                fallbacks.append(bool(out.meta.get("segmentation_fallback")))
                l = self._update_on_song(song, a)
                if l is not None:
                    losses.append(l)
        if self.optimizer is not None:                                        # hook D
            self.optimizer.step()
            self.optimizer.zero_grad()
        return {
            "step": step,
            "reward_mean": float(np.mean(rmean)),
            "reward_max": float(np.max(rmax)),
            "axis_mean": np.mean(np.stack(axes), axis=0).tolist(),
            "penalty_mean": {k: float(np.mean([p[k] for p in pens])) for k in pens[0]},
            "segmentation_fallback_rate": float(np.mean(fallbacks)),
            "zero_advantage_groups": zero_adv_groups,
            "loss_mean": float(np.mean(losses)) if losses else None,
        }

    def train(self):
        return [self.train_step(s) for s in range(self.cfg.max_steps)]


# ---- reward-hacking audit -------------------------------------------------

def audit_axis_growth(history_axes: list, window: int = 10) -> dict:
    """Which axis improved fastest (mean of the last `window` steps vs the
    first `window`)? If variation races ahead of the others, inspect the audio
    for reward hacking before trusting the gain."""
    A = np.asarray(history_axes, dtype=float)
    if len(A) < 2:
        return {}
    w = max(1, min(window, len(A) // 2))
    growth = A[-w:].mean(0) - A[:w].mean(0)
    names = ["distinctness", "similarity", "variation"]
    fastest = int(np.argmax(growth))
    return {
        "growth_per_axis": dict(zip(names, np.round(growth, 4).tolist())),
        "fastest_axis": names[fastest],
        "variation_racing": bool(fastest == 2 and growth[2] > 2 * max(growth[0], growth[1], 1e-6)),
    }
