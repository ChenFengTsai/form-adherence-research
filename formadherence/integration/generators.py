"""Music generators behind one interface, so the reward function and the RL
loop never depend on a specific model's internals.

`MusicGenerator.generate(target_form, ...)` returns a GeneratedSong: the audio
plus everything downstream needs (sample rate, and -- crucially for RL -- the
generated token ids and per-step log-probs when the backend exposes them).

Included:
  * YuEGenerator   -- wraps the official multimodal-art-projection/YuE inference.
  * CommandLineGenerator -- shells out to YuE's infer.py if you'd rather not
                            import its code; audio-only (no token log-probs).

Why the token fields matter: YuE is autoregressive over audio tokens, so PPO/
GRPO need the sampled token ids and their log-probs to compute the policy-
gradient objective. A pure audio-in/audio-out wrapper is fine for *evaluation*
but not for on-policy RL -- you must generate through the same policy object you
update. YuEGenerator is structured so you can plug the RL loop into its sampling
step; the CLI wrapper deliberately cannot, and says so.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import numpy as np


@dataclass
class GeneratedSong:
    audio: np.ndarray                  # mono float32 waveform
    sample_rate: int
    target_form: str
    # Optional, present only for RL-capable backends:
    token_ids: np.ndarray | None = None        # (L,) sampled audio-token ids
    token_logprobs: np.ndarray | None = None   # (L,) log-prob of each sample
    meta: dict = field(default_factory=dict)


class MusicGenerator(ABC):
    @abstractmethod
    def generate(self, target_form: str, lyrics: str, genre: str = "",
                 seed: int | None = None) -> GeneratedSong:
        ...


# ---- prompt construction -------------------------------------------------

def form_to_structure_tags(target_form: str) -> list[str]:
    """Map a letter form to YuE-style section tags.

    YuE conditions on lyrics annotated with [verse]/[chorus]/[bridge]/... section
    labels. We assign the first-seen letter to 'verse', the second to 'chorus',
    then bridge/outro/etc., and reuse the same tag for repeats of a letter so the
    *intended* repetition is expressed in the prompt. The metric later checks
    whether the model actually honoured it.
    """
    role_order = ["verse", "chorus", "bridge", "pre-chorus", "outro",
                  "intro", "interlude", "refrain"]
    mapping: dict[str, str] = {}
    tags = []
    for letter in target_form:
        if letter not in mapping:
            idx = len(mapping)
            mapping[letter] = role_order[idx] if idx < len(role_order) \
                else f"section-{idx}"
        tags.append(mapping[letter])
    return tags


def build_structured_lyrics(target_form: str, lyrics_by_letter: dict[str, str]
                            ) -> str:
    """Compose a YuE lyrics prompt with one tagged block per section in the
    target form. Repeats of a letter reuse that letter's lyric block, encoding
    the requested same/different structure into the prompt itself."""
    tags = form_to_structure_tags(target_form)
    letters = list(target_form)
    blocks = []
    for letter, tag in zip(letters, tags):
        body = lyrics_by_letter.get(letter, f"[{letter} section lyrics]")
        blocks.append(f"[{tag}]\n{body.strip()}")
    return "\n\n".join(blocks)


# ---- YuE wrapper ---------------------------------------------------------

class YuEGenerator(MusicGenerator):
    """Wraps the official YuE inference pipeline.

    This targets multimodal-art-projection/YuE. Its inference lives in a script
    (infer.py) rather than a pip package, so you point this class at your cloned
    repo and it imports the pieces it needs. The exact symbol names in YuE's code
    may drift; the two TODO hooks below are the only places you touch when they
    do -- everything else in the metric/reward/RL stack is insulated from it.
    """

    def __init__(self, yue_repo_path: str, stage1_model: str, stage2_model: str,
                 device: str = "cuda", max_new_tokens: int = 3000):
        import sys
        sys.path.insert(0, yue_repo_path)
        import torch
        self.torch = torch
        self.device = device
        self.max_new_tokens = max_new_tokens

        # TODO(hook 1): load YuE's stage-1 (semantic) and stage-2 (acoustic)
        # models and codec here, following the current infer.py. Kept abstract
        # because the loader signature has changed across YuE releases.
        #   from models import ...   /   self.stage1 = load(...)
        self.stage1_model_path = stage1_model
        self.stage2_model_path = stage2_model
        self._loaded = False

    def _lazy_load(self):
        if self._loaded:
            return
        # TODO(hook 1 continued): populate self.stage1, self.stage2, self.codec
        raise NotImplementedError(
            "Wire YuE model loading here following your cloned infer.py "
            "(hook 1). See README_integration.md."
        )

    def generate(self, target_form: str, lyrics: str, genre: str = "",
                 seed: int | None = None) -> GeneratedSong:
        self._lazy_load()
        torch = self.torch
        if seed is not None:
            torch.manual_seed(seed)

        prompt = self._format_prompt(target_form, lyrics, genre)

        # TODO(hook 2): run YuE's two-stage generation. For EVALUATION you only
        # need the decoded audio. For RL you additionally need, from stage 1's
        # autoregressive sampling loop, the sampled token ids and their per-step
        # log-probs -- return them so PPO/GRPO can use them. Structure:
        #   tokens, logprobs = self._sample_stage1(prompt)   # keep grads/logps
        #   audio = self._decode(tokens)                     # stage 2 + codec
        raise NotImplementedError(
            "Wire YuE two-stage generation here (hook 2). Return audio, and for "
            "RL also token_ids + token_logprobs from the stage-1 sampler."
        )

    def _format_prompt(self, target_form: str, lyrics: str, genre: str) -> str:
        # YuE takes a genre/description line plus tagged lyrics. If `lyrics`
        # already contains [verse]/[chorus] tags we pass it through; otherwise we
        # wrap it into one block per section of the target form.
        if "[" in lyrics and "]" in lyrics:
            structured = lyrics
        else:
            structured = build_structured_lyrics(
                target_form, {l: lyrics for l in set(target_form)})
        return f"{genre}\n\n{structured}".strip()


# ---- CLI fallback (evaluation only) -------------------------------------

class CommandLineGenerator(MusicGenerator):
    """Shell out to YuE's infer.py and read the wav it writes. Simple and robust
    for building the *evaluation* corpus (proposal §5). Cannot expose token log-
    probs, so it is NOT usable for on-policy RL -- generate() sets those to None.
    """

    def __init__(self, infer_cmd_template: str, out_dir: str,
                 read_wav):
        # infer_cmd_template: a shell command with {prompt_file} and {out_dir}
        #   placeholders. read_wav: a callable(path)->(waveform, sr), e.g. wrap
        #   soundfile.read, kept injectable to avoid a hard dep here.
        self.tpl = infer_cmd_template
        self.out_dir = out_dir
        self.read_wav = read_wav

    def generate(self, target_form: str, lyrics: str, genre: str = "",
                 seed: int | None = None) -> GeneratedSong:
        import os, subprocess, tempfile, glob
        os.makedirs(self.out_dir, exist_ok=True)
        structured = build_structured_lyrics(
            target_form, {l: lyrics for l in set(target_form)})
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write(f"{genre}\n\n{structured}")
            prompt_file = f.name
        cmd = self.tpl.format(prompt_file=prompt_file, out_dir=self.out_dir)
        subprocess.run(cmd, shell=True, check=True)
        wavs = sorted(glob.glob(os.path.join(self.out_dir, "*.wav")),
                      key=os.path.getmtime)
        if not wavs:
            raise RuntimeError("YuE produced no wav in out_dir")
        audio, sr = self.read_wav(wavs[-1])
        return GeneratedSong(audio=np.asarray(audio, np.float32), sample_rate=sr,
                             target_form=target_form,
                             meta={"wav_path": wavs[-1]})
