"""Music generators behind one interface.

`MusicGenerator.generate(target_form, lyrics, ...)` returns a GeneratedSong: the
audio plus, for RL-capable backends, the sampled token ids and log-probs.

  * YuEGenerator          -- in-process YuE (needed for RL). Hooks 1-2 + A to wire.
  * CommandLineGenerator  -- shells out to YuE's infer.py (evaluation corpus only).

Fixes in v0.2
-------------
1. LYRICS PER LETTER. The old code built prompts with {l: lyrics for l in
   set(form)} -- every letter got the SAME lyrics, so the prompt itself asked
   for verse == chorus. Now `lyrics` can be:
     * dict  letter -> text            (same text at every repeat; chorus-like)
     * dict  letter -> [text1, text2]  (one text per occurrence; verse-like)
     * plain text with blank-line-separated paragraphs: one paragraph per
       DISTINCT letter, in first-appearance order (error if too few)
     * already-tagged text ("[verse]\n...") -- passed through, and its tag
       pattern is checked against target_form.
2. YuE generates only --run_n_segments lyric sections (the README example uses
   2). The CLI template now gets {n_segments} = len(target_form); without it an
   ABACA prompt produces a 2-section song and the metric scores nonsense.
3. CLI: genre and lyrics go to SEPARATE files ({genre_file}, {lyrics_file}) as
   YuE's infer.py expects; each call writes into its own fresh sub-folder (no
   picking up a stale file); finds wav/mp3/flac recursively, preferring a
   "mix" file; returns mono audio; passes {seed}.
"""

from __future__ import annotations

import re
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import numpy as np


@dataclass
class GeneratedSong:
    audio: np.ndarray                  # mono float32 waveform
    sample_rate: int
    target_form: str
    # Optional, present only for RL-capable backends:
    token_ids: np.ndarray | None = None        # (L,) sampled stage-1 token ids
    token_logprobs: np.ndarray | None = None   # (L,) behaviour log-prob per token
    meta: dict = field(default_factory=dict)


class MusicGenerator(ABC):
    @abstractmethod
    def generate(self, target_form: str, lyrics, genre: str = "",
                 seed: int | None = None) -> GeneratedSong:
        ...


# ---- prompt construction -------------------------------------------------

DEFAULT_ROLES = ["verse", "chorus", "bridge", "prechorus", "outro",
                 "intro", "interlude", "refrain"]


def form_to_structure_tags(target_form: str, role_order=None) -> list[str]:
    """First new letter -> verse, second -> chorus, then bridge, ...; repeats
    reuse their letter's tag. Check the tag names your YuE version expects."""
    roles = role_order or DEFAULT_ROLES
    mapping: dict[str, str] = {}
    tags = []
    for letter in target_form:
        if letter not in mapping:
            k = len(mapping)
            mapping[letter] = roles[k] if k < len(roles) else f"section{k}"
        tags.append(mapping[letter])
    return tags


def _pattern(seq) -> str:
    """Repetition pattern: ['verse','chorus','verse'] -> 'ABA'."""
    m: dict = {}
    return "".join(m.setdefault(x, chr(ord("A") + len(m))) for x in seq)


def parse_tagged_lyrics(text: str) -> list[str]:
    return [t.strip().lower() for t in re.findall(r"^\s*\[([^\]]+)\]", text, flags=re.M)]


def lyrics_by_occurrence(target_form: str, lyrics) -> list[str]:
    """Resolve `lyrics` (see module docstring) to one text per section."""
    letters = list(target_form)
    distinct = list(dict.fromkeys(letters))
    if isinstance(lyrics, dict):
        missing = [l for l in distinct if l not in lyrics]
        if missing:
            raise ValueError(f"lyrics dict has no text for letters {missing}")
        seen: dict[str, int] = {}
        out = []
        for l in letters:
            v = lyrics[l]
            k = seen.get(l, 0)
            seen[l] = k + 1
            if isinstance(v, (list, tuple)):
                out.append(v[min(k, len(v) - 1)])
            else:
                out.append(v)
        return [str(x).strip() for x in out]
    paras = [p.strip() for p in re.split(r"\n\s*\n", str(lyrics)) if p.strip()]
    if len(distinct) > 1 and len(paras) < len(distinct):
        raise ValueError(
            f"form {target_form!r} has {len(distinct)} distinct letters but the "
            f"lyrics have {len(paras)} paragraph(s). Give one blank-line-separated "
            "paragraph per distinct letter, or a dict letter -> text. (The old "
            "code silently gave every letter the same lyrics.)")
    by_letter = {l: paras[i] if paras else "" for i, l in enumerate(distinct)}
    return [by_letter[l] for l in letters]


def build_structured_lyrics(target_form: str, lyrics, role_order=None) -> str:
    """YuE lyrics prompt: one tagged block per section of the target form."""
    if isinstance(lyrics, str) and parse_tagged_lyrics(lyrics):
        tags = parse_tagged_lyrics(lyrics)
        if _pattern(tags) != _pattern(target_form):
            warnings.warn(f"tagged lyrics pattern {_pattern(tags)} ({tags}) does not "
                          f"match target form {target_form!r}")
        return lyrics.strip()
    tags = form_to_structure_tags(target_form, role_order)
    texts = lyrics_by_occurrence(target_form, lyrics)
    return "\n\n".join(f"[{t}]\n{x}" for t, x in zip(tags, texts))


# ---- YuE wrapper ---------------------------------------------------------

class YuEGenerator(MusicGenerator):
    """In-process YuE (multimodal-art-projection/YuE). Needed for RL because
    on-policy training must sample through the same model it updates.

    Hooks to wire, following your cloned infer.py:
      hook 1  load stage-1, stage-2 and the codec       (_lazy_load)
      hook 2  two-stage generation, returning audio +    (generate)
              stage-1 token ids and per-token log-probs
      hook A  score_tokens(): log-probs of given token ids under a model, with
              or without gradients. The RL loop calls it on the policy (with
              grads) and on a frozen reference copy (without) -- that covers
              the old hooks A and B.
    RL updates stage 1 only. Stage-2 / codec sampling still adds randomness to
    the reward; fix their seeds per rollout to reduce reward noise.
    Pass n_segments = len(target_form) (YuE's run_n_segments) when generating.
    """

    def __init__(self, yue_repo_path: str, stage1_model: str, stage2_model: str,
                 device: str = "cuda", max_new_tokens: int = 3000):
        import sys
        sys.path.insert(0, yue_repo_path)
        import torch
        self.torch = torch
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.stage1_model_path = stage1_model
        self.stage2_model_path = stage2_model
        self._loaded = False

    def _lazy_load(self):
        if self._loaded:
            return
        raise NotImplementedError(
            "hook 1: load YuE stage-1/stage-2 models and codec here, following "
            "your cloned infer.py")

    def generate(self, target_form: str, lyrics, genre: str = "",
                 seed: int | None = None) -> GeneratedSong:
        self._lazy_load()
        if seed is not None:
            self.torch.manual_seed(seed)
        prompt = self._format_prompt(target_form, lyrics, genre)  # noqa: F841
        raise NotImplementedError(
            "hook 2: run YuE two-stage generation with run_n_segments = "
            "len(target_form); return GeneratedSong(audio, sr, target_form, "
            "token_ids, token_logprobs) from the stage-1 sampler")

    def score_tokens(self, song: GeneratedSong, model=None, with_grad: bool = True):
        """hook A: per-token log-probs of song.token_ids under `model` (default:
        the policy's stage 1), conditioned on the same prompt. Return a 1-D
        torch tensor; keep the graph when with_grad=True."""
        raise NotImplementedError("hook A: teacher-forced log-probs of song.token_ids")

    def _format_prompt(self, target_form: str, lyrics, genre: str) -> str:
        return f"{genre}\n\n{build_structured_lyrics(target_form, lyrics)}".strip()


# ---- CLI fallback (evaluation only) -------------------------------------

class CommandLineGenerator(MusicGenerator):
    """Run YuE's infer.py as a subprocess and read the audio it writes.
    Evaluation corpus only (no token log-probs -> not usable for RL).

    infer_cmd_template placeholders: {genre_file} {lyrics_file} {out_dir}
    {seed} {n_segments} (and legacy {prompt_file} = genre + lyrics in one file).
    Example (check flag names against your infer.py):

      cd external/YuE/inference && python infer.py --cuda_idx 0 \\
        --stage1_model m-a-p/YuE-s1-7B-anneal-en-cot \\
        --stage2_model m-a-p/YuE-s2-1B-general \\
        --genre_txt {genre_file} --lyrics_txt {lyrics_file} \\
        --run_n_segments {n_segments} --stage2_batch_size 4 \\
        --output_dir {out_dir} --max_new_tokens 3000 --seed {seed}
    """

    AUDIO_EXTS = (".wav", ".mp3", ".flac")

    def __init__(self, infer_cmd_template: str, out_dir: str, read_wav=None,
                 role_order=None):
        self.tpl = infer_cmd_template
        self.out_dir = out_dir
        self.read_wav = read_wav
        self.role_order = role_order

    def _read(self, path):
        if self.read_wav is not None:
            return self.read_wav(path)
        from formadherence.audio_io import read_audio
        return read_audio(path)

    def generate(self, target_form: str, lyrics, genre: str = "",
                 seed: int | None = None) -> GeneratedSong:
        import glob
        import os
        import subprocess
        import uuid
        from .embeddings import _to_mono

        seed = 0 if seed is None else int(seed)
        run_dir = os.path.join(self.out_dir, f"{target_form}_{seed}_{uuid.uuid4().hex[:6]}")
        os.makedirs(run_dir, exist_ok=True)
        structured = build_structured_lyrics(target_form, lyrics, self.role_order)
        files = {"genre_file": os.path.join(run_dir, "genre.txt"),
                 "lyrics_file": os.path.join(run_dir, "lyrics.txt"),
                 "prompt_file": os.path.join(run_dir, "prompt.txt")}
        with open(files["genre_file"], "w") as f:
            f.write(genre.strip() + "\n")
        with open(files["lyrics_file"], "w") as f:
            f.write(structured + "\n")
        with open(files["prompt_file"], "w") as f:
            f.write(f"{genre}\n\n{structured}\n")
        cmd = self.tpl.format(out_dir=os.path.abspath(run_dir), seed=seed,
                              n_segments=len(target_form),
                              **{k: os.path.abspath(v) for k, v in files.items()})
        subprocess.run(cmd, shell=True, check=True)

        found = [p for p in glob.glob(os.path.join(run_dir, "**", "*"), recursive=True)
                 if p.lower().endswith(self.AUDIO_EXTS)]
        if not found:
            raise RuntimeError(f"YuE produced no audio under {run_dir}")
        mixes = [p for p in found if "mix" in os.path.basename(p).lower()]
        path = max(mixes or found, key=os.path.getmtime)
        audio, sr = self._read(path)
        return GeneratedSong(audio=_to_mono(audio), sample_rate=int(sr),
                             target_form=target_form,
                             meta={"audio_path": path, "run_dir": run_dir,
                                   "seed": seed, "lyrics": structured, "genre": genre})
