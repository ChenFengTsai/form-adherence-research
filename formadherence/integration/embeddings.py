"""Audio -> (T, d) per-frame embeddings: the extraction stage that the metric
assumes has already happened.

The proposal calls for two complementary foundation models:
  * CLAP    -- semantic / high-level content (what the section 'is')
  * EnCodec -- fine-grained acoustic detail (how it actually sounds)

This module provides a common `EmbeddingExtractor` interface plus concrete CLAP
and EnCodec extractors, and a `ConcatExtractor` that fuses them (the proposal's
intent). Everything imports its heavy deps lazily so the metric package stays
usable without a GPU; you only pay for torch/transformers when you actually
extract.

IMPORTANT: run this on your GPU box. It is written against the real APIs
(transformers ClapModel, the `encodec` package) but was not executed in the
authoring environment, which has no network or weights. Treat the first run as a
smoke test: check the printed shapes match what score_form_adherence expects.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np


class EmbeddingExtractor(ABC):
    """Turns a mono waveform into a (T, d) sequence of per-frame embeddings.

    T is the number of frames (a few per second, not per audio-sample) and d is
    the embedding dimension. Frames SHOULD be L2-normalized so the metric's
    cosine ground metric is well behaved -- the base class does this for you if
    you call `_finalize`.
    """

    #: frames produced per second of audio -- set by each concrete extractor.
    frame_rate: float = 0.0

    @abstractmethod
    def extract(self, waveform: np.ndarray, sample_rate: int) -> np.ndarray:
        ...

    @staticmethod
    def _finalize(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2:
            raise ValueError(f"expected (T, d), got shape {x.shape}")
        n = np.linalg.norm(x, axis=1, keepdims=True)
        return x / (n + 1e-12)


class ClapExtractor(EmbeddingExtractor):
    """Windowed CLAP audio embeddings -- semantic content.

    CLAP produces one embedding per clip, so to get a *sequence* we slide a
    window (default 1s, 0.5s hop) and embed each window. That window rate is the
    frame rate the metric segments over.
    """

    def __init__(self, model_name: str = "laion/clap-htsat-unfused",
                 window_s: float = 1.0, hop_s: float = 0.5, device: str = "cuda"):
        import torch
        from transformers import ClapModel, ClapProcessor
        self.torch = torch
        self.device = device
        self.model = ClapModel.from_pretrained(model_name, use_safetensors=True).to(device).eval()
        self.processor = ClapProcessor.from_pretrained(model_name)
        self.window_s = window_s
        self.hop_s = hop_s
        self.frame_rate = 1.0 / hop_s

    def extract(self, waveform: np.ndarray, sample_rate: int) -> np.ndarray:
        torch = self.torch
        target_sr = 48000  # CLAP expects 48 kHz
        wav = _to_mono(waveform)
        if sample_rate != target_sr:
            wav = _resample(wav, sample_rate, target_sr)
            sample_rate = target_sr

        win = int(self.window_s * sample_rate)
        hop = int(self.hop_s * sample_rate)
        if len(wav) < win:
            wav = np.pad(wav, (0, win - len(wav)))
        starts = list(range(0, len(wav) - win + 1, hop)) or [0]

        embs = []
        with torch.no_grad():
            for s in starts:
                chunk = wav[s:s + win]
                inputs = self.processor(audio=chunk, sampling_rate=sample_rate,
                                        return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                e = self.model.get_audio_features(**inputs)
                if not isinstance(e, torch.Tensor):
                    e = e.pooler_output
                    if e is None:
                        raise TypeError(
                            "get_audio_features returned a non-tensor without .audio_embeds; "
                            "inspect the object to find the embedding field")
                embs.append(e.squeeze(0).float().cpu().numpy())
        return self._finalize(np.stack(embs))


class EncodecExtractor(EmbeddingExtractor):
    """EnCodec continuous latent frames -- fine-grained acoustic detail.

    We take the *continuous* encoder output (pre-quantization latent) rather than
    the discrete codes, giving a dense (T, d) sequence at EnCodec's native frame
    rate (~75 Hz at 24 kHz). That is a lot of frames for a 3-minute song; the
    metric's max_frames_per_segment cap keeps the OT tractable, and you can also
    average-pool in time here to lower the frame rate if you prefer.
    """

    def __init__(self, bandwidth: float = 6.0, device: str = "cuda",
                 pool: int = 1):
        import torch
        from encodec import EncodecModel
        self.torch = torch
        self.device = device
        self.pool = max(1, pool)
        self.model = EncodecModel.encodec_model_24khz().to(device).eval()
        self.model.set_target_bandwidth(bandwidth)
        self.target_sr = 24000
        self.frame_rate = 75.0 / self.pool

    def extract(self, waveform: np.ndarray, sample_rate: int) -> np.ndarray:
        torch = self.torch
        wav = _to_mono(waveform)
        if sample_rate != self.target_sr:
            wav = _resample(wav, sample_rate, self.target_sr)
        x = torch.tensor(wav, dtype=torch.float32,
                         device=self.device)[None, None, :]
        with torch.no_grad():
            latent = self.model.encoder(x)          # (1, d, T)
        seq = latent.squeeze(0).transpose(0, 1).float().cpu().numpy()  # (T, d)
        if self.pool > 1:
            seq = _avg_pool_time(seq, self.pool)
        return self._finalize(seq)


class ConcatExtractor(EmbeddingExtractor):
    """Fuse several extractors by resampling each to a common frame grid and
    concatenating along the feature dimension. This realizes the proposal's
    'CLAP for semantic + EnCodec for acoustic' design as one (T, d) array.
    """

    def __init__(self, extractors: list[EmbeddingExtractor],
                 target_frame_rate: float | None = None):
        if not extractors:
            raise ValueError("need at least one extractor")
        self.extractors = extractors
        self.frame_rate = target_frame_rate or min(e.frame_rate
                                                    for e in extractors)

    def extract(self, waveform: np.ndarray, sample_rate: int) -> np.ndarray:
        seqs = [e.extract(waveform, sample_rate) for e in self.extractors]
        # Resample each (T_i, d_i) to a common T on the target frame grid.
        dur = len(_to_mono(waveform)) / sample_rate
        T = max(1, int(round(dur * self.frame_rate)))
        aligned = [_resample_sequence(s, T) for s in seqs]
        fused = np.concatenate(aligned, axis=1)
        return self._finalize(fused)


# ---- small dependency-free helpers --------------------------------------

def _to_mono(w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=np.float32)
    if w.ndim == 2:
        w = w.mean(axis=0 if w.shape[0] < w.shape[1] else 1)
    return w


def _resample(w: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    if sr_from == sr_to:
        return w
    from math import gcd
    g = gcd(sr_from, sr_to)
    up, down = sr_to // g, sr_from // g
    try:
        from scipy.signal import resample_poly
        return resample_poly(w, up, down).astype(np.float32)
    except Exception:
        n = int(round(len(w) * sr_to / sr_from))
        xp = np.linspace(0, 1, len(w))
        x = np.linspace(0, 1, n)
        return np.interp(x, xp, w).astype(np.float32)


def _avg_pool_time(seq: np.ndarray, k: int) -> np.ndarray:
    T = (len(seq) // k) * k
    if T == 0:
        return seq
    return seq[:T].reshape(T // k, k, seq.shape[1]).mean(axis=1)


def _resample_sequence(seq: np.ndarray, T_out: int) -> np.ndarray:
    """Linear-interpolate a (T_in, d) sequence to (T_out, d) along time."""
    T_in, d = seq.shape
    if T_in == T_out:
        return seq
    xp = np.linspace(0.0, 1.0, T_in)
    x = np.linspace(0.0, 1.0, T_out)
    out = np.empty((T_out, d))
    for c in range(d):
        out[:, c] = np.interp(x, xp, seq[:, c])
    return out
