"""Audio -> (T, d) per-frame embeddings.

  * CLAP    -- semantic content (windowed clip embeddings)
  * EnCodec -- acoustic detail (continuous encoder latent)
  * Chroma+MFCC -- classical structure-analysis features (harmony + timbre)
  * ConcatExtractor -- puts several on one frame grid and concatenates

Heavy deps (torch, transformers, encodec, librosa) are imported lazily.

Fixes in v0.2
-------------
1. TIME ALIGNMENT. The old ConcatExtractor stretched every sequence end-to-end
   onto the target grid. But a CLAP frame describes the window CENTRE: with a
   4 s window / 2 s hop the first frame is at t=2 s and the last at dur-2 s, so
   stretching put frames up to 2 s (4 frames at 2 fps) away from where they
   belong near the start/end -- right where boundaries are compared. Every
   extractor now reports the time of each frame (extract_with_times) and
   ConcatExtractor interpolates by time onto frame centres (k + 0.5) / fps.
   At the default 1 s window the shift was < 0.25 s; at 4-8 s it mattered.
2. CLAP windows are embedded in batches (much faster), with a per-window
   fallback if the processor rejects a batch.
3. ConcatExtractor records block_dims, so the balance check knows where CLAP
   ends and EnCodec starts (CLAP width depends on the transformers version).

Things to know (not bugs)
-------------------------
* Each block is L2-normalized per frame before concatenation, so CLAP and
  EnCodec get equal weight in the cosine distance regardless of width.
* CLAP's processor pads clips shorter than 10 s by REPEATING them
  ("repeatpad"): a 1 s window is tiled 10x before CLAP sees it. CLAP was
  trained on ~10 s clips, so short windows are out of distribution. Windows
  longer than 10 s are randomly cropped (truncation="rand_trunc") -> not
  deterministic; keep window_s <= 10.
* For a music-specific model try model_name="laion/larger_clap_music".
* EnCodec: we use the continuous encoder output, BEFORE quantization, so the
  `bandwidth` setting has no effect on these features.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np


class EmbeddingExtractor(ABC):
    """Mono waveform -> (T, d) per-frame embeddings (L2-normalized rows)."""

    #: nominal frames per second -- set by each concrete extractor.
    frame_rate: float = 0.0

    @abstractmethod
    def extract(self, waveform: np.ndarray, sample_rate: int) -> np.ndarray:
        ...

    def extract_with_times(self, waveform, sample_rate):
        """(embeddings, frame_centre_times_in_seconds). Default: frames assumed
        evenly spread over the audio. Concrete extractors override this."""
        seq = self.extract(waveform, sample_rate)
        dur = len(_to_mono(waveform)) / sample_rate
        return seq, (np.arange(len(seq)) + 0.5) * dur / max(len(seq), 1)

    @staticmethod
    def _finalize(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2:
            raise ValueError(f"expected (T, d), got shape {x.shape}")
        n = np.linalg.norm(x, axis=1, keepdims=True)
        return x / (n + 1e-12)


class ClapExtractor(EmbeddingExtractor):
    """Windowed CLAP audio embeddings (one embedding per sliding window)."""

    def __init__(self, model_name: str = "laion/clap-htsat-unfused",
                 window_s: float = 1.0, hop_s: float = 0.5, device: str = "cuda",
                 batch_size: int = 16):
        import torch
        from transformers import ClapModel, ClapProcessor
        if window_s > 10:
            raise ValueError("CLAP randomly crops inputs longer than 10 s; "
                             "use window_s <= 10 for deterministic embeddings")
        self.torch = torch
        self.device = device
        self.model = ClapModel.from_pretrained(model_name, use_safetensors=True).to(device).eval()
        self.processor = ClapProcessor.from_pretrained(model_name)
        self.window_s = window_s
        self.hop_s = hop_s
        self.batch_size = max(1, batch_size)
        self.frame_rate = 1.0 / hop_s

    def _embed(self, chunks, sample_rate):
        torch = self.torch
        inputs = self.processor(audio=chunks if len(chunks) > 1 else chunks[0],
                                sampling_rate=sample_rate, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        e = self.model.get_audio_features(**inputs)
        if not isinstance(e, torch.Tensor):
            # newer transformers return a model-output object. pooler_output
            # first = same choice as the original patched code, so embeddings
            # stay in the space your earlier runs used.
            out = getattr(e, "pooler_output", None)
            if out is None:
                out = getattr(e, "audio_embeds", None)
            e = out
            if e is None:
                raise TypeError("get_audio_features returned no tensor; inspect "
                                "the output object for the embedding field")
        return e.reshape(len(chunks), -1).float().cpu().numpy()

    def extract_with_times(self, waveform, sample_rate):
        target_sr = 48000  # CLAP expects 48 kHz
        wav = _to_mono(waveform)
        if sample_rate != target_sr:
            wav = _resample(wav, sample_rate, target_sr)
            sample_rate = target_sr
        win = int(self.window_s * sample_rate)
        hop = int(self.hop_s * sample_rate)
        n_orig = len(wav)
        if len(wav) < win:
            wav = np.pad(wav, (0, win - len(wav)))
        starts = list(range(0, len(wav) - win + 1, hop)) or [0]

        embs = []
        with self.torch.no_grad():
            for b in range(0, len(starts), self.batch_size):
                chunks = [wav[s:s + win] for s in starts[b:b + self.batch_size]]
                try:
                    embs.append(self._embed(chunks, sample_rate))
                except Exception:
                    if len(chunks) == 1:
                        raise
                    embs.append(np.concatenate([self._embed([c], sample_rate)
                                                for c in chunks]))
        times = np.minimum((np.asarray(starts) + win / 2) / sample_rate,
                           n_orig / sample_rate)
        return self._finalize(np.concatenate(embs)), times

    def extract(self, waveform, sample_rate):
        return self.extract_with_times(waveform, sample_rate)[0]


class EncodecExtractor(EmbeddingExtractor):
    """EnCodec continuous encoder latent (pre-quantization), 75 Hz at 24 kHz,
    optionally average-pooled in time."""

    def __init__(self, bandwidth: float = 6.0, device: str = "cuda", pool: int = 1):
        import torch
        from encodec import EncodecModel
        self.torch = torch
        self.device = device
        self.pool = max(1, pool)
        self.model = EncodecModel.encodec_model_24khz().to(device).eval()
        self.model.set_target_bandwidth(bandwidth)   # no effect on encoder latent
        self.target_sr = 24000
        self.native_rate = 75.0
        self.frame_rate = self.native_rate / self.pool

    def extract_with_times(self, waveform, sample_rate):
        torch = self.torch
        wav = _to_mono(waveform)
        if sample_rate != self.target_sr:
            wav = _resample(wav, sample_rate, self.target_sr)
        x = torch.tensor(wav, dtype=torch.float32, device=self.device)[None, None, :]
        with torch.no_grad():
            latent = self.model.encoder(x)                                # (1, d, T)
        seq = latent.squeeze(0).transpose(0, 1).float().cpu().numpy()     # (T, d)
        if self.pool > 1:
            seq = _avg_pool_time(seq, self.pool)
        times = (np.arange(len(seq)) * self.pool + self.pool / 2) / self.native_rate
        return self._finalize(seq), times

    def extract(self, waveform, sample_rate):
        return self.extract_with_times(waveform, sample_rate)[0]


class ChromaMfccExtractor(EmbeddingExtractor):
    """Chroma (12) + MFCC (n_mfcc) [+ MFCC deltas], standardized per song per
    dimension, then L2-normalized per frame."""

    def __init__(self, frame_rate: float = 2.0, n_mfcc: int = 20,
                 use_delta: bool = True, target_sr: int = 22050):
        self.frame_rate = frame_rate
        self.n_mfcc = n_mfcc
        self.use_delta = use_delta
        self.target_sr = target_sr

    def extract_with_times(self, waveform, sample_rate):
        import librosa
        wav = _to_mono(waveform)
        if sample_rate != self.target_sr:
            wav = _resample(wav, sample_rate, self.target_sr)
            sample_rate = self.target_sr
        # chroma_cqt needs hop to be a multiple of 2**n_octaves (128)
        hop = max(1, int(round(sample_rate / self.frame_rate / 128))) * 128
        chroma = librosa.feature.chroma_cqt(y=wav, sr=sample_rate, hop_length=hop)
        mfcc = librosa.feature.mfcc(y=wav, sr=sample_rate, n_mfcc=self.n_mfcc,
                                    hop_length=hop)
        feats = [chroma, mfcc] + ([librosa.feature.delta(mfcc)] if self.use_delta else [])
        T = min(f.shape[1] for f in feats)
        X = np.concatenate([f[:, :T] for f in feats], axis=0).T
        X = (X - X.mean(0, keepdims=True)) / (X.std(0, keepdims=True) + 1e-8)
        times = np.arange(T) * hop / sample_rate      # librosa frames are centred
        return self._finalize(X), times

    def extract(self, waveform, sample_rate):
        return self.extract_with_times(waveform, sample_rate)[0]


class ConcatExtractor(EmbeddingExtractor):
    """Put several extractors on one frame grid (frame k centred at
    (k + 0.5) / frame_rate) by TIME-based interpolation, then concatenate."""

    def __init__(self, extractors: list[EmbeddingExtractor],
                 target_frame_rate: float | None = None):
        if not extractors:
            raise ValueError("need at least one extractor")
        self.extractors = extractors
        self.frame_rate = target_frame_rate or min(e.frame_rate for e in extractors)
        self.block_dims: list[int] | None = None

    def extract_with_times(self, waveform, sample_rate):
        dur = len(_to_mono(waveform)) / sample_rate
        T = max(1, int(round(dur * self.frame_rate)))
        grid = (np.arange(T) + 0.5) / self.frame_rate
        aligned = []
        for e in self.extractors:
            seq, times = e.extract_with_times(waveform, sample_rate)
            aligned.append(_resample_to_times(seq, times, grid))
        self.block_dims = [a.shape[1] for a in aligned]
        # re-normalize each block after interpolation so blocks stay equal-weight
        aligned = [self._finalize(a) for a in aligned]
        return self._finalize(np.concatenate(aligned, axis=1)), grid

    def extract(self, waveform, sample_rate):
        return self.extract_with_times(waveform, sample_rate)[0]


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
    g = gcd(int(sr_from), int(sr_to))
    up, down = int(sr_to) // g, int(sr_from) // g
    try:
        from scipy.signal import resample_poly
        return resample_poly(w, up, down).astype(np.float32)
    except Exception:
        n = int(round(len(w) * sr_to / sr_from))
        return np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(w)), w).astype(np.float32)


def _avg_pool_time(seq: np.ndarray, k: int) -> np.ndarray:
    T = (len(seq) // k) * k
    if T == 0:
        return seq
    return seq[:T].reshape(T // k, k, seq.shape[1]).mean(axis=1)


def _resample_to_times(seq: np.ndarray, src_times, tgt_times) -> np.ndarray:
    """Interpolate a (T_in, d) sequence sampled at src_times onto tgt_times.
    Outside the source range the nearest frame is held."""
    seq = np.asarray(seq, dtype=np.float64)
    src = np.asarray(src_times, dtype=np.float64)
    tgt = np.asarray(tgt_times, dtype=np.float64)
    if len(seq) == 1:
        return np.repeat(seq, len(tgt), axis=0)
    out = np.empty((len(tgt), seq.shape[1]))
    for c in range(seq.shape[1]):
        out[:, c] = np.interp(tgt, src, seq[:, c])
    return out


def _resample_sequence(seq: np.ndarray, T_out: int) -> np.ndarray:
    """OLD end-to-end stretch, kept only for comparison/tests."""
    T_in, d = seq.shape
    if T_in == T_out:
        return seq
    xp, x = np.linspace(0.0, 1.0, T_in), np.linspace(0.0, 1.0, T_out)
    return np.stack([np.interp(x, xp, seq[:, c]) for c in range(d)], axis=1)
