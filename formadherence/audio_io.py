"""Audio reading, the default extractor, and the embedding cache.

One place for what every script used to copy-paste.

CACHE FIX: the old cache file was <sha1(audio)[:16]>.npy -- the extractor
settings were not part of the name, so changing the CLAP window (or anything
else) and rebuilding into the same cache_dir silently returned the OLD
embeddings. Now it is <sha1(audio)[:16]>_<sha1(extractor spec)[:8]>.npy and a
<same>.json sidecar records the spec.

Old caches: pass reuse_legacy=True (scripts: --reuse_legacy_cache) to accept an
old <hash>.npy when the new name is missing. Only do that if those files were
made with the settings you are asking for now (the old default: CLAP 1 s / 0.5 s
hop + EnCodec 6 kbps pool 8 @ 2 fps). v0.2 also changed how CLAP/EnCodec frames
are time-aligned: at the 1 s default the difference is < 0.25 s (half a frame),
so reusing is acceptable; for 4-8 s CLAP windows, re-extract.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, asdict

import numpy as np

AUDIO_EXTS = (".wav", ".mp3", ".flac", ".m4a", ".opus", ".ogg")


def read_audio(path):
    """Read wav/mp3/... as (samples, sample_rate). soundfile first, librosa
    (ffmpeg/audioread) as fallback for formats libsndfile can't decode."""
    import soundfile as sf
    try:
        wav, sr = sf.read(path)
        return np.asarray(wav), sr
    except Exception:
        import librosa
        wav, sr = librosa.load(path, sr=None, mono=False)
        if wav.ndim == 2:
            wav = wav.T
        return np.asarray(wav), sr


def write_wav(path, wav, sr):
    """Write float audio as 16-bit PCM wav."""
    import soundfile as sf
    sf.write(path, np.asarray(wav), int(sr), subtype="PCM_16")


def index_audio_dir(audio_dir: str) -> dict[str, str]:
    """stem -> path for every audio file in a directory."""
    idx = {}
    for fn in os.listdir(audio_dir):
        stem, ext = os.path.splitext(fn)
        if ext.lower() in AUDIO_EXTS:
            idx[stem] = os.path.join(audio_dir, fn)
    return idx


def track_id(item: dict):
    """The per-track id of a manifest entry, whatever dataset wrote it."""
    for k in ("salami_id", "id"):
        if item.get(k) is not None:
            return str(item[k])
    return None


def file_hash(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()[:16]


@dataclass(frozen=True)
class ExtractorSpec:
    """Everything that changes the embedding. Part of the cache key."""
    sources: tuple[str, ...] = ("clap", "encodec")
    clap_window_s: float = 1.0
    clap_hop_s: float = 0.5
    encodec_bandwidth: float = 6.0
    encodec_pool: int = 8
    frame_rate: float = 2.0
    # "t2" = v0.2 extractors (time-based alignment in ConcatExtractor). Bump by
    # hand if you patch extractor code again; it busts the cache.
    version: str = "t2"

    def tag(self) -> str:
        blob = json.dumps(asdict(self), sort_keys=True).encode()
        return hashlib.sha1(blob).hexdigest()[:8]

    def describe(self) -> str:
        parts = []
        if "clap" in self.sources:
            parts.append(f"CLAP w={self.clap_window_s}s hop={self.clap_hop_s}s")
        if "encodec" in self.sources:
            parts.append(f"EnCodec {self.encodec_bandwidth}kbps pool={self.encodec_pool}")
        if "chroma_mfcc" in self.sources:
            parts.append("chroma+MFCC")
        return " + ".join(parts) + f" @ {self.frame_rate} fps"


def add_extractor_args(ap):
    """Standard CLI flags for the extractor, shared by every script."""
    ap.add_argument("--sources", default="clap,encodec",
                    help="comma list from {clap, encodec, chroma_mfcc}")
    ap.add_argument("--clap_window", type=float, default=1.0)
    ap.add_argument("--clap_hop", type=float, default=None,
                    help="default: clap_window / 2")
    ap.add_argument("--encodec_bandwidth", type=float, default=6.0)
    ap.add_argument("--encodec_pool", type=int, default=8)
    ap.add_argument("--frame_rate", type=float, default=2.0)
    ap.add_argument("--extractor_version", default="t2",
                    help="bump if you patched extractor code (busts the cache)")
    ap.add_argument("--reuse_legacy_cache", action="store_true",
                    help="accept old <hash>.npy cache files (see audio_io.py)")


def spec_from_args(args) -> ExtractorSpec:
    hop = args.clap_hop if args.clap_hop is not None else args.clap_window / 2.0
    return ExtractorSpec(
        sources=tuple(s.strip() for s in args.sources.split(",") if s.strip()),
        clap_window_s=args.clap_window, clap_hop_s=hop,
        encodec_bandwidth=args.encodec_bandwidth,
        encodec_pool=args.encodec_pool, frame_rate=args.frame_rate,
        version=args.extractor_version)


def build_extractor(spec: ExtractorSpec):
    """Instantiate the (lazy, heavy) extractor described by a spec."""
    from formadherence.integration import (
        ClapExtractor, EncodecExtractor, ConcatExtractor)
    parts = []
    for s in spec.sources:
        if s == "clap":
            parts.append(ClapExtractor(window_s=spec.clap_window_s,
                                       hop_s=spec.clap_hop_s))
        elif s == "encodec":
            parts.append(EncodecExtractor(bandwidth=spec.encodec_bandwidth,
                                          pool=spec.encodec_pool))
        elif s == "chroma_mfcc":
            try:
                from formadherence.integration.embeddings import ChromaMfccExtractor
            except ImportError:
                from chroma_mfcc_extractor import ChromaMfccExtractor
            parts.append(ChromaMfccExtractor(frame_rate=spec.frame_rate))
        else:
            raise ValueError(f"unknown source {s!r}")
    return ConcatExtractor(parts, target_frame_rate=spec.frame_rate)


class EmbeddingCache:
    """Get-or-compute embeddings, keyed by audio content AND extractor spec."""

    def __init__(self, cache_dir: str, spec: ExtractorSpec,
                 reuse_legacy: bool = False):
        self.cache_dir = cache_dir
        self.spec = spec
        self.reuse_legacy = reuse_legacy
        self._extractor = None
        os.makedirs(cache_dir, exist_ok=True)

    def path_for(self, audio_path: str) -> str:
        return os.path.join(self.cache_dir,
                            f"{file_hash(audio_path)}_{self.spec.tag()}.npy")

    def get(self, audio_path: str) -> tuple[np.ndarray, str]:
        """Return (embeddings, cache_path); extracts and saves on a miss."""
        out = self.path_for(audio_path)
        if os.path.exists(out):
            return np.load(out), out
        if self.reuse_legacy:
            legacy = os.path.join(self.cache_dir, f"{file_hash(audio_path)}.npy")
            if os.path.exists(legacy):
                return np.load(legacy), legacy
        if self._extractor is None:
            self._extractor = build_extractor(self.spec)
        wav, sr = read_audio(audio_path)
        emb = self._extractor.extract(np.asarray(wav), sr)
        np.save(out, emb)
        with open(out[:-4] + ".json", "w") as f:
            json.dump({"audio": os.path.abspath(audio_path),
                       "spec": asdict(self.spec)}, f, indent=2)
        return emb, out


def block_norm_report(emb: np.ndarray, first_dim: int) -> str:
    # NOTE: ConcatExtractor L2-normalizes each block, so this should report a
    # ratio near 1 (block_dims tells you first_dim); a far-off ratio means the
    # extractor code differs from v0.2.
    """Compare the average size of the first `first_dim` columns (e.g. CLAP)
    with the rest (e.g. EnCodec). Cosine distance on the concatenation is
    dominated by whichever block is bigger."""
    a = np.linalg.norm(emb[:, :first_dim], axis=1).mean()
    b = np.linalg.norm(emb[:, first_dim:], axis=1).mean()
    ratio = a / max(b, 1e-12)
    msg = (f"block norms: first {first_dim} dims={a:.3f}, rest={b:.3f}, "
           f"ratio={ratio:.2f}")
    if ratio > 3 or ratio < 1 / 3:
        msg += ("  <-- UNBALANCED: cosine distance is dominated by one block. "
                "L2-normalize each block inside ConcatExtractor before concat.")
    return msg
