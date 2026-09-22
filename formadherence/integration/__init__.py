"""Integration layer: audio embeddings, music generators, and the RL reward."""

from .embeddings import (
    EmbeddingExtractor, ClapExtractor, EncodecExtractor, ConcatExtractor,
    ChromaMfccExtractor,
)
from .generators import (
    MusicGenerator, GeneratedSong, YuEGenerator, CommandLineGenerator,
    form_to_structure_tags, build_structured_lyrics,
)
from .reward import FormAdherenceReward, RewardConfig, RewardOutput

__all__ = [
    "EmbeddingExtractor", "ClapExtractor", "EncodecExtractor", "ConcatExtractor",
    "MusicGenerator", "GeneratedSong", "YuEGenerator", "CommandLineGenerator",
    "form_to_structure_tags", "build_structured_lyrics",
    "FormAdherenceReward", "RewardConfig", "RewardOutput",
]
