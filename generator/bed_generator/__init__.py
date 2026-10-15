"""BedBench voxel-object generator."""
from .config import GeneratorConfig
from .pipeline import generate

__all__ = ["GeneratorConfig", "generate"]
