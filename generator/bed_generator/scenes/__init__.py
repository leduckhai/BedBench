"""Multi-object scene generator for T13 (separate pool and random stream from the voxel objects)."""
from .config import SceneConfig
from .pipeline import generate

__all__ = ["SceneConfig", "generate"]
