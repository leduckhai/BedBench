"""Scene-generator configuration. Defaults reproduce the released T13 scene pool."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SceneConfig:
    """All knobs of the multi-object scene generator (Appendix C, "Multi-object scenes")."""

    # Scene construction.
    n_objects: tuple[int, int] = (6, 8)
    scene_attempts: int = 300
    object_attempts: int = 600
    attach_prob: float = 0.45            # attach to an existing primitive (top or side) instead of the floor
    floor_range: float = 3.0             # floor placements in [-r, r]^2

    # Primitive sizes.
    extent_range: tuple[float, float] = (1.0, 2.1)   # box and cylinder extents
    cube_edge_range: tuple[float, float] = (1.1, 1.55)
    cylinder_segments: int = 24

    # Query views (azimuth, elevation) in degrees.
    query_views: tuple[tuple[float, float], ...] = ((45, 30), (150, 40))

    # Visibility checks on a depth-aware raster.
    visibility_px: int = 220
    min_visible_px_query: int = 220
    min_visible_px_canonical: int = 180
    min_visible_fraction: float = 0.08
    max_hidden_ratio_query: float = 0.92
    max_hidden_ratio_canonical: float = 0.95
    pad: float = 0.8                     # view-limit padding in world units

    # Scene distractors.
    min_distractor_iou: float = 0.55     # silhouette IoU strictly between min and max
    max_distractor_iou: float = 0.88
    distractor_attempts: int = 120

    # Rendering.
    dpi: int = 140
    query_figsize: float = 4.0
    canonical_figsize: float = 3.0
    edge_width: float = 1.2
    uncolored: str = "#bfbfbf"


# Primitive colors (the voxel palette without white and black).
COLORS = ("yellow", "cyan", "green", "magenta", "red", "orange",
          "orchid", "teal", "slategray", "gold", "blue", "purple")
PLANES = ("xy", "yz", "zx")
