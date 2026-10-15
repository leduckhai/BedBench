"""Generator configuration. Defaults reproduce the released BedBench voxel split."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GeneratorConfig:
    """All knobs of the voxel-object generator (Appendix C, Dataset Generation Details)."""

    # Connected voxel objects (Algorithm "Grow a Connected Voxel Object").
    n_blocks: tuple[int, int] = (10, 15)
    xy_radius: int = 4
    max_height: int = 6
    horizontal_growth_prob: float = 0.72
    growth_attempts: int = 9000

    # Query-view selection (Algorithm "Select Two Query Views").
    azimuth_range: tuple[float, float] = (0.0, 360.0)
    elevation_range: tuple[float, float] = (18.0, 55.0)
    min_axis_sep_deg: float = 45.0
    # A random second view is tried first; with min_axis_sep_deg=45 its constraints are
    # (practically) unsatisfiable, so every released pair uses the opposite-azimuth fallback.
    second_azimuth_delta: tuple[float, float] = (140.0, 220.0)
    second_elevation_delta: tuple[float, float] = (12.0, 25.0)
    min_azimuth_sep_deg: float = 110.0
    min_elevation_sep_deg: float = 10.0
    fallback_elevation_offset: float = 12.0
    view_pair_tries: int = 200
    view_attempts: int = 2600
    footprint_min_span: float = 2.2
    footprint_min_area: float = 6.0
    footprint_max_aspect: float = 4.2

    # Silhouette masks shared by both filters.
    mask_px: int = 256
    mask_threshold: float = 0.985

    # Hidden-block filter (Algorithm "Filter Hidden-Block Objects").
    hidden_block_iou: float = 0.9998
    hidden_block_min_pixels: int = 8

    # Sampled ambiguity check (Algorithm "Reject Two-View Collisions").
    collision_trials: int = 500
    collision_iou: float = 0.995
    object_attempts: int = 120

    # Colors (Algorithm "Assign Adversarial Colors").
    color_mode: str = "disegno_failure"

    # Projection distractors (Algorithm "Generate Projection Distractors").
    neg_edits: tuple[int, int] = (1, 1)
    neg_ops: tuple[str, ...] = ("add", "remove", "shift")
    neg_max_count_delta: int = 1
    neg_sampling_pad: int = 2
    neg_edit_tries: int = 400
    neg_similarity_tries: int = 600
    neg_max_iou_vs_other: float = 0.85
    neg_min_symdiff_vs_other: int = 2
    neg_similarity_pad: int = 1
    thin_bar_max_len: int = 3

    # Rendering.
    linewidth: float = 1.2
    uniform_color: tuple[float, float, float] = (0.70, 0.70, 0.70)
    image_px: int = 256
    image_dpi: int = 160
    view_pad: float = 0.8
