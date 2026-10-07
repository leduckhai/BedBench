"""Generate the voxel-object split: accepted objects, renders, and manifest.csv."""
from __future__ import annotations

import csv
import json
import logging
import os
from typing import Optional, Sequence

import numpy as np

from .colors import assign_colors
from .config import GeneratorConfig
from .distractors import make_distractor
from .filters import hidden_block_ok, is_ambiguous
from .geometry import PLANES, Block, Cell, frontmost_colors, plane_bounds, projection_cells
from .render import draw_cells, draw_object, save_panel, save_query_view
from .targets import color_counts, container_box, extents
from .views import View, select_query_views
from .voxels import sample_object

logger = logging.getLogger(__name__)

MANIFEST_COLUMNS = [
    "sample_id", "sample_dir", "query1_az", "query1_el", "query2_az", "query2_el",
    "n_blocks_base", "base_blocks_json",
    "pos_zx_cells_json", "pos_yz_cells_json", "pos_xy_cells_json",
    "neg_zx_type", "neg_yz_type", "neg_xy_type",
    "neg_zx_cells_json", "neg_yz_cells_json", "neg_xy_cells_json",
    "filtered_collision_trials", "filtered_tau_iou",
    "query_footprint_min_span", "query_footprint_min_area", "query_footprint_max_aspect",
    "resample_attempts_used", "used_ghost_filter",
    "neg_max_iou_vs_other_pos", "neg_min_symdiff_vs_other_pos",
    "plane_bounds_xy_world", "plane_bounds_yz_world", "plane_bounds_zx_world",
    "colors_json", "task_volume", "box_dims_json", "fits_box", "color_total", "color_distribution_json",
]


def _compact(obj) -> str:
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


def _cells_json(cells: Optional[set[Cell]]) -> str:
    return "null" if cells is None else _compact(sorted([[int(u), int(v)] for u, v in cells], key=lambda t: (t[1], t[0])))


def accept_object(rng: np.random.Generator, cfg: GeneratorConfig) -> Optional[tuple[list[Block], View, View, int]]:
    """Sample objects until one passes the hidden-block filter and the ambiguity check."""
    for attempt in range(1, cfg.object_attempts + 1):
        blocks = sample_object(rng, cfg)
        q1, q2 = select_query_views(rng, blocks, cfg)
        if hidden_block_ok(blocks, q1, q2, cfg) and not is_ambiguous(rng, blocks, q1, q2, cfg):
            return blocks, q1, q2, attempt
    return None


def _nearest_colors(cells: set[Cell], known: dict[Cell, str]) -> dict[Cell, str]:
    """Distractor cells reuse the color of their own or the nearest (L1) positive cell."""
    keys = list(known)
    return {c: known[c] if c in known else known[min(keys, key=lambda k: abs(k[0] - c[0]) + abs(k[1] - c[1]))]
            for c in cells}


def write_renders(sample_dir: str, plain: Sequence[Block], colored: Sequence[Block], q1: View, q2: View,
                  negatives: dict[str, Optional[set[Cell]]], cfg: GeneratorConfig) -> None:
    """Query views (colored/uncolored x axes/no axes) and positive/negative projections."""
    for tag, blocks in (("", colored), ("_uncolored", plain)):
        for axes_tag, show_axes in (("", False), ("_axes", True)):
            for i, q in ((1, q1), (2, q2)):
                save_query_view(os.path.join(sample_dir, f"query{i}{tag}{axes_tag}.png"), blocks, q, cfg, show_axes)
    for plane in ("xy", "yz", "zx"):
        for folder, blocks in (("pos", colored), ("pos_uncolored", plain)):
            save_panel(os.path.join(sample_dir, folder, f"{plane}.png"),
                       lambda ax, b=blocks, p=plane: draw_object(ax, b, cfg, view=p), cfg)
        neg = negatives[plane]
        if neg is not None:
            fill = _nearest_colors(neg, frontmost_colors(colored, plane))
            save_panel(os.path.join(sample_dir, "neg", f"{plane}.png"), lambda ax: draw_cells(ax, neg, cfg, fill), cfg)
            save_panel(os.path.join(sample_dir, "neg_uncolored", f"{plane}.png"), lambda ax: draw_cells(ax, neg, cfg), cfg)


def generate(out_dir: str, n_samples: int, seed: int = 42, cfg: GeneratorConfig = GeneratorConfig()) -> str:
    """Write n_samples accepted samples and their manifest; returns the manifest path."""
    rng = np.random.default_rng(seed)
    os.makedirs(os.path.join(out_dir, "samples"), exist_ok=True)
    manifest = os.path.join(out_dir, "manifest.csv")
    with open(manifest, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(MANIFEST_COLUMNS)
        sample_id = 0
        while sample_id < n_samples:
            accepted = accept_object(rng, cfg)
            if accepted is None:
                continue
            plain, q1, q2, attempts = accepted
            colored = assign_colors(rng, plain)
            positives = {p: projection_cells(colored, p) for p in PLANES}
            negatives, status = {}, {}
            for p in PLANES:
                others = [positives[o] for o in PLANES if o != p]
                negatives[p], status[p] = make_distractor(rng, positives[p], others, cfg)
            box, box_fits = container_box(rng, extents(colored))
            counts = color_counts(colored)

            rel_dir = os.path.join("samples", f"sample_{sample_id:05d}")
            write_renders(os.path.join(out_dir, rel_dir), plain, colored, q1, q2, negatives, cfg)
            writer.writerow([
                sample_id, rel_dir, f"{q1[0]:.4f}", f"{q1[1]:.4f}", f"{q2[0]:.4f}", f"{q2[1]:.4f}",
                len(colored), _compact([[b.x, b.y, b.z] for b in colored]),
                *(_cells_json(positives[p]) for p in PLANES),
                *(status[p] for p in PLANES),
                *(_cells_json(negatives[p]) for p in PLANES),
                cfg.collision_trials, f"{cfg.collision_iou:.6f}",
                f"{cfg.footprint_min_span:.4f}", f"{cfg.footprint_min_area:.4f}", f"{cfg.footprint_max_aspect:.4f}",
                attempts, 1, f"{cfg.neg_max_iou_vs_other:.4f}", cfg.neg_min_symdiff_vs_other,
                *(_compact(list(plane_bounds(p, cfg.xy_radius, cfg.max_height))) for p in ("xy", "yz", "zx")),
                _compact([b.color for b in colored]), len(colored), _compact(list(box)), int(box_fits),
                len(counts), _compact(counts),
            ])
            f.flush()
            logger.info("sample %d accepted after %d attempt(s)", sample_id, attempts)
            sample_id += 1
    return manifest
