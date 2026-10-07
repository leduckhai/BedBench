"""Painter's-algorithm rendering of voxel objects and projection cells (matplotlib, Agg)."""
from __future__ import annotations

import os
from typing import Callable, Optional, Sequence

import matplotlib.patheffects as path_effects
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Polygon, Rectangle

from .config import GeneratorConfig
from .geometry import CUBE_FACES, Block, Cell, camera_matrix, cube_vertices, project, view_limits
from .voxels import cell_bounds

Limits = tuple[float, float, float, float]
_AXIS_LENGTH = 50.0
_LABEL_MARGIN = 0.15


def _faces(blocks: Sequence[Block], view: str, camera: Optional[np.ndarray]) -> list[dict]:
    items = []
    for b in blocks:
        xy, depth = project(cube_vertices(b), view, camera)
        for face in CUBE_FACES:
            idx = list(face)
            items.append({"type": "face", "poly": xy[idx], "depth": float(np.mean(depth[idx])), "color": b.color})
    return items


def _axis_items(view: str, camera: Optional[np.ndarray], limits: Optional[Limits]) -> list[dict]:
    """Dashed X/Y/Z axes from the origin, clipped to the viewport with labels at the edge."""
    origin = np.zeros((1, 3))
    tips = np.eye(3) * _AXIS_LENGTH
    p0, d0 = project(origin, view, camera)
    p_tips, d_tips = project(tips, view, camera)
    items = []
    for i, label in enumerate("XYZ"):
        x0, y0 = p0[0]
        dx, dy = p_tips[i, 0] - x0, p_tips[i, 1] - y0
        item = {"type": "axis", "label": label, "depth": (d0[0] + d_tips[i]) / 2.0,
                "coords": ([x0, p_tips[i, 0]], [y0, p_tips[i, 1]]),
                "text_pos": p_tips[i] + (p_tips[i] - p0[0]) * 0.1, "ha": "center", "va": "center"}
        if limits is not None:
            xmin, xmax, ymin, ymax = limits
            t_max, hit = 1.0, None
            for edge, is_x, is_min in ((xmin, True, True), (xmax, True, False), (ymin, False, True), (ymax, False, False)):
                step = dx if is_x else dy
                if abs(step) < 1e-9:
                    continue
                t = (edge - (x0 if is_x else y0)) / step
                if (step > 0) != is_min and t < t_max:
                    t_max, hit = t, (is_x, is_min)
            item["coords"] = ([x0, x0 + t_max * dx], [y0, y0 + t_max * dy])
            item["text_pos"] = np.array([x0 + t_max * dx, y0 + t_max * dy])
            if hit is not None:
                is_x, is_min = hit
                if is_x:
                    item["ha"] = "left" if is_min else "right"
                    item["text_pos"][0] += _LABEL_MARGIN if is_min else -_LABEL_MARGIN
                else:
                    item["va"] = "bottom" if is_min else "top"
                    item["text_pos"][1] += _LABEL_MARGIN if is_min else -_LABEL_MARGIN
        items.append(item)
    return items


def draw_object(ax, blocks: Sequence[Block], cfg: GeneratorConfig, view: str = "query",
                camera_view: Optional[tuple[float, float]] = None, limits: Optional[Limits] = None,
                show_axes: bool = False) -> None:
    """Draw faces (and optional axes) far to near; colorless blocks use the uniform gray."""
    ax.set_facecolor((1.0, 1.0, 1.0))
    camera = camera_matrix(*camera_view) if view == "query" else None
    items = _faces(blocks, view, camera)
    if show_axes:
        items += _axis_items(view, camera, limits)
    items.sort(key=lambda it: it["depth"])
    for it in items:
        if it["type"] == "face":
            patch = Polygon(it["poly"], closed=True, fill=True, linewidth=cfg.linewidth, alpha=1.0)
            patch.set_facecolor(it["color"] or cfg.uniform_color)
            patch.set_edgecolor((0, 0, 0))
            ax.add_patch(patch)
        else:
            ax.plot(*it["coords"], color="black", linewidth=1, linestyle="--", alpha=0.9, zorder=100)
            text = ax.text(*it["text_pos"], it["label"], color="black", fontsize=12,
                           ha=it["ha"], va=it["va"], zorder=101)
            text.set_path_effects([path_effects.withStroke(linewidth=3, foreground="white")])
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    if limits is None:
        pts = np.vstack([it["poly"] for it in items if it["type"] == "face"])
        limits = (pts[:, 0].min() - 0.8, pts[:, 0].max() + 0.8, pts[:, 1].min() - 0.8, pts[:, 1].max() + 0.8)
    ax.set_xlim(float(limits[0]), float(limits[1]))
    ax.set_ylim(float(limits[2]), float(limits[3]))


def draw_cells(ax, cells: set[Cell], cfg: GeneratorConfig, colors: Optional[dict[Cell, str]] = None) -> None:
    """Draw projection cells as unit squares; uniform gray unless a color map is given."""
    ax.set_facecolor((1.0, 1.0, 1.0))
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    for cell in cells:
        color = (colors or {}).get(cell, cfg.uniform_color)
        ax.add_patch(Rectangle(cell, 1, 1, linewidth=cfg.linewidth, edgecolor=(0, 0, 0),
                               facecolor=color, alpha=1.0, fill=True))
    umin, umax, vmin, vmax = cell_bounds(cells, pad=1)
    ax.set_xlim(float(umin - 0.6), float(umax + 1 + 0.6))
    ax.set_ylim(float(vmin - 0.6), float(vmax + 1 + 0.6))


def _figure(px: int) -> tuple[Figure, object]:
    fig = Figure(figsize=(px / 100.0, px / 100.0), dpi=100)
    FigureCanvasAgg(fig)
    return fig, fig.add_axes([0, 0, 1, 1])


def save_panel(path: str, draw: Callable, cfg: GeneratorConfig) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig, ax = _figure(cfg.image_px)
    draw(ax)
    fig.canvas.draw()
    fig.savefig(path, dpi=cfg.image_dpi)


def save_query_view(path: str, blocks: Sequence[Block], view: tuple[float, float], cfg: GeneratorConfig,
                    show_axes: bool = False) -> None:
    limits = view_limits(blocks, view, cfg.view_pad)
    save_panel(path, lambda ax: draw_object(ax, blocks, cfg, "query", view, limits, show_axes), cfg)


def silhouette(blocks: Sequence[Block], view: tuple[float, float], limits: Limits, cfg: GeneratorConfig) -> np.ndarray:
    """Boolean foreground mask of a query view rendered at cfg.mask_px within fixed limits."""
    fig, ax = _figure(cfg.mask_px)
    draw_object(ax, blocks, cfg, "query", view, limits)
    fig.canvas.draw()
    rgb = np.asarray(fig.canvas.buffer_rgba())[..., :3].astype(np.float32) / 255.0
    return np.any(rgb < cfg.mask_threshold, axis=-1)
