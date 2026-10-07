"""Painter's-algorithm rendering of query views and canonical projections."""
from __future__ import annotations

from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402

from .config import SceneConfig  # noqa: E402
from .geometry import Limits, Primitive, mesh, project  # noqa: E402


def save_view(path: str, scene: Sequence[Primitive], view: str, lims: Limits, cfg: SceneConfig,
              R: Optional[np.ndarray] = None, color: Optional[str] = None) -> None:
    """Draw all faces back to front (mean depth key), each in its primitive's color or `color`, with black edges."""
    size = cfg.query_figsize if view == "query" else cfg.canonical_figsize
    fig, ax = plt.subplots(figsize=(size, size))
    ax.set_facecolor("white")
    ax.axis("off")
    faces = []
    for p in scene:
        verts, idx = mesh(p, cfg.cylinder_segments)
        xy, depth = project(verts, view, R)
        faces += [(float(np.mean(depth[list(f)])), xy[list(f)], color or p.color) for f in idx]
    for _, poly, fc in sorted(faces, key=lambda f: f[0]):
        ax.add_patch(Polygon(poly, closed=True, fc=fc, ec="black", lw=cfg.edge_width))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(lims[0], lims[1])
    ax.set_ylim(lims[2], lims[3])
    fig.savefig(path, dpi=cfg.dpi, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
