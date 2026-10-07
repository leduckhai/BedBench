"""Depth-aware rasterizer for visibility checks and silhouette IoU."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .config import SceneConfig
from .geometry import Limits, Primitive, mesh, project


@dataclass
class Raster:
    id_map: np.ndarray          # index of the frontmost primitive per pixel, -1 for background
    projected_px: np.ndarray    # per primitive: pixels covered when drawn alone
    visible_px: np.ndarray      # per primitive: pixels where it is frontmost

    @property
    def mask(self) -> np.ndarray:
        return self.id_map >= 0


def rasterize(scene: Sequence[Primitive], view: str, lims: Limits, res: int, R: Optional[np.ndarray] = None,
              cylinder_segments: int = 24) -> Raster:
    """Fan-triangulate every face and z-buffer pixel centers (barycentric test with a 1e-9 tolerance).

    Pixels of one triangle are processed together; triangles are processed in order, so the result equals a
    per-pixel loop.
    """
    xmin, xmax, ymin, ymax = lims
    scale_x = (res - 1) / max(xmax - xmin, 1e-12)
    scale_y = (res - 1) / max(ymax - ymin, 1e-12)
    depth = np.full((res, res), -np.inf)
    id_map = np.full((res, res), -1, dtype=int)
    projected = np.zeros(len(scene), dtype=np.int64)

    for obj_id, p in enumerate(scene):
        verts, faces = mesh(p, cylinder_segments)
        xy, zbuf = project(verts, view, R)
        pix = np.empty_like(xy)
        pix[:, 0] = (xy[:, 0] - xmin) * scale_x
        pix[:, 1] = (xy[:, 1] - ymin) * scale_y
        covered = np.zeros((res, res), dtype=bool)
        for face in faces:
            for k in range(1, len(face) - 1):
                tri = [face[0], face[k], face[k + 1]]
                (ax, ay), (bx, by), (cx, cy) = pix[tri]
                z0, z1, z2 = zbuf[tri]
                x0, x1 = max(0, int(np.floor(pix[tri, 0].min()))), min(res - 1, int(np.ceil(pix[tri, 0].max())))
                y0, y1 = max(0, int(np.floor(pix[tri, 1].min()))), min(res - 1, int(np.ceil(pix[tri, 1].max())))
                v0x, v0y, v1x, v1y = cx - ax, cy - ay, bx - ax, by - ay
                den = v0x * v1y - v1x * v0y
                if x1 < x0 or y1 < y0 or abs(den) < 1e-12:
                    continue
                inv = 1.0 / den
                v2x = (np.arange(x0, x1 + 1) + 0.5 - ax)[None, :]
                v2y = (np.arange(y0, y1 + 1) + 0.5 - ay)[:, None]
                u = (v2x * v1y - v1x * v2y) * inv
                v = (v0x * v2y - v2x * v0y) * inv
                w = 1.0 - u - v
                inside = (u >= -1e-9) & (v >= -1e-9) & (w >= -1e-9)
                z = w * z0 + v * z1 + u * z2
                covered[y0:y1 + 1, x0:x1 + 1] |= inside
                win = inside & (z > depth[y0:y1 + 1, x0:x1 + 1])
                depth[y0:y1 + 1, x0:x1 + 1][win] = z[win]
                id_map[y0:y1 + 1, x0:x1 + 1][win] = obj_id
        projected[obj_id] = int(covered.sum())

    visible = np.array([(id_map == i).sum() for i in range(len(scene))], dtype=np.int64)
    return Raster(id_map, projected, visible)


def view_ok(scene: Sequence[Primitive], view: str, lims: Limits, cfg: SceneConfig,
            R: Optional[np.ndarray] = None) -> tuple[bool, Raster]:
    """Every primitive keeps enough visible pixels and enough of its projected area in this view."""
    r = rasterize(scene, view, lims, cfg.visibility_px, R, cfg.cylinder_segments)
    query = view == "query"
    min_px = cfg.min_visible_px_query if query else cfg.min_visible_px_canonical
    max_hidden = cfg.max_hidden_ratio_query if query else cfg.max_hidden_ratio_canonical
    if np.any(r.projected_px <= 0):
        return False, r
    visible_fraction = r.visible_px / r.projected_px
    ok = bool(np.all(r.visible_px >= min_px) and np.all(visible_fraction >= cfg.min_visible_fraction)
              and np.max(1.0 - visible_fraction) <= max_hidden)
    return ok, r


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union > 0 else 1.0
