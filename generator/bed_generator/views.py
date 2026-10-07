"""Query-view selection (Algorithm "Select Two Query Views")."""
from __future__ import annotations

from typing import Sequence

import numpy as np

from .config import GeneratorConfig
from .geometry import Block, axis_separation_ok, pick, projected_vertices

View = tuple[float, float]


def footprint_ok(blocks: Sequence[Block], view: View, cfg: GeneratorConfig) -> bool:
    """The projected object must be large enough and not too elongated."""
    pts = projected_vertices(blocks, view)
    span_x = max(1e-9, float(pts[:, 0].max()) - float(pts[:, 0].min()))
    span_y = max(1e-9, float(pts[:, 1].max()) - float(pts[:, 1].min()))
    return (min(span_x, span_y) >= cfg.footprint_min_span
            and span_x * span_y >= cfg.footprint_min_area
            and max(span_x / span_y, span_y / span_x) <= cfg.footprint_max_aspect)


def _azimuth_gap(a: float, b: float) -> float:
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def _sample_pair(rng: np.random.Generator, cfg: GeneratorConfig) -> tuple[View, View]:
    """First view far from the canonical axes; second view on the opposite side."""
    first = (35.0, 35.0)
    for _ in range(cfg.view_pair_tries):
        az = float(rng.uniform(*cfg.azimuth_range))
        el = float(rng.uniform(*cfg.elevation_range))
        if axis_separation_ok(az, el, cfg.min_axis_sep_deg):
            first = (az, el)
            break
    az1, el1 = first
    lo, hi = cfg.elevation_range
    for _ in range(cfg.view_pair_tries):
        az2 = float((az1 + float(rng.uniform(*cfg.second_azimuth_delta))) % 360.0)
        sign = pick(rng, (-1.0, 1.0))
        el2 = float(np.clip(el1 + float(sign * rng.uniform(*cfg.second_elevation_delta)), lo, hi))
        if (_azimuth_gap(az1, az2) >= cfg.min_azimuth_sep_deg
                and abs(el2 - el1) >= cfg.min_elevation_sep_deg
                and axis_separation_ok(az2, el2, cfg.min_axis_sep_deg)):
            return first, (az2, el2)
    # Fallback used by every released pair: opposite azimuth, slightly higher elevation.
    return first, (float((az1 + 180.0) % 360.0), float(np.clip(el1 + cfg.fallback_elevation_offset, lo, hi)))


def select_query_views(rng: np.random.Generator, blocks: Sequence[Block], cfg: GeneratorConfig) -> tuple[View, View]:
    """Resample view pairs until both views pass the footprint checks."""
    for _ in range(cfg.view_attempts):
        q1, q2 = _sample_pair(rng, cfg)
        if footprint_ok(blocks, q1, cfg) and footprint_ok(blocks, q2, cfg):
            return q1, q2
    return _sample_pair(rng, cfg)  # not reached for the released split
