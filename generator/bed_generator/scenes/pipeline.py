"""Generate the T13 scene pool: scenes, scene distractors, renders, and manifest.json."""
from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict

import numpy as np

from .config import PLANES, SceneConfig
from .geometry import Primitive, rotation, view_limits
from .render import save_view
from .sampling import sample_scene, scene_distractor

logger = logging.getLogger(__name__)


def save_scene(path: str, scene: list[Primitive]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump([asdict(p) for p in scene], f, indent=2)


def generate(out_dir: str, n_scenes: int, seed: int = 42, cfg: SceneConfig = SceneConfig()) -> str:
    """Write n_scenes scenes; returns the manifest path.

    Layout per scene: queries/view_{1,2}.png and queries_uncolored/ (two query views), pos/{xy,yz,zx}.png and
    pos_uncolored/ (canonical projections), neg/ and neg_uncolored/ (projections of the scene distractor of
    each plane, drawn in the same view limits as pos/), scene.json, and neg/{plane}_scene.json.
    """
    rng = np.random.default_rng(seed)
    samples = []
    for sid in range(n_scenes):
        d = os.path.join(out_dir, f"sample_{sid:04d}")
        for sub in ("queries", "queries_uncolored", "pos", "pos_uncolored", "neg", "neg_uncolored"):
            os.makedirs(os.path.join(d, sub), exist_ok=True)
        scene = sample_scene(rng, cfg)
        save_scene(os.path.join(d, "scene.json"), scene)
        for i, (az, el) in enumerate(cfg.query_views, start=1):
            R = rotation(az, el)
            lims = view_limits([scene], "query", R, cfg.pad, cfg.cylinder_segments)
            save_view(os.path.join(d, "queries", f"view_{i}.png"), scene, "query", lims, cfg, R)
            save_view(os.path.join(d, "queries_uncolored", f"view_{i}.png"), scene, "query", lims, cfg, R, cfg.uncolored)
        ious = {}
        for plane in PLANES:
            found = scene_distractor(rng, scene, plane, cfg)
            if found is None:
                raise RuntimeError(f"no scene distractor for sample {sid}, plane {plane}")
            neg, lims, ious[plane] = found
            for folder, s, color in (("pos", scene, None), ("pos_uncolored", scene, cfg.uncolored),
                                     ("neg", neg, None), ("neg_uncolored", neg, cfg.uncolored)):
                save_view(os.path.join(d, folder, f"{plane}.png"), s, plane, lims, cfg, color=color)
            save_scene(os.path.join(d, "neg", f"{plane}_scene.json"), neg)
        samples.append({"sample_id": sid, "path": f"sample_{sid:04d}", "n_objects": len(scene),
                        "distractor_iou": ious})
        logger.info("scene %d: %d primitives", sid, len(scene))
    manifest = os.path.join(out_dir, "manifest.json")
    with open(manifest, "w", encoding="utf-8") as f:
        json.dump({"config": asdict(cfg), "seed": seed, "samples": samples}, f, indent=2)
    return manifest
