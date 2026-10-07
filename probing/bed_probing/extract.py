"""Extract frozen-encoder features for the probed BED configurations and cache them as .npz.

    python -m bed_probing.extract --encoder siglip2 --dataset DATASET --cache-dir features
    python -m bed_probing.extract --vlm Qwen/Qwen3.5-27B --alias qwen27b-tower \
        --dataset DATASET --cache-dir features

--encoder runs a pretrained encoder (encoders.py); --vlm runs the vision tower inside a VLM checkpoint
without loading its language model (towers.py). DATASET is the directory written by
`python -m bed_generator.build_tasks` (one subdirectory per configuration) or a Hugging Face dataset id.
Output: CACHE/<alias>/<config>.npz (pooled__<column>, patch_mean__<column>) and <config>.targets.json
(the non-image fields of every row).
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List

import numpy as np

logger = logging.getLogger(__name__)

# Single-object probe configurations; T1 variants double as retrieval configurations.
PROBE_CONFIGS: List[str] = [
    "T1_color", "T1_uncolor", "T1_uncolor_axes",
    "T2_uncolor", "T2_uncolor_axes",
    "T4_color", "T4_color_axes",
    "T5_color", "T5_color_axes",
    "T6_uncolor", "T6_uncolor_axes",
    "T7_color", "T7_color_axes",
    "T8_uncolor", "T8_uncolor_axes",
    "T9_uncolor", "T9_uncolor_axes",
    "T10_color", "T10_color_axes",
]
# Object pairs for the distance-vs-structure analysis.
PAIR_CONFIGS: List[str] = ["T3_color_axes", "T3_uncolor_axes"]
# Candidate-selection tasks for zero-shot ranking, default condition (uncolored).
CANDIDATE_CONFIGS: List[str] = ["T11_invalid_uncolor", "T11_valid_uncolor", "T12_uncolor",
                                "T13_invalid_uncolor", "T13_valid_uncolor"]
ALL_CONFIGS: List[str] = PROBE_CONFIGS + PAIR_CONFIGS + CANDIDATE_CONFIGS


def load_config(dataset: str, config: str):
    """A local save_to_disk directory (DATASET/CONFIG) or a Hugging Face Hub dataset."""
    import datasets

    local = Path(dataset) / config
    return datasets.load_from_disk(str(local)) if local.exists() else datasets.load_dataset(dataset, config, split="train")


def extract_config(encoder, alias: str, config: str, dataset: str, cache_dir: Path) -> Path:
    """Embed every image column of one configuration; save features and non-image fields."""
    from datasets import Image as ImageFeature

    out_dir = cache_dir / alias
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path, meta_path = out_dir / f"{config}.npz", out_dir / f"{config}.targets.json"
    if npz_path.exists() and meta_path.exists():
        logger.info("[skip] %s / %s already cached", alias, config)
        return npz_path

    ds = load_config(dataset, config)
    image_cols = [name for name, feat in ds.features.items() if isinstance(feat, ImageFeature)]
    arrays = {}
    for col in image_cols:
        for rep, mat in encoder.encode(list(ds[col])).items():
            arrays[f"{rep}__{col}"] = mat
        logger.info("%s: encoded %s", config, col)
    non_image = [c for c in ds.column_names if c not in image_cols]
    targets = [{c: row[c] for c in non_image} for row in ds.remove_columns(image_cols)]
    np.savez_compressed(npz_path, **arrays)
    meta_path.write_text(json.dumps(targets))
    return npz_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--encoder", help="pretrained encoder alias (siglip2, dinov2, clip, depthanything, ...)")
    source.add_argument("--vlm", help="Hugging Face id of a VLM whose vision tower is probed")
    parser.add_argument("--alias", help="cache folder name for --vlm, e.g. qwen27b-tower")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--configs", nargs="*", default=ALL_CONFIGS)
    parser.add_argument("--cache-dir", type=Path, default=Path("features"))
    parser.add_argument("--dtype", default="float16", choices=["bfloat16", "float16", "float32"],
                        help="--vlm only: float16 for Turing GPUs, bfloat16 on Ampere or newer")
    parser.add_argument("--max-pixels", type=int, default=None,
                        help="--vlm only: cap dynamic-resolution processors (unset matches VLM evaluation)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.vlm:
        import torch
        from .towers import TowerEncoder

        if not args.alias:
            parser.error("--vlm needs --alias")
        encoder, alias = TowerEncoder(args.vlm, None, getattr(torch, args.dtype), max_pixels=args.max_pixels), args.alias
    else:
        from .encoders import build_encoder

        encoder, alias = build_encoder(args.encoder), args.encoder
    for config in args.configs:
        extract_config(encoder, alias, config, args.dataset, args.cache_dir)


if __name__ == "__main__":
    main()
