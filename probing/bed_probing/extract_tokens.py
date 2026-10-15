"""Extract frozen tower or Qwen connector tokens for the T5 train/validation/test splits. See README.md for usage."""
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

# Count-vector order used by the original token experiment and evaluation harness.
PALETTE = ["yellow", "cyan", "green", "magenta", "red", "orange", "orchid", "teal", "slategray", "gold", "blue", "purple"]

logger = logging.getLogger("extract_tokens")


def load_splits(dataset: Path) -> dict:
    import datasets

    ft = datasets.load_from_disk(str(dataset / "T5_finetune"))
    return {"train": ft["train"], "val": ft["validation"], "test": datasets.load_from_disk(str(dataset / "T5_color"))}


class TokenTower:
    """Wraps bed_probing.towers.TowerEncoder and captures the full token matrix of each image."""

    def __init__(self, model_id: str, dtype: torch.dtype, stage: str = "tower") -> None:
        from .towers import TowerEncoder

        self.stage = stage

        self.enc = TowerEncoder(model_id, None, dtype)
        self._out = None
        self.enc.tower.register_forward_hook(lambda m, i, o: setattr(self, "_out", o))

    def tokens(self, image) -> torch.Tensor:
        feat = self.enc.encode_one(image)  # paper feature (mean over tokens)
        out = self._out
        if self.stage == "connector":
            # Qwen3.5 vision model returns the patch-merger output as pooler_output; it is what the LLM receives.
            merged = getattr(out, "pooler_output", None)
            if merged is None:
                raise RuntimeError("Tower output has no pooler_output (connector output); this model/transformers "
                                   "version does not expose it. Use --stage tower.")
            merged = merged.flatten(0, -2) if merged.dim() > 2 else merged
            merged = merged.float().cpu().half()
            if not torch.isfinite(merged).all():
                raise RuntimeError("connector output overflows float16; store it in a wider dtype")
            return merged
        hidden = getattr(out, "last_hidden_state", None)  # same selection as towers.encode_one
        if hidden is None:
            hidden = out if isinstance(out, torch.Tensor) else out[0]
        if hidden.dim() == 3:
            hidden = hidden.flatten(0, 1)
        hidden = hidden.float().cpu()
        assert np.allclose(hidden.mean(0).numpy(), feat, atol=1e-4), "token mean != paper feature"
        return hidden.half()


class TokenEncoder:
    """Wraps bed_probing.encoders.FrozenEncoder (pretrained encoders); keeps the patch tokens."""

    def __init__(self, alias: str) -> None:
        from .encoders import build_encoder

        self.enc = build_encoder(alias)
        self._patches = None
        split = self.enc._split_outputs

        def capture(out):
            pooled, patches = split(out)
            self._patches = patches
            return pooled, patches

        self.enc._split_outputs = capture

    def tokens(self, image) -> torch.Tensor:
        self.enc.encode([image], batch_size=1)
        return self._patches[0].float().cpu().half()


def main() -> None:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--vlm", help="HF id of the VLM whose vision tower is used")
    src.add_argument("--encoder", help="pretrained encoder alias (dinov2, siglip2, ...)")
    ap.add_argument("--alias", help="output folder name (also used for optional feature-cache lookup)")
    ap.add_argument("--stage", default="tower", choices=["tower", "connector"],
                    help="--vlm only: tower = encoder output (paper probes); connector = patch-merger output fed to the LLM")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"],
                    help="tower dtype; float32 is safest on Apple MPS")
    ap.add_argument("--dataset", type=Path, required=True, help="local dataset directory containing T5_finetune and T5_color")
    ap.add_argument("--out", type=Path, default=Path("tokens"))
    ap.add_argument("--paper-features", type=Path, help="optional pooled-feature cache for self-check")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    alias = args.alias or args.encoder
    if args.vlm and not args.alias:
        ap.error("--vlm needs --alias")
    if args.encoder and args.stage != "tower":
        ap.error("--stage connector requires --vlm")
    enc = TokenTower(args.vlm, getattr(torch, args.dtype), args.stage) if args.vlm else TokenEncoder(args.encoder)

    out_dir = args.out / alias
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, ds in load_splits(args.dataset).items():
        path = out_dir / f"{split}.pt"
        if path.exists():
            logger.info("[skip] %s exists", path)
            continue
        toks, counts = [], []
        for i, row in enumerate(ds):
            toks.append((enc.tokens(row["query_image_1"]), enc.tokens(row["query_image_2"])))
            c = json.loads(row["color_distribution_json"])
            counts.append([int(c.get(k, 0)) for k in PALETTE])
            if (i + 1) % 50 == 0:
                logger.info("%s %s: %d/%d", alias, split, i + 1, len(ds))
        logger.info("%s %s: %d objects, tokens per view %s, dim %d", alias, split, len(toks),
                    tuple(t.shape[0] for t in toks[0]), toks[0][0].shape[1])
        torch.save({"tokens": toks, "counts": torch.tensor(counts), "sample_id": list(ds["sample_id"])}, path)

    # Self-check against the paper's cached mean-pooled features (test split = benchmark T5_color).
    ref = args.paper_features / alias / "T5_color.npz" if args.paper_features else None
    if args.stage == "connector":
        logger.info("connector stage: no paper features to self-check against")
    elif ref is not None and ref.exists():
        d, test = np.load(ref), torch.load(out_dir / "test.pt")
        for v in (0, 1):
            mine = np.stack([t[v].float().mean(0).numpy() for t in test["tokens"]])
            paper = d[f"patch_mean__query_image_{v + 1}"]
            err = np.abs(mine - paper).max()
            logger.info("self-check view %d: max |token mean - paper patch_mean| = %.2e", v + 1, err)
            if err > 1e-2:
                logger.warning("Large difference from the paper features (dtype/device?). Report this.")
    else:
        logger.warning("No paper features at %s; self-check skipped", ref)


if __name__ == "__main__":
    main()
