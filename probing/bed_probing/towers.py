"""Features from the vision tower inside a VLM checkpoint (Qwen3.5, Gemma4).

The full VLM is instantiated on the meta device, the vision tower is located by name, and only the
safetensors shards holding tower weights are downloaded and materialized, so the language model is never
loaded. Images are encoded one at a time (variable-resolution towers); the feature is the mean over the
tower's output tokens, stored as both `patch_mean` and `pooled` (towers have no pooling head).
Used by `python -m bed_probing.extract --vlm MODEL --alias NAME`.
"""

import inspect
import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch

logger = logging.getLogger("tower_extract")

VISION_MODULE_CANDIDATES = (
    "visual", "vision_tower", "vision_model",
    "model.visual", "model.vision_tower", "model.vision_model",
)


# --------------------------------------------------------------------------
# Tower loading (language model never materialized)
# --------------------------------------------------------------------------

def _resolve_module(model: torch.nn.Module, dotted: str) -> torch.nn.Module | None:
    mod: torch.nn.Module = model
    for part in dotted.split("."):
        if not hasattr(mod, part):
            return None
        mod = getattr(mod, part)
    return mod if isinstance(mod, torch.nn.Module) else None


def find_vision_module(model: torch.nn.Module) -> Tuple[str, torch.nn.Module]:
    """Locate the vision tower inside a VLM by common attribute names."""
    for name in VISION_MODULE_CANDIDATES:
        mod = _resolve_module(model, name)
        if mod is not None:
            return name, mod
    for name, mod in model.named_modules():
        cls = type(mod).__name__.lower()
        if name.count(".") <= 2 and ("vision" in cls or "visual" in cls):
            return name, mod
    raise RuntimeError("Could not locate a vision tower; inspect model.named_modules().")


def _weight_files(model_id: str, revision: str | None) -> List[Tuple[str, List[str]]]:
    """[(filename, [tensor keys])] for the checkpoint's safetensors files."""
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    try:
        index_path = hf_hub_download(model_id, "model.safetensors.index.json",
                                     revision=revision)
        weight_map: Dict[str, str] = json.loads(Path(index_path).read_text())["weight_map"]
        by_file: Dict[str, List[str]] = {}
        for key, fname in weight_map.items():
            by_file.setdefault(fname, []).append(key)
        return sorted(by_file.items())
    except Exception:  # single-file checkpoint (no index)
        logger.info("No index json; falling back to model.safetensors")
        path = hf_hub_download(model_id, "model.safetensors", revision=revision)
        with safe_open(path, framework="pt") as f:
            return [("model.safetensors", list(f.keys()))]


def load_tower(model_id: str, device: str, dtype: torch.dtype,
               revision: str | None = None) -> Tuple[torch.nn.Module, str]:
    """Meta-device skeleton -> locate tower -> load only its shards.

    Parameters go on the meta device; buffers are REAL and computed by the
    module's __init__ (include_buffers=False). Non-persistent buffers such as
    rotary inv_freq tables are never stored in checkpoints, so materializing
    them via to_empty() would leave garbage and NaN outputs.
    """
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    from transformers import AutoConfig, AutoModelForImageTextToText

    config = AutoConfig.from_pretrained(model_id, revision=revision,
                                        trust_remote_code=True)
    with init_empty_weights(include_buffers=False):
        skeleton = AutoModelForImageTextToText.from_config(config,
                                                           trust_remote_code=True)
    prefix, tower = find_vision_module(skeleton)
    n_params = sum(p.numel() for p in tower.parameters())
    logger.info("Vision tower found at '%s' (%.1fM params)", prefix, n_params / 1e6)

    # Checkpoint keys may carry, lack, or drop a leading "model." relative to
    # the module path (e.g. attr 'model.visual' vs keys 'visual.*').
    cands = {prefix, f"model.{prefix}"}
    if prefix.startswith("model."):
        cands.add(prefix[len("model."):])
    prefixes = {f"{c}." for c in cands}
    loaded = 0
    for fname, keys in _weight_files(model_id, revision):
        wanted = [k for k in keys if any(k.startswith(p) for p in prefixes)]
        if not wanted:
            continue
        path = hf_hub_download(model_id, fname, revision=revision)
        with safe_open(path, framework="pt") as f:
            for key in wanted:
                stripped = next(key[len(p):] for p in prefixes if key.startswith(p))
                set_module_tensor_to_device(tower, stripped, device="cpu",
                                            value=f.get_tensor(key).to(dtype))
                loaded += 1
        logger.info("  loaded %d tower tensors from %s", len(wanted), fname)
    if loaded == 0:
        raise RuntimeError(f"No checkpoint tensors matched prefixes {prefixes}.")

    still_meta = [n for n, p in tower.named_parameters() if p.device.type == "meta"]
    if still_meta:
        raise RuntimeError(f"Parameters not found in checkpoint: {still_meta[:8]}")
    tower.eval().to(device=device, dtype=dtype)
    for p in tower.parameters():
        p.requires_grad_(False)
    return tower, prefix


# --------------------------------------------------------------------------
# Feature extraction (batch size 1: robust to variable-resolution towers)
# --------------------------------------------------------------------------

class TowerEncoder:
    """Runs single images through a VLM vision tower; returns patch-mean features."""

    def __init__(self, model_id: str, device: str | None, dtype: torch.dtype,
                 revision: str | None = None, max_pixels: int | None = None) -> None:
        from transformers import AutoImageProcessor

        if device is None:
            device = ("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available() else "cpu")
        self.device = device
        if (device == "cuda" and dtype == torch.bfloat16
                and torch.cuda.get_device_capability()[0] < 8):
            logger.warning("This GPU (e.g. Turing) has no native bf16 -- expect a "
                           "10-40x slowdown. Use --dtype float16 instead.")
        self.tower, self.prefix = load_tower(model_id, device, dtype, revision)
        proc_kwargs = {"max_pixels": max_pixels} if max_pixels else {}
        # Prefer the full AutoProcessor: some repos (e.g. Gemma4) keep image
        # processing in processor_config.json and emit extra tensors the
        # tower requires (pixel_position_ids etc.).
        from transformers import AutoProcessor

        try:
            self.processor = AutoProcessor.from_pretrained(
                model_id, revision=revision, trust_remote_code=True, **proc_kwargs)
        except Exception as exc:  # fall back to bare image processor
            logger.warning("AutoProcessor failed (%s); using AutoImageProcessor", exc)
            self.processor = AutoImageProcessor.from_pretrained(
                model_id, revision=revision, trust_remote_code=True, **proc_kwargs)
        self._logged_tokens = False
        self.forward_params = set(
            inspect.signature(self.tower.forward).parameters)
        logger.info("Tower forward accepts: %s", sorted(self.forward_params))
        logger.info("Using device=%s dtype=%s", self.device,
                    next(self.tower.parameters()).dtype)
        if self.device == "cpu":
            logger.warning("Running on CPU: expect ~10-30s per image.")

    def _run_processor(self, image: "object") -> dict:
        try:
            return self.processor(images=image, return_tensors="pt")
        except (TypeError, ValueError):
            # Some multimodal processors insist on a text argument.
            return self.processor(text="", images=image, return_tensors="pt")

    @torch.inference_mode()
    def encode_one(self, image: "object") -> np.ndarray:
        inputs = self._run_processor(image.convert("RGB"))
        pixel_values = inputs["pixel_values"]
        # Tiling processors (e.g. Idefics3) emit (batch, crops, C, H, W).
        if pixel_values.dim() == 5:
            pixel_values = pixel_values.flatten(0, 1)
        pixel_values = pixel_values.to(
            self.device, dtype=next(self.tower.parameters()).dtype)
        # Qwen-style towers take pre-patchified pixels as `hidden_states`.
        for arg_name in ("pixel_values", "hidden_states"):
            if arg_name in self.forward_params:
                kwargs = {arg_name: pixel_values}
                break
        else:
            raise RuntimeError(
                f"Tower forward has neither pixel_values nor hidden_states: "
                f"{sorted(self.forward_params)}")
        # Pass through every processor tensor the tower's forward asks for
        # (pixel_position_ids, patch_attention_mask, ...); keep integer dtypes.
        for name, value in inputs.items():
            if (name in self.forward_params and name not in kwargs
                    and isinstance(value, torch.Tensor)
                    and name not in ("input_ids", "attention_mask")):
                kwargs[name] = value.to(self.device)
        # Known processor-output -> forward-argument name mismatches.
        # Only REQUIRED parameters are aliased; optional ones (attention
        # masks etc.) have safe internal defaults and injecting them with
        # mismatched shapes does more harm than good.
        aliases = {
            "grid_thw": ("image_grid_thw",),
            "pixel_position_ids": ("image_position_ids", "position_ids"),
        }
        for param, sources in aliases.items():
            if param in self.forward_params and param not in kwargs:
                for src in sources:
                    if src in inputs:
                        kwargs[param] = inputs[src].to(self.device)
                        break

        sig = inspect.signature(self.tower.forward)
        missing = [n for n, p in sig.parameters.items()
                   if p.default is inspect.Parameter.empty
                   and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
                   and n not in kwargs]
        if missing:
            raise RuntimeError(
                f"Tower requires {missing} but the processor only produced "
                f"{sorted(inputs.keys())}. Inspect the model's processor to "
                f"see how these are constructed.")
        out = self.tower(**kwargs)
        hidden = getattr(out, "last_hidden_state", None)
        if hidden is None:
            hidden = out if isinstance(out, torch.Tensor) else out[0]
        # (crops, N, D) -> (crops*N, D); variable-resolution towers give (N, D).
        if hidden.dim() == 3:
            hidden = hidden.flatten(0, 1)
        if not self._logged_tokens:
            logger.info("    first image -> %d tokens x %d dims "
                        "(use --max-pixels to cap if token count is huge)",
                        hidden.shape[0], hidden.shape[1])
            self._logged_tokens = True
        feat = hidden.float().mean(dim=0).cpu().numpy()
        if not np.isfinite(feat).all():
            raise RuntimeError(
                "Non-finite features. If running float16, the tower likely "
                "overflows: use --dtype float32 (Turing) or bfloat16 (Ampere+).")
        return feat

    def encode(self, images: Iterable["object"], log_every: int = 25) -> Dict[str, np.ndarray]:
        feats: List[np.ndarray] = []
        for i, img in enumerate(images, 1):
            feats.append(self.encode_one(img))
            if i % log_every == 0:
                logger.info("    %d images encoded ...", i)
        feats = np.stack(feats)
        return {"patch_mean": feats, "pooled": feats}
