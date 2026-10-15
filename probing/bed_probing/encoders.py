"""Vision-encoder registry for BedBench probing.

Each encoder yields two representations per image:
  - ``pooled``: the encoder's own pooled/CLS embedding
  - ``patch_mean``: mean over patch tokens

All encoders are frozen; inference only.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EncoderSpec:
    """Immutable description of one probe-able encoder."""

    alias: str
    hf_id: str
    family: str  # "siglip2" | "dinov2" | "clip"


ENCODER_REGISTRY: Dict[str, EncoderSpec] = {
    "siglip2": EncoderSpec("siglip2", "google/siglip2-so400m-patch14-384", "siglip2"),
    "siglip2-base": EncoderSpec("siglip2-base", "google/siglip2-base-patch16-224", "siglip2"),
    "dinov2": EncoderSpec("dinov2", "facebook/dinov2-large", "dinov2"),
    "clip": EncoderSpec("clip", "openai/clip-vit-large-patch14", "clip"),
    # Depth-supervised upper bound: DINOv2-L fine-tuned for depth estimation,
    # so (dinov2 vs depthanything) isolates the effect of 3D-aware training
    # on the same backbone.
    "depthanything": EncoderSpec(
        "depthanything", "depth-anything/Depth-Anything-V2-Large-hf", "depthanything"),
}


class FrozenEncoder:
    """Wraps a HF vision encoder and exposes batched feature extraction."""

    def __init__(self, spec: EncoderSpec, device: str | None = None) -> None:
        self.spec = spec
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model, self.processor = self._load()
        self.model.eval().to(self.device, dtype=self.dtype)
        for p in self.model.parameters():
            p.requires_grad_(False)
        logger.info("Loaded %s (%s) on %s", spec.alias, spec.hf_id, self.device)

    def _load(self) -> Tuple[torch.nn.Module, "object"]:
        from transformers import AutoImageProcessor, AutoModel

        processor = AutoImageProcessor.from_pretrained(self.spec.hf_id)
        if self.spec.family == "siglip2":
            model = AutoModel.from_pretrained(self.spec.hf_id).vision_model
        elif self.spec.family == "clip":
            from transformers import CLIPVisionModel

            model = CLIPVisionModel.from_pretrained(self.spec.hf_id)
        elif self.spec.family == "dinov2":
            model = AutoModel.from_pretrained(self.spec.hf_id)
        elif self.spec.family == "depthanything":
            from transformers import AutoModelForDepthEstimation

            model = AutoModelForDepthEstimation.from_pretrained(self.spec.hf_id).backbone
        else:
            raise ValueError(f"Unknown encoder family: {self.spec.family}")
        return model, processor

    def _split_outputs(self, out: "object") -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (pooled [B, D], patch tokens [B, N, D]) for the family."""
        if self.spec.family == "depthanything":
            # Deepest backbone stage; layout depends on reshape_hidden_states:
            # (B, C, h, w) spatial maps or (B, N, D) tokens with a CLS at 0.
            fmap = out.feature_maps[-1]
            if fmap.dim() == 4:
                patches = fmap.flatten(2).transpose(1, 2)
            else:
                patches = fmap[:, 1:]
            return patches.mean(dim=1), patches
        hidden = out.last_hidden_state
        if self.spec.family == "siglip2":
            # SigLIP has no CLS token; pooler is a learned attention pool.
            pooled = out.pooler_output
            patches = hidden
        else:
            # DINOv2 / CLIP: token 0 is CLS, the rest are patches.
            pooled = getattr(out, "pooler_output", None)
            if pooled is None:
                pooled = hidden[:, 0]
            patches = hidden[:, 1:]
        return pooled, patches

    @torch.inference_mode()
    def encode(self, images: List[Image.Image], batch_size: int = 16) -> Dict[str, np.ndarray]:
        """Encode PIL images into ``pooled`` and ``patch_mean`` feature matrices."""
        pooled_chunks: List[np.ndarray] = []
        patch_chunks: List[np.ndarray] = []
        for start in range(0, len(images), batch_size):
            batch = [img.convert("RGB") for img in images[start : start + batch_size]]
            inputs = self.processor(images=batch, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(self.device, dtype=self.dtype)
            out = self.model(pixel_values=pixel_values)
            pooled, patches = self._split_outputs(out)
            pooled_chunks.append(pooled.float().cpu().numpy())
            patch_chunks.append(patches.float().mean(dim=1).cpu().numpy())
        return {
            "pooled": np.concatenate(pooled_chunks, axis=0),
            "patch_mean": np.concatenate(patch_chunks, axis=0),
        }


def build_encoder(alias: str, device: str | None = None) -> FrozenEncoder:
    """Factory: instantiate a frozen encoder by registry alias."""
    if alias not in ENCODER_REGISTRY:
        raise KeyError(f"Unknown encoder '{alias}'. Choices: {sorted(ENCODER_REGISTRY)}")
    return FrozenEncoder(ENCODER_REGISTRY[alias], device=device)
