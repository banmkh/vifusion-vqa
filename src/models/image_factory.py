from __future__ import annotations

from .image_backbones import (
    DinoBackbone,
    SwinBackbone,
    ConvNeXtBackbone,
    BeitBackbone,
    EvaBackbone,
    SigLIPBackbone,
)


class ImageEncoderFactory:
    def __init__(self, embedding_dim: int = 768, device: str = "cuda", image_weights: dict[str, str] | None = None) -> None:
        self.embedding_dim = embedding_dim
        self.device = device
        self.image_weights = image_weights or {}

    def get_encoder(self, name: str):
        name = name.lower()
        if name == "dino":
            return DinoBackbone(self.embedding_dim, self.device, self.image_weights.get("dino"))
        if name == "swin":
            return SwinBackbone(self.embedding_dim, self.device, self.image_weights.get("swin"))
        if name == "convnext":
            return ConvNeXtBackbone(self.embedding_dim, self.device, self.image_weights.get("convnext"))
        if name == "beit":
            return BeitBackbone(self.device)
        if name == "eva":
            return EvaBackbone(device=self.device, embedding_dim=self.embedding_dim, weights_path=self.image_weights.get("eva"))
        if name == "siglip":
            return SigLIPBackbone(self.device)
        raise ValueError(f"Unknown encoder: {name}")
