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
    def __init__(self, embedding_dim: int = 768, device: str = "cuda") -> None:
        self.embedding_dim = embedding_dim
        self.device = device

    def get_encoder(self, name: str):
        name = name.lower()
        if name == "dino":
            return DinoBackbone(self.embedding_dim, self.device)
        if name == "swin":
            return SwinBackbone(self.embedding_dim, self.device)
        if name == "convnext":
            return ConvNeXtBackbone(self.embedding_dim, self.device)
        if name == "beit":
            return BeitBackbone(self.device)
        if name == "eva":
            return EvaBackbone(device=self.device, embedding_dim=self.embedding_dim)
        if name == "siglip":
            return SigLIPBackbone(self.device)
        raise ValueError(f"Unknown encoder: {name}")
