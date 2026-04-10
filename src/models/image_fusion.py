from __future__ import annotations

import torch
import torch.nn as nn

from .image_factory import ImageEncoderFactory


class GatedFusion(nn.Module):
    def __init__(self, embedding_dim: int = 768):
        super().__init__()
        self.gate = nn.Linear(embedding_dim * 2, 2)
        self.proj = nn.Linear(embedding_dim * 3, embedding_dim)

    def forward(self, emb1: torch.Tensor, emb2: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([emb1, emb2], dim=-1)
        gate = torch.softmax(self.gate(combined), dim=-1)
        fused = gate[..., 0:1] * emb1 + gate[..., 1:2] * emb2
        return self.proj(torch.cat([fused, combined], dim=-1))


class ImageEmbedding(nn.Module):
    def __init__(
        self,
        encoders: list[str] | tuple[str, ...] = ("dino", "beit"),
        fusion: str = "gated",
        embedding_dim: int = 768,
        device: str = "cuda",
    ):
        super().__init__()
        factory = ImageEncoderFactory(embedding_dim, device)
        self.encoders = nn.ModuleList([factory.get_encoder(name) for name in encoders])

        if fusion == "linear":
            self.fusion = nn.Linear(len(encoders) * embedding_dim, embedding_dim)
        elif fusion == "attention":
            self.fusion = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=8, batch_first=True)
        elif fusion == "gated":
            if len(encoders) != 2:
                raise ValueError("Gated fusion chỉ hỗ trợ 2 encoder")
            self.fusion = GatedFusion(embedding_dim)
        else:
            raise ValueError("fusion must be linear, attention or gated")

        self.fusion_type = fusion

    def forward(self, image: torch.Tensor, image_ids=None):
        embeddings = []
        for encoder in self.encoders:
            emb = encoder(image)
            embeddings.append(emb.unsqueeze(1))

        if len(embeddings) > 1:
            if self.fusion_type == "linear":
                fused = self.fusion(torch.cat(embeddings, dim=-1))
            elif self.fusion_type == "attention":
                combined = torch.cat(embeddings, dim=1)
                fused, _ = self.fusion(combined, combined, combined)
                fused = fused.mean(dim=1, keepdim=True)
            elif self.fusion_type == "gated":
                fused = self.fusion(embeddings[0], embeddings[1])
        else:
            fused = embeddings[0]

        return fused, image_ids
