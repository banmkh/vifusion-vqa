from __future__ import annotations
import torch
import torch.nn as nn
from .image_factory import ImageEncoderFactory

class GatedFusion(nn.Module):
    def __init__(self, embedding_dim: int = 768):
        super().__init__()
        # Gate dựa trên cả 2 đầu vào để quyết định trọng số
        self.gate = nn.Linear(embedding_dim * 2, 2)
        self.proj = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, emb1: torch.Tensor, emb2: torch.Tensor) -> torch.Tensor:
        # Giả định đầu vào (B, D)
        combined = torch.cat([emb1, emb2], dim=-1)
        gate = torch.softmax(self.gate(combined), dim=-1)
        
        # Weighted sum
        fused = gate[..., 0:1] * emb1 + gate[..., 1:2] * emb2
        return self.proj(fused)

class CrossAttentionFusion(nn.Module):
    def __init__(self, dim=768, num_heads=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)

    def forward(self, emb1, emb2):
        # Đảm bảo đầu vào là (B, 1, D)
        e1 = emb1.unsqueeze(1) if emb1.dim() == 2 else emb1
        e2 = emb2.unsqueeze(1) if emb2.dim() == 2 else emb2

        out1, _ = self.attn(e1, e2, e2) # e1 query vào e2
        out2, _ = self.attn(e2, e1, e1) # e2 query vào e1

        return (out1 + out2) / 2

class ImageEmbedding(nn.Module):
    def __init__(
        self,
        encoders: list[str] | tuple[str, ...] = ("dino", "beit"),
        fusion: str = "gated",
        embedding_dim: int = 768,
        device: str = "cuda",
        image_weights: dict[str, str] | None = None,
    ):
        super().__init__()
        factory = ImageEncoderFactory(embedding_dim, device, image_weights)
        self.encoders = nn.ModuleList([factory.get_encoder(name) for name in encoders])
        self.fusion_type = fusion
        
        if len(encoders) == 1:
            fusion = "linear"

        if fusion == "linear":
            self.fusion = nn.Linear(len(encoders) * embedding_dim, embedding_dim)
        elif fusion == "attention":
            self.fusion = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=8, batch_first=True)
        elif fusion == "gated":
            if len(encoders) != 2:
                raise ValueError("Gated fusion chỉ hỗ trợ 2 encoder")
            self.fusion = GatedFusion(embedding_dim)
        elif fusion == "cross-attention":
            if len(encoders) != 2:
                raise ValueError("Cross-attention chỉ hỗ trợ 2 encoder")
            self.fusion = CrossAttentionFusion(embedding_dim)
        else:
            raise ValueError(f"Fusion type '{fusion}' is not supported.")

    def forward(self, image: torch.Tensor, image_ids=None):
        embeddings = [encoder(image) for encoder in self.encoders]
        
        # Đảm bảo tất cả embeddings là (B, D) trước khi fusion
        # (Giả sử encoder trả về pool output 2D)
        
        if len(embeddings) > 1:
            if self.fusion_type == "linear":
                fused = self.fusion(torch.cat(embeddings, dim=-1)).unsqueeze(1)
            elif self.fusion_type == "attention":
                # Stack để tạo chiều sequence: (B, num_encoders, D)
                stacked = torch.stack(embeddings, dim=1)
                fused, _ = self.fusion(stacked, stacked, stacked)
                fused = fused.mean(dim=1, keepdim=True)
            elif self.fusion_type == "gated":
                fused = self.fusion(embeddings[0], embeddings[1]).unsqueeze(1)
            elif self.fusion_type == "cross-attention":
                fused = self.fusion(embeddings[0], embeddings[1])
        else:
            fused = embeddings[0].unsqueeze(1)

        # Trả về (B, 1, D) đồng nhất
        return fused, image_ids