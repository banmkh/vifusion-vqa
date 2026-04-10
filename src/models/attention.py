from __future__ import annotations

import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, d: int = 768, num_heads: int = 8, dropout: float = 0.5):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=d, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.dropout = nn.Dropout(p=dropout) if dropout else None
        self.layer_norm = nn.LayerNorm(d)

    def forward(self, vi: torch.Tensor, vq: torch.Tensor) -> torch.Tensor:
        combined_input = torch.cat([vq, vi], dim=1)
        attn_output, _ = self.attention(combined_input, combined_input, combined_input)

        if self.dropout:
            attn_output = self.dropout(attn_output)
        attn_output = self.layer_norm(attn_output)

        vi_attended = attn_output[:, 1:, :]
        u = vi_attended.sum(dim=1) + vq.squeeze(1)
        return u
