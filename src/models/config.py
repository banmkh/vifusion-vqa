from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelConfig:
    image_encoders: tuple[str, ...] = ("dino", "eva")
    fusion: str = "gated"
    text_model: str = "vinai/phobert-base"
    use_safetensors: bool = True
    local_files_only: bool = False
    image_weights: dict[str, str] = field(default_factory=dict)

    d_model: int = 768
    ffn_hidden: int = 2048
    num_heads: int = 8
    num_layers: int = 5
    num_att_layers: int = 4

    max_len: int = 27
    dropout: float = 0.3
