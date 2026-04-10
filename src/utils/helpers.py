from __future__ import annotations

from typing import Mapping

import torch


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_vocab_swap(vocab: Mapping[str, int]) -> dict[int, str]:
    return {value: key for key, value in vocab.items()}
