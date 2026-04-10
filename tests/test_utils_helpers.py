from __future__ import annotations

import torch

from src.utils.helpers import count_parameters, build_vocab_swap


def test_count_parameters():
    model = torch.nn.Linear(4, 3)
    assert count_parameters(model) == (4 * 3 + 3)


def test_build_vocab_swap():
    vocab = {"a": 0, "b": 1}
    swap = build_vocab_swap(vocab)
    assert swap[0] == "a"
    assert swap[1] == "b"
