from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.utils.visualize import decode_prediction, display_image_with_text


def test_decode_prediction_strips_tokens():
    assert decode_prediction("<s> xin chao </s>") == "xin chao"


def test_display_image_with_text_runs():
    fig, ax = plt.subplots(1, 1)
    image = torch.zeros(3, 10, 10)
    display_image_with_text(image, "Q", "A", "P", ax, is_correct=True)
    plt.close(fig)
