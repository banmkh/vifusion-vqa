from __future__ import annotations

import random
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch


def decode_prediction(prediction: str) -> str:
    tokens = prediction.split()
    tokens = [token for token in tokens if token not in ["<s>", "</s>"]]
    return " ".join(tokens)


def display_image_with_text(image, question: str, ground_truth: str, predicted: str, ax, is_correct=None):
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
        if image.shape[0] in [1, 3]:
            image = np.transpose(image, (1, 2, 0))

    alpha = 0.35
    image = image * alpha + (1 - alpha)
    image = np.clip(image, 0, 1)

    ax.imshow(image)
    ax.axis("off")

    edge_color = "green" if is_correct else "red" if is_correct is not None else "black"
    text = f"Q: {question}\nGT: {ground_truth}\nPred: {predicted}"
    ax.text(
        0.5,
        -0.1,
        text,
        fontsize=10,
        color="black",
        ha="center",
        va="top",
        transform=ax.transAxes,
        bbox=dict(facecolor="none", edgecolor=edge_color, boxstyle="round,pad=0.5", lw=2),
    )


def display_samples_grid(samples: Iterable[dict], model, device, n: int = 20):
    samples = list(samples)
    random_samples = random.sample(samples, min(n, len(samples)))

    correct_samples = []
    incorrect_samples = []

    with torch.no_grad():
        for sample in random_samples:
            image = sample["image"].unsqueeze(0).to(device)
            question = [sample["question"]]
            answer = [sample["answer"]]
            anno = [sample["anno_id"]]

            pred_token, _ = model(image, question, answer, anno_ids=anno, mask=True)
            pred_ids = torch.argmax(pred_token[0], dim=1)
            pred_sentence = " ".join([str(idx.item()) for idx in pred_ids])
            pred_sentence = decode_prediction(pred_sentence)

            sample_out = {**sample, "pred": pred_sentence}
            if pred_sentence.strip() == sample["answer"].strip():
                correct_samples.append(sample_out)
            else:
                incorrect_samples.append(sample_out)

    all_samples = correct_samples[:10] + incorrect_samples[:10]
    fig, axes = plt.subplots(4, 5, figsize=(20, 16))
    axes = axes.flatten()

    for ax, sample in zip(axes, all_samples):
        is_correct = sample["pred"].strip() == sample["answer"].strip()
        display_image_with_text(
            sample["image"],
            sample["question"],
            sample["answer"],
            sample["pred"],
            ax,
            is_correct=is_correct,
        )

    plt.tight_layout()
    plt.show()
