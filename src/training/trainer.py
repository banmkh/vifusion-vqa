from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim


@dataclass
class TrainConfig:
    epochs: int = 5
    lr: float = 1e-4
    weight_decay: float = 0.001


def train_one_epoch(model, loader, criterion, optimizer, device, max_len: int):
    model.train()
    total_loss = 0.0
    for _, batch in enumerate(loader):
        _, _, images, questions, answers = batch
        images = images.to(device)

        logits, targets = model(images, questions, answers, anno_ids=None, mask=True, max_len=max_len)
        loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(1, len(loader))


def evaluate_one_epoch(model, loader, criterion, device, max_len: int):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            _, _, images, questions, answers = batch
            images = images.to(device)
            logits, targets = model(images, questions, answers, anno_ids=None, mask=True, max_len=max_len)
            loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            total_loss += loss.item()
    return total_loss / max(1, len(loader))


def build_optimizer(model, cfg: TrainConfig):
    return optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)


def build_scheduler(optimizer, epochs: int):
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
