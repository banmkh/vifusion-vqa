from __future__ import annotations

import torch
import torch.nn as nn

from src.training import TrainConfig, train_one_epoch, evaluate_one_epoch, build_optimizer, build_scheduler


class DummyModel(torch.nn.Module):
    def __init__(self, vocab_size=11, d_model=8):
        super().__init__()
        self.linear = torch.nn.Linear(d_model, vocab_size)

    def forward(self, images, questions, answers, anno_ids=None, mask=True, max_len=5):
        batch = images.size(0)
        x = torch.zeros(batch, max_len, self.linear.in_features)
        logits = self.linear(x)
        targets = torch.zeros(batch, max_len, dtype=torch.long)
        return logits, targets


def make_loader(batch_size=2, max_len=5):
    images = torch.randn(batch_size, 3, 8, 8)
    questions = ["q"] * batch_size
    answers = ["a"] * batch_size
    batch = (None, None, images, questions, answers)
    return [batch]


def test_train_and_eval_one_epoch():
    model = DummyModel()
    loader = make_loader()
    criterion = nn.CrossEntropyLoss()
    cfg = TrainConfig(epochs=1, lr=1e-3, weight_decay=0.0)
    optimizer = build_optimizer(model, cfg)

    train_loss = train_one_epoch(model, loader, criterion, optimizer, device="cpu", max_len=5)
    eval_loss = evaluate_one_epoch(model, loader, criterion, device="cpu", max_len=5)

    assert train_loss >= 0.0
    assert eval_loss >= 0.0


def test_build_scheduler_runs():
    model = DummyModel()
    cfg = TrainConfig(epochs=2, lr=1e-3, weight_decay=0.0)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg.epochs)
    scheduler.step()
