from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch

from src.data import DataConfig
from src.models import ModelConfig
from src.training import TrainConfig


@dataclass(frozen=True)
class AppConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    seed: int = 1105

    def resolve(self, root: str | Path = ".") -> "AppConfig":
        return AppConfig(
            data=self.data.resolve(root),
            model=self.model,
            train=self.train,
            seed=self.seed,
        )


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
