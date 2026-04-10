from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


@dataclass(frozen=True)
class DataConfig:
    limit: int = 10**2

    train_images_path: str = "data/raw/training-images"
    test_images_path: str = "data/raw/test-images"
    dev_images_path: str = "data/raw/dev-images"

    train_json_path: str = "data/raw/training-annotations.json"
    test_json_path: str = "data/raw/test-annotations.json"
    dev_json_path: str = "data/raw/dev-annotations.json"

    train_csv: str = "data/processed/train.csv"
    test_csv: str = "data/processed/test.csv"
    dev_csv: str = "data/processed/dev.csv"

    seed: int = 1105
    max_len: int = 27
    train_batch_size: int = 16
    num_workers: int = os.cpu_count() or 0

    def resolve(self, root: str | Path = ".") -> "DataConfig":
        root = Path(root)
        return DataConfig(
            limit=self.limit,
            train_images_path=str(root / self.train_images_path),
            test_images_path=str(root / self.test_images_path),
            dev_images_path=str(root / self.dev_images_path),
            train_json_path=str(root / self.train_json_path),
            test_json_path=str(root / self.test_json_path),
            dev_json_path=str(root / self.dev_json_path),
            train_csv=str(root / self.train_csv),
            test_csv=str(root / self.test_csv),
            dev_csv=str(root / self.dev_csv),
            seed=self.seed,
            max_len=self.max_len,
            train_batch_size=self.train_batch_size,
            num_workers=self.num_workers,
        )
