from __future__ import annotations

from pathlib import Path

from src.config import AppConfig


def test_app_config_resolve(tmp_path: Path):
    cfg = AppConfig().resolve(tmp_path)
    assert str(tmp_path) in cfg.data.train_csv
