from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data import DataConfig, convert_json_to_csv, normalize_qa_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess VQA data into CSV files.")
    parser.add_argument("--root", type=str, default=".", help="Project root path")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples")
    parser.add_argument("--skip-normalize", action="store_true", help="Skip text normalization")
    return parser.parse_args()


def preprocess_split(json_path: str, images_path: str, csv_path: str, limit: int | None, normalize: bool) -> None:
    convert_json_to_csv(json_path, images_path, csv_path, limit=limit)
    if not normalize:
        return
    df = pd.read_csv(csv_path)
    df = normalize_qa_df(df)
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)


def main() -> None:
    args = parse_args()
    cfg = DataConfig().resolve(args.root)
    limit = args.limit if args.limit is not None else cfg.limit
    normalize = not args.skip_normalize

    preprocess_split(cfg.train_json_path, cfg.train_images_path, cfg.train_csv, limit, normalize)
    preprocess_split(cfg.test_json_path, cfg.test_images_path, cfg.test_csv, limit, normalize)
    preprocess_split(cfg.dev_json_path, cfg.dev_images_path, cfg.dev_csv, limit, normalize)


if __name__ == "__main__":
    main()
