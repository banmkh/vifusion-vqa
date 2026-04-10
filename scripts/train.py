from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

from src.data import DataConfig, build_image_transform, build_dataloaders, normalize_qa_df
from src.models import ModelConfig, VQAModel
from src.training import TrainConfig, train_one_epoch, evaluate_one_epoch, build_optimizer, build_scheduler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train VQA model")
    parser.add_argument("--root", type=str, default=".", help="Project root path")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--save", type=str, default="outputs/checkpoints/ViFusion.pt")
    parser.add_argument("--skip-normalize", action="store_true")
    parser.add_argument(
        "--image-encoders",
        type=str,
        default=None,
        help="Comma-separated list of image encoders, e.g. dino,eva,beit",
    )
    parser.add_argument(
        "--fusion",
        type=str,
        default=None,
        choices=["gated", "attention", "linear"],
        help="Fusion method for image encoders",
    )
    parser.add_argument(
        "--encoder-weights",
        type=str,
        default=None,
        help="Comma-separated encoder weights: dino=/path/a.safetensors,eva=/path/b.safetensors",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Only load model weights from local files (no downloads)",
    )
    return parser.parse_args()


def parse_encoder_weights(arg: str | None) -> dict[str, str]:
    if not arg:
        return {}
    items = [x.strip() for x in arg.split(",") if x.strip()]
    out: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError("encoder-weights must be in name=path format")
        name, path = item.split("=", 1)
        out[name.strip().lower()] = path.strip()
    return out


def main() -> None:
    args = parse_args()
    root = Path(args.root)

    data_cfg = DataConfig().resolve(root)
    model_cfg = ModelConfig()
    train_cfg = TrainConfig(epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay)

    if args.image_encoders:
        image_encoders = [e.strip() for e in args.image_encoders.split(",") if e.strip()]
    else:
        image_encoders = list(model_cfg.image_encoders)

    fusion = args.fusion or model_cfg.fusion
    image_weights = parse_encoder_weights(args.encoder_weights) or model_cfg.image_weights

    batch_size = args.batch_size or data_cfg.train_batch_size
    num_workers = args.num_workers if args.num_workers is not None else data_cfg.num_workers

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    if not Path(data_cfg.train_csv).exists():
        raise FileNotFoundError(
            f"{data_cfg.train_csv} not found. Run `python -m scripts.preprocess --root .` first."
        )

    df_train = pd.read_csv(data_cfg.train_csv)
    df_dev = pd.read_csv(data_cfg.dev_csv) if Path(data_cfg.dev_csv).exists() else df_train.copy()

    if not args.skip_normalize:
        df_train = normalize_qa_df(df_train)
        df_dev = normalize_qa_df(df_dev)

    transform = build_image_transform()
    train_loader, _, dev_loader = build_dataloaders(
        df_train,
        df_train,
        df_dev,
        transform,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    model = VQAModel(
        vocab_size=None,
        text_model=model_cfg.text_model,
        image_encoders=image_encoders,
        fusion=fusion,
        image_weights=image_weights,
        use_safetensors=model_cfg.use_safetensors,
        local_files_only=args.local_files_only or model_cfg.local_files_only,
        d_model=model_cfg.d_model,
        ffn_hidden=model_cfg.ffn_hidden,
        num_heads=model_cfg.num_heads,
        num_layers=model_cfg.num_layers,
        num_att_layers=model_cfg.num_att_layers,
        dropout=model_cfg.dropout,
        device=device,
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=1)
    optimizer = build_optimizer(model, train_cfg)
    scheduler = build_scheduler(optimizer, train_cfg.epochs)

    for epoch in range(train_cfg.epochs):
        avg_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, max_len=data_cfg.max_len
        )
        scheduler.step()
        print(f"Epoch {epoch + 1}/{train_cfg.epochs} - train loss: {avg_loss:.4f}")

        avg_dev = evaluate_one_epoch(
            model, dev_loader, criterion, device, max_len=data_cfg.max_len
        )
        print(f"Epoch {epoch + 1}/{train_cfg.epochs} - dev loss: {avg_dev:.4f}")

    save_path = Path(args.save)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "model_cfg": {
                **model_cfg.__dict__,
                "image_encoders": image_encoders,
                "fusion": fusion,
                "image_weights": image_weights,
                "local_files_only": args.local_files_only or model_cfg.local_files_only,
            },
            "data_cfg": data_cfg.__dict__,
            "train_cfg": train_cfg.__dict__,
        },
        save_path,
    )
    print(f"Saved checkpoint to {save_path}")


if __name__ == "__main__":
    main()
