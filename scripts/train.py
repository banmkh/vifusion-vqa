from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

from src.data import DataConfig, build_image_transform, build_dataloaders, normalize_qa_df
from src.models import ModelConfig, VQAModel
from src.training import (
    TrainConfig,
    train_one_epoch,
    evaluate_one_epoch,
    evaluate_benchmark_epoch,
    build_optimizer,
    build_scheduler,
)


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
        choices=["gated", "attention", "linear","cross-attention"],
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
    parser.add_argument(
        "--benchmark-interval",
        type=int,
        default=2,
        help="Compute benchmark metrics every N epochs (default: 2)",
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


def print_model_summary(
    model: nn.Module,
    train_loader,
    device: str,
    batch_size: int,
    epochs: int,
    optimizer: torch.optim.Optimizer,
) -> None:
    """In parameter count, memory footprint, và ước tính training time."""
    # --- Parameter count ---
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable

    # --- Memory footprint ---
    # fp32: 4 bytes/param; gradient: thêm 4 bytes; Adam: thêm 8 bytes (m + v)
    param_mb = total * 4 / 1024 ** 2
    grad_mb = trainable * 4 / 1024 ** 2
    n_optimizer_states = sum(
        p.numel() for group in optimizer.param_groups for p in group["params"] if p.requires_grad
    )
    adam_mb = n_optimizer_states * 8 / 1024 ** 2
    total_mem_mb = param_mb + grad_mb + adam_mb

    # --- GPU memory (nếu có) ---
    gpu_alloc_mb = 0.0
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
        gpu_alloc_mb = torch.cuda.memory_allocated() / 1024 ** 2

    # --- Ước tính training time: đo 3 batch forward ---
    model.train()
    times: list[float] = []
    criterion_tmp = nn.CrossEntropyLoss(ignore_index=1)
    sample_batches = []
    for i, batch in enumerate(train_loader):
        if i >= 3:
            break
        sample_batches.append(batch)

    for batch in sample_batches:
        _, _, images, questions, answers = batch
        images = images.to(device)
        t0 = time.perf_counter()
        logits, ans_vocab = model(images, questions, answers, max_len=27)
        B, T, V = logits.shape
        loss = criterion_tmp(logits.reshape(B * T, V), ans_vocab.reshape(B * T))
        loss.backward()
        optimizer.zero_grad()
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    avg_batch_s = sum(times) / len(times)
    steps_per_epoch = len(train_loader)
    epoch_s = avg_batch_s * steps_per_epoch
    total_s = epoch_s * epochs

    def fmt_time(s: float) -> str:
        if s < 60:
            return f"{s:.1f}s"
        if s < 3600:
            return f"{s/60:.1f}min"
        return f"{s/3600:.1f}h"

    # --- In ra ---
    sep = "=" * 60
    print(sep)
    print("  MODEL SUMMARY TRƯỚC KHI TRAIN")
    print(sep)
    print(f"  Parameters")
    print(f"    Total      : {total:>15,}")
    print(f"    Trainable  : {trainable:>15,}")
    print(f"    Frozen     : {frozen:>15,}")
    print()
    print(f"  Memory footprint (ước tính)")
    print(f"    Params (fp32)  : {param_mb:>8.1f} MB")
    print(f"    Gradients      : {grad_mb:>8.1f} MB")
    print(f"    Optimizer (Adam): {adam_mb:>7.1f} MB")
    print(f"    Tổng ước tính  : {total_mem_mb:>8.1f} MB  (~{total_mem_mb/1024:.2f} GB)")
    if gpu_alloc_mb > 0:
        print(f"    GPU đang dùng  : {gpu_alloc_mb:>8.1f} MB")
    print()
    print(f"  Training time (ước tính, batch_size={batch_size})")
    print(f"    Avg / batch    : {avg_batch_s*1000:>8.1f} ms")
    print(f"    Steps / epoch  : {steps_per_epoch:>8,}")
    print(f"    Thời gian / epoch: {fmt_time(epoch_s):>8}")
    print(f"    Tổng {epochs} epoch  : {fmt_time(total_s):>8}")
    print(sep)
    print()


def print_sample_predictions(model, loader, device: str, max_len: int, n: int = 5) -> None:
    """In n sample (câu hỏi / ground truth / dự đoán) lấy từ batch đầu tiên của loader."""
    model.eval()
    tokenizer = model.ans_model.tokenizer

    batch = next(iter(loader))
    _, _, images, questions, answers = batch

    images = images[:n].to(device)
    questions = questions[:n]
    answers = answers[:n]

    with torch.no_grad():
        generated_ids = model.generate(images, questions, max_len=max_len)

    print("-" * 72)
    for i in range(len(questions)):
        predicted = tokenizer.decode(generated_ids[i], skip_special_tokens=True).strip()
        raw_ids = generated_ids[i].tolist()
        print(f"  [{i + 1}] Câu hỏi  : {questions[i]}")
        print(f"       Ground truth: {answers[i]}")
        print(f"       Dự đoán     : {predicted if predicted else '(rỗng)'}")
        print(f"       Raw IDs     : {raw_ids[:12]}")
        print()
    print("-" * 72)


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

    print_model_summary(model, train_loader, device, batch_size, train_cfg.epochs, optimizer)

    for epoch in range(train_cfg.epochs):
        # Scheduled Sampling: bắt đầu 100% teacher forcing, giảm dần đến 50%
        # Nửa đầu: pure teacher forcing (model học cơ bản)
        # Nửa sau: giảm dần để model quen với input tự generate
        progress = epoch / max(1, train_cfg.epochs - 1)
        tf_ratio = max(0.5, 1.0 - 0.5 * progress)

        avg_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            max_len=data_cfg.max_len, teacher_forcing_ratio=tf_ratio,
        )
        scheduler.step()
        print(f"Epoch {epoch + 1}/{train_cfg.epochs} - train loss: {avg_loss:.4f} (tf_ratio={tf_ratio:.2f})")

        avg_dev = evaluate_one_epoch(
            model, dev_loader, criterion, device, max_len=data_cfg.max_len
        )
        print(f"Epoch {epoch + 1}/{train_cfg.epochs} - dev loss: {avg_dev:.4f}")

        # Compute benchmark metrics every N epochs (starting from epoch 1)
        if (epoch + 1) % args.benchmark_interval == 0 or epoch == train_cfg.epochs - 1:
            print(f"\nComputing benchmark metrics for epoch {epoch + 1}...")
            benchmark_metrics = evaluate_benchmark_epoch(
                model, dev_loader, device, max_len=data_cfg.max_len
            )
            print(f"Epoch {epoch + 1}/{train_cfg.epochs} - Benchmark Results:")
            print(f"  - Precision:  {benchmark_metrics['precision']:.4f}")
            print(f"  - Recall:     {benchmark_metrics['recall']:.4f}")
            print(f"  - F1-Score:   {benchmark_metrics['f1']:.4f}")
            print(f"  - Accuracy:   {benchmark_metrics['accuracy']:.4f}")
            print(f"  - BLEU-1:     {benchmark_metrics['bleu_1']:.4f}")
            print(f"  - BLEU-2:     {benchmark_metrics['bleu_2']:.4f}")
            print(f"  - BLEU-3:     {benchmark_metrics['bleu_3']:.4f}")
            print(f"  - BLEU-4:     {benchmark_metrics['bleu_4']:.4f}")
            print(f"  - ROUGE-L:    {benchmark_metrics['rouge']:.4f}")
            print(f"  - CIDER:      {benchmark_metrics['cider']:.4f}\n")

            print(f"Sample predictions (epoch {epoch + 1}):")
            print_sample_predictions(model, dev_loader, device, max_len=data_cfg.max_len)

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
