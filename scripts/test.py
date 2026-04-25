from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch

from src.data import DataConfig, build_image_transform, build_dataloaders, normalize_qa_df
from src.models import ModelConfig, VQAModel
from src.training import evaluate_benchmark_epoch
from src.utils import build_vocab_swap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test VQA model from checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint")
    parser.add_argument("--root", type=str, default=".", help="Project root path")
    parser.add_argument("--split", type=str, default="test", choices=["train", "dev", "test"],
                        help="Which data split to evaluate on")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--samples", type=int, default=10, help="Number of sample predictions to print")
    parser.add_argument("--output-json", type=str, default=None,
                        help="Optional path to save full predictions as JSON")
    parser.add_argument("--skip-normalize", action="store_true")
    return parser.parse_args()


def load_model_from_checkpoint(ckpt: dict, device: str) -> VQAModel:
    mcfg = ckpt["model_cfg"]
    model = VQAModel(
        vocab_size=None,
        text_model=mcfg.get("text_model", "vinai/phobert-base"),
        image_encoders=mcfg.get("image_encoders", ["dino", "eva"]),
        fusion=mcfg.get("fusion", "gated"),
        image_weights=mcfg.get("image_weights", {}),
        use_safetensors=mcfg.get("use_safetensors", True),
        local_files_only=mcfg.get("local_files_only", False),
        d_model=mcfg.get("d_model", 768),
        ffn_hidden=mcfg.get("ffn_hidden", 2048),
        num_heads=mcfg.get("num_heads", 8),
        num_layers=mcfg.get("num_layers", 5),
        num_att_layers=mcfg.get("num_att_layers", 4),
        dropout=mcfg.get("dropout", 0.3),
        device=device,
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    return model


def print_samples(model, loader, device: str, max_len: int, n: int) -> None:
    model.eval()
    tokenizer = model.ans_model.tokenizer

    batch = next(iter(loader))
    _, _, images, questions, answers = batch
    n = min(n, len(questions))

    images = images[:n].to(device)
    questions = questions[:n]
    answers = answers[:n]

    with torch.no_grad():
        generated_ids = model.generate(images, questions, max_len=max_len)

    print("=" * 72)
    print("SAMPLE PREDICTIONS")
    print("=" * 72)
    for i in range(n):
        predicted = tokenizer.decode(generated_ids[i], skip_special_tokens=True).strip()
        print(f"  [{i + 1}] Câu hỏi    : {questions[i]}")
        print(f"       Ground truth: {answers[i]}")
        print(f"       Dự đoán     : {predicted if predicted else '(rỗng)'}")
        print()
    print("=" * 72)


def collect_predictions(model, loader, device: str, max_len: int) -> list[dict]:
    model.eval()
    tokenizer = model.ans_model.tokenizer
    records = []
    with torch.no_grad():
        for batch in loader:
            anno_ids, image_ids, images, questions, answers = batch
            images = images.to(device)
            generated_ids = model.generate(images, questions, max_len=max_len)
            for i in range(len(questions)):
                predicted = tokenizer.decode(generated_ids[i], skip_special_tokens=True).strip()
                records.append({
                    "anno_id": anno_ids[i] if not isinstance(anno_ids[i], torch.Tensor) else anno_ids[i].item(),
                    "image_id": image_ids[i] if not isinstance(image_ids[i], torch.Tensor) else image_ids[i].item(),
                    "question": questions[i],
                    "ground_truth": answers[i],
                    "prediction": predicted,
                })
    return records


def print_benchmark(metrics: dict) -> None:
    print("\n" + "=" * 72)
    print("BENCHMARK RESULTS")
    print("=" * 72)
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1-Score  : {metrics['f1']:.4f}")
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  BLEU-1    : {metrics['bleu_1']:.4f}")
    print(f"  BLEU-2    : {metrics['bleu_2']:.4f}")
    print(f"  BLEU-3    : {metrics['bleu_3']:.4f}")
    print(f"  BLEU-4    : {metrics['bleu_4']:.4f}")
    print(f"  ROUGE-L   : {metrics['rouge']:.4f}")
    print(f"  CIDEr     : {metrics['cider']:.4f}")
    print("=" * 72)


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    ckpt_path = Path(args.checkpoint)

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data_cfg = DataConfig().resolve(root)
    saved_data_cfg = ckpt.get("data_cfg", {})
    max_len = saved_data_cfg.get("max_len", data_cfg.max_len)

    csv_map = {"train": data_cfg.train_csv, "dev": data_cfg.dev_csv, "test": data_cfg.test_csv}
    csv_path = csv_map[args.split]
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"{csv_path} not found. Run scripts/preprocess.py first.")

    df = pd.read_csv(csv_path)
    if not args.skip_normalize:
        from src.data import normalize_qa_df
        df = normalize_qa_df(df)

    print(f"Evaluating on '{args.split}' split: {len(df)} samples")

    batch_size = args.batch_size or saved_data_cfg.get("train_batch_size", 16)
    num_workers = args.num_workers if args.num_workers is not None else saved_data_cfg.get("num_workers", 0)

    transform = build_image_transform()
    _, _, loader = build_dataloaders(
        df, df, df, transform, batch_size=batch_size, num_workers=num_workers
    )

    print("Building model from checkpoint config...")
    model = load_model_from_checkpoint(ckpt, device)
    model.eval()

    mcfg = ckpt["model_cfg"]
    print(f"  Encoders : {mcfg.get('image_encoders')}")
    print(f"  Fusion   : {mcfg.get('fusion')}")
    print(f"  Text     : {mcfg.get('text_model')}")
    print(f"  Max len  : {max_len}")

    if args.samples > 0:
        print_samples(model, loader, device, max_len, args.samples)

    print("\nRunning benchmark evaluation...")
    t0 = time.time()
    metrics = evaluate_benchmark_epoch(model, loader, device, max_len=max_len)
    elapsed = time.time() - t0
    print(f"Evaluation time: {elapsed:.1f}s")

    print_benchmark(metrics)

    if args.output_json:
        print(f"\nCollecting full predictions for JSON output...")
        records = collect_predictions(model, loader, device, max_len)
        out = {"split": args.split, "checkpoint": str(ckpt_path), "metrics": metrics, "predictions": records}
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"Saved predictions to {out_path}")


if __name__ == "__main__":
    main()
