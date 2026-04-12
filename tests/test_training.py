from __future__ import annotations

import torch
import torch.nn as nn

from src.training import TrainConfig, train_one_epoch, evaluate_one_epoch, build_optimizer, build_scheduler
from src.training.trainer import evaluate_benchmark_epoch
from src.utils.metrics import evaluation_benchmark


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
    criterion = nn.CrossEntropyLoss(ignore_index=1)
    cfg = TrainConfig(epochs=1, lr=1e-3, weight_decay=0.0)
    optimizer = build_optimizer(model, cfg)

    train_loss = train_one_epoch(model, loader, criterion, optimizer, device="cpu", max_len=5)
    eval_loss = evaluate_one_epoch(model, loader, criterion, device="cpu", max_len=5)

    assert train_loss >= 0.0
    assert eval_loss >= 0.0


def test_loss_uses_shifted_targets():
    """
    Loss phải tính logits[:, :-1, :] vs targets[:, 1:] (next-token prediction).
    Kiểm tra bằng cách so sánh với loss tính thủ công theo đúng cách đó.
    """
    torch.manual_seed(0)
    vocab_size = 11
    model = DummyModel(vocab_size=vocab_size, d_model=8)
    criterion = nn.CrossEntropyLoss(ignore_index=1)

    loader = make_loader(batch_size=2, max_len=5)
    batch = loader[0]
    _, _, images, questions, answers = batch

    model.eval()
    with torch.no_grad():
        logits, targets = model(images, questions, answers, max_len=5)

    # Loss đúng: shift trái 1
    expected_loss = criterion(
        logits[:, :-1, :].contiguous().view(-1, vocab_size),
        targets[:, 1:].contiguous().view(-1),
    )
    # Loss sai (không shift): sẽ khác
    wrong_loss = criterion(
        logits.contiguous().view(-1, vocab_size),
        targets.contiguous().view(-1),
    )

    # Tính loss từ trainer (phải bằng expected, không bằng wrong)
    cfg = TrainConfig(epochs=1, lr=0.0, weight_decay=0.0)
    optimizer = build_optimizer(model, cfg)
    actual_loss = train_one_epoch(model, loader, criterion, optimizer, device="cpu", max_len=5)

    assert abs(actual_loss - expected_loss.item()) < 1e-5, \
        "trainer phải dùng shifted loss (logits[:, :-1] vs targets[:, 1:])"
    # Đảm bảo loss thực sự khác với cách tính sai (trừ khi ngẫu nhiên bằng nhau)
    if abs(expected_loss.item() - wrong_loss.item()) > 1e-6:
        assert abs(actual_loss - wrong_loss.item()) > 1e-5, \
            "trainer KHÔNG được dùng unshifted loss"


def test_build_scheduler_runs():
    model = DummyModel()
    cfg = TrainConfig(epochs=2, lr=1e-3, weight_decay=0.0)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg.epochs)
    scheduler.step()


# ---------------------------------------------------------------------------
# Helpers cho test evaluation_benchmark với generate()
# ---------------------------------------------------------------------------

class _MockTokenizer:
    bos_token_id = 0
    eos_token_id = 2
    pad_token_id = 1

    def get_vocab(self):
        # vocab tối giản: PAD=1, EOS=2, "a"=3
        return {"<pad>": 1, "</s>": 2, "a": 3, "<s>": 0}


class _MockAnsModel(nn.Module):
    """Giả lập ans_model có tokenizer và phobert_embed — đủ cho generate()."""
    def __init__(self, d_model: int = 8):
        super().__init__()
        self.d_model = d_model
        self.tokenizer = _MockTokenizer()
        self._embed = nn.Embedding(10, d_model)

        # phobert_embed: nhận input_ids tensor, trả về embeddings
        class _PhobertEmbed(nn.Module):
            def __init__(self, embed):
                super().__init__()
                self._embed = embed

            def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
                return self._embed(input_ids)

        self.phobert_embed = _PhobertEmbed(self._embed)

    def forward(self, answers, max_len: int):
        raise AssertionError("evaluation_benchmark không được gọi ans_model.forward()")


class _DummyModelWithGenerate(nn.Module):
    """
    Model đủ nhỏ để test evaluation_benchmark.
    - forward(): dùng cho train/eval loss (không liên quan test này)
    - generate(): trả về token IDs xác định (BOS + EOS + PADs)
    """
    def __init__(self, vocab_size: int = 10, d_model: int = 8, max_len: int = 5):
        super().__init__()
        self._vocab_size = vocab_size
        self._d_model = d_model
        self._max_len = max_len
        self.ans_model = _MockAnsModel(d_model)
        # forward dùng cho train_one_epoch / evaluate_one_epoch
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, images, questions, answers, anno_ids=None, mask=True, max_len=5):
        B = images.size(0)
        logits = torch.zeros(B, max_len, self._vocab_size)
        targets = torch.zeros(B, max_len, dtype=torch.long)
        return logits, targets

    @torch.no_grad()
    def generate(self, images: torch.Tensor, questions: list, max_len: int = 5) -> torch.Tensor:
        """Sinh chuỗi cố định: [BOS, EOS, PAD, PAD, ...]."""
        B = images.size(0)
        pad_id = self.ans_model.tokenizer.pad_token_id  # 1
        bos_id = self.ans_model.tokenizer.bos_token_id  # 0
        eos_id = self.ans_model.tokenizer.eos_token_id  # 2
        seq = torch.full((B, max_len), pad_id, dtype=torch.long)
        seq[:, 0] = bos_id
        seq[:, 1] = eos_id
        return seq


def _make_benchmark_loader(batch_size: int = 2):
    images = torch.randn(batch_size, 3, 8, 8)
    questions = ["câu hỏi"] * batch_size
    answers = ["câu trả lời"] * batch_size
    batch = (["id1", "id2"][:batch_size], None, images, questions, answers)
    return [batch]


# ---------------------------------------------------------------------------
# Tests cho evaluation_benchmark với generate()
# ---------------------------------------------------------------------------

def test_evaluation_benchmark_uses_generate_not_forward():
    """
    evaluation_benchmark phải gọi model.generate() chứ KHÔNG gọi
    model.forward(images, questions, answers).
    _DummyModelWithGenerate.ans_model.forward() raise nếu bị gọi.
    """
    model = _DummyModelWithGenerate()
    loader = _make_benchmark_loader()
    vocab_swap = {v: k for k, v in model.ans_model.tokenizer.get_vocab().items()}

    # Nếu evaluation_benchmark vẫn dùng teacher forcing → AssertionError
    metrics = evaluation_benchmark(model, loader, None, vocab_swap, device="cpu", max_len=5)
    assert isinstance(metrics, dict)
    assert set(metrics.keys()) == {
        "precision", "recall", "f1", "accuracy",
        "rouge", "bleu_1", "bleu_2", "bleu_3", "bleu_4", "cider",
    }


def test_evaluation_benchmark_returns_valid_metric_ranges():
    """Tất cả metrics phải nằm trong khoảng hợp lệ [0, ∞)."""
    model = _DummyModelWithGenerate()
    loader = _make_benchmark_loader()
    vocab_swap = {v: k for k, v in model.ans_model.tokenizer.get_vocab().items()}

    metrics = evaluation_benchmark(model, loader, None, vocab_swap, device="cpu", max_len=5)

    for key in ("precision", "recall", "f1", "accuracy", "rouge", "bleu_1", "bleu_2", "bleu_3", "bleu_4"):
        assert 0.0 <= metrics[key] <= 1.0, f"{key}={metrics[key]} nằm ngoài [0,1]"
    assert metrics["cider"] >= 0.0


def test_evaluate_benchmark_epoch_calls_evaluation_benchmark(monkeypatch):
    """evaluate_benchmark_epoch trong trainer gọi evaluation_benchmark đúng cách."""
    called_with_model = []

    def _mock_evaluation_benchmark(model, loader, criterion, vocab_swap, device, max_len):
        called_with_model.append(model)
        return {
            "precision": 0.5, "recall": 0.5, "f1": 0.5, "accuracy": 0.5,
            "rouge": 0.5, "bleu_1": 0.5, "bleu_2": 0.5,
            "bleu_3": 0.5, "bleu_4": 0.5, "cider": 1.0,
        }

    import src.training.trainer as trainer_module
    monkeypatch.setattr(trainer_module, "evaluation_benchmark", _mock_evaluation_benchmark)

    model = _DummyModelWithGenerate()
    loader = _make_benchmark_loader()
    result = evaluate_benchmark_epoch(model, loader, device="cpu", max_len=5)

    assert len(called_with_model) == 1
    assert called_with_model[0] is model
    assert result["accuracy"] == 0.5
