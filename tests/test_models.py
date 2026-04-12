from __future__ import annotations

import torch
import torch.nn as nn

from src.models.attention import Attention
from src.models.decoder import Decoder
from src.models.vqa import build_causal_mask
import src.models.vqa as vqa_module


def test_build_causal_mask_shape():
    mask = build_causal_mask(4, device=torch.device("cpu"))
    assert mask.shape == (4, 4)
    assert torch.isfinite(mask[0, 0])
    assert mask[0, 1].item() < -1e6


def test_attention_output_shape():
    att = Attention(d=32, num_heads=4, dropout=0.0)
    vi = torch.randn(2, 1, 32)
    vq = torch.randn(2, 1, 32)
    out = att(vi, vq)
    assert out.shape == (2, 32)


def test_decoder_output_shape():
    decoder = Decoder(d_model=32, ffn_hidden=64, num_heads=4, drop_prob=0.1, num_layers=2)
    x = torch.randn(2, 5, 32)
    y = torch.randn(2, 5, 32)
    mask = build_causal_mask(5, device=torch.device("cpu"))
    out = decoder(x, y, mask)
    assert out.shape == (2, 5, 32)


def test_vqa_model_forward(monkeypatch):
    class DummyImageEmbedding(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

        def forward(self, images, image_ids=None):
            batch = images.size(0)
            return torch.zeros(batch, 1, 16), image_ids

    class DummyQuesEmbedding(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

        def forward(self, questions, max_len: int):
            batch = len(questions)
            return torch.zeros(batch, 16)

    class DummyAnsEmbedding(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

        def forward(self, answers, max_len: int):
            batch = len(answers)
            token_ids = torch.zeros(batch, max_len, dtype=torch.long)
            emb = torch.zeros(batch, max_len, 16)
            return token_ids, emb

    monkeypatch.setattr(vqa_module, "ImageEmbedding", DummyImageEmbedding)
    monkeypatch.setattr(vqa_module, "QuesEmbedding", DummyQuesEmbedding)
    monkeypatch.setattr(vqa_module, "AnsEmbedding", DummyAnsEmbedding)

    model = vqa_module.VQAModel(
        vocab_size=10,
        text_model="dummy",
        image_encoders=["dino"],
        fusion="gated",
        d_model=16,
        ffn_hidden=32,
        num_heads=4,
        num_layers=2,
        num_att_layers=2,
        dropout=0.1,
        device="cpu",
    )

    images = torch.randn(2, 3, 224, 224)
    questions = ["q1", "q2"]
    answers = ["a1", "a2"]
    logits, vocab = model(images, questions, answers, anno_ids=None, mask=True, max_len=5)
    assert logits.shape == (2, 5, 10)
    assert vocab.shape == (2, 5)


# ---------------------------------------------------------------------------
# Helpers dùng chung cho các test generate()
# ---------------------------------------------------------------------------

class _MockTokenizer:
    """Tokenizer tối giản với 3 special token IDs chuẩn PhoBERT."""
    bos_token_id = 0
    eos_token_id = 2
    pad_token_id = 1


class _MockPhobertEmbed(nn.Module):
    """Trả về zero embeddings (B, seq_len, d_model) — đủ để decoder chạy."""
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, L = input_ids.shape
        return torch.zeros(B, L, self.d_model)


class _DummyImageEmbedding(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, images, image_ids=None):
        return torch.zeros(images.size(0), 1, 16), image_ids


class _DummyQuesEmbedding(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, questions, max_len: int):
        return torch.zeros(len(questions), 16)


class _DummyAnsEmbedding(nn.Module):
    """Hỗ trợ cả forward() lẫn generate() (có tokenizer và phobert_embed)."""
    def __init__(self, d_model: int = 16, *args, **kwargs):
        super().__init__()
        self.d_model = d_model
        self.tokenizer = _MockTokenizer()
        self.phobert_embed = _MockPhobertEmbed(d_model)

    def forward(self, answers, max_len: int):
        batch = len(answers)
        token_ids = torch.zeros(batch, max_len, dtype=torch.long)
        emb = torch.zeros(batch, max_len, self.d_model)
        return token_ids, emb


class _DummyAnsEmbeddingNoForward(nn.Module):
    """Dùng để chứng minh generate() KHÔNG gọi forward() với ground truth."""
    def __init__(self, d_model: int = 16, *args, **kwargs):
        super().__init__()
        self.d_model = d_model
        self.tokenizer = _MockTokenizer()
        self.phobert_embed = _MockPhobertEmbed(d_model)

    def forward(self, answers, max_len: int):
        raise AssertionError(
            "generate() không được gọi ans_model.forward() với ground truth answers"
        )


def _make_generate_model(monkeypatch, ans_cls=_DummyAnsEmbedding, vocab_size=10):
    """Tạo VQAModel nhỏ với tất cả dummy components để test generate()."""
    monkeypatch.setattr(vqa_module, "ImageEmbedding", _DummyImageEmbedding)
    monkeypatch.setattr(vqa_module, "QuesEmbedding", _DummyQuesEmbedding)
    monkeypatch.setattr(vqa_module, "AnsEmbedding", ans_cls)
    return vqa_module.VQAModel(
        vocab_size=vocab_size,
        text_model="dummy",
        image_encoders=["dino"],
        fusion="gated",
        d_model=16,
        ffn_hidden=32,
        num_heads=4,
        num_layers=2,
        num_att_layers=2,
        dropout=0.0,
        device="cpu",
    )


# ---------------------------------------------------------------------------
# Tests cho generate()
# ---------------------------------------------------------------------------

def test_vqa_model_generate_output_shape(monkeypatch):
    """generate() trả về LongTensor shape (B, max_len)."""
    model = _make_generate_model(monkeypatch)
    model.eval()
    images = torch.randn(2, 3, 224, 224)
    generated = model.generate(images, ["q1", "q2"], max_len=5)
    assert generated.shape == (2, 5)
    assert generated.dtype == torch.long


def test_vqa_model_generate_first_token_is_bos(monkeypatch):
    """Token đầu tiên luôn là BOS (id=0), không phụ thuộc vào input."""
    model = _make_generate_model(monkeypatch)
    model.eval()
    images = torch.randn(3, 3, 224, 224)
    generated = model.generate(images, ["q1", "q2", "q3"], max_len=6)
    bos_id = _MockTokenizer.bos_token_id
    assert (generated[:, 0] == bos_id).all(), "Tất cả sequences phải bắt đầu bằng BOS"


def test_vqa_model_generate_stops_after_eos(monkeypatch):
    """Sau khi sinh EOS, các vị trí còn lại phải là PAD."""
    model = _make_generate_model(monkeypatch, vocab_size=10)
    model.eval()

    eos_id = _MockTokenizer.eos_token_id   # 2
    pad_id = _MockTokenizer.pad_token_id   # 1

    # Monkeypatch MLP: luôn trả logit cao nhất tại vị trí EOS
    class _AlwaysEosMLP(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            B, L, _ = x.shape
            logits = torch.full((B, L, 10), -1e9)
            logits[:, :, eos_id] = 1e9
            return logits

    model.mlp = _AlwaysEosMLP()

    images = torch.randn(2, 3, 224, 224)
    generated = model.generate(images, ["q1", "q2"], max_len=5)

    # step=0 → next_token=EOS → generated[:,1]=EOS, finished=True → loop dừng
    assert (generated[:, 0] == 0).all(),   "position 0 phải là BOS"
    assert (generated[:, 1] == eos_id).all(), "position 1 phải là EOS (dự đoán đầu tiên)"
    assert (generated[:, 2:] == pad_id).all(), "các position sau EOS phải là PAD"


def test_vqa_model_generate_no_teacher_forcing(monkeypatch):
    """generate() KHÔNG truyền ground truth answers vào ans_model.forward()."""
    # _DummyAnsEmbeddingNoForward.forward() raise AssertionError nếu bị gọi
    model = _make_generate_model(monkeypatch, ans_cls=_DummyAnsEmbeddingNoForward)
    model.eval()

    images = torch.randn(2, 3, 224, 224)
    # Nếu generate() gọi self.ans_model(answers, ...) → AssertionError → test fail
    # Nếu chỉ gọi self.ans_model.phobert_embed(input_ids=...) → pass
    generated = model.generate(images, ["q1", "q2"], max_len=5)
    assert generated.shape == (2, 5)
