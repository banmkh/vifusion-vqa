from __future__ import annotations

import torch

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
