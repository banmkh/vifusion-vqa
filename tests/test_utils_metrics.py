from __future__ import annotations

import pytest
import torch

from src.utils.metrics import (
    decode_subwords,
    decode_predictions,
    evaluate_vqa_benchmark,
    compute_rouge,
    compute_cider,
)


def test_decode_subwords():
    assert decode_subwords(["xin", "chao"]) == "xin chao"


def test_decode_predictions_strips_tokens():
    preds = ["<s> xin chao </s>"]
    out = decode_predictions(preds)
    assert out == ["xin chao"]


def test_evaluate_vqa_benchmark_basic():
    """Kết quả trung bình của 3 samples: 2 đúng, 1 sai (single-token)."""
    refs = ["a", "b", "c"]
    hyps = ["a", "x", "c"]
    precision, recall, f1, acc = evaluate_vqa_benchmark(refs, hyps)
    # Sample 1 (a/a): P=1, R=1, F1=1, acc=1
    # Sample 2 (b/x): P=0, R=0, F1=0, acc=0
    # Sample 3 (c/c): P=1, R=1, F1=1, acc=1  → avg = 2/3
    assert precision == pytest.approx(2 / 3)
    assert recall == pytest.approx(2 / 3)
    assert f1 == pytest.approx(2 / 3)
    assert acc == pytest.approx(2 / 3)


# ---------------------------------------------------------------------------
# Tests mới xác nhận hành vi per-sample (fix vừa thực hiện)
# ---------------------------------------------------------------------------

def test_evaluate_vqa_benchmark_token_overlap_partial():
    """Prediction chứa subset của reference → precision=1, recall<1 (P≠R)."""
    refs = ["xe do dep"]   # 3 tokens
    hyps = ["xe do"]       # 2 tokens, cả 2 đều đúng
    precision, recall, f1, acc = evaluate_vqa_benchmark(refs, hyps)
    # common=2, precision=2/2=1.0, recall=2/3
    assert precision == pytest.approx(1.0)
    assert recall == pytest.approx(2 / 3)
    expected_f1 = 2 * 1.0 * (2 / 3) / (1.0 + 2 / 3)
    assert f1 == pytest.approx(expected_f1)
    assert acc == pytest.approx(0.0)   # không exact match


def test_evaluate_vqa_benchmark_precision_ne_recall():
    """Sau fix, P ≠ R là hợp lệ — không còn bị force bằng nhau như set-level cũ."""
    refs = ["cat sat on mat"]  # 4 tokens
    hyps = ["cat sat"]         # 2 tokens, precision=1 nhưng recall=0.5
    precision, recall, f1, acc = evaluate_vqa_benchmark(refs, hyps)
    assert precision == pytest.approx(1.0)
    assert recall == pytest.approx(0.5)
    assert precision != pytest.approx(recall)


def test_evaluate_vqa_benchmark_accuracy_exact_match():
    """Accuracy là exact string match, không phải set overlap."""
    refs = ["xe do", "xe do"]
    hyps = ["xe do", "xe xanh"]   # sample 1 đúng, sample 2 sai hoàn toàn
    _, _, _, acc = evaluate_vqa_benchmark(refs, hyps)
    assert acc == pytest.approx(0.5)


def test_evaluate_vqa_benchmark_partial_token_overlap_accuracy():
    """Câu đúng 1 nửa token không được tính là exact match."""
    refs = ["con meo vang"]
    hyps = ["con meo"]     # 2/3 token đúng nhưng không exact
    _, _, _, acc = evaluate_vqa_benchmark(refs, hyps)
    assert acc == pytest.approx(0.0)


def test_evaluate_vqa_benchmark_all_correct():
    """Tất cả đúng → P=R=F1=Accuracy=1.0."""
    refs = ["xe do", "con meo"]
    hyps = ["xe do", "con meo"]
    precision, recall, f1, acc = evaluate_vqa_benchmark(refs, hyps)
    assert precision == pytest.approx(1.0)
    assert recall == pytest.approx(1.0)
    assert f1 == pytest.approx(1.0)
    assert acc == pytest.approx(1.0)


def test_evaluate_vqa_benchmark_all_wrong():
    """Tất cả sai (không có token chung nào) → P=R=F1=Accuracy=0.0."""
    # "meo", "vang" vs "xe", "do" → không có token chung
    refs = ["meo vang", "cho den"]
    hyps = ["xe do",   "bau troi"]
    precision, recall, f1, acc = evaluate_vqa_benchmark(refs, hyps)
    assert precision == pytest.approx(0.0)
    assert recall == pytest.approx(0.0)
    assert f1 == pytest.approx(0.0)
    assert acc == pytest.approx(0.0)


def test_evaluate_vqa_benchmark_duplicate_tokens_in_answer():
    """Token lặp lại được tính đúng bằng Counter intersection."""
    refs = ["do do xanh"]       # "do" xuất hiện 2 lần
    hyps = ["do xanh"]          # "do" xuất hiện 1 lần → common("do")=min(2,1)=1
    precision, recall, f1, acc = evaluate_vqa_benchmark(refs, hyps)
    # common = 1("do") + 1("xanh") = 2
    # precision = 2/2 = 1.0, recall = 2/3
    assert precision == pytest.approx(1.0)
    assert recall == pytest.approx(2 / 3)


def test_compute_rouge_runs():
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    refs = [["xin", "chao"], ["hello"]]
    hyps = [["xin", "chao"], ["hi"]]
    score = compute_rouge(refs, hyps, scorer)
    assert 0.0 <= score <= 1.0


def test_compute_cider_runs():
    from pycocoevalcap.cider.cider import Cider

    cider = Cider()
    refs = [["xin", "chao"], ["hello"]]
    hyps = [["xin", "chao"], ["hello"]]
    score = compute_cider(refs, hyps, cider)
    assert score >= 0.0
