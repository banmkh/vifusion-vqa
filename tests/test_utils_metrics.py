from __future__ import annotations

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
    refs = ["a", "b", "c"]
    hyps = ["a", "x", "c"]
    precision, recall, f1, acc = evaluate_vqa_benchmark(refs, hyps)
    assert precision == 2 / 3
    assert recall == 2 / 3
    assert round(f1, 5) == round(2 * (2/3) * (2/3) / ((2/3) + (2/3)), 5)
    assert acc == 2 / 3


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
