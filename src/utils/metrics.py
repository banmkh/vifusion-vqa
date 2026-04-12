from __future__ import annotations

from collections import Counter
from typing import Iterable, Sequence

import torch
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from rouge_score import rouge_scorer
from pycocoevalcap.cider.cider import Cider


def decode_subwords(predicted_tokens: Sequence[str]) -> str:
    decoded_tokens = []
    for token in predicted_tokens:
        if token.endswith("@@ "):
            decoded_tokens.append(token[:-2])
        else:
            decoded_tokens.append(token)
    decoded_sentence = "".join([word if word.startswith("@@ ") else " " + word for word in decoded_tokens]).strip()
    return decoded_sentence.replace("@@ ", "")


def decode_predictions(predictions: Iterable[str]) -> list[str]:
    decoded_predictions = []
    for pred in predictions:
        tokens = pred.split()
        tokens = [token for token in tokens if token not in ["<s>", "</s>"]]
        decoded_predictions.append(decode_subwords(tokens))
    return decoded_predictions


def evaluate_vqa_benchmark(references: Sequence[str], hypotheses: Sequence[str]):
    """
    Tính Precision / Recall / F1 / Accuracy theo từng sample (per-sample),
    sau đó lấy trung bình — thay vì so sánh set của cả batch.

    - Precision, Recall, F1: token-level overlap (Counter intersection)
    - Accuracy: exact string match
    """
    total_precision = total_recall = total_f1 = total_accuracy = 0.0

    hyp_decoded = decode_predictions(list(hypotheses))

    for ref, hyp in zip(references, hyp_decoded):
        ref_tokens = ref.strip().split()
        hyp_tokens = hyp.strip().split()

        ref_counter = Counter(ref_tokens)
        hyp_counter = Counter(hyp_tokens)
        common = sum((ref_counter & hyp_counter).values())

        precision = common / len(hyp_tokens) if hyp_tokens else 0.0
        recall    = common / len(ref_tokens)  if ref_tokens  else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)
        accuracy  = 1.0 if ref.strip() == hyp.strip() else 0.0

        total_precision += precision
        total_recall    += recall
        total_f1        += f1
        total_accuracy  += accuracy

    n = max(1, len(references))
    return total_precision / n, total_recall / n, total_f1 / n, total_accuracy / n


def compute_rouge(references: Sequence[Sequence[str]], hypotheses: Sequence[Sequence[str]], scorer):
    total_rouge_l = 0.0
    for ref, hyp in zip(references, hypotheses):
        scores = scorer.score(" ".join(hyp), " ".join(ref))
        total_rouge_l += scores["rougeL"].fmeasure
    return total_rouge_l / max(1, len(references))


def compute_cider(references: Sequence[Sequence[str]], hypotheses: Sequence[Sequence[str]], scorer):
    gts = {i: [" ".join(ref)] for i, ref in enumerate(references)}
    res = {i: [" ".join(hyp)] for i, hyp in enumerate(hypotheses)}
    score, _ = scorer.compute_score(gts, res)
    return score


def evaluation_benchmark(model, test_loader, criterion, vocab_swap, device, max_len: int):
    """
    Evaluation thực sự: dùng model.generate() (autoregressive) thay vì
    truyền ground truth answers vào decoder (teacher forcing).
    """
    model.eval()
    total_precision = 0.0
    total_recall = 0.0
    total_f1 = 0.0
    total_accuracy = 0.0
    total_rouge = 0.0
    total_bleu_1 = 0.0
    total_bleu_2 = 0.0
    total_bleu_3 = 0.0
    total_bleu_4 = 0.0
    total_cider = 0.0

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    cider_scorer = Cider()
    smoother = SmoothingFunction()

    # EOS token id của PhoBERT = 2 (</s>)
    eos_id = model.ans_model.tokenizer.eos_token_id

    with torch.no_grad():
        for _, batch in enumerate(test_loader):
            _, _, images, questions, answers = batch

            # --- Autoregressive generation, không dùng ground truth ---
            generated_ids = model.generate(images.to(device), questions, max_len=max_len)
            # generated_ids: (B, max_len) token IDs

            references = [answer.split() for answer in answers]
            golden_answer = list(answers)
            hypotheses = []
            predicted = []

            for i in range(len(answers)):
                predicted_sentence = ""
                for idx in generated_ids[i]:
                    token_id = idx.item()
                    predicted_sentence += vocab_swap[token_id] + " "
                    if token_id == eos_id:
                        break
                predicted_sentence = decode_predictions([predicted_sentence])[0]
                hypotheses.append(predicted_sentence.split())
                predicted.append(predicted_sentence)

            rouge_score = compute_rouge(references, hypotheses, scorer)
            bleu_score_1 = corpus_bleu(
                [[ref] for ref in references],
                hypotheses,
                weights=(1, 0, 0, 0),
                smoothing_function=smoother.method1,
            )
            bleu_score_2 = corpus_bleu(
                [[ref] for ref in references],
                hypotheses,
                weights=(0.5, 0.5, 0, 0),
                smoothing_function=smoother.method1,
            )
            bleu_score_3 = corpus_bleu(
                [[ref] for ref in references],
                hypotheses,
                weights=(0.34, 0.33, 0.33, 0),
                smoothing_function=smoother.method1,
            )
            bleu_score_4 = corpus_bleu(
                [[ref] for ref in references],
                hypotheses,
                weights=(0.25, 0.25, 0.25, 0.25),
                smoothing_function=smoother.method1,
            )
            cider_score = compute_cider(references, hypotheses, cider_scorer)

            precision, recall, f1, accuracy = evaluate_vqa_benchmark(golden_answer, predicted)

            total_precision += precision
            total_recall += recall
            total_f1 += f1
            total_accuracy += accuracy
            total_rouge += rouge_score
            total_bleu_1 += bleu_score_1
            total_bleu_2 += bleu_score_2
            total_bleu_3 += bleu_score_3
            total_bleu_4 += bleu_score_4
            total_cider += cider_score

    num_batches = max(1, len(test_loader))
    return {
        "precision": total_precision / num_batches,
        "recall": total_recall / num_batches,
        "f1": total_f1 / num_batches,
        "accuracy": total_accuracy / num_batches,
        "rouge": total_rouge / num_batches,
        "bleu_1": total_bleu_1 / num_batches,
        "bleu_2": total_bleu_2 / num_batches,
        "bleu_3": total_bleu_3 / num_batches,
        "bleu_4": total_bleu_4 / num_batches,
        "cider": total_cider / num_batches,
    }
