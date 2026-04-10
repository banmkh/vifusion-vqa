from __future__ import annotations

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
    pa = hypotheses
    ga = references

    pa_decoded = decode_predictions(pa)
    ga_set = set(ga)
    pa_set = set(pa_decoded)

    intersection = ga_set.intersection(pa_set)

    precision = len(intersection) / len(pa_set) if pa_set else 0.0
    recall = len(intersection) / len(ga_set) if ga_set else 0.0

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    accuracy = len(intersection) / len(ga_set) if ga_set else 0.0
    return precision, recall, f1, accuracy


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

    with torch.no_grad():
        for _, batch in enumerate(test_loader):
            anno_id, _, images, questions, answers = batch
            predicted_tokens, ans_embedds = model(
                images.to(device), questions, answers, anno_ids=anno_id, mask=True, max_len=max_len
            )
            predicted_tokens = predicted_tokens.float()
            ans_embedds = ans_embedds.long()

            references = [answer.split() for answer in answers]
            golden_answer = [answer for answer in answers]
            hypotheses = []
            predicted = []

            for i in range(len(answers)):
                sentence_predicted = torch.argmax(predicted_tokens[i], axis=1)
                predicted_sentence = ""
                for idx in sentence_predicted:
                    predicted_sentence += vocab_swap[idx.item()] + " "
                    if idx == 2:
                        break
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
