from __future__ import annotations

from typing import Callable


def _safe_underthesea():
    try:
        from underthesea import text_normalize, word_tokenize  # type: ignore
    except Exception:  # pragma: no cover - optional dependency
        return None, None
    return text_normalize, word_tokenize


def normalize_text(text: str) -> str:
    text_normalize, _ = _safe_underthesea()
    if text_normalize is None:
        return str(text).strip()
    return text_normalize(text)


def normalize_qa_df(df, question_col: str = "Question", answer_col: str = "Answer"):
    text_normalize, _ = _safe_underthesea()
    if text_normalize is None:
        df[question_col] = [str(x).strip() for x in df[question_col]]
        df[answer_col] = [str(x).strip() for x in df[answer_col]]
        return df

    df[question_col] = [text_normalize(x) for x in df[question_col]]
    df[answer_col] = [text_normalize(str(x)) for x in df[answer_col]]
    return df


def segment_text(text: str) -> list[str]:
    _, word_tokenize = _safe_underthesea()
    if word_tokenize is None:
        return str(text).split()
    return word_tokenize(text)
