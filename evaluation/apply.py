# evaluation/apply.py
from __future__ import annotations
import pandas as pd
from typing import List
from evaluation.fresheval import eval_relaxed_llm
from evaluation.robust import eval_robust


def evaluate_dataframe(
    df: pd.DataFrame,
    question_col: str,
    response_col: str,
    correct_col: str,
    mode: str = "robust",  # "robust" or "relaxed-llm"
    model: str = "gpt-4o",
) -> pd.DataFrame:
    """
    correct_col can be:
      - list of strings, or
      - a single string (will be wrapped into a list)
    """
    out = df.copy()
    labels, reasons = [], []
    for _, row in out.iterrows():
        question = str(row[question_col])
        response = str(row[response_col])
        ca = row[correct_col]
        correct_answers: List[str] = ca if isinstance(ca, list) else [str(ca)]
        if mode == "relaxed-llm":
            r = eval_relaxed_llm(correct_answers, response, model=model)
            labels.append(r["label"])
            reasons.append(r["raw"])
        else:
            r = eval_robust(question, response, correct_answers)
            labels.append(r["label"])
            reasons.append(r["reason"])
    out[f"eval_label_{mode}"] = labels
    out[f"eval_reason_{mode}"] = reasons
    return out
