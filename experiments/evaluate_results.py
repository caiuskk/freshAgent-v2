# experiments/evaluate_results.py
import os
import json
import argparse
import pandas as pd
from typing import List

from evaluation.apply import evaluate_dataframe  # uses fresheval/robust modules


def _parse_correct_answers(cell) -> List[str]:
    # Accept list, JSON string, or pipe-separated string
    if isinstance(cell, list):
        return [str(x).strip() for x in cell if str(x).strip()]
    if isinstance(cell, str):
        t = cell.strip()
        # Try JSON list
        try:
            obj = json.loads(t)
            if isinstance(obj, list):
                return [str(x).strip() for x in obj if str(x).strip()]
        except Exception:
            pass
        # Pipe-separated fallback
        if "|" in t:
            return [s.strip() for s in t.split("|") if s.strip()]
        if t:
            return [t]
        return []
    if pd.isna(cell):
        return []
    return [str(cell).strip()]


def main():
    ap = argparse.ArgumentParser(
        description="Evaluate model responses with Robust / FreshEval (relaxed-llm)."
    )
    ap.add_argument("--input", required=True, help="Input CSV path")
    ap.add_argument("--output", required=True, help="Output CSV path")
    ap.add_argument("--question_col", default="question", help="Question column name")
    ap.add_argument(
        "--ground_truth_col", default="ground_truth", help="Ground-truth column name"
    )
    ap.add_argument(
        "--response_cols",
        nargs="+",
        required=True,
        help="One or more response columns to evaluate",
    )
    ap.add_argument(
        "--modes",
        nargs="+",
        default=["robust"],
        choices=["robust", "relaxed-llm"],
        help="Evaluation modes",
    )
    ap.add_argument("--model", default="gpt-4o", help="LLM model for relaxed-llm")
    ap.add_argument(
        "--limit", type=int, default=None, help="Optional row limit for quick tests"
    )
    args = ap.parse_args()

    df = pd.read_csv(args.input)

    # Normalize ground truth column to a list for downstream helpers
    gt_col = args.ground_truth_col
    if gt_col not in df.columns:
        raise RuntimeError(f"Ground-truth column '{gt_col}' not found in {args.input}")

    df["_gt_list"] = df[gt_col].apply(_parse_correct_answers)

    if args.limit is not None:
        df = df.iloc[: args.limit].copy()

    # Evaluate per response column and per mode
    out = df.copy()
    for resp_col in args.response_cols:
        if resp_col not in out.columns:
            print(f"[warn] response column '{resp_col}' not found. Skipping.")
            continue
        for mode in args.modes:
            # Build a minimal frame for evaluate_dataframe API
            sub = out[[args.question_col, resp_col, "_gt_list"]].rename(
                columns={
                    args.question_col: "question",
                    resp_col: "response",
                    "_gt_list": "ground_truth",
                }
            )
            evaluated = evaluate_dataframe(
                sub,
                question_col="question",
                response_col="response",
                correct_col="ground_truth",
                mode=mode,
                model=args.model,
            )
            out[f"eval_label_{mode}_{resp_col}"] = evaluated[f"eval_label_{mode}"]
            out[f"eval_reason_{mode}_{resp_col}"] = evaluated[f"eval_reason_{mode}"]

    # Write results
    out.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"[ok] saved: {args.output}")

    # Summary
    for resp_col in args.response_cols:
        for mode in args.modes:
            col = f"eval_label_{mode}_{resp_col}"
            if col in out.columns:
                s = out[col].value_counts(dropna=False)
                total = int(s.sum())
                acc = float((out[col] == "correct").sum()) / total if total else 0.0
                unk = int((out[col] == "unknown").sum())
                print(
                    f"[summary] {mode} on {resp_col}: acc={acc:.3f}, unknown={unk}/{total}"
                )


if __name__ == "__main__":
    main()
