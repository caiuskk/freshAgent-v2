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
        "--response_cols",
        nargs="+",
        required=True,
        help="One or more response columns to evaluate (evaluates exactly these columns)",
    )
    ap.add_argument(
        "--mode",
        default="robust",
        choices=["robust", "relaxed-llm"],
        help="Evaluation mode (single run)",
    )
    ap.add_argument("--model", default="gpt-4o", help="LLM model for relaxed-llm")
    ap.add_argument(
        "--limit", type=int, default=None, help="Optional row limit for quick tests"
    )
    args = ap.parse_args()

    df = pd.read_csv(args.input)

    # Build ground truth list from answer_0..answer_9 columns (auto-detect)
    ans_cols = [c for c in df.columns if c.startswith("answer_")]
    if not ans_cols:
        raise RuntimeError(
            "No answer_* columns found; cannot evaluate without ground truth"
        )

    def _answers_row_to_list(row):
        vals = []
        for c in ans_cols:
            v = row.get(c)
            if pd.notna(v) and str(v).strip():
                vals.append(str(v).strip())
        return vals

    df["_gt_list"] = df.apply(_answers_row_to_list, axis=1)

    if args.limit is not None:
        df = df.iloc[: args.limit].copy()

    # Evaluate per response column using the chosen mode
    out = df.copy()
    for resp_col in args.response_cols:
        if resp_col not in out.columns:
            print(f"[warn] response column '{resp_col}' not found. Skipping.")
            continue
        eval_col = resp_col  # evaluate exactly what user passed
        # Build a minimal frame for evaluate_dataframe API
        sub = out[[args.question_col, eval_col, "_gt_list"]].rename(
            columns={
                args.question_col: "question",
                eval_col: "response",
                "_gt_list": "ground_truth",
            }
        )
        evaluated = evaluate_dataframe(
            sub,
            question_col="question",
            response_col="response",
            correct_col="ground_truth",
            mode=args.mode,
            model=args.model,
        )
        # Rename to requested pattern: <col>_evaluated and <col>_explaination
        out[f"{eval_col}_evaluated"] = evaluated[f"eval_label_{args.mode}"]
        out[f"{eval_col}_explaination"] = evaluated[f"eval_reason_{args.mode}"]

    # Write results
    out.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"[ok] saved: {args.output}")

    # Summary
    for resp_col in args.response_cols:
        eval_col = resp_col
        col = f"{eval_col}_evaluated"
        if col in out.columns:
            s = out[col].value_counts(dropna=False)
            total = int(s.sum())
            acc = float((out[col] == "correct").sum()) / total if total else 0.0
            unk = int((out[col] == "unknown").sum())
            print(
                f"[summary] {args.mode} on {eval_col}: acc={acc:.3f}, unknown={unk}/{total}"
            )


if __name__ == "__main__":
    main()
