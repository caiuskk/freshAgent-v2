# experiments/run_baseline.py
import os, pandas as pd
from dotenv import load_dotenv
from core.search_engine import call_search_engine
from core.search_result_formatters import freshprompt_format
from core.llm_api import call_llm_api

load_dotenv()
MODEL = os.getenv("MODEL_NAME", "gpt-4o")
INPUT = "data/fresheval_relaxed_gpt-4o_top_184.csv"  # 放你的测试集；先用前5条
OUT = "data/freshQA_since24_first5.csv"


def _hp(model):
    if str(model).startswith("gpt-4"):
        return dict(n_org=15, n_rel=3, n_qa=3, n_evd=15, max_tokens=256, chat=True)
    return dict(n_org=15, n_rel=2, n_qa=2, n_evd=5, max_tokens=256, chat=True)


def build_question_block(q, model, check_premise=True):
    hp = _hp(model)
    suffix = (
        "\nPlease check if the question contains a valid premise before answering.\nanswer: "
        if check_premise
        else "\nanswer: "
    )
    sdata = call_search_engine(q)  # serper→标准键
    prompt = freshprompt_format(
        q, sdata, suffix, hp["n_org"], hp["n_rel"], hp["n_qa"], hp["n_evd"]
    )
    return prompt, hp


def main():
    os.makedirs("data/results", exist_ok=True)
    df = pd.read_csv(INPUT).iloc[:5].copy()
    if "model_response" not in df.columns:
        df["model_response"] = None
    for i, q in df["question"].items():
        prompt, hp = build_question_block(str(q), MODEL, True)
        ans = call_llm_api(prompt, MODEL, 0.0, hp["max_tokens"], hp["chat"])
        df.at[i, "model_response"] = ans
        print(f"[ok] {i}: {str(q)[:60]}...")
    df.to_csv(OUT, index=False, encoding="utf-8-sig")
    print("saved:", OUT)


if __name__ == "__main__":
    main()
