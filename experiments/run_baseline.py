# experiments/run_baseline.py
import os, pandas as pd
from dotenv import load_dotenv
from core.search_engine import call_search_engine
from core.search_result_formatters import freshprompt_format
from core.llm_api import call_llm_api
from freshagent.freshprompt_demos import build_freshprompt_demo  # NEW

load_dotenv()
MODEL = os.getenv("MODEL_NAME", "gpt-4o")
INPUT = "data/results/freshQA_since24_full.csv"
OUT = "data/results/freshQA_since24_full.csv"
TMP = OUT + ".tmp"
RESULT_COL = "model_response_freshprompt"
CHECKPOINT_EVERY = 184

# Controls for FreshPrompt variations
CHECK_PREMISE = True  # original “check premise” line
PROMPT_DEMO = False  # set True to prepend demo examples
PROMPT_DEMO_VERBOSE = False  # set True for verbose demo answers
PROVIDER = "serper"  # search provider for both demo and question


def _hp(model):
    if str(model).startswith("gpt-4"):
        return dict(n_org=15, n_rel=3, n_qa=3, n_evd=15, max_tokens=256, chat=True)
    return dict(n_org=15, n_rel=2, n_qa=2, n_evd=5, max_tokens=256, chat=True)


def _suffix():
    parts = []
    if CHECK_PREMISE:
        parts.append(
            "\nPlease check if the question contains a valid premise before answering."
        )
    parts.append("\nanswer: ")
    return "".join(parts)


def build_question_block(q, model):
    hp = _hp(model)
    sdata = call_search_engine(q, provider=PROVIDER)
    prompt = freshprompt_format(
        q, sdata, _suffix(), hp["n_org"], hp["n_rel"], hp["n_qa"], hp["n_evd"]
    )
    return prompt, hp


def main():
    os.makedirs("data/results", exist_ok=True)
    df = pd.read_csv(INPUT).copy()  # full dataset
    if RESULT_COL not in df.columns:
        df[RESULT_COL] = None

    demo_block = ""
    if PROMPT_DEMO:
        demo_block = build_freshprompt_demo(
            MODEL, provider=PROVIDER, verbose=PROMPT_DEMO_VERBOSE
        )

    processed = 0
    for i, q in df["question"].items():
        # resume: skip filled
        val = df.at[i, RESULT_COL]
        if pd.notna(val) and str(val).strip():
            continue

        q_prompt, hp = build_question_block(str(q), MODEL)
        full_prompt = (demo_block + q_prompt) if demo_block else q_prompt
        try:
            ans = call_llm_api(full_prompt, MODEL, 0.0, hp["max_tokens"], hp["chat"])
        except Exception as e:
            ans = f"[ERROR] {type(e).__name__}: {e}"
        df.at[i, RESULT_COL] = ans
        processed += 1
        print(f"[freshprompt] {i}: {str(q)[:60]}...")
        if processed % CHECKPOINT_EVERY == 0:
            df.to_csv(TMP, index=False, encoding="utf-8-sig")
            print(f"[checkpoint] wrote {TMP}")

    df.to_csv(OUT, index=False, encoding="utf-8-sig")
    print("saved:", OUT)


if __name__ == "__main__":
    main()
