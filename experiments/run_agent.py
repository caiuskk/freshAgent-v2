# experiments/run_agent.py
import os
import time
import pandas as pd
from dotenv import load_dotenv
from freshagent.agent import Agent, AgentConfig, extract_direct_answer

MODEL = os.getenv("MODEL_NAME", "gpt-4o")
INPUT = "data/results/freshQA_since24_full.csv"
OUT = "data/results/freshQA_since24_full.csv"
TMP = OUT + ".tmp"
# Store both full and direct answers
RESULT_COL = "model_response_agent_v1"
RESULT_COL_DIRECT = RESULT_COL + "_direct"
CHECKPOINT_EVERY = 186


def preflight():
    load_dotenv()
    missing = []
    if not os.getenv("OPENAI_API_KEY"):
        missing.append("OPENAI_API_KEY")
    if not os.getenv("SERPER_API_KEY"):
        missing.append("SERPER_API_KEY")
    if missing:
        raise RuntimeError(f"Missing env vars: {', '.join(missing)}")


def main():
    preflight()
    os.makedirs("data/results", exist_ok=True)
    df = pd.read_csv(INPUT).copy()
    if RESULT_COL not in df.columns:
        df[RESULT_COL] = None
    if RESULT_COL_DIRECT not in df.columns:
        df[RESULT_COL_DIRECT] = None

    agent = Agent(AgentConfig(model=MODEL, provider="serper"))

    processed = 0
    for i, q in df["question"].items():
        # Skip if both full and direct exist
        full_ok = pd.notna(df.at[i, RESULT_COL]) and str(df.at[i, RESULT_COL]).strip()
        direct_ok = (
            pd.notna(df.at[i, RESULT_COL_DIRECT])
            and str(df.at[i, RESULT_COL_DIRECT]).strip()
        )
        if full_ok and direct_ok:
            continue

        # If we already have full but missing direct, derive locally without LLM
        if full_ok and not direct_ok:
            try:
                direct = extract_direct_answer(str(df.at[i, RESULT_COL]))
            except Exception:
                direct = ""
            df.at[i, RESULT_COL_DIRECT] = direct
            processed += 1
            preview = (direct or str(df.at[i, RESULT_COL]))[:80]
            print(f"[agent-fill-direct] {i}: {str(q)[:60]}... -> {preview}")
            if processed % CHECKPOINT_EVERY == 0:
                df.to_csv(TMP, index=False, encoding="utf-8-sig")
                print(f"[checkpoint] wrote {TMP} at {time.strftime('%H:%M:%S')}")
            continue

        try:
            parts = agent.run_parts(str(q), dbg=False)
            full, direct = parts.get("full", ""), parts.get("direct", "")
        except Exception as e:
            full, direct = f"[ERROR] {type(e).__name__}: {e}", ""
        df.at[i, RESULT_COL] = full
        df.at[i, RESULT_COL_DIRECT] = direct
        processed += 1

        preview = (direct or full)[:80]
        print(f"[agent] {i}: {str(q)[:60]}... -> {preview}")
        if processed % CHECKPOINT_EVERY == 0:
            df.to_csv(TMP, index=False, encoding="utf-8-sig")
            print(f"[checkpoint] wrote {TMP} at {time.strftime('%H:%M:%S')}")

    df.to_csv(OUT, index=False, encoding="utf-8-sig")
    print("saved:", OUT)


if __name__ == "__main__":
    main()
