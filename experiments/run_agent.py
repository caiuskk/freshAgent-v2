# experiments/run_agent.py
import os, pandas as pd
from dotenv import load_dotenv
from freshagent.agent import Agent, AgentConfig

load_dotenv()
MODEL = os.getenv("MODEL_NAME", "gpt-4o")
INPUT = "data/freshQA_since24_first5.csv"  # 先用你已有的小样本
OUT = "data/results/freshQA_since24_first5.csv"
result_column = "model_response_agent"


def main():
    os.makedirs("data/results", exist_ok=True)
    df = pd.read_csv(INPUT).copy()
    if result_column not in df.columns:
        df[result_column] = None

    agent = Agent(AgentConfig(model=MODEL, provider="serper"))
    for i, q in df["question"].items():
        ans = agent.run(str(q), dbg=False)
        df.at[i, result_column] = ans
        print(f"[agent] {i}: {str(q)[:60]}...")

    df.to_csv(OUT, index=False, encoding="utf-8-sig")
    print("saved:", OUT)


if __name__ == "__main__":
    main()
