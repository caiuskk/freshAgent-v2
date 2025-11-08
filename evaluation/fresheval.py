# evaluation/fresheval.py
from __future__ import annotations
from typing import List, Dict, Tuple
from core.llm_api import call_llm_api

# Relaxed evaluation instruction (from your fresheval notebook)
RELAXED_INSTRUCTION = (
    "Please evaluate the response to a question under relaxed evaluation, where "
    "hallucinations, outdated information, and ill-formed answers are allowed, "
    "as long as the primary answer is accurate. Please credit the response "
    "only if it provides a confident and definitive answer, or the correct "
    "answer can be obviously inferred from the response. The primary or final "
    "answer when standing alone must be accurate. Any additional information "
    "that is provided must not contradict the primary answer or reshape one's "
    "perception of it. For false-premise questions, the response must point "
    "out the presence of a false premise to receive credit. For answers that "
    "involve names of entities (e.g., people), complete names or commonly "
    "recognized names are expected. Regarding numerical answers, approximate "
    "numbers are generally not accepted unless explicitly included in the "
    "ground-truth answers. We accept ill-formed responses (including those in "
    "a non-English language), as well as hallucinated or outdated information "
    "that does not significantly impact the primary answer."
)

# Compact few-shot demos (question omitted in the pattern; notebook shows this format)
# Each tuple = (correct_answers_line, response_line, comment_line, evaluation_line)
RELAXED_DEMOS: List[Tuple[str, str, str, str]] = [
    (
        "correct answer(s): 117 years old | 117",
        "response: As of today <DATE>, the most up-to-date and relevant information regarding this query is as follows. The oldest verified living person is Maria Branyas Morera, who was born on March 4, 1907, making her 117 years old.",
        "comment: The primary answer (117 years old) is accurate; information is up-to-date. Credit.",
        "evaluation: correct",
    ),
    (
        "correct answer(s): The United Kingdom has never adopted the Euro.",
        "response: The UK has never adopted the Euro as its official currency. The country has retained the British pound sterling (GBP).",
        "comment: False premise; response debunks it explicitly. Credit.",
        "evaluation: correct",
    ),
    (
        "correct answer(s): She was released in December 2022 as part of a prisoner swap.",
        "response: I'm sorry, but I have no information to suggest that Brittney Griner is currently in a Russian prison...",
        "comment: False premise not debunked explicitly; lacks a confident, definitive answer. Do not credit.",
        "evaluation: incorrect",
    ),
    (
        "correct answer(s): English",
        "response: 1. Mandarin 2. Spanish 3. English",
        "comment: Correct answer can be obviously inferred. Credit.",
        "evaluation: correct",
    ),
    (
        "correct answer(s): No",
        "response: No, it isn't. The stock price is currently at $257.",
        "comment: Additional information contradicts the primary answer (257 > 250). Do not credit.",
        "evaluation: incorrect",
    ),
]


def _format_demo_block() -> str:
    lines = [RELAXED_INSTRUCTION, "\n--- DEMONSTRATIONS ---"]
    for ca, resp, cmt, ev in RELAXED_DEMOS:
        lines += [ca, resp, cmt, ev, ""]
    lines += ["--- END DEMOS ---"]
    return "\n".join(lines)


def build_relaxed_prompt(
    correct_answers: List[str], response: str, use_demos: bool = True
) -> str:
    """
    Build the relaxed evaluator prompt. The notebook uses a compact format:
      correct answer(s): <a | b | ...>
      response: <model response>
      ... and expects a final 'evaluation: correct|incorrect'
    """
    ca_line = " | ".join([str(x).strip() for x in correct_answers if str(x).strip()])
    parts = []
    parts.append(_format_demo_block() if use_demos else RELAXED_INSTRUCTION)
    parts.append("\nNow evaluate the following response:")
    parts.append(f"correct answer(s): {ca_line}")
    parts.append(f"response: {response.strip()}")
    parts.append("\nPlease output exactly one line:\nevaluation: <correct|incorrect>")
    return "\n".join(parts)


def parse_relaxed_label(text: str) -> str:
    """
    Parse 'evaluation: correct|incorrect' from grader output.
    Returns 'correct' or 'incorrect'; falls back to 'unknown'.
    """
    t = (text or "").lower()
    if "evaluation:" in t:
        tail = t.split("evaluation:", 1)[1].strip()
        if tail.startswith("correct"):
            return "correct"
        if tail.startswith("incorrect"):
            return "incorrect"
    # tolerant fallback
    if "correct" in t and "incorrect" not in t:
        return "correct"
    if "incorrect" in t and "correct" not in t:
        return "incorrect"
    return "unknown"


def eval_relaxed_llm(
    correct_answers: List[str],
    response: str,
    model: str = "gpt-4o",
    temperature: float = 0.0,
    max_tokens: int = 128,
    use_demos: bool = True,
) -> Dict[str, str]:
    """
    Run the LLM-based relaxed evaluation and parse a label.
    Returns: {"label": "correct|incorrect|unknown", "raw": <grader_text>}
    """
    prompt = build_relaxed_prompt(correct_answers, response, use_demos=use_demos)
    raw = call_llm_api(prompt, model, temperature, max_tokens, chat_completions=True)
    return {"label": parse_relaxed_label(raw), "raw": raw}
