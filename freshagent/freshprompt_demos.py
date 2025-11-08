# freshagent/freshprompt_demos.py
from __future__ import annotations
import datetime
from zoneinfo import ZoneInfo
from typing import List
from core.search_engine import call_search_engine
from core.search_result_formatters import freshprompt_format

DEMO_QUESTIONS: List[str] = [
    "What year is considered Albert Einstein's annus mirabilis?",
    "Which photographer took the most expensive photograph in the world?",
    "How many days are left until the 2023 Grammy Awards?",
    "How many years ago did the Boxing Day Tsunami happen?",
    "When did Amazon become the first publicly traded company to exceed a market value of $3 trillion?",
]

# Bare demo answers from the notebook (before applying the "As of today ..." prefix)
CONCISE_DEMO_RA_BARE: List[str] = [
    "1905 is considered Albert Einstein's annus mirabilis, his miraculous year.",
    'The most expensive photograph in the world is "Le Violon d\'Ingres". The photograph was created by Man Ray.',
    "The 2023 Grammy Awards ceremony was held on February 5, 2023. Thus, the ceremony has already taken place.",
    "The disaster occurred on December 26, 2004. Thus, it happened 19 years ago.",
    "Amazon's market capitalization has never exceeded $3 trillion.",
]

VERBOSE_DEMO_RA_BARE: List[str] = [
    "In the year of 1905, Albert Einstein published four groundbreaking papers that revolutionized scientific understanding of the universe. Thus, scientists call 1905 Albert Einstein's annus mirabilis — his year of miracles.",
    "Man Ray's famed \"Le Violon d'Ingres\" became the most expensive photograph ever to sell at auction, sold for $12.4 million on May 14th, 2022 at Christie's New York. The black and white image, taken in 1924 by the American surrealist artist, transforms a woman's naked body into a violin by overlaying the picture of her back with f-holes. Thus, Man Ray is the photographer who took the most expensive photograph in the world.",
    "The 2023 Grammy Awards, officially known as the 65th Annual Grammy Awards ceremony, was held in Los Angeles on February 5, 2023. Thus, the event has already taken place.",
    "The Boxing Day Tsunami refers to the 2004 Indian Ocean earthquake and tsunami, which is one of the deadliest natural disasters in recorded history, killing an estimated 230,000 people across 14 countries. The disaster occurred on December 26, 2004, which is 19 years ago.",
    "Amazon's market capitalization hit a peak of roughly $1.9 trillion in July 2021. In 2022, Amazon became the first public company ever to lose $1 trillion in market value. Thus, Amazon's market value has never exceeded $3 trillion. In fact, Apple became the first publicly traded U.S. company to exceed a market value of $3 trillion in January 2022.",
]


def _today_date_str(tz_name: str = "America/Chicago") -> str:
    try:
        tz = ZoneInfo(tz_name)
        now = datetime.datetime.now(tz)
    except Exception:
        now = datetime.datetime.now()
    return now.strftime("%b %d, %Y")


def _hp(model: str):
    # Match your baseline hp choices
    if str(model).startswith("gpt-4"):
        return dict(n_org=15, n_rel=3, n_qa=3, n_evd=15)
    return dict(n_org=15, n_rel=2, n_qa=2, n_evd=5)


def build_freshprompt_demo(
    model: str,
    provider: str = "serper",
    verbose: bool = False,
) -> str:
    """
    Build the concatenated FreshPrompt demo block as in the TQA notebook:
    - Uses 5 demo questions
    - Formats evidence for each with freshprompt_format(...)
    - Applies the 'As of today ... answer:' prefix to each demo answer
    """
    hp = _hp(model)
    current_date = _today_date_str()
    prefix = (
        f"\nanswer: As of today {current_date}, the most up-to-date and relevant"
        " information regarding this query is as follows. "
    )
    ra_list = VERBOSE_DEMO_RA_BARE if verbose else CONCISE_DEMO_RA_BARE

    demo_prompts: List[str] = []
    for q, ra in zip(DEMO_QUESTIONS, ra_list):
        sdata = call_search_engine(q, provider=provider)
        prompt = freshprompt_format(
            question=q,
            search_data=sdata,
            reasoning_and_answer=prefix + ra,
            num_organic_results=hp["n_org"],
            num_related_questions=hp["n_rel"],
            num_questions_and_answers=hp["n_qa"],
            num_retrieved_evidences=hp["n_evd"],
        )
        demo_prompts.append(prompt)

    return "".join(demo_prompts).strip()
