# evaluation/robust.py
from __future__ import annotations
import re
from typing import List, Dict, Optional, Tuple, Set

_ALLOWED_PREMISE = {"TRUE", "FALSE", "UNCERTAIN"}
_ALLOWED_VERDICTS = {"YES", "NO", "UNCERTAIN"}


def _parse_contract(final_text: str) -> Dict[str, Optional[str]]:
    """Extract Premise, Verdict, Direct Answer fields if present."""
    t = (final_text or "").strip()
    if "Final Answer:" in t:
        t = t.split("Final Answer:", 1)[1].strip()

    def _grab(label: str) -> Optional[str]:
        m = re.search(rf"{label}:\s*(.+)", t, re.I)
        return m.group(1).strip() if m else None

    return {
        "premise": _grab("Premise"),
        "verdict": _grab("Verdict"),
        "direct": _grab("Direct Answer"),
    }


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip().lower()


def _is_bool_word(s: str) -> bool:
    ss = _norm(s)
    return ss in {"yes", "no", "uncertain"}


def _bool_from_text(s: str) -> Optional[str]:
    ss = _norm(s)
    if ss in {"yes", "no", "uncertain"}:
        return ss.upper()
    return None


def _any_contains(hay: str, needles: List[str]) -> bool:
    H = _norm(hay)
    return any(_norm(n) in H for n in needles if str(n).strip())


def _direct_answer_aligns(direct_answer: str, correct_answers: List[str]) -> bool:
    if not direct_answer:
        return False
    # Exact-ish containment match (relaxed inference)
    if _any_contains(direct_answer, correct_answers):
        return True
    # Special case: boolean answers
    da_bool = _bool_from_text(direct_answer)
    if da_bool and any(_bool_from_text(x) == da_bool for x in correct_answers):
        return True
    return False


def _has_contradictory_polarity(text: str) -> bool:
    t = " " + _norm(text) + " "
    # crude polarity collision: contains both yes and no
    return (" yes " in t) and (" no " in t)


def eval_robust(
    question: str,
    response: str,
    correct_answers: List[str],
) -> Dict[str, object]:
    """
    Rule-based evaluator:
      - Prefer Answer Contract 'Direct Answer' alignment with correct_answers
      - Else obvious-inference via containment/boolean checks
      - Penalize contradictory polarity (contains both yes and no)
    Returns:
      {
        "label": "correct|incorrect|unknown",
        "reason": str,
        "contract": {"premise":..., "verdict":..., "direct":...}
      }
    """
    contract = _parse_contract(response)
    direct = contract.get("direct") or ""
    verdict = (contract.get("verdict") or "").strip()
    prem = (contract.get("premise") or "").strip()

    # If contract fields present, sanity check the header fields
    if prem:
        p0 = prem.split()[0].upper()
        if p0 not in _ALLOWED_PREMISE:
            return {
                "label": "incorrect",
                "reason": "invalid premise field",
                "contract": contract,
            }
    if verdict:
        v0 = verdict.split()[0].upper()
        if v0 not in _ALLOWED_VERDICTS:
            return {
                "label": "incorrect",
                "reason": "invalid verdict field",
                "contract": contract,
            }

    # Prefer direct answer alignment when present
    if direct.strip():
        ok = _direct_answer_aligns(direct, correct_answers)
        if ok and not _has_contradictory_polarity(response):
            return {
                "label": "correct",
                "reason": "direct answer aligns",
                "contract": contract,
            }
        if ok and _has_contradictory_polarity(response):
            return {
                "label": "incorrect",
                "reason": "polarity contradiction",
                "contract": contract,
            }
        return {
            "label": "incorrect",
            "reason": "direct answer does not align",
            "contract": contract,
        }

    # Fallback: obvious inference from the whole response
    if _any_contains(response, correct_answers) and not _has_contradictory_polarity(
        response
    ):
        return {
            "label": "correct",
            "reason": "answer inferable from response",
            "contract": contract,
        }

    # Boolean fallback
    truth_bools: Set[str] = {(_bool_from_text(x) or "") for x in correct_answers} - {""}
    if truth_bools:
        found = _bool_from_text(response)
        if found and found in truth_bools and not _has_contradictory_polarity(response):
            return {
                "label": "correct",
                "reason": "boolean aligns",
                "contract": contract,
            }

    return {"label": "unknown", "reason": "no alignment detected", "contract": contract}
