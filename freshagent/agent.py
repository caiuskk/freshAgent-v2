# freshagent/agent.py
import json
import datetime
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from core.llm_api import call_llm_api
from .prompts import REACT_PROMPT, REFLECT_AFTER_TOOL, FINAL_SYNTH_PROMPT
from .tools import TOOL_REGISTRY


# --------------------------- #
#       Helper functions      #
# --------------------------- #


def _final_context_from_prompt(user_query: str, messages: List[Dict[str, Any]]) -> str:
    """Format FINAL_SYNTH_PROMPT with the user question and aggregated evidence."""
    evidences: List[str] = [
        m["content"]
        for m in messages
        if m.get("role") == "system" and "EVIDENCE" in (m.get("content") or "")
    ]
    evidence_text = (
        "\n\n---\n\n".join(evidences) if evidences else "[no evidence gathered]"
    )
    return FINAL_SYNTH_PROMPT.format(
        USER_QUESTION=user_query.strip(),
        EVIDENCE_TEXT=evidence_text,
    )


def _today_context(tz_name: str = "America/Chicago") -> str:
    """Return current datetime string with timezone name."""
    try:
        from zoneinfo import ZoneInfo  # Python 3.9+

        tz = ZoneInfo(tz_name)
        now = datetime.datetime.now(tz)
    except Exception:
        now = datetime.datetime.now()
    return now.strftime("%a, %b %d, %Y %H:%M %Z")


def build_react_messages(
    user_query: str, context: Optional[str] = None
) -> List[Dict[str, str]]:
    """Build initial ReAct conversation messages."""
    sys = f"Today is {_today_context()}.\n\n" + REACT_PROMPT
    if context:
        sys = context.strip() + "\n\n" + sys
    return [
        {"role": "system", "content": sys},
        {"role": "user", "content": user_query},
    ]


def _concat_chat(messages: List[Dict[str, Any]]) -> str:
    """
    Flatten a list of chat messages into a single plain-text prompt.
    Each message is prefixed by its role name for clarity.
    """
    out_lines = []
    for m in messages:
        role = m.get("role", "unknown").upper()
        content = m.get("content", "").strip()
        out_lines.append(f"[{role}]\n{content}\n")
    return "\n".join(out_lines)


def _inject_evidence(
    messages: List[Dict[str, Any]], result: Dict[str, Any], source_name: str = "google"
) -> None:
    """
    Insert evidence blocks into the conversation.
    Supports FreshPrompt-style dicts (with 'prompt') or raw JSON fallback.
    """
    try:
        if isinstance(result, dict) and result.get("prompt"):
            evidence_text = result["prompt"]
            messages.append(
                {
                    "role": "system",
                    "content": (
                        f"EVIDENCE BLOCK (from {source_name}):\n"
                        + (
                            evidence_text
                            if isinstance(evidence_text, str)
                            else json.dumps(evidence_text, ensure_ascii=False, indent=2)
                        )
                        + "\n\nInstructions: Base your next reasoning ONLY on the above evidence. "
                        "If the evidence looks stale for a time-varying query, either search again or say Uncertain."
                    ),
                }
            )
            return

        # Fallback to pretty JSON dump
        fallback = json.dumps(result, ensure_ascii=False, indent=2)
        messages.append(
            {
                "role": "system",
                "content": (
                    f"EVIDENCE (raw, from {source_name}):\n{fallback}\n\n"
                    "Use ONLY the above evidence to continue; if it is inadequate or stale, "
                    "search again or mark the result as Uncertain."
                ),
            }
        )
    except Exception:
        pass


# --------------------------- #
#        Agent Config         #
# --------------------------- #
@dataclass
class AgentConfig:
    model: str = "gpt-4o"
    provider: str = "serper"
    max_steps: int = 8
    temperature: float = 0.0
    dbg: bool = False


def _hp(model: str):
    """Return default search parameters depending on model family."""
    if str(model).startswith("gpt-4"):
        return dict(n_org=15, n_rel=3, n_qa=3, n_evd=15, max_tokens=256, chat=True)
    return dict(n_org=15, n_rel=2, n_qa=2, n_evd=5, max_tokens=256, chat=True)


# --------------------------- #
#        Main  Agent          #
# --------------------------- #
class Agent:
    """Minimal ReAct-style temporal reasoning agent."""

    def __init__(self, config: AgentConfig):
        self.cfg = config

    def run(self, query: str, dbg: bool = False) -> str:
        """
        Multi-round ReAct loop with exactly one tool call per round.
        Assumes helper functions already exist in this module:
          - build_react_messages(...)
          - _inject_evidence(...)
          - _build_final_context_block(...)
        Also assumes TOOL_REGISTRY and prompts are imported.
        """

        # Small model-specific limits; adjust as needed
        def _hp(model: str):
            if str(model).startswith("gpt-4"):
                return dict(max_tokens=384, chat=True)
            return dict(max_tokens=320, chat=True)

        hp = _hp(self.cfg.model)

        # 0) bootstrap conversation
        messages: List[Dict[str, Any]] = build_react_messages(query)

        # Tool-call directive embedded per round. We constrain the model to emit either:
        #   - a single line: TOOL_CALL: {"name": "<tool>", "arguments": {...}}
        #   - or a single line: FINALIZE
        tool_names = ", ".join(sorted(TOOL_REGISTRY.keys()))
        TOOL_CALL_INSTR = (
            "Plan your next step. You may use at most ONE tool in this round.\n"
            f"Available tools: {tool_names}\n"
            "If you need a tool, output EXACTLY ONE line:\n"
            'TOOL_CALL: {"name": "<tool-name>", "arguments": {...}}\n'
            "If you do not need any tool, output exactly: FINALIZE\n"
            "Do not output anything else on that line."
        )

        # 1) iterative loop
        for step in range(1, self.cfg.max_steps + 1):
            messages.append({"role": "system", "content": TOOL_CALL_INSTR})

            # Ask the model for the next action (tool or finalize)
            planner = call_llm_api(
                prompt=_concat_chat(messages),
                model=self.cfg.model,
                temperature=self.cfg.temperature,
                max_tokens=hp["max_tokens"],
                chat_completions=True,
            )
            if dbg:
                print(f"\n[Step {step} Planner]\n{planner}")

            # Decide whether to finalize or run a tool
            lines = [ln.strip() for ln in (planner or "").splitlines() if ln.strip()]
            tool_line = next((ln for ln in lines if ln.startswith("TOOL_CALL:")), "")
            wants_finalize = any(ln == "FINALIZE" for ln in lines)

            if wants_finalize and not tool_line:
                break  # proceed to final synthesis

            if not tool_line:
                # No well-formed tool call produced; stop looping and finalize
                break

            # Parse tool call (single line JSON payload after "TOOL_CALL:")
            payload = tool_line.split("TOOL_CALL:", 1)[1].strip()
            call = json.loads(payload)  # let it error if malformed during development
            name = str(call.get("name", "")).strip()
            args = call.get("arguments", {}) or {}

            if name not in TOOL_REGISTRY:
                # Unknown tool; stop and finalize
                break

            # Execute exactly one tool
            tool = TOOL_REGISTRY[name]
            # Always pass provider so search tools can route correctly
            if "provider" not in args:
                args["provider"] = self.cfg.provider

            result = tool.func(args)

            # Inject evidence for the next reasoning step
            _inject_evidence(messages, result, source_name=name)

            # Force a reflection step before any further tool usage
            messages.append({"role": "system", "content": REFLECT_AFTER_TOOL})
            reflection = call_llm_api(
                prompt=_concat_chat(messages),
                model=self.cfg.model,
                temperature=self.cfg.temperature,
                max_tokens=hp["max_tokens"],
                chat_completions=True,
            )
            messages.append({"role": "assistant", "content": reflection})
            if dbg:
                print(f"\n[Step {step} Reflection]\n{reflection}")

            # loop continues to next round; still at most one tool in each round

        # 2) final synthesis (Answer Contract)
        final_ctx = self._build_final_context_block(query, messages)
        messages.append({"role": "system", "content": final_ctx})

        final = call_llm_api(
            prompt=_concat_chat(messages),
            model=self.cfg.model,
            temperature=self.cfg.temperature,
            max_tokens=hp["max_tokens"],
            chat_completions=True,
        )
        if dbg:
            print("\n[Final]\n", final)

        return final

    # --------------------------- #
    #       Internal helpers      #
    # --------------------------- #
    def _as_chat(self, messages: List[Dict[str, Any]]) -> str:
        """Serialize message list into a single text prompt (for text-only LLM API)."""
        lines = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            lines.append(f"[{role.upper()}]\n{content}\n")
        return "\n".join(lines)
