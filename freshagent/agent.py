# freshagent/agent.py
import json
import datetime
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from .prompts import render_react_prompt, REFLECT_AFTER_TOOL, FINAL_SYNTH_PROMPT
from core.llm_api import call_llm_messages
from .tools import TOOL_REGISTRY, tools_to_openai_format
from freshagent.debug import pretty_debug


# --------------------------- #
#       Helper functions      #
# --------------------------- #
def _assistant_message_to_dict(m) -> dict:
    """Convert OpenAI message object to a plain dict we can append to messages."""
    out = {"role": "assistant", "content": getattr(m, "content", "") or ""}
    tcs = getattr(m, "tool_calls", None)
    if tcs:
        conv = []
        for tc in tcs:
            fn = getattr(tc, "function", None)
            conv.append(
                {
                    "id": getattr(tc, "id", ""),
                    "type": getattr(tc, "type", "function"),
                    "function": {
                        "name": getattr(fn, "name", "") if fn else "",
                        "arguments": getattr(fn, "arguments", "") if fn else "",
                    },
                }
            )
        out["tool_calls"] = conv
    return out


def _extract_latest_reflection(messages: List[Dict[str, Any]]) -> str:
    """Return the last assistant content (used as 'reflection' snapshot seed)."""
    for m in reversed(messages):
        if m.get("role") == "assistant":
            return (m.get("content") or "").strip()
    return ""


def _build_context_snapshot(user_query: str, reflection_text: str) -> str:
    """Build a short system snapshot to keep model focused on the target."""
    rt = reflection_text.strip()
    if not rt:
        return (
            f"SNAPSHOT: Focus on answering the user's question.\nQuestion: {user_query}"
        )
    return (
        "SNAPSHOT: You must stay focused on the user's question.\n"
        f"Question: {user_query}\n"
        "Recent reflection summary (for focus, not for quoting):\n"
        f"{rt}\n"
        "Use exactly ONE tool in the next step if needed; do not drift."
    )


def _final_context_from_prompt(user_query: str, messages: list[dict]) -> str:
    """Format FINAL_SYNTH_PROMPT with the user question and aggregated evidence."""
    evidences: list[str] = [
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


def extract_direct_answer(final_text: str) -> str:
    """Best-effort extractor for the 'Direct Answer' field from an Answer Contract.

    Fallbacks gracefully to heuristics when the contract is missing:
      - If 'Direct Answer:' line is present, return its content (single line)
      - Else if 'Final Answer:' header present, return the first non-empty line after it
      - Else return the first non-empty line of the text
    """
    import re

    t = (final_text or "").strip()
    if not t:
        return ""

    # 1) Direct Answer line (allow bullets and different separators)
    m = re.search(r"(?im)^[\s\-•>*]*Direct\s*Answer\s*[:\-–]\s*(.*)$", t)
    if m:
        val = m.group(1).strip()
        if val:
            return val
        # If empty on the same line, try the next non-empty line
        after = t[m.end():].splitlines()
        for line in after:
            ls = line.strip()
            if ls:
                return ls
        return ""

    # 2) After a 'Final Answer' header, take next non-empty line
    fa = re.search(r"(?im)^[\s\-•>*]*Final\s*Answer\s*[:\-–]?(.*)$", t)
    if fa:
        tail = t[fa.end() :].strip()
        for line in tail.splitlines():
            ls = line.strip()
            if ls:
                return ls

    # 3) If Verdict is present, return it (useful fallback for yes/no questions)
    v = re.search(r"(?im)^[\s\-•>*]*Verdict\s*[:\-–]\s*(.+)$", t)
    if v:
        return v.group(1).strip()

    # 4) Fallback: first non-empty line
    for line in t.splitlines():
        ls = line.strip()
        if ls:
            return ls
    return t


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
    """Build initial ReAct conversation messages with proper date injection."""
    sys = render_react_prompt(_today_context())
    if context:
        sys = context.strip() + "\n\n" + sys
    return [
        {"role": "system", "content": sys},
        {"role": "user", "content": user_query},
    ]


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


# TODO： evaluate token limitation
def _hp(model: str):
    """Return default search/token parameters depending on model family."""
    if str(model).startswith("gpt-4"):
        return dict(
            n_org=15,
            n_rel=3,
            n_qa=3,
            n_evd=15,
            max_tokens=256,  # 反思/中间轮
            max_tokens_final=512,  # 最后一轮 Answer Contract
            chat=True,
        )
    return dict(
        n_org=15,
        n_rel=2,
        n_qa=2,
        n_evd=5,
        max_tokens=256,
        max_tokens_final=512,
        chat=True,
    )


# --------------------------- #
#        Main  Agent          #
# --------------------------- #
class Agent:
    """Minimal ReAct-style temporal reasoning agent."""

    def __init__(self, config: AgentConfig):
        self.cfg = config

    def run(self, query: str, dbg: bool = False) -> str:
        """
        Multi-round ReAct with OpenAI function-calling tools:
          - One tool call at most per round
          - Last round: force final synthesis (no tools)
        """
        # Allow enabling debug from either the call site or config
        dbg = dbg or getattr(self.cfg, "dbg", False)
        hp = _hp(self.cfg.model)
        messages: List[Dict[str, Any]] = build_react_messages(query)
        tools_spec = tools_to_openai_format(TOOL_REGISTRY) if TOOL_REGISTRY else None

        if dbg:
            pretty_debug(messages, title="INIT")

        for step in range(1, self.cfg.max_steps + 1):
            steps_left = self.cfg.max_steps - step + 1

            # Snapshot keeps the model focused (only if not the last step)
            if step > 1 and steps_left > 1:
                latest_reflection = _extract_latest_reflection(messages)
                if latest_reflection:
                    snapshot_text = _build_context_snapshot(query, latest_reflection)
                    messages.append({"role": "system", "content": snapshot_text})
                    if dbg:
                        pretty_debug(messages, title="SNAPSHOT ADDED")

            # Final round: inject final synthesis context and disable tools
            use_tools = tools_spec if steps_left > 1 else None
            if steps_left == 1:
                final_ctx = _final_context_from_prompt(query, messages)
                messages.append({"role": "system", "content": final_ctx})
                if dbg:
                    pretty_debug(messages, title="FINAL CONTEXT")

            # 1) Call LLM
            this_max = hp["max_tokens_final"] if steps_left == 1 else hp["max_tokens"]
            out = call_llm_messages(
                messages=messages,
                model=self.cfg.model,
                temperature=self.cfg.temperature,
                max_tokens=this_max,
                tools=use_tools,
            )
            m = out.get("message")
            if m is None:
                return "[ERROR] LLM did not return a message."

            # 2) Append assistant message
            assistant_dict = _assistant_message_to_dict(m)
            messages.append(assistant_dict)

            if dbg:
                pretty_debug(messages, title="AFTER ASSISTANT")

            content = (assistant_dict.get("content") or "").strip()
            tool_calls = assistant_dict.get("tool_calls") or []

            # 3) Early finish if final answer is present
            if ("Final Answer:" in content) or (
                "Premise:" in content and "Verdict:" in content
            ):
                return content

            # 4) Tool call branch (only if not last step)
            if tool_calls and steps_left > 1:
                first_tc = tool_calls[0]
                fn = first_tc.get("function", {}) or {}
                fn_name = (fn.get("name") or "").strip()
                args_str = fn.get("arguments") or "{}"
                try:
                    args = json.loads(args_str)
                except Exception:
                    args = {}

                # Provide provider for google if missing
                if fn_name == "google" and "provider" not in args:
                    args["provider"] = self.cfg.provider

                if fn_name not in TOOL_REGISTRY:
                    available = ", ".join(sorted(TOOL_REGISTRY.keys()))
                    messages.append(
                        {
                            "role": "system",
                            "content": f"Requested tool '{fn_name}' is not available. Available: {available}.",
                        }
                    )
                    continue

                tool = TOOL_REGISTRY[fn_name]
                result = tool.func(args) or {}

                # Append tool result as a 'tool' message (with tool_call_id)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": first_tc.get("id") or "",
                        "content": json.dumps(result, ensure_ascii=False),
                    }
                )

                # Inject readable evidence and require reflection
                _inject_evidence(messages, result, source_name=fn_name)
                messages.append({"role": "system", "content": REFLECT_AFTER_TOOL})
                if dbg:
                    pretty_debug(messages, title=f"TOOL '{fn_name}' RESULT + EVIDENCE")
                continue

            # 5) No tools, no final — continue next round
            continue

        # 6) Fallback if max steps exhausted
        if messages and messages[-1].get("role") == "assistant":
            return messages[-1].get("content", "[Stopped: max steps reached]")
        return "[Stopped: max steps reached]"

    def run_parts(self, query: str, dbg: bool = False) -> Dict[str, str]:
        """Run the agent and return both full text and extracted direct answer.

        Returns a dict with keys:
          - 'full': the complete assistant output (Answer Contract or final text)
          - 'direct': best-effort extracted direct answer string
        """
        full = self.run(query, dbg=dbg)
        direct = extract_direct_answer(full)
        return {"full": full, "direct": direct}
