# freshagent/debug.py
from __future__ import annotations
import json
import datetime
from typing import Any, Dict, List, Optional

SEPARATOR = "=" * 80
SUBSEP = "-" * 80


def _ts() -> str:
    return datetime.datetime.now().strftime("%H:%M:%S")


def _truncate(text: str, limit: int) -> str:
    if text is None:
        return ""
    t = str(text)
    if len(t) <= limit:
        return t
    return t[:limit] + f"\n...[truncated {len(t) - limit} chars]"


def _is_evidence_block(content: str) -> bool:
    c = (content or "").upper()
    return (
        "EVIDENCE BLOCK" in c or "EVIDENCE (RAW" in c or "FINAL SYNTHESIS CONTEXT" in c
    )


def _render_tool_calls(m: Dict[str, Any]) -> Optional[str]:
    tcs = m.get("tool_calls")
    if not tcs:
        return None
    lines = ["tool_calls:"]
    for tc in tcs:
        fn = tc.get("function", {}) or {}
        lines.append(f"  - id: {tc.get('id', '')}")
        lines.append(f"    type: {tc.get('type', 'function')}")
        lines.append(f"    function.name: {fn.get('name', '')}")
        # Arguments are usually a JSON string; truncate for readability
        args = fn.get("arguments", "")
        lines.append(
            "    function.arguments: " + _truncate(args, 400).replace("\n", " ")
        )
    return "\n".join(lines)


def _render_tool_message(m: Dict[str, Any]) -> str:
    content = m.get("content", "")
    try:
        obj = json.loads(content)
        pretty = json.dumps(obj, ensure_ascii=False, indent=2)
    except Exception:
        pretty = content
    return _truncate(pretty, 2000)


def pretty_debug(
    messages: List[Dict[str, Any]],
    title: Optional[str] = None,
    max_content_chars: int = 2000,
) -> None:
    """
    Print the current message trace in a structured way.
    - Groups by role, highlights EVIDENCE/SNAPSHOT/FINAL blocks
    - Summarizes assistant tool_calls and tool message content
    - Truncates long content for readability
    """
    print(SEPARATOR)
    head = f"[{_ts()}] TRACE" + (f" — {title}" if title else "")
    print(head)
    print(SEPARATOR)

    for idx, m in enumerate(messages):
        role = m.get("role", "").upper()
        print(f"[{idx:02d}] ROLE={role}")
        if role == "ASSISTANT":
            calls = _render_tool_calls(m)
            if calls:
                print(calls)
        if role == "TOOL":
            tcid = m.get("tool_call_id", "")
            print(f"tool_call_id: {tcid}")
            print(SUBSEP)
            print(_render_tool_message(m))
            print(SUBSEP)
            continue

        content = m.get("content", "") or ""
        if _is_evidence_block(content):
            # Highlight big blocks: EVIDENCE, FINAL SYNTHESIS CONTEXT, SNAPSHOT
            print(SUBSEP)
            # Print first chunk fully, then truncate
            print(_truncate(content, max_content_chars))
            print(SUBSEP)
        else:
            # Regular short messages
            print(_truncate(content, 800))

    print(SEPARATOR)
    print()


def compact_summary(messages: List[Dict[str, Any]], limit: int = 10) -> str:
    """
    Returns a short, single-string summary of the last few messages (for logging).
    """
    tail = messages[-limit:]
    lines = []
    for m in tail:
        role = m.get("role", "").upper()
        content = (m.get("content", "") or "").strip().replace("\n", " ")
        content = _truncate(content, 200)
        lines.append(f"[{role}] {content}")
    return " | ".join(lines)


def save_trace(
    messages: List[Dict[str, Any]], path_txt: str, path_json: Optional[str] = None
) -> None:
    """
    Save a human-readable TXT trace, and optionally the raw messages as JSON.
    """
    with open(path_txt, "w", encoding="utf-8") as f:
        f.write(SEPARATOR + "\n")
        f.write(f"TRACE saved at {_ts()}\n")
        f.write(SEPARATOR + "\n\n")
        for i, m in enumerate(messages):
            f.write(f"[{i:02d}] ROLE={m.get('role', '').upper()}\n")
            if m.get("role", "").lower() == "assistant" and m.get("tool_calls"):
                f.write((_render_tool_calls(m) or "") + "\n")
            if m.get("role", "").lower() == "tool":
                f.write("tool_call_id: " + str(m.get("tool_call_id", "")) + "\n")
                f.write(SUBSEP + "\n")
                f.write(_render_tool_message(m) + "\n")
                f.write(SUBSEP + "\n")
            else:
                content = m.get("content", "") or ""
                f.write(_truncate(content, 4000) + "\n")
            f.write("\n")
    if path_json:
        import json as _json

        with open(path_json, "w", encoding="utf-8") as fj:
            _json.dump(messages, fj, ensure_ascii=False, indent=2)
