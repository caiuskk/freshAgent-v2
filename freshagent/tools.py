from dataclasses import dataclass
from typing import Callable, Dict, Any, List
from core.search_engine import call_search_engine
from core.search_result_formatters import freshprompt_format


@dataclass
class Tool:
    name: str
    description: str
    json_schema: Dict[str, Any]
    func: Callable[[Dict[str, Any]], Dict[str, Any]]


def tools_to_openai_format(registry: Dict[str, Tool]) -> List[Dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.json_schema,
            },
        }
        for t in registry.values()
    ]


_TOP_ORGANIC = 5
_TOP_RELATED = 3
_TOP_QA = 3
_TOP_EVID = 6
_SUFFIX = "\n\nReasoning: <agent to fill>\nAnswer: <agent to fill>"  # TODO: need here??


def _formatted_google_search(question: str, provider: str = "serper") -> str:
    sdata = call_search_engine(question, provider=provider)
    prompt = freshprompt_format(
        question=question,
        search_data=sdata,
        reasoning_and_answer=_SUFFIX,
        num_organic_results=_TOP_ORGANIC,
        num_related_questions=_TOP_RELATED,
        num_questions_and_answers=_TOP_QA,
        num_retrieved_evidences=_TOP_EVID,
    )
    return prompt


def _google_impl(args: Dict[str, Any]) -> Dict[str, Any]:
    q = args.get("question", "")
    provider = args.get("provider", "serper")
    try:
        p = _formatted_google_search(q, provider=provider)
        return {"ok": True, "question": q, "prompt": p}
    except Exception as e:
        return {"ok": False, "error": str(e)}


google_tool = Tool(
    name="google",
    description="Search the web and return a FreshPrompt-style evidence block.",
    json_schema={
        "type": "object",
        "properties": {
            "question": {"type": "string"},
            "provider": {"type": "string", "enum": ["serper", "serpapi"]},
        },
        "required": ["question"],
    },
    func=_google_impl,
)


def _calc_impl(args: Dict[str, Any]) -> Dict[str, Any]:
    expr = str(args.get("expression", ""))
    try:
        result = eval(expr, {"__builtins__": {}})
        return {"ok": True, "result": result}
    except Exception as e:
        return {"ok": False, "error": str(e)}


calculator_tool = Tool(
    name="calculator",
    description="Safely evaluate a simple arithmetic expression like '2+2*3'.",
    json_schema={
        "type": "object",
        "properties": {"expression": {"type": "string"}},
        "required": ["expression"],
    },
    func=_calc_impl,
)

TOOL_REGISTRY: Dict[str, Tool] = {
    google_tool.name: google_tool,
    calculator_tool.name: calculator_tool,
}
