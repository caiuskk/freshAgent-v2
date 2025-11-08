"""
search_engine.py

Unified search interface for FreshAgent.

Supports both Serper.dev and SerpAPI providers.
Serper results are normalized into SerpAPI-style keys
so that downstream modules (e.g., FreshPrompt formatters)
can use a consistent schema.

Requirements:
    - requests
    - serpapi (install via `pip install google-search-results`) [only when provider="serpapi"]

Environment variables:
    - SERPER_API_KEY
    - SERPAPI_API_KEY
"""

import os
import json
import requests

try:
    from serpapi import GoogleSearch  # only when provider="serpapi"
except Exception:
    GoogleSearch = None


def _serper_request(query: str, gl="us", hl="en", timeout=30):
    """
    Perform a search request using the Serper.dev API.

    Args:
        query (str): Search query string.
        gl (str): Geolocation code (default: "us").
        hl (str): Language code (default: "en").
        timeout (int): Request timeout in seconds.

    Returns:
        dict: A response dict with keys:
            - ok (bool)
            - data (dict, present when ok=True)
            - error/detail (present when ok=False)
    """
    api_key = os.getenv("SERPER_API_KEY")
    if not api_key:
        return {"ok": False, "error": "SERPER_API_KEY not found in environment"}

    url = "https://google.serper.dev/search"
    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}
    payload = {"q": query, "gl": gl, "hl": hl}

    try:
        resp = requests.post(
            url, headers=headers, data=json.dumps(payload), timeout=timeout
        )
        resp.raise_for_status()
        return {"ok": True, "data": resp.json() if resp.content else {}}
    except requests.HTTPError:
        try:
            detail = resp.json()
        except Exception:
            detail = {"message": resp.text}
        return {"ok": False, "error": f"HTTP {resp.status_code}", "detail": detail}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def call_search_engine(
    query: str,
    provider: str = "serper",  # "serper" or "serpapi"
    gl: str = "us",
    hl: str = "en",
    timeout: int = 30,
):
    """
    Unified search interface used by FreshAgent.

    This function provides a consistent SerpAPI-like structure regardless of provider. Though it assumes serper.dev in default, to keep compatibility and Expandability with existing other Freshprompt components, the output schema follows SerpAPI's format.
    When using Serper.dev, it maps fields to match SerpAPI output schema:
        - organic_results
        - answer_box
        - knowledge_graph
        - related_questions
        - questions_and_answers
        - news
        - images

    Args:
        query (str): Search query.
        provider (str): Either "serper" or "serpapi" (default: "serper").
        gl (str): Country code (default: "us").
        hl (str): Language code (default: "en").
        timeout (int): HTTP timeout in seconds.

    Returns:
        dict: A normalized search result dictionary:
            {
                "ok": bool,
                "organic_results": list,
                "answer_box": dict,
                "knowledge_graph": dict,
                "related_questions": list,
                "questions_and_answers": list,
                "news": list,
                "images": list,
                "_raw": dict,
            }
    """

    prov = (provider or "serper").lower()

    # ----- SerpAPI provider -----
    if prov == "serpapi":
        if GoogleSearch is None:
            raise RuntimeError(
                "SerpAPI not installed. Run `pip install google-search-results`."
            )
        serpapi_api_key = os.getenv("SERPAPI_API_KEY")
        if not serpapi_api_key:
            return {"ok": False, "error": "SERPAPI_API_KEY not set in environment"}

        params = {
            "q": query,
            "hl": hl,
            "gl": gl,
            "google_domain": "google.com",
            "api_key": serpapi_api_key,
        }
        search = GoogleSearch(params)
        raw = search.get_dict() or {}

        # Ensure all expected keys exist to avoid KeyError downstream
        return {
            "ok": True,
            "organic_results": raw.get("organic_results", []) or [],
            "answer_box": raw.get("answer_box", {}) or {},
            "knowledge_graph": raw.get("knowledge_graph", {}) or {},
            "related_questions": raw.get("related_questions", []) or [],
            "questions_and_answers": raw.get("questions_and_answers", []) or [],
            "news": raw.get("news_results", raw.get("news", [])) or [],
            "images": raw.get("images_results", raw.get("images", [])) or [],
            "_raw": raw,
        }

    # ----- Serper.dev provider -----
    ser = _serper_request(query, gl=gl, hl=hl, timeout=timeout)
    if not ser.get("ok"):
        return ser

    raw = ser.get("data") or {}

    # --- 1. Map organic → organic_results (normalize displayed_link field)
    organic_results = []
    for item in raw.get("organic", []) or []:
        displayed = item.get("source") or item.get("displayed_link")
        if not displayed and item.get("link"):
            try:
                displayed = item["link"].split("//", 1)[-1].split("/", 1)[0]
            except Exception:
                displayed = None

        organic_results.append({**item, "displayed_link": displayed})

    # --- 2. Map answerBox → answer_box
    answer_box = raw.get("answerBox", {}) or {}

    # --- 3. Map knowledgeGraph → knowledge_graph
    knowledge_graph = raw.get("knowledgeGraph", {}) or {}

    # --- 4. Map peopleAlsoAsk → related_questions + questions_and_answers
    paa = raw.get("peopleAlsoAsk", []) or []
    related_questions, questions_and_answers = [], []
    for x in paa:
        related_questions.append(
            {
                "question": x.get("question"),
                "snippet": x.get("snippet") or x.get("answer"),
                "displayed_link": x.get("source") or x.get("link"),
            }
        )
        questions_and_answers.append(
            {
                "question": x.get("question"),
                "answer": x.get("answer") or x.get("snippet"),
                "link": x.get("link") or x.get("source"),
            }
        )

    # --- 5. Pass through news/images if available
    news = raw.get("news", []) or []
    images = raw.get("images", []) or []

    # --- Return normalized SerpAPI-like structure
    return {
        "ok": True,
        "organic_results": organic_results,
        "answer_box": answer_box,
        "knowledge_graph": knowledge_graph,
        "related_questions": related_questions,
        "questions_and_answers": questions_and_answers,
        "news": news,
        "images": images,
        "_raw": raw,
    }
