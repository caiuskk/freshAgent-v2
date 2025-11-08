import datetime
import re
from typing import Any, Dict, Optional

import pandas as pd

# dateutil fallback
try:
    import dateutil  # type: ignore
except Exception:
    from dateutil import parser as _parser  # type: ignore

    class _DU:
        pass

    dateutil = _DU()  # type: ignore
    dateutil.parser = _parser  # type: ignore

# default timezone and reference date
try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None  # type: ignore

_TZ = ZoneInfo("America/Chicago") if ZoneInfo else None
CURRENT_DATE_STR = datetime.datetime.now(_TZ).strftime("%b %d, %Y")


def is_date(string, fuzzy=False):
    """Return True if string can be parsed as a date."""
    try:
        dateutil.parser.parse(string, fuzzy=fuzzy)  # type: ignore
        return True
    except Exception:
        return False


def format_date(d):
    """Normalize date strings to '%b %d, %Y'."""
    if d is None:
        return None

    today_str = CURRENT_DATE_STR
    d_str = str(d).strip()
    d_lc = d_str.lower()

    for t in ["second", "minute", "hour"]:
        if f"{t} ago" in d_lc or f"{t}s ago" in d_lc:
            return today_str

    if "day ago" in d_lc or "days ago" in d_lc:
        m = re.search(r"(\d+)\s+days?\s+ago", d_lc)
        if m:
            try:
                n_days = int(m.group(1))
                return (
                    datetime.datetime.strptime(today_str, "%b %d, %Y")
                    - datetime.timedelta(days=n_days)
                ).strftime("%b %d, %Y")
            except Exception:
                return today_str
        return today_str

    try:
        return dateutil.parser.parse(d_str, fuzzy=True).strftime("%b %d, %Y")  # type: ignore
    except Exception:
        for x in d_str.split():
            if is_date(x):
                try:
                    return dateutil.parser.parse(x, fuzzy=True).strftime("%b %d, %Y")  # type: ignore
                except Exception:
                    continue
    return None


def extract_source_webpage(link):
    """Return domain name from a URL-like string."""
    if not link:
        return None
    try:
        return (
            str(link)
            .strip()
            .replace("https://www.", "")
            .replace("http://www.", "")
            .replace("https://", "")
            .replace("http://", "")
            .split("/")[0]
        )
    except Exception:
        return None


def simplify_displayed_link(displayed_link):
    """Simplify displayed link to its main domain."""
    if displayed_link is None:
        return None
    try:
        left = str(displayed_link).split(" › ")[0]
        return extract_source_webpage(left)
    except Exception:
        return extract_source_webpage(displayed_link)


def format_search_results(search_data, title_field=None, highlight_field=None):
    """Normalize heterogeneous search results into a unified evidence schema."""
    sd = dict(search_data or {})

    field = "snippet_highlighted_words"
    if field in sd and isinstance(sd[field], list):
        sd[field] = " | ".join(str(x) for x in sd[field])

    if "displayed_link" in sd:
        sd["displayed_link"] = simplify_displayed_link(sd["displayed_link"])
    else:
        fallback = sd.get("source") or extract_source_webpage(sd.get("link"))
        sd["displayed_link"] = simplify_displayed_link(fallback)

    source = date = title = snippet = highlight = None

    if sd.get("type") == "local_time":
        source = sd.get("displayed_link")
        date = format_date(sd.get("date"))
        title = sd.get("title")
        snippet = sd.get("snippet")
        if snippet is None and "result" in sd:
            if "extensions" in sd and isinstance(sd["extensions"], list):
                snippet = "\n\t".join(
                    [sd["result"]] + [str(x) for x in sd["extensions"]]
                )
            else:
                snippet = sd["result"]
        highlight = sd.get("snippet_highlighted_words") or sd.get("result")

    elif sd.get("type") == "population_result":
        source = sd.get("displayed_link")
        if source is None and "sources" in sd:
            try:
                if (
                    isinstance(sd["sources"], list)
                    and sd["sources"]
                    and "link" in sd["sources"][0]
                ):
                    source = extract_source_webpage(sd["sources"][0]["link"])
            except Exception:
                pass
        date = format_date(sd.get("date")) or format_date(sd.get("year"))
        title = sd.get("title")
        snippet = sd.get("snippet")
        if snippet is None and "population" in sd:
            if "place" in sd:
                snippet = "\n\t".join(
                    [f"{sd.get('place') or ''} / Population", str(sd.get("population"))]
                )
            else:
                snippet = str(sd.get("population"))
        highlight = sd.get("snippet_highlighted_words") or str(sd.get("population"))

    else:
        source = sd.get("displayed_link")
        date = format_date(sd.get("date"))
        title = sd.get("title") if title_field is None else sd.get(title_field)
        highlight = (
            sd.get("snippet_highlighted_words")
            if highlight_field is None
            else sd.get(highlight_field)
        )
        snippet = sd.get("snippet", "")

        if "rich_snippet" in sd:
            try:
                for key in ["top", "bottom"]:
                    if (
                        key in sd["rich_snippet"]
                        and "extensions" in sd["rich_snippet"][key]
                    ):
                        exts = sd["rich_snippet"][key]["extensions"]
                        if isinstance(exts, list):
                            snippet = "\n\t".join([snippet] + [str(x) for x in exts])
            except Exception:
                pass

        if "list" in sd and isinstance(sd["list"], list):
            snippet = "\n\t".join([snippet] + [str(x) for x in sd["list"]])

        if (
            "contents" in sd
            and isinstance(sd["contents"], dict)
            and "table" in sd["contents"]
        ):
            tbl = sd["contents"]["table"]
            snippet = (snippet or "") + "\n"
            if isinstance(tbl, list):
                for row in tbl:
                    if isinstance(row, list):
                        snippet += f"\n{','.join(str(x) for x in row)}"
                    else:
                        snippet += f"\n{str(row)}"

        if snippet is not None and str(snippet).strip() == "":
            snippet = None

    return {
        "source": source,
        "date": date,
        "title": title,
        "snippet": snippet,
        "highlight": highlight,
    }


def format_knowledge_graph(search_data):
    """Normalize a knowledge-graph result into the evidence schema."""
    sd = dict(search_data or {})
    source = None
    try:
        src = sd.get("source")
        if isinstance(src, dict) and "link" in src:
            source = extract_source_webpage(src["link"])
    except Exception:
        pass
    date = None
    title = None
    if "title" in sd:
        title = str(sd["title"])
        if "type" in sd:
            title += f"\n\t{str(sd['type'])}"
    snippet_lines = []
    for field, val in sd.items():
        if (
            field not in ["title", "type", "kgmid"]
            and "_link" not in field
            and "_stick" not in field
            and isinstance(val, str)
            and not val.startswith("http")
        ):
            snippet_lines.append(f"\t{field}: {val}")
    snippet = "\n".join(snippet_lines).strip() if snippet_lines else None
    return {
        "source": source,
        "date": date,
        "title": title,
        "snippet": snippet,
        "highlight": None,
    }


def format_questions_and_answers(search_data):
    """Normalize Q&A-style results."""
    sd = dict(search_data or {})
    source = extract_source_webpage(sd["link"]) if "link" in sd else None
    return {
        "source": source,
        "date": None,
        "title": sd.get("question"),
        "snippet": sd.get("answer"),
        "highlight": None,
    }


def freshprompt_format(
    question,
    search_data,
    reasoning_and_answer,
    num_organic_results,
    num_related_questions,
    num_questions_and_answers,
    num_retrieved_evidences,
):
    """Assemble evidence strings into the FreshPrompt text block."""
    df = pd.DataFrame(columns=["source", "date", "title", "snippet", "highlight"])
    sd = search_data or {}

    def _append(rows):
        nonlocal df
        for d in rows[::-1]:
            df = pd.concat([df, pd.DataFrame([d])], ignore_index=True)

    # organic
    organic = [
        format_search_results(sd["organic_results"][k])
        if "organic_results" in sd and len(sd["organic_results"]) > k
        else format_search_results({})
        for k in range(num_organic_results)
    ]
    _append(organic)

    # related
    related = [
        format_search_results(sd["related_questions"][k], title_field="question")
        if "related_questions" in sd and len(sd["related_questions"]) > k
        else format_search_results({})
        for k in range(num_related_questions)
    ]
    _append(related)

    # QA
    qa = [
        format_questions_and_answers(sd["questions_and_answers"][k])
        if "questions_and_answers" in sd and len(sd["questions_and_answers"]) > k
        else format_questions_and_answers({})
        for k in range(num_questions_and_answers)
    ]
    _append(qa)

    # knowledge graph
    kg = (
        format_knowledge_graph(sd["knowledge_graph"])
        if "knowledge_graph" in sd
        else format_knowledge_graph({})
    )
    _append([kg])

    # answer box
    ab = (
        format_search_results(sd["answer_box"], highlight_field="answer")
        if "answer_box" in sd
        else format_search_results({})
    )
    _append([ab])

    df["date"] = df["date"].apply(format_date)
    df["datetime"] = pd.to_datetime(df["date"], errors="coerce")
    df = (
        df.sort_values(by="datetime", na_position="first")
        .replace({pd.NaT: None})
        .dropna(how="all")
    )

    evidences = []
    if not df.empty:
        for _, row in df.tail(num_retrieved_evidences).iterrows():
            evidences.append(
                "\n\n"
                f"source: {row.get('source')}\n"
                f"date: {row.get('date')}\n"
                f"title: {row.get('title')}\n"
                f"snippet: {row.get('snippet')}\n"
                f"highlight: {row.get('highlight')}"
            )

    return (
        "".join(["\n\n\nquery: " + question] + evidences)
        + f"\n\nquestion: {question}{reasoning_and_answer}"
    )
