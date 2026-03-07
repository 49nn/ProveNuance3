"""
LLM-assisted drafting of evaluation case queries.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from runtime_env import get_required_env

_PROMPT_TEMPLATE_PATH = Path(__file__).with_name("draft_case_queries_prompt_template.txt")
_QUERY_RE = re.compile(r"^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:\((.*)\))?\s*$")
_EXPECTED_RESULTS = {"proved", "not_proved", "blocked", "unknown"}


@lru_cache(maxsize=1)
def _load_prompt_template() -> str:
    return _PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8")


def _format_predicate_signatures(predicate_positions: dict[str, list[str]]) -> str:
    lines: list[str] = []
    for predicate, roles in sorted(predicate_positions.items()):
        signature = f"{predicate}({', '.join(role.upper() for role in roles)})" if roles else f"{predicate}()"
        lines.append(f"- {signature}")
    return "\n".join(lines)


def _format_preferred_predicates(predicates: list[str]) -> str:
    if not predicates:
        return "- none"
    return "\n".join(f"- {predicate}" for predicate in sorted(dict.fromkeys(predicates)))


def build_case_query_draft_prompt(
    *,
    case_id: str,
    title: str,
    case_text: str,
    predicate_positions: dict[str, list[str]],
    preferred_predicates: list[str],
    max_queries: int,
    year: int,
) -> str:
    template = _load_prompt_template()
    return (
        template
        .replace("{{CASE_ID}}", case_id)
        .replace("{{CASE_TITLE}}", title or case_id)
        .replace("{{CASE_TEXT}}", case_text)
        .replace("{{PREDICATE_SIGNATURES}}", _format_predicate_signatures(predicate_positions))
        .replace("{{PREFERRED_PREDICATES}}", _format_preferred_predicates(preferred_predicates))
        .replace("{{MAX_QUERIES}}", str(max_queries))
        .replace("{{YEAR}}", str(year))
    )


def build_case_query_draft_schema() -> dict[str, Any]:
    query_schema = {
        "type": "OBJECT",
        "properties": {
            "query": {"type": "STRING"},
            "expected_result": {"type": "STRING"},
            "notes": {"type": "STRING"},
            "rationale": {"type": "STRING"},
        },
        "required": ["query", "expected_result"],
    }
    return {
        "type": "OBJECT",
        "properties": {
            "queries": {"type": "ARRAY", "items": query_schema},
        },
        "required": ["queries"],
    }


def parse_case_query_draft_response(
    data: dict[str, Any],
    *,
    predicate_positions: dict[str, list[str]],
    max_queries: int,
) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    seen_queries: set[str] = set()

    for item in data.get("queries", []):
        query = str(item.get("query", "")).strip()
        expected = str(item.get("expected_result", "")).strip().lower()
        if not query or expected not in _EXPECTED_RESULTS:
            continue

        match = _QUERY_RE.match(query)
        if match is None:
            continue
        predicate = match.group(1).strip().lower()
        args_blob = match.group(2) or ""
        args = [arg.strip() for arg in args_blob.split(",") if arg.strip()]
        expected_roles = predicate_positions.get(predicate)
        if expected_roles is None:
            continue
        if len(args) != len(expected_roles):
            continue

        normalized_query = (
            f"{predicate}({','.join(args)})"
            if args else predicate
        )
        if normalized_query in seen_queries:
            continue
        seen_queries.add(normalized_query)

        out.append({
            "query": normalized_query,
            "expected_result": expected,
            "notes": str(item.get("notes", "")).strip(),
            "rationale": str(item.get("rationale", "")).strip(),
        })
        if len(out) >= max_queries:
            break

    return out


class CaseQueryDrafter:
    def __init__(
        self,
        predicate_positions: dict[str, list[str]],
        config: Any,
        *,
        year: int = 2026,
        preferred_predicates: list[str] | None = None,
    ) -> None:
        try:
            from google import genai
            from google.genai import types as genai_types
        except ImportError as exc:
            raise ImportError(
                "CaseQueryDrafter wymaga pakietu google-genai. "
                "Zainstaluj: pip install google-genai"
            ) from exc

        if not predicate_positions:
            raise ValueError("CaseQueryDrafter wymaga aktywnej ontologii z predykatami.")

        api_key = get_required_env(config.api_key_env)
        timeout_ms = int(config.preflight_timeout_s * 1000)
        self._http_options = genai_types.HttpOptions(timeout=timeout_ms)
        self._genai_types = genai_types
        self._client = genai.Client(
            api_key=api_key,
            http_options=self._http_options,
        )
        self._predicate_positions = {
            predicate.lower(): [role.upper() for role in roles]
            for predicate, roles in predicate_positions.items()
        }
        self._preferred_predicates = [
            predicate.lower() for predicate in (preferred_predicates or [])
        ]
        self._config = config
        self._year = year
        self._response_schema = build_case_query_draft_schema()

    def draft(
        self,
        *,
        case_id: str,
        title: str,
        case_text: str,
        max_queries: int = 3,
    ) -> list[dict[str, str]]:
        prompt = build_case_query_draft_prompt(
            case_id=case_id,
            title=title,
            case_text=case_text,
            predicate_positions=self._predicate_positions,
            preferred_predicates=self._preferred_predicates,
            max_queries=max_queries,
            year=self._year,
        )
        request_config = self._genai_types.GenerateContentConfig(
            httpOptions=self._http_options,
            temperature=0.0,
            responseMimeType="application/json",
            responseSchema=self._response_schema,
        )
        response = self._client.models.generate_content(
            model=self._config.gemini_model,
            contents=prompt,
            config=request_config,
        )
        response_text = response.text
        if response_text is None:
            raise ValueError("Gemini returned an empty response body")
        return parse_case_query_draft_response(
            json.loads(response_text),
            predicate_positions=self._predicate_positions,
            max_queries=max_queries,
        )

    def close(self) -> None:
        close = getattr(self._client, "close", None)
        if callable(close):
            close()
