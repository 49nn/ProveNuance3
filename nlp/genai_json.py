from __future__ import annotations

import json
from enum import Enum
from typing import Any


def parse_json_response(
    response: object,
    *,
    expect_object: bool = True,
) -> dict[str, Any] | list[Any]:
    parsed = getattr(response, "parsed", None)
    if parsed is not None:
        if hasattr(parsed, "model_dump"):
            parsed = parsed.model_dump(mode="python")
        elif isinstance(parsed, Enum):
            parsed = parsed.value
        return _validate_payload(parsed, expect_object=expect_object)

    response_text = getattr(response, "text", None)
    if response_text is None:
        raise ValueError("Gemini returned an empty response body.")

    raw_text = str(response_text).strip()
    if raw_text.startswith("```"):
        raw_text = _strip_code_fence(raw_text)

    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "Gemini returned invalid JSON "
            f"(line={exc.lineno}, col={exc.colno}, pos={exc.pos}): {exc.msg}"
        ) from exc

    return _validate_payload(payload, expect_object=expect_object)


def _validate_payload(
    payload: object,
    *,
    expect_object: bool,
) -> dict[str, Any] | list[Any]:
    if expect_object:
        if not isinstance(payload, dict):
            raise ValueError(
                f"Gemini returned payload of type {type(payload).__name__}, expected JSON object."
            )
        return payload

    if not isinstance(payload, (dict, list)):
        raise ValueError(
            f"Gemini returned payload of type {type(payload).__name__}, expected JSON object or array."
        )
    return payload


def _strip_code_fence(text: str) -> str:
    lines = text.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()
