from __future__ import annotations

import pytest

from nlp.genai_json import parse_json_response


class _FakeResponse:
    def __init__(self, *, parsed=None, text=None) -> None:
        self.parsed = parsed
        self.text = text


def test_parse_json_response_prefers_parsed_payload() -> None:
    response = _FakeResponse(
        parsed={"entity_types": [], "predicates": [], "clusters": [], "rules": []}
    )

    payload = parse_json_response(response)

    assert payload["rules"] == []


def test_parse_json_response_strips_code_fences() -> None:
    response = _FakeResponse(
        text="```json\n{\"entity_types\": [], \"predicates\": [], \"clusters\": [], \"rules\": []}\n```"
    )

    payload = parse_json_response(response)

    assert payload["entity_types"] == []


def test_parse_json_response_reports_invalid_json_cleanly() -> None:
    response = _FakeResponse(text='{"entity_types": ["unterminated]}')

    with pytest.raises(ValueError) as excinfo:
        parse_json_response(response)

    assert "invalid JSON" in str(excinfo.value)
