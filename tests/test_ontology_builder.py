from __future__ import annotations

from nlp.ontology_builder import (
    build_ontology_prompt,
    build_ontology_correction_prompt,
    parse_ontology_response,
)


def test_parse_ontology_response_flags_empty_rules() -> None:
    result = parse_ontology_response(
        {
            "entity_types": [
                {"name": "CUSTOMER", "description": "Customer."},
            ],
            "predicates": [
                {
                    "name": "CONSUMER_CAN_WITHDRAW",
                    "description": "Consumer may withdraw.",
                    "roles": [
                        {"position": 0, "role": "CUSTOMER", "entity_type": "CUSTOMER"},
                    ],
                },
            ],
            "clusters": [],
            "rules": [],
        },
        source_id="regulation",
    )

    assert result.rules == []
    assert any("brak poprawnych regul" in err.lower() for err in result.validation_errors)


def test_parse_ontology_response_reports_malformed_rule() -> None:
    result = parse_ontology_response(
        {
            "entity_types": [
                {"name": "CUSTOMER", "description": "Customer."},
            ],
            "predicates": [
                {
                    "name": "CONSUMER_CAN_WITHDRAW",
                    "description": "Consumer may withdraw.",
                    "roles": [
                        {"position": 0, "role": "CUSTOMER", "entity_type": "CUSTOMER"},
                    ],
                },
            ],
            "clusters": [],
            "rules": [
                {
                    "rule_id": "core.consumer_can_withdraw",
                    "module": "core",
                    "head": {"predicate": "consumer_can_withdraw"},
                    "body": [],
                },
            ],
        },
        source_id="regulation",
    )

    assert result.rules == []
    assert any("glowa reguly" in err.lower() for err in result.validation_errors)


def test_build_ontology_correction_prompt_keeps_full_base_instructions() -> None:
    prompt = build_ontology_correction_prompt(
        "Konsument moze odstapic od umowy.",
        ["REGULA core.can_withdraw: brak predykatu glowy"],
    )

    assert "## REGULY HORN + NAF" in prompt
    assert "Nie zwracaj pustej listy rules" in prompt
    assert "brak predykatu glowy" in prompt


def test_build_ontology_prompt_distinguishes_observable_and_derived_predicates() -> None:
    prompt = build_ontology_prompt("Konsument moze odstapic od umowy.")
    lowered = prompt.lower()

    assert "Predykaty obserwowalne" in prompt
    assert "Predykaty konkluzyjne" in prompt
    assert "nie wolno modelowac skutku prawnego tylko jako samotnego predykatu bez reguly" in lowered
    assert "nie wolno mieszac nazw" in lowered
