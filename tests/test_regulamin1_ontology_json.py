from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from data_model.rule import Rule
from nlp.ontology_builder import parse_ontology_response
from sv.stratification import validate_stratification
from sv.temporal import temporal_constraints_to_rules


def _load_get_temporal_constraints():
    module_path = Path("pipeline/temporal_config.py")
    spec = importlib.util.spec_from_file_location("test_temporal_config", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.get_temporal_constraints


def test_regulamin1_ontology_json_parses_without_validation_errors() -> None:
    ontology_path = Path("docs/regulamin1_ontology.json")
    raw = json.loads(ontology_path.read_text(encoding="utf-8"))

    result = parse_ontology_response(raw, source_id="regulamin1_json")

    assert result.validation_errors == []
    assert len(result.entity_types) >= 10
    assert len(result.predicates) >= 25
    assert len(result.clusters) >= 5
    assert len(result.rules) >= 20
    assert any(rule.rule_id == "core.customer_can_withdraw_physical" for rule in result.rules)
    assert any(rule.rule_id == "core.chargeback_not_applicable_cod" for rule in result.rules)


def test_regulamin1_ontology_json_is_stratified_with_temporal_rules() -> None:
    ontology_path = Path("docs/regulamin1_ontology.json")
    raw = json.loads(ontology_path.read_text(encoding="utf-8"))
    result = parse_ontology_response(raw, source_id="regulamin1_json")
    get_temporal_constraints = _load_get_temporal_constraints()

    predicate_positions = {
        predicate.name.lower(): [role.role.upper() for role in predicate.roles]
        for predicate in result.predicates
    }
    temporal_rules = temporal_constraints_to_rules(
        get_temporal_constraints(predicate_positions),
        predicate_positions,
    )
    rules = [
        Rule.model_validate(
            {
                "rule_id": spec.rule_id,
                "head": spec.head,
                "body": spec.body,
                "metadata": {"stratum": spec.stratum, "learned": False},
            }
        )
        for spec in result.rules
    ]

    validate_stratification(rules + temporal_rules)
