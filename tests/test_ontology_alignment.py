from __future__ import annotations

from datetime import datetime

import pytest

pytest.importorskip("torch")

from data_model.common import RoleArg, Span, TruthDistribution
from data_model.entity import Entity
from data_model.fact import Fact, FactSource, FactStatus
from nlp.ontology_alignment import align_extraction_to_ontology
from nlp.result import ExtractionResult
from nn.graph_builder import ClusterSchema, ClusterStateRow


def test_aligns_fact_roles_to_current_ontology() -> None:
    result = ExtractionResult(
        entities=[],
        facts=[
            Fact(
                fact_id="f1",
                predicate="PAYMENT_SELECTED",
                arity=3,
                args=[
                    RoleArg(role="CUSTOMER", entity_id="CUST1"),
                    RoleArg(role="ORDER", entity_id="O1"),
                    RoleArg(role="METHOD", literal_value="BLIK"),
                ],
                truth=TruthDistribution(domain=["T", "F", "U"], value="T", confidence=1.0),
                status=FactStatus.observed,
                source=FactSource(
                    source_id="TC-1",
                    spans=[Span(start=0, end=10, text="wybrałem BLIK")],
                    extractor="TextExtractor",
                    confidence=1.0,
                ),
            )
        ],
        cluster_states=[],
        source_id="TC-1",
    )

    aligned = align_extraction_to_ontology(
        result,
        predicate_positions={"payment_selected": ["ORDER", "PAYMENT_METHOD"]},
        cluster_schemas=[],
        year=2026,
    )

    assert len(aligned.facts) == 1
    fact = aligned.facts[0]
    assert fact.predicate == "PAYMENT_SELECTED"
    assert [(arg.role, arg.entity_id, arg.literal_value) for arg in fact.args] == [
        ("ORDER", "O1", None),
        ("PAYMENT_METHOD", None, "BLIK"),
    ]


def test_emits_alias_fact_from_cluster_state() -> None:
    result = ExtractionResult(
        entities=[
            Entity(
                entity_id="CUST1",
                type="CUSTOMER",
                canonical_name="Klient",
                created_at=datetime(2026, 1, 1),
                provenance=[],
            )
        ],
        facts=[],
        cluster_states=[
            ClusterStateRow(
                entity_id="CUST1",
                cluster_name="customer_type",
                logits=[10.0, -10.0],
                is_clamped=True,
                clamp_hard=True,
                clamp_source="text",
                source_span=Span(start=0, end=11, text="konsumentem"),
            )
        ],
        source_id="TC-2",
    )

    aligned = align_extraction_to_ontology(
        result,
        predicate_positions={"customer_is_consumer": ["CUSTOMER"]},
        cluster_schemas=[
            ClusterSchema(
                cluster_id=1,
                name="customer_type",
                entity_type="CUSTOMER",
                domain=["CONSUMER", "BUSINESS"],
            )
        ],
        year=2026,
    )

    assert len(aligned.facts) == 1
    fact = aligned.facts[0]
    assert fact.predicate == "CUSTOMER_IS_CONSUMER"
    assert [(arg.role, arg.entity_id) for arg in fact.args] == [("CUSTOMER", "CUST1")]
