from __future__ import annotations

from cli.pn3train import _pseudo_fact_label_stats
from data_model.common import RoleArg, TruthDistribution
from data_model.fact import Fact, FactStatus
from data_model.self_training import PseudoFactLabel


def _pseudo_fact_label(fact_id: str, value: str) -> PseudoFactLabel:
    fact = Fact(
        fact_id=fact_id,
        predicate="sample_predicate",
        arity=1,
        args=[RoleArg(role="ENTITY", entity_id="E1")],
        truth=TruthDistribution(
            domain=["T", "F", "U"],
            value=value,  # type: ignore[arg-type]
            confidence=0.9,
        ),
        status=FactStatus.proved if value == "T" else FactStatus.rejected,
    )
    return PseudoFactLabel(
        round_id="R1",
        case_id="TC-001",
        fact_key=f"{fact.predicate}|ENTITY=E1|{fact_id}",
        fact=fact,
        truth_confidence=0.9,
    )


def test_pseudo_fact_label_stats_counts_truth_values() -> None:
    stats = _pseudo_fact_label_stats([
        _pseudo_fact_label("f1", "T"),
        _pseudo_fact_label("f2", "F"),
        _pseudo_fact_label("f3", "U"),
    ])

    assert stats == {
        "total": 3,
        "t": 1,
        "f": 1,
        "u": 1,
    }
