from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

import torch
from torch_geometric.data import HeteroData

from cli.pn3train import _attach_fact_supervision, _collect_case_pseudo_labels
from data_model.common import RoleArg, TruthDistribution
from data_model.fact import Fact, FactStatus
from nn.graph_builder import GraphNodeIndex
from sv.types import CandidateFeedback, GroundAtom


def _fact(
    fact_id: str,
    predicate: str,
    args: list[RoleArg],
    *,
    value: str,
    confidence: float,
    status: FactStatus,
) -> Fact:
    return Fact(
        fact_id=fact_id,
        predicate=predicate,
        arity=len(args),
        args=args,
        truth=TruthDistribution(
            domain=["T", "F", "U"],
            value=value,  # type: ignore[arg-type]
            confidence=confidence,
        ),
        status=status,
    )


def test_collect_case_pseudo_labels_includes_blocked_negative() -> None:
    positive = _fact(
        "f_pos",
        "RISK_TRANSFERRED",
        [
            RoleArg(role="PRODUCT", entity_id="P1"),
            RoleArg(role="CUSTOMER", entity_id="C1"),
            RoleArg(role="DATE", entity_id="D1"),
        ],
        value="T",
        confidence=1.0,
        status=FactStatus.proved,
    )
    blocked = _fact(
        "f_neg",
        "CUSTOMER_BEARS_RETURN_COST",
        [
            RoleArg(role="CUSTOMER", entity_id="C1"),
            RoleArg(role="PRODUCT", entity_id="P1"),
        ],
        value="F",
        confidence=1.0,
        status=FactStatus.rejected,
    )

    feedback = [
        CandidateFeedback(
            fact_id="f_neg",
            predicate="customer_bears_return_cost",
            outcome="blocked",
            atom=GroundAtom(
                "customer_bears_return_cost",
                tuple(sorted((("CUSTOMER", "c1"), ("PRODUCT", "p1")))),
            ),
        )
    ]

    pseudo_facts, pseudo_clusters = _collect_case_pseudo_labels(
        round_id="R1",
        case_id="TC-X",
        gold_facts=[],
        gold_states=[],
        result_facts=[positive, blocked],
        candidate_feedback=feedback,
        result_states=[],
        schemas=[],
        fact_conf_threshold=0.95,
        cluster_top1_threshold=0.95,
        cluster_margin_threshold=0.80,
    )

    assert pseudo_clusters == []
    assert len(pseudo_facts) == 2
    by_key = {label.fact_key: label for label in pseudo_facts}
    assert by_key["RISK_TRANSFERRED|CUSTOMER=C1|DATE=D1|PRODUCT=P1"].fact.status == FactStatus.proved
    assert by_key["CUSTOMER_BEARS_RETURN_COST|CUSTOMER=C1|PRODUCT=P1"].fact.status == FactStatus.rejected
    assert by_key["CUSTOMER_BEARS_RETURN_COST|CUSTOMER=C1|PRODUCT=P1"].fact.truth.value == "F"


def test_attach_fact_supervision_marks_targets_for_pseudo_facts() -> None:
    data = HeteroData()
    data["fact"].x = torch.zeros(2, 3)
    data["fact"].is_clamped = torch.ones(2, dtype=torch.bool)
    data["fact"].clamp_hard = torch.ones(2, dtype=torch.bool)
    node_index = GraphNodeIndex(
        fact_node_to_idx={"f_pos": 0, "f_neg": 1},
        idx_to_fact_node={0: "f_pos", 1: "f_neg"},
    )

    facts = [
        _fact(
            "f_pos",
            "RISK_TRANSFERRED",
            [RoleArg(role="PRODUCT", entity_id="P1")],
            value="T",
            confidence=0.9,
            status=FactStatus.proved,
        ),
        _fact(
            "f_neg",
            "CUSTOMER_BEARS_RETURN_COST",
            [RoleArg(role="PRODUCT", entity_id="P1")],
            value="F",
            confidence=1.0,
            status=FactStatus.rejected,
        ),
    ]

    _attach_fact_supervision(data, node_index, facts, pseudo_fact_weight=0.5)

    assert data["fact"].supervision_target.tolist() == [0, 1]
    assert data["fact"].supervision_weight.tolist() == pytest.approx([0.45, 0.5])
    assert data["fact"].is_clamped.tolist() == [False, False]
    assert data["fact"].clamp_hard.tolist() == [False, False]
