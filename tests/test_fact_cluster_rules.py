from __future__ import annotations

from datetime import datetime

import pytest

pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

import torch

from data_model.cluster import ClusterSchema
from data_model.common import ConstTerm, RoleArg, RuleArg, TruthDistribution, VarTerm
from data_model.entity import Entity
from data_model.fact import Fact, FactStatus
from data_model.rule import LiteralType, Rule, RuleBodyLiteral, RuleHead, RuleMetadata
from nn.graph_builder import EdgeTypeSpec, GraphBuilder, supports_relation
from nn.message_passing import HeteroMessagePassingBank
from nn.rule_extractor import RuleExtractionConfig, extract_rules_from_mp_bank


def test_extract_rules_from_mp_bank_includes_fact_to_cluster_rule() -> None:
    schema = ClusterSchema(
        cluster_id=1,
        name="customer_type",
        entity_type="CUSTOMER",
        domain=["CONSUMER", "BUSINESS"],
        entity_role="CUSTOMER",
        value_role="VALUE",
    )
    spec = EdgeTypeSpec(
        src_type="fact",
        relation=supports_relation("ordered_by", "CUSTOMER"),
        dst_type="c_customer_type",
        src_dim=3,
        dst_dim=schema.dim,
    )
    mp_bank = HeteroMessagePassingBank([spec])
    module = mp_bank.get_module(spec)
    with torch.no_grad():
        module.W_pos.zero_()
        module.W_neg_raw.fill_(-20.0)
        module.W_pos[0, 0] = 0.85

    rules = extract_rules_from_mp_bank(
        mp_bank=mp_bank,
        cluster_schemas=[schema],
        config=RuleExtractionConfig(min_weight=0.5, top_k_per_source_value=1),
        predicate_positions={"ordered_by": ["ORDER", "CUSTOMER"]},
    )

    assert len(rules) == 1
    rule = rules[0]
    assert rule.body[0].predicate == "ordered_by"
    assert rule.head.predicate == "customer_type"
    assert isinstance(rule.head.args[0].term, VarTerm)
    assert rule.head.args[0].term.var == "E"
    assert isinstance(rule.head.args[1].term, ConstTerm)
    assert rule.head.args[1].term.const == "consumer"
    assert isinstance(rule.body[0].args[0].term, VarTerm)
    assert rule.body[0].args[0].term.var != "E"
    assert isinstance(rule.body[0].args[1].term, VarTerm)
    assert rule.body[0].args[1].term.var == "E"


def test_graph_builder_uses_role_specific_supports_relation_for_learned_rule() -> None:
    schema = ClusterSchema(
        cluster_id=1,
        name="customer_type",
        entity_type="CUSTOMER",
        domain=["CONSUMER", "BUSINESS"],
        entity_role="CUSTOMER",
        value_role="VALUE",
    )
    customer = Entity(
        entity_id="C1",
        type="CUSTOMER",
        canonical_name="Customer 1",
        created_at=datetime.utcnow(),
    )
    fact = Fact(
        fact_id="f1",
        predicate="ORDERED_BY",
        arity=2,
        args=[
            RoleArg(role="ORDER", entity_id="O1"),
            RoleArg(role="CUSTOMER", entity_id="C1"),
        ],
        truth=TruthDistribution(domain=["T", "F", "U"], value="T", confidence=1.0),
        status=FactStatus.observed,
    )
    rule = Rule(
        rule_id="learned.ordered_by.customer__to__customer_type.consumer",
        head=RuleHead(
            predicate="customer_type",
            args=[
                RuleArg(role="CUSTOMER", term=VarTerm(var="E")),
                RuleArg(role="VALUE", term=ConstTerm(const="consumer")),
            ],
        ),
        body=[
            RuleBodyLiteral(
                literal_type=LiteralType.pos,
                predicate="ordered_by",
                args=[
                    RuleArg(role="ORDER", term=VarTerm(var="ORDER")),
                    RuleArg(role="CUSTOMER", term=VarTerm(var="E")),
                ],
            )
        ],
        metadata=RuleMetadata(stratum=0, learned=True, weight=0.8),
    )

    builder = GraphBuilder([schema])
    data, _node_index, specs = builder.build(
        entities=[customer],
        facts=[fact],
        rules=[rule],
        cluster_states=[],
        memory_biases=None,
    )

    relation = supports_relation("ordered_by", "CUSTOMER")
    assert ("fact", relation, "c_customer_type") in data.edge_types
    assert data["fact", relation, "c_customer_type"].edge_index.tolist() == [[0], [0]]
    assert any(
        spec.src_type == "fact"
        and spec.relation == relation
        and spec.dst_type == "c_customer_type"
        for spec in specs
    )
