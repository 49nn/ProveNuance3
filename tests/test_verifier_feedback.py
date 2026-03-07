from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("clingo")

from data_model.cluster import ClusterSchema
from data_model.common import RoleArg, RuleArg, TruthDistribution, VarTerm
from data_model.fact import Fact, FactStatus
from data_model.rule import LiteralType, Rule, RuleBodyLiteral, RuleHead, RuleMetadata
from pipeline.runner import ProposeVerifyRunner
from sv.types import CandidateFeedback, GroundAtom, VerifyResult
from sv.verifier import SymbolicVerifier


def _fact(
    fact_id: str,
    predicate: str,
    args: list[RoleArg],
    *,
    value: str = "T",
    confidence: float = 1.0,
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


def _rule_arg(role: str, var: str) -> RuleArg:
    return RuleArg(role=role, term=VarTerm(var=var))


def test_verifier_emits_blocked_feedback_for_candidate_fact() -> None:
    verifier = SymbolicVerifier(
        cluster_schemas=[],
        predicate_positions={
            "order_placed": ["ORDER"],
            "ab_can_withdraw": ["CUSTOMER", "ORDER"],
            "can_withdraw": ["CUSTOMER", "ORDER"],
        },
    )

    rules = [
        Rule(
            rule_id="r_can_withdraw",
            head=RuleHead(
                predicate="can_withdraw",
                args=[_rule_arg("CUSTOMER", "C"), _rule_arg("ORDER", "O")],
            ),
            body=[
                RuleBodyLiteral(
                    literal_type=LiteralType.pos,
                    predicate="order_placed",
                    args=[_rule_arg("ORDER", "O")],
                ),
                RuleBodyLiteral(
                    literal_type=LiteralType.naf,
                    predicate="ab_can_withdraw",
                    args=[_rule_arg("CUSTOMER", "C"), _rule_arg("ORDER", "O")],
                ),
            ],
            metadata=RuleMetadata(stratum=1, learned=False),
        )
    ]

    facts = [
        _fact(
            "f_order",
            "ORDER_PLACED",
            [RoleArg(role="ORDER", entity_id="O1")],
            status=FactStatus.observed,
        ),
        _fact(
            "f_ab",
            "AB_CAN_WITHDRAW",
            [
                RoleArg(role="CUSTOMER", entity_id="C1"),
                RoleArg(role="ORDER", entity_id="O1"),
            ],
            status=FactStatus.observed,
        ),
        _fact(
            "f_candidate",
            "CAN_WITHDRAW",
            [
                RoleArg(role="CUSTOMER", entity_id="C1"),
                RoleArg(role="ORDER", entity_id="O1"),
            ],
            status=FactStatus.inferred_candidate,
            confidence=0.95,
        ),
    ]

    result = verifier.verify(facts, rules, cluster_states=[])

    assert len(result.candidate_feedback) == 1
    feedback = result.candidate_feedback[0]
    assert feedback.fact_id == "f_candidate"
    assert feedback.outcome == "blocked"
    assert tuple(atom.predicate for atom in feedback.violated_naf) == ("ab_can_withdraw",)
    assert feedback.supporting_rule_ids == ("r_can_withdraw",)


def test_verifier_explains_query_atom_even_without_candidate_fact() -> None:
    verifier = SymbolicVerifier(
        cluster_schemas=[],
        predicate_positions={
            "order_placed": ["ORDER"],
            "ab_can_withdraw": ["CUSTOMER", "ORDER"],
            "can_withdraw": ["CUSTOMER", "ORDER"],
        },
    )

    rules = [
        Rule(
            rule_id="r_can_withdraw",
            head=RuleHead(
                predicate="can_withdraw",
                args=[_rule_arg("CUSTOMER", "C"), _rule_arg("ORDER", "O")],
            ),
            body=[
                RuleBodyLiteral(
                    literal_type=LiteralType.pos,
                    predicate="order_placed",
                    args=[_rule_arg("ORDER", "O")],
                ),
                RuleBodyLiteral(
                    literal_type=LiteralType.naf,
                    predicate="ab_can_withdraw",
                    args=[_rule_arg("CUSTOMER", "C"), _rule_arg("ORDER", "O")],
                ),
            ],
            metadata=RuleMetadata(stratum=1, learned=False),
        )
    ]

    facts = [
        _fact(
            "f_order",
            "ORDER_PLACED",
            [RoleArg(role="ORDER", entity_id="O1")],
            status=FactStatus.observed,
        ),
        _fact(
            "f_ab",
            "AB_CAN_WITHDRAW",
            [
                RoleArg(role="CUSTOMER", entity_id="C1"),
                RoleArg(role="ORDER", entity_id="O1"),
            ],
            status=FactStatus.observed,
        ),
    ]

    result = verifier.verify(facts, rules, cluster_states=[])
    query_feedback = verifier.explain_query_atom(
        GroundAtom(
            "can_withdraw",
            tuple(sorted((("CUSTOMER", "c1"), ("ORDER", "o1")))),
        ),
        derived_atoms=result.derived_atoms,
        proof_nodes=result.proof_nodes,
        rules=rules,
    )

    assert query_feedback.outcome == "blocked"
    assert tuple(atom.predicate for atom in query_feedback.violated_naf) == ("ab_can_withdraw",)
    assert query_feedback.supporting_rule_ids == ("r_can_withdraw",)


def test_runner_refines_with_proved_and_blocked_feedback() -> None:
    runner = ProposeVerifyRunner.from_schemas(
        cluster_schemas=[
            ClusterSchema(
                cluster_id=1,
                name="customer_type",
                entity_type="CUSTOMER",
                domain=["CONSUMER", "BUSINESS"],
            )
        ],
        predicate_positions={"seen": ["ENTITY"]},
        max_refinement_rounds=2,
    )

    base_fact = _fact(
        "f_base",
        "SEEN",
        [RoleArg(role="ENTITY", entity_id="E1")],
        status=FactStatus.observed,
    )
    candidate_proved = _fact(
        "f_proved",
        "CAN_APPLY",
        [RoleArg(role="ENTITY", entity_id="E1")],
        status=FactStatus.inferred_candidate,
        confidence=0.9,
    )
    candidate_blocked = _fact(
        "f_blocked",
        "CAN_REFUND",
        [RoleArg(role="ENTITY", entity_id="E1")],
        status=FactStatus.inferred_candidate,
        confidence=0.9,
    )
    new_fact = _fact(
        "f_new",
        "DERIVED_FACT",
        [RoleArg(role="ENTITY", entity_id="E1")],
        status=FactStatus.proved,
    )

    propose_inputs: list[list[Fact]] = []
    verify_calls = {"count": 0}

    def fake_propose(_entities, facts, _rules, _states):
        propose_inputs.append(list(facts))
        if len(propose_inputs) == 1:
            return [base_fact, candidate_proved, candidate_blocked], []
        return facts, []

    def fake_verify(facts, _rules, _states):
        verify_calls["count"] += 1
        if verify_calls["count"] == 1:
            return VerifyResult(
                updated_facts=[
                    base_fact,
                    candidate_proved.model_copy(update={"status": FactStatus.proved}),
                    candidate_blocked,
                ],
                new_facts=[new_fact],
                derived_atoms=frozenset(),
                proof_nodes={},
                candidate_feedback=[
                    CandidateFeedback(
                        fact_id="f_proved",
                        predicate="can_apply",
                        outcome="proved",
                        atom=GroundAtom("can_apply", (("ENTITY", "e1"),)),
                    ),
                    CandidateFeedback(
                        fact_id="f_blocked",
                        predicate="can_refund",
                        outcome="blocked",
                        atom=GroundAtom("can_refund", (("ENTITY", "e1"),)),
                    ),
                ],
            )
        return VerifyResult(
            updated_facts=list(facts),
            new_facts=[],
            derived_atoms=frozenset(),
            proof_nodes={},
            candidate_feedback=[],
        )

    runner.nn_inference.propose = fake_propose  # type: ignore[method-assign]
    runner.verifier.verify = fake_verify  # type: ignore[method-assign]

    result = runner.run(entities=[], facts=[base_fact], rules=[], cluster_states=[])

    assert result.rounds == 2
    assert len(propose_inputs) == 2

    second_round = {fact.fact_id: fact for fact in propose_inputs[1]}
    assert set(second_round) == {"f_base", "f_proved", "f_blocked", "f_new"}
    assert second_round["f_base"].status == FactStatus.observed
    assert second_round["f_proved"].status == FactStatus.proved
    assert second_round["f_new"].status == FactStatus.proved
    assert second_round["f_blocked"].status == FactStatus.rejected
    assert second_round["f_blocked"].truth.value == "F"

    feedback_by_id = {item.fact_id: item for item in result.candidate_feedback}
    assert feedback_by_id["f_proved"].outcome == "proved"
    assert feedback_by_id["f_blocked"].outcome == "blocked"
