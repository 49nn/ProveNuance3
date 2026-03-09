"""
ProposeVerifyRunner - facade connecting NeuralInference to SymbolicVerifier.

Workflow (single run):
  1. NeuralInference.propose()  -> nn_facts (inferred_candidate), nn_states
  2. SymbolicVerifier.verify()  -> proved facts, new facts, proof DAG, candidate feedback
  3. Optional refinement round with verifier-fed proved/blocked signals
  4. Merge into PipelineResult
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

log = logging.getLogger(__name__)

from data_model.cluster import ClusterSchema, ClusterStateRow
from data_model.common import ConstTerm, TruthDistribution
from data_model.fact import FactStatus
from data_model.rule import LiteralType
from nn.config import NNConfig
from nn.entity_memory import EntityMemoryBiasEncoder
from nn.gating import ExceptionGateBank
from nn.graph_builder import (
    EdgeTypeSpec,
    GraphBuilder,
    find_head_entity_term,
    get_support_binding_roles,
    is_supports_relation,
    supports_relation,
)
from nn.inference import NeuralInference
from nn.message_passing import HeteroMessagePassingBank
from nn.proposer import NeuralProposer
from sv.temporal import AnyTemporalConstraint
from sv.verifier import SymbolicVerifier

from .result import PipelineResult

if TYPE_CHECKING:
    from data_model.entity import Entity
    from data_model.fact import Fact
    from data_model.rule import Rule


_KEEP_INPUT_FACT_STATUSES = {
    FactStatus.observed,
    FactStatus.proved,
    FactStatus.rejected,
    FactStatus.retracted,
}


class ProposeVerifyRunner:
    """
    Propose-verify facade with a small verifier-guided refinement loop.

    Args:
        nn_inference: constructed NeuralInference (with ready NeuralProposer)
        verifier: constructed SymbolicVerifier
    """

    def __init__(
        self,
        nn_inference: NeuralInference,
        verifier: SymbolicVerifier,
        max_refinement_rounds: int = 2,
    ) -> None:
        self.nn_inference = nn_inference
        self.verifier = verifier
        self.max_refinement_rounds = max(1, int(max_refinement_rounds))

    @classmethod
    def from_schemas(
        cls,
        cluster_schemas: list[ClusterSchema],
        config: NNConfig | None = None,
        predicate_positions: dict[str, list[str]] | None = None,
        max_refinement_rounds: int = 2,
        temporal_constraints: list[AnyTemporalConstraint] | None = None,
    ) -> ProposeVerifyRunner:
        """
        Build runner from cluster schemas (without case data).

        Default edge specs include:
          - role_of:  c_* -> fact
          - implies:  c_src -> c_dst (same entity_type)
          - supports: fact -> c_*

        This lets default run-case path use learned relation edges
        without custom runner wiring.
        """
        config = config or NNConfig()

        role_specs: list[EdgeTypeSpec] = [
            EdgeTypeSpec(
                src_type=f"c_{s.name}",
                relation="role_of",
                dst_type="fact",
                src_dim=s.dim,
                dst_dim=GraphBuilder.FACT_DIM,
            )
            for s in cluster_schemas
        ]

        implies_specs: list[EdgeTypeSpec] = [
            EdgeTypeSpec(
                src_type=f"c_{src.name}",
                relation="implies",
                dst_type=f"c_{dst.name}",
                src_dim=src.dim,
                dst_dim=dst.dim,
            )
            for src in cluster_schemas
            for dst in cluster_schemas
            if src.name != dst.name and src.entity_type == dst.entity_type
        ]

        cluster_names = {schema.name.lower() for schema in cluster_schemas}
        supports_specs: list[EdgeTypeSpec] = [
            EdgeTypeSpec(
                src_type="fact",
                relation=supports_relation(predicate, role),
                dst_type=f"c_{dst.name}",
                src_dim=GraphBuilder.FACT_DIM,
                dst_dim=dst.dim,
            )
            for predicate, roles in sorted((predicate_positions or {}).items())
            if predicate.lower() not in cluster_names
            if not predicate.lower().startswith("_sv_")
            if not predicate.lower().startswith("ab_")
            for role in roles
            for dst in cluster_schemas
        ]

        edge_type_specs = role_specs + implies_specs + supports_specs

        mp_bank = HeteroMessagePassingBank(edge_type_specs)
        gate_bank = ExceptionGateBank(gate_specs=[])
        cluster_type_dims = {s.name: s.dim for s in cluster_schemas}

        proposer = NeuralProposer(config, mp_bank, gate_bank, cluster_type_dims)
        graph_builder = GraphBuilder(cluster_schemas)
        memory_encoder = EntityMemoryBiasEncoder(cluster_schemas, config)

        nn_inference = NeuralInference(
            proposer,
            graph_builder,
            memory_encoder,
            config,
            predicate_positions=predicate_positions,
        )
        verifier = SymbolicVerifier(
            cluster_schemas,
            predicate_positions=predicate_positions,
            temporal_constraints=temporal_constraints,
        )

        return cls(
            nn_inference,
            verifier,
            max_refinement_rounds=max_refinement_rounds,
        )

    @staticmethod
    def _const_value_in_domain(
        rule_args,
        domain: list[str],
    ) -> str | None:
        """
        Return first ConstTerm value that belongs to cluster domain.
        Domain matching is case-insensitive; return value is normalized to upper-case.
        """
        domain_set = {v.upper() for v in domain}
        for arg in rule_args:
            term = arg.term
            if isinstance(term, ConstTerm):
                value = term.const.upper()
                if value in domain_set:
                    return value
        return None

    def _apply_learned_rule_weights(self, rules: list[Rule]) -> None:
        """
        Seed message-passing matrices for learned relation edges from rule metadata.weight.

        Scope:
          - relation='implies' (cluster -> cluster)
          - relation='supports' (fact -> cluster, truth row = 'T')

        Strategy:
          1. Reset implies/supports modules to neutral effective matrix.
          2. Fill positive entries from learned rules.
        """
        mp_bank = self.nn_inference.proposer.mp_bank
        schemas = self.nn_inference.graph_builder.cluster_schemas
        schema_by_name = {s.name: s for s in schemas}

        module_by_key = {
            (spec.src_type, spec.relation, spec.dst_type): mp_bank.get_module(spec)
            for spec in mp_bank.specs
        }

        truth_domain = tuple(self.nn_inference.proposer.config.truth_domain)
        truth_t_idx = truth_domain.index("T") if "T" in truth_domain else 0
        neutral_neg_raw = -20.0  # softplus(-20) ~= 0

        with torch.no_grad():
            # 1) Neutralize implies/supports matrices each run to avoid drift across cases.
            for spec in mp_bank.specs:
                if spec.relation != "implies" and not is_supports_relation(spec.relation):
                    continue
                module = mp_bank.get_module(spec)
                module.W_pos.zero_()
                module.W_neg_raw.fill_(neutral_neg_raw)

            # 2) Inject learned weights from rule table.
            for rule in rules:
                if not rule.metadata.learned:
                    continue

                weight = float(rule.metadata.weight or 0.0)
                if weight <= 0.0:
                    continue

                dst_schema = schema_by_name.get(rule.head.predicate)
                if dst_schema is None:
                    continue

                dst_value = self._const_value_in_domain(rule.head.args, dst_schema.domain)
                if dst_value is None:
                    continue
                dst_idx = dst_schema.domain.index(dst_value)

                for lit in rule.body:
                    if lit.literal_type != LiteralType.pos:
                        continue

                    src_schema = schema_by_name.get(lit.predicate)
                    if src_schema is not None:
                        src_value = self._const_value_in_domain(lit.args, src_schema.domain)
                        if src_value is None:
                            continue
                        src_idx = src_schema.domain.index(src_value)
                        module = module_by_key.get(
                            (f"c_{src_schema.name}", "implies", f"c_{dst_schema.name}")
                        )
                        if module is None:
                            continue
                        current = float(module.W_pos[src_idx, dst_idx].item())
                        module.W_pos[src_idx, dst_idx] = max(current, weight)
                        continue

                    head_entity_term = find_head_entity_term(
                        rule.head.args,
                        dst_schema.resolved_entity_role,
                    )
                    support_roles = get_support_binding_roles(lit.args, head_entity_term)
                    if not support_roles:
                        continue

                    for support_role in support_roles:
                        module = module_by_key.get(
                            (
                                "fact",
                                supports_relation(lit.predicate, support_role),
                                f"c_{dst_schema.name}",
                            )
                        )
                        if module is None:
                            continue
                        current = float(module.W_pos[truth_t_idx, dst_idx].item())
                        module.W_pos[truth_t_idx, dst_idx] = max(current, weight)

    @staticmethod
    def _fact_signature(facts: list[Fact]) -> tuple[tuple[str, str, str], ...]:
        return tuple(sorted(
            (fact.fact_id, fact.status.value, fact.truth.value or "")
            for fact in facts
        ))

    @staticmethod
    def _make_blocked_negative_fact(fact: Fact) -> Fact:
        provenance = None
        if fact.provenance is not None:
            provenance = fact.provenance.model_copy(update={"proof_id": None})
        return fact.model_copy(update={
            "truth": TruthDistribution(
                domain=["T", "F", "U"],
                value="F",
                confidence=1.0,
            ),
            "status": FactStatus.rejected,
            "provenance": provenance,
        })

    def _build_refinement_facts(
        self,
        base_facts: list[Fact],
        updated_facts: list[Fact],
        new_facts: list[Fact],
        blocked_fact_ids: set[str],
    ) -> list[Fact]:
        facts_by_id: dict[str, Fact] = {
            fact.fact_id: fact
            for fact in base_facts
            if fact.status in _KEEP_INPUT_FACT_STATUSES
        }
        updated_by_id = {fact.fact_id: fact for fact in updated_facts}

        for fact in updated_facts:
            if fact.status == FactStatus.proved:
                facts_by_id[fact.fact_id] = fact

        for fact_id in blocked_fact_ids:
            blocked_fact = updated_by_id.get(fact_id)
            if blocked_fact is None:
                continue
            facts_by_id[fact_id] = self._make_blocked_negative_fact(blocked_fact)

        for fact in new_facts:
            facts_by_id[fact.fact_id] = fact

        return list(facts_by_id.values())

    def run(
        self,
        entities: list[Entity],
        facts: list[Fact],
        rules: list[Rule],
        cluster_states: list[ClusterStateRow],
    ) -> PipelineResult:
        """
        Execute one propose-verify pass.

        Returns:
            PipelineResult with:
              - facts = updated_facts + new_facts
              - cluster_states = neural-updated states
              - new_facts = symbolically derived facts
              - proof_nodes = proof DAG
        """
        # Rebuild gate bank from current rules (learned defaults + ab_*).
        cluster_type_dims = {
            cname: int(param.numel())
            for cname, param in self.nn_inference.proposer.cluster_biases.items()
        }
        fact_dim = int(self.nn_inference.proposer.fact_bias.numel())
        self.nn_inference.proposer.gate_bank = ExceptionGateBank.from_rules(
            rules=rules,
            cluster_type_dims=cluster_type_dims,
            fact_dim=fact_dim,
        )
        self._apply_learned_rule_weights(rules)
        current_facts = list(facts)
        current_states = list(cluster_states)
        final_states = current_states
        final_result = None
        feedback_by_fact_id = {}
        # Accumulated proof_nodes: DERIVED nodes (rule_id != None) from any round
        # take priority over BASE nodes from later rounds where the atom is already proved.
        accumulated_proof_nodes: dict = {}
        rounds_run = 0

        for round_idx in range(self.max_refinement_rounds):
            rounds_run = round_idx + 1
            log.debug(
                "Faza NN propose [round=%d]: %d encji, %d faktow, %d regul, %d stanow",
                rounds_run, len(entities), len(current_facts), len(rules), len(current_states),
            )
            nn_facts, nn_states = self.nn_inference.propose(
                entities, current_facts, rules, current_states
            )
            log.debug(
                "Faza NN propose [round=%d]: zwrocono %d faktow kandydatow",
                rounds_run, len(nn_facts),
            )

            log.debug("Faza SV verify [round=%d]: start", rounds_run)
            sv_result = self.verifier.verify(nn_facts, rules, nn_states)
            log.debug(
                "Faza SV verify [round=%d]: %d faktow proved, %d nowych faktow, %d feedback items",
                rounds_run,
                len(sv_result.updated_facts),
                len(sv_result.new_facts),
                len(sv_result.candidate_feedback),
            )

            for item in sv_result.candidate_feedback:
                feedback_by_fact_id[item.fact_id] = item

            # Merge proof_nodes: keep DERIVED (rule_id != None) nodes from any round.
            # A node derived in round N should not be overwritten by a BASE entry in round N+1
            # (when the atom was promoted to proved and became a base atom).
            for atom, node in sv_result.proof_nodes.items():
                existing = accumulated_proof_nodes.get(atom)
                if existing is None or (node.rule_id is not None and existing.rule_id is None):
                    accumulated_proof_nodes[atom] = node

            final_result = sv_result
            final_states = nn_states

            if rounds_run >= self.max_refinement_rounds:
                break

            blocked_fact_ids = {
                item.fact_id
                for item in sv_result.candidate_feedback
                if item.outcome == "blocked"
            }
            refined_facts = self._build_refinement_facts(
                base_facts=current_facts,
                updated_facts=sv_result.updated_facts,
                new_facts=sv_result.new_facts,
                blocked_fact_ids=blocked_fact_ids,
            )
            if self._fact_signature(refined_facts) == self._fact_signature(current_facts):
                break

            current_facts = refined_facts
            current_states = nn_states

        assert final_result is not None
        all_facts = final_result.updated_facts + final_result.new_facts

        return PipelineResult(
            facts=all_facts,
            cluster_states=final_states,
            new_facts=final_result.new_facts,
            proof_nodes=accumulated_proof_nodes,
            derived_atoms=final_result.derived_atoms,
            candidate_feedback=list(feedback_by_fact_id.values()),
            rounds=rounds_run,
        )
