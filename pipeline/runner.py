"""
ProposeVerifyRunner - facade connecting NeuralInference to SymbolicVerifier.

Workflow (single run):
  1. NeuralInference.propose()  -> nn_facts (inferred_candidate), nn_states
  2. SymbolicVerifier.verify()  -> proved facts, new facts, proof DAG
  3. Merge into PipelineResult
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

log = logging.getLogger(__name__)

from data_model.common import ConstTerm
from data_model.rule import LiteralType
from nn.config import NNConfig
from nn.entity_memory import EntityMemoryBiasEncoder
from nn.gating import ExceptionGateBank
from nn.graph_builder import ClusterSchema, ClusterStateRow, EdgeTypeSpec, GraphBuilder
from nn.inference import NeuralInference
from nn.message_passing import HeteroMessagePassingBank
from nn.proposer import NeuralProposer
from sv.verifier import SymbolicVerifier

from .result import PipelineResult

if TYPE_CHECKING:
    from data_model.entity import Entity
    from data_model.fact import Fact
    from data_model.rule import Rule


class ProposeVerifyRunner:
    """
    Propose-verify facade: one neural pass followed by symbolic verification.

    Args:
        nn_inference: constructed NeuralInference (with ready NeuralProposer)
        verifier: constructed SymbolicVerifier
    """

    def __init__(
        self,
        nn_inference: NeuralInference,
        verifier: SymbolicVerifier,
    ) -> None:
        self.nn_inference = nn_inference
        self.verifier = verifier

    @classmethod
    def from_schemas(
        cls,
        cluster_schemas: list[ClusterSchema],
        config: NNConfig | None = None,
        predicate_positions: dict[str, list[str]] | None = None,
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

        supports_specs: list[EdgeTypeSpec] = [
            EdgeTypeSpec(
                src_type="fact",
                relation="supports",
                dst_type=f"c_{dst.name}",
                src_dim=GraphBuilder.FACT_DIM,
                dst_dim=dst.dim,
            )
            for dst in cluster_schemas
        ]

        edge_type_specs = role_specs + implies_specs + supports_specs

        mp_bank = HeteroMessagePassingBank(edge_type_specs)
        gate_bank = ExceptionGateBank(gate_specs=[])
        cluster_type_dims = {s.name: s.dim for s in cluster_schemas}

        proposer = NeuralProposer(config, mp_bank, gate_bank, cluster_type_dims)
        graph_builder = GraphBuilder(cluster_schemas)
        memory_encoder = EntityMemoryBiasEncoder(cluster_schemas, config)

        nn_inference = NeuralInference(proposer, graph_builder, memory_encoder, config)
        verifier = SymbolicVerifier(
            cluster_schemas,
            predicate_positions=predicate_positions,
        )

        return cls(nn_inference, verifier)

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
                if spec.relation not in {"implies", "supports"}:
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

                    module = module_by_key.get(("fact", "supports", f"c_{dst_schema.name}"))
                    if module is None:
                        continue
                    current = float(module.W_pos[truth_t_idx, dst_idx].item())
                    module.W_pos[truth_t_idx, dst_idx] = max(current, weight)

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

        # Phase 1: neural propose
        log.debug(
            "Faza NN propose: %d encji, %d faktow, %d regul, %d stanow",
            len(entities), len(facts), len(rules), len(cluster_states),
        )
        nn_facts, nn_states = self.nn_inference.propose(
            entities, facts, rules, cluster_states
        )
        log.debug("Faza NN propose: zwrocono %d faktow kandydatow", len(nn_facts))

        # Phase 2: symbolic verify
        log.debug("Faza SV verify: start")
        sv_result = self.verifier.verify(nn_facts, rules, nn_states)
        log.debug(
            "Faza SV verify: %d faktow proved, %d nowych faktow",
            len(sv_result.updated_facts), len(sv_result.new_facts),
        )

        # Merge: updated facts + newly derived facts
        all_facts = sv_result.updated_facts + sv_result.new_facts

        return PipelineResult(
            facts=all_facts,
            cluster_states=nn_states,
            new_facts=sv_result.new_facts,
            proof_nodes=dict(sv_result.proof_nodes),
        )
