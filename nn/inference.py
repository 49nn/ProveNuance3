"""
NeuralInference — entry point pipeline propose-verify.

Workflow:
  1. Oblicz bias pamięci encji (EntityMemoryBiasEncoder).
  2. Zbuduj HeteroData (GraphBuilder).
  3. Zastosuj clamp (apply_clamp).
  4. Uruchom NeuralProposer.forward() z aktywnym NeuralTracer.
  5. Zdekoduj logity faktów → Fact z status=inferred_candidate i neural_trace.
  6. Zdekoduj logity klastrów → zaktualizowane ClusterStateRow.

Deterministyczny: @torch.no_grad(), brak Dropout, stały T.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData

from data_model.common import TruthDistribution
from data_model.fact import Fact, FactProvenance, FactStatus, NeuralTraceItem
from data_model.rule import Rule

from .clamp import apply_clamp
from .config import NNConfig
from .entity_memory import EntityMemoryBiasEncoder
from .graph_builder import (
    ClusterSchema,
    ClusterStateRow,
    GraphBuilder,
    GraphNodeIndex,
)
from .proposer import NeuralProposer
from .trace import NeuralTracer

TRUTH_ORDER = ("T", "F", "U")

# Obserwowane fakty zachowują swój status — nie nadpisujemy
_KEEP_STATUS = {FactStatus.observed, FactStatus.proved, FactStatus.rejected}


class NeuralInference:
    """
    Fasada: przyjmuje dane domenowe, zwraca wzbogacone Fact + zaktualizowane ClusterStateRow.
    """

    def __init__(
        self,
        proposer: NeuralProposer,
        graph_builder: GraphBuilder,
        memory_encoder: EntityMemoryBiasEncoder,
        config: NNConfig,
    ) -> None:
        self.proposer = proposer
        self.graph_builder = graph_builder
        self.memory_encoder = memory_encoder
        self.config = config

    # ------------------------------------------------------------------

    def propose(
        self,
        entities: list,     # list[Entity]
        facts: list[Fact],
        rules: list[Rule],
        cluster_states: list[ClusterStateRow],
    ) -> tuple[list[Fact], list[ClusterStateRow]]:
        """
        Główny punkt wejściowy.

        Zwraca:
            updated_facts:   Fact z zaktualizowanymi truth.logits, truth.value,
                             truth.confidence, provenance.neural_trace;
                             status = inferred_candidate (jeśli nie był observed/proved).
            updated_states:  ClusterStateRow z zaktualizowanymi logitami.
        """
        # 1. Bias pamięci encji
        memory_biases = self.memory_encoder.compute_memory_bias(entities, _dummy_index(self.graph_builder))
        # Pełny indeks budujemy przez build()

        # 2. Buduj graf
        data, node_index, _ = self.graph_builder.build(
            entities=entities,
            facts=facts,
            rules=rules,
            cluster_states=cluster_states,
            memory_biases=None,  # zastąpimy poniżej
        )

        # Oblicz właściwy memory_bias z gotowym node_index
        memory_biases = self.memory_encoder.compute_memory_bias(entities, node_index)

        # Dodaj memory_bias do data
        for schema in self.graph_builder.cluster_schemas:
            node_type = f"c_{schema.name}"
            if schema.name in memory_biases:
                data[node_type].memory_bias = memory_biases[schema.name]

        # 3. Zastosuj clamp (hard → frozen)
        for node_type in data.node_types:
            x = data[node_type].x
            is_clamped = data[node_type].get("is_clamped", torch.zeros(x.size(0), dtype=torch.bool))
            clamp_hard = data[node_type].get("clamp_hard", torch.zeros(x.size(0), dtype=torch.bool))
            logits_out, frozen = apply_clamp(x, is_clamped, clamp_hard, self.config)
            data[node_type].x = logits_out
            data[node_type].is_clamped = is_clamped
            data[node_type].clamp_hard = clamp_hard  # frozen wynika z is_clamped & clamp_hard

        # 4. Forward pass z tracer
        tracer = NeuralTracer(
            top_k=self.config.top_k_trace,
            truth_domain=self.config.truth_domain,
        )
        self.proposer.eval()

        with torch.no_grad():
            logits_cluster, logits_fact = self.proposer(data, node_index, tracer=tracer)

        # 5. Dekoduj fakty
        updated_facts = [
            self._decode_fact(fact, logits_fact, node_index, tracer)
            for fact in facts
        ]

        # 6. Dekoduj klastry
        updated_states = self._decode_cluster_states(
            logits_cluster, node_index, cluster_states
        )

        return updated_facts, updated_states

    # ------------------------------------------------------------------
    # Dekodowanie faktów
    # ------------------------------------------------------------------

    def _decode_fact(
        self,
        fact: Fact,
        logits_fact: torch.Tensor,  # [N_fact, 3]
        node_index: GraphNodeIndex,
        tracer: NeuralTracer,
    ) -> Fact:
        """Produkuje zaktualizowany Fact (immutable — używa model_copy)."""
        idx = node_index.fact_node_to_idx.get(fact.fact_id)
        if idx is None or logits_fact.numel() == 0:
            return fact  # fakt nie był w grafie — bez zmian

        logits_3 = logits_fact[idx]  # [3]
        p = F.softmax(logits_3, dim=-1)

        tv_idx = int(p.argmax().item())
        tv = TRUTH_ORDER[tv_idx]
        confidence = float(p[tv_idx].item())

        truth_updated = TruthDistribution(
            domain=list(TRUTH_ORDER),
            value=tv,
            confidence=confidence,
            logits={
                "T": float(logits_3[0].item()),
                "F": float(logits_3[1].item()),
                "U": float(logits_3[2].item()),
            },
        )

        neural_trace: list[NeuralTraceItem] = tracer.finalize(fact.fact_id)

        new_provenance = FactProvenance(
            proof_id=fact.provenance.proof_id if fact.provenance else None,
            neural_trace=neural_trace,
        )

        new_status = (
            fact.status
            if fact.status in _KEEP_STATUS
            else FactStatus.inferred_candidate
        )

        return fact.model_copy(
            update={
                "truth": truth_updated,
                "status": new_status,
                "provenance": new_provenance,
            }
        )

    # ------------------------------------------------------------------
    # Dekodowanie stanów klastrów
    # ------------------------------------------------------------------

    def _decode_cluster_states(
        self,
        logits_cluster: dict[str, torch.Tensor],  # "c_{cname}" -> [N, dim]
        node_index: GraphNodeIndex,
        original_states: list[ClusterStateRow],
    ) -> list[ClusterStateRow]:
        """
        Produkuje zaktualizowane ClusterStateRow z finalnymi logitami.
        """
        # Indeks oryginałów do szybkiego wyszukiwania
        orig_lookup: dict[tuple[str, str], ClusterStateRow] = {
            (s.entity_id, s.cluster_name): s for s in original_states
        }

        updated: list[ClusterStateRow] = []

        for node_type, logits in logits_cluster.items():
            cname = node_type[2:] if node_type.startswith("c_") else node_type
            idx_to_entity = node_index.idx_to_cluster_node.get(cname, {})

            for node_idx, entity_id in idx_to_entity.items():
                row_logits = logits[node_idx].tolist()
                orig = orig_lookup.get((entity_id, cname))

                updated.append(
                    ClusterStateRow(
                        entity_id=entity_id,
                        cluster_name=cname,
                        logits=row_logits,
                        is_clamped=orig.is_clamped if orig else False,
                        clamp_hard=orig.clamp_hard if orig else False,
                        clamp_source=orig.clamp_source if orig else "manual",
                    )
                )

        return updated


# ---------------------------------------------------------------------------
# Pomocnicze
# ---------------------------------------------------------------------------

def _dummy_index(builder: GraphBuilder) -> GraphNodeIndex:
    """Zwraca pusty GraphNodeIndex — używany gdy potrzebujemy schematu bez danych."""
    from .graph_builder import GraphNodeIndex
    return GraphNodeIndex()
