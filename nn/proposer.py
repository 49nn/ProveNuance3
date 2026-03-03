"""
NeuralProposer — T-krokowa pętla message passing.

Pełni rolę "Neural Proposer" z project desc:
  s_v^(t+1) = s_v^(t) + Σ Δs_{u→v}^(t) + b_v

Parametry:
  - W+, W- per typ krawędzi  (w HeteroMessagePassingBank)
  - u  per bramka wyjątku    (w ExceptionGateBank)
  - b_v per typ klastra      (cluster_biases — bias węzła, nie per instancja)

Deterministyczny: brak Dropout, stały T, @torch.no_grad() podczas inference.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData

from .config import NNConfig
from .gating import ExceptionGateBank
from .graph_builder import EdgeTypeSpec, GraphNodeIndex
from .message_passing import HeteroMessagePassingBank
from .trace import NeuralTracer


class NeuralProposer(nn.Module):
    """
    Moduł główny Neural Proposer.

    Przechowuje:
      mp_bank       — W+/W- per typ krawędzi
      gate_bank     — bramki wyjątków
      cluster_biases — nn.ParameterDict; jeden bias per typ klastra (nazwa klastra → Tensor[dim])
    """

    def __init__(
        self,
        config: NNConfig,
        mp_bank: HeteroMessagePassingBank,
        gate_bank: ExceptionGateBank,
        cluster_type_dims: dict[str, int],  # cluster_name -> dim (np. "customer_type" -> 2)
    ) -> None:
        super().__init__()
        self.config = config
        self.mp_bank = mp_bank
        self.gate_bank = gate_bank

        # b_v per typ klastra (jeden wektor na cały typ, nie per instancja)
        biases: dict[str, nn.Parameter] = {}
        for cname, dim in cluster_type_dims.items():
            biases[cname] = nn.Parameter(torch.zeros(dim))
        self.cluster_biases = nn.ParameterDict(biases)

        # b_v dla faktów
        self.fact_bias = nn.Parameter(torch.zeros(3))  # T / F / U

    # ------------------------------------------------------------------

    def forward(
        self,
        data: HeteroData,
        node_index: GraphNodeIndex,
        tracer: NeuralTracer | None = None,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """
        Uruchamia T kroków message passing.

        Returns:
            logits_cluster: dict cluster_name -> Tensor[N, dim]
            logits_fact:    Tensor[N_fact, 3]
        """
        # --- Inicjalizacja logitów ---
        logits: dict[str, torch.Tensor] = {}
        frozen: dict[str, torch.BoolTensor] = {}

        for node_type in data.node_types:
            x = data[node_type].x.clone()
            # Dodaj bias pamięci encji do logitów startowych
            x = x + data[node_type].memory_bias
            logits[node_type] = x
            # Zamrożone = hard-clamped
            is_clamped = data[node_type].get("is_clamped", torch.zeros(x.size(0), dtype=torch.bool))
            clamp_hard = data[node_type].get("clamp_hard", torch.zeros(x.size(0), dtype=torch.bool))
            frozen[node_type] = is_clamped & clamp_hard

        # --- T kroków ---
        for t in range(self.config.T):
            # 1. Oblicz Δs dla wszystkich typów docelowych
            delta = self.mp_bank(data, logits, frozen)

            # 2. Wyhamuj bramkami wyjątków
            delta = self.gate_bank.apply_gates(data, delta, logits, node_index)

            # 3. Zaktualizuj logity (zamrożone węzły pomijają update)
            for node_type, delta_v in delta.items():
                mask_update = (~frozen[node_type]).float().unsqueeze(-1)  # [N, 1]
                bias_v = self._get_bias(node_type)                        # [dim]

                logits[node_type] = (
                    logits[node_type]
                    + delta_v * mask_update
                    + bias_v.unsqueeze(0) * mask_update
                )

            # 4. Rejestruj trace (tylko podczas inference)
            if tracer is not None and "fact" in delta:
                self._record_trace(tracer, data, logits, node_index, t)

        logits_cluster = {
            k: v for k, v in logits.items() if k != "fact"
        }
        logits_fact = logits.get("fact", torch.zeros(0, 3))

        return logits_cluster, logits_fact

    # ------------------------------------------------------------------
    # Bias
    # ------------------------------------------------------------------

    def _get_bias(self, node_type: str) -> torch.Tensor:
        """
        Zwraca bias dla danego typu węzła.
        Dla "c_{cluster_name}" → cluster_biases[cluster_name].
        Dla "fact" → fact_bias.
        """
        if node_type == "fact":
            return self.fact_bias
        if node_type.startswith("c_"):
            cname = node_type[2:]  # usuń prefix "c_"
            if cname in self.cluster_biases:
                return self.cluster_biases[cname]
        # Brak biasu — zero
        return torch.zeros(1)

    # ------------------------------------------------------------------
    # Neural trace
    # ------------------------------------------------------------------

    def _record_trace(
        self,
        tracer: NeuralTracer,
        data: HeteroData,
        logits: dict[str, torch.Tensor],
        node_index: GraphNodeIndex,
        step: int,
    ) -> None:
        """
        Dla każdej krawędzi prowadzącej do węzła 'fact':
        rejestruje (cluster_id, edge_type, Δs_per_edge, step).
        """
        per_edge = self.mp_bank.compute_per_edge_deltas(data, logits)

        for spec, src_indices, delta_per_edge in per_edge:
            if spec.dst_type != "fact":
                continue

            edge_index = data[spec.src_type, spec.relation, spec.dst_type].edge_index
            dst_indices = edge_index[1]  # [E]

            # Buduj from_cluster_id z src_type i src_indices
            cname = spec.src_type[2:] if spec.src_type.startswith("c_") else spec.src_type
            idx_to_entity = node_index.idx_to_cluster_node.get(cname, {})

            edge_type_str = f"{spec.src_type}__{spec.relation}__{spec.dst_type}"

            for e in range(delta_per_edge.size(0)):
                src_idx = int(src_indices[e].item())
                dst_idx = int(dst_indices[e].item())

                fact_id = node_index.idx_to_fact_node.get(dst_idx)
                if fact_id is None:
                    continue

                entity_id = idx_to_entity.get(src_idx)
                cluster_id = f"{cname}:{entity_id}" if entity_id else f"{cname}:{src_idx}"

                tracer.record(
                    target_fact_id=fact_id,
                    from_fact_id=None,
                    from_cluster_id=cluster_id,
                    edge_type=edge_type_str,
                    delta=delta_per_edge[e],
                    step=step,
                )
