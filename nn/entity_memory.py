"""
EntityMemoryBiasEncoder — konwersja memory_slots encji na bias logitów.

Dla każdej encji szuka slotu o nazwie == cluster_name.
Najlepszy wpis (wg source_rank * confidence, potem valid_from) dostaje
addytywny bias: b[entity_idx, domain_pos] += clamp_soft_M * confidence.

Bias jest soft — nie zamraża węzła, tylko przesuwa rozkład startowy.
Wzór: s_v^(0) += b_v^mem  (project desc §Clamp)
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .config import NNConfig
from .graph_builder import ClusterSchema, GraphNodeIndex

if TYPE_CHECKING:
    from data_model.entity import Entity, MemorySlotEntry


class EntityMemoryBiasEncoder:
    """
    Mapuje Entity.memory_slots → dict cluster_name → Tensor[N, dim].

    Zwraca bias do dodania do logitów klastrów przy inicjalizacji grafu.
    """

    def __init__(
        self,
        cluster_schemas: list[ClusterSchema],
        config: NNConfig,
    ) -> None:
        self.cluster_schemas = cluster_schemas
        self.config = config

    def compute_memory_bias(
        self,
        entities: list[Entity],
        node_index: GraphNodeIndex,
    ) -> dict[str, torch.Tensor]:
        """
        Zwraca dict: cluster_name -> Tensor[N_cluster_of_type, dim].

        Dla klastrów bez pasującego memory_slot zwraca tensor zer.
        """
        result: dict[str, torch.Tensor] = {}

        for schema in self.cluster_schemas:
            cluster_map = node_index.cluster_node_to_idx.get(schema.name, {})
            n = len(cluster_map)
            dim = schema.dim
            bias = torch.zeros(n, dim)

            for entity in entities:
                if entity.type != schema.entity_type:
                    continue
                idx = cluster_map.get(entity.entity_id)
                if idx is None:
                    continue

                entries = entity.memory_slots.get(schema.name, [])
                if not entries:
                    continue

                best = self._select_best_slot_entry(entries)
                if best is None:
                    continue

                # Normalizuj wartość i szukaj w domenie klastra
                raw_value = best.normalized if best.normalized is not None else best.value
                value_str = str(raw_value).upper()

                if value_str not in schema.domain:
                    continue

                domain_pos = schema.domain.index(value_str)
                confidence = float(best.confidence) if best.confidence is not None else 1.0
                bias[idx, domain_pos] += self.config.clamp_soft_M * confidence

            result[schema.name] = bias

        return result

    # ------------------------------------------------------------------

    @staticmethod
    def _select_best_slot_entry(entries: list[MemorySlotEntry]) -> MemorySlotEntry | None:
        """
        Wybiera najlepszy wpis wg (source_rank * confidence) malejąco,
        następnie valid_from malejąco (nowszy = ważniejszy).
        """
        if not entries:
            return None

        def score(e: MemorySlotEntry) -> tuple[float, float]:
            sr = float(e.source_rank) if e.source_rank is not None else 0.5
            cf = float(e.confidence) if e.confidence is not None else 1.0
            vf = e.valid_from.timestamp() if e.valid_from is not None else 0.0
            return (sr * cf, vf)

        return max(entries, key=score)
