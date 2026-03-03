"""
NeuralTracer — rejestruje wkłady krawędzi message-passing do logitów faktów.

Produkuje listę NeuralTraceItem (model Pydantic z data_model/fact.py).
Aktywny tylko podczas inference — podczas treningu tracer=None (oszczędność pamięci).

Dla każdego faktu docelowego: zachowuje top-k wpisów wg |Δs|₂ malejąco.

Kontrakt NeuralTraceItem (z fact.py):
  - dokładnie jedno z from_fact_id / from_cluster_id musi być ustawione
  - edge_type: str min_length=1
  - delta_logits: dict[TruthValue, float] — klucze T/F/U
  - step: int ≥ 0
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field

import torch

from data_model.fact import NeuralTraceItem

TRUTH_ORDER = ("T", "F", "U")


# ---------------------------------------------------------------------------
# Wewnętrzny bufor
# ---------------------------------------------------------------------------

@dataclass
class _RawEntry:
    """Surowy wpis przed konwersją na NeuralTraceItem."""
    target_fact_id: str
    from_fact_id: str | None
    from_cluster_id: str | None   # format "{cluster_name}:{entity_id}"
    edge_type: str
    delta: tuple[float, float, float]  # (T, F, U) — już zdetachowane
    step: int
    magnitude: float               # |Δs|₂ — używane do rankingu top-k

    def __lt__(self, other: _RawEntry) -> bool:
        # min-heap wg magnitude — usuwamy najmniejsze przy przepełnieniu
        return self.magnitude < other.magnitude


# ---------------------------------------------------------------------------
# NeuralTracer
# ---------------------------------------------------------------------------

class NeuralTracer:
    """
    Akumuluje wpisy śladu podczas forward pass i produkuje list[NeuralTraceItem].

    Utrzymuje per-fakt min-heap rozmiaru top_k (heap na najmniejszą magnitude
    żeby łatwo wyrzucić najsłabszy wpis).
    """

    def __init__(self, top_k: int, truth_domain: tuple[str, ...] = TRUTH_ORDER) -> None:
        self.top_k = top_k
        self.truth_domain = truth_domain
        # fact_id -> min-heap[_RawEntry]
        self._heaps: dict[str, list[_RawEntry]] = {}

    # ------------------------------------------------------------------

    def record(
        self,
        target_fact_id: str,
        from_fact_id: str | None,
        from_cluster_id: str | None,
        edge_type: str,
        delta: torch.Tensor,   # [3] — (T, F, U) dla węzła faktu; detachowane przez wywołującego
        step: int,
    ) -> None:
        """
        Rejestruje jeden wkład krawędzi do logitów faktu docelowego.

        Dokładnie jedno z from_fact_id / from_cluster_id musi być nie-None
        (weryfikowane przy finalize → Pydantic). Tu nie sprawdzamy żeby
        uniknąć narzutu w pętli.
        """
        d = delta.tolist()
        magnitude = float(torch.linalg.norm(delta).item())

        entry = _RawEntry(
            target_fact_id=target_fact_id,
            from_fact_id=from_fact_id,
            from_cluster_id=from_cluster_id,
            edge_type=edge_type,
            delta=(d[0], d[1], d[2]),
            step=step,
            magnitude=magnitude,
        )

        heap = self._heaps.setdefault(target_fact_id, [])

        if len(heap) < self.top_k:
            heapq.heappush(heap, entry)
        elif magnitude > heap[0].magnitude:
            heapq.heapreplace(heap, entry)

    # ------------------------------------------------------------------

    def record_batch(
        self,
        target_fact_ids: list[str],
        from_cluster_ids: list[str],    # format "{cluster_name}:{entity_id}"
        edge_type: str,
        delta_per_edge: torch.Tensor,   # [E, 3] — już detachowane
        step: int,
        dst_fact_indices: torch.Tensor, # [E] — indeksy faktów docelowych
        idx_to_fact_id: dict[int, str],
    ) -> None:
        """
        Wektorowe rejestrowanie wielu krawędzi naraz (używane przez proposer.py).

        target_fact_ids jest wyznaczany wewnętrznie z dst_fact_indices + idx_to_fact_id.
        """
        for e_idx in range(delta_per_edge.size(0)):
            dst_node_idx = int(dst_fact_indices[e_idx].item())
            fact_id = idx_to_fact_id.get(dst_node_idx)
            if fact_id is None:
                continue

            delta_e = delta_per_edge[e_idx]   # [3]
            cluster_id = from_cluster_ids[e_idx] if e_idx < len(from_cluster_ids) else None

            self.record(
                target_fact_id=fact_id,
                from_fact_id=None,
                from_cluster_id=cluster_id,
                edge_type=edge_type,
                delta=delta_e,
                step=step,
            )

    # ------------------------------------------------------------------

    def finalize(self, fact_id: str) -> list[NeuralTraceItem]:
        """
        Zwraca top-k NeuralTraceItem dla danego faktu,
        posortowane wg magnitude malejąco.

        Każdy wpis jest walidowany przez Pydantic (XOR from_fact_id/from_cluster_id).
        """
        heap = self._heaps.get(fact_id, [])
        entries = sorted(heap, key=lambda e: e.magnitude, reverse=True)

        items: list[NeuralTraceItem] = []
        for entry in entries:
            dlogits = {
                "T": entry.delta[0],
                "F": entry.delta[1],
                "U": entry.delta[2],
            }
            item = NeuralTraceItem(
                from_fact_id=entry.from_fact_id,
                from_cluster_id=entry.from_cluster_id,
                edge_type=entry.edge_type,
                delta_logits=dlogits,
                step=entry.step,
            )
            items.append(item)

        return items

    def reset(self) -> None:
        """Czyści wszystkie buforowane wpisy (między przypadkami)."""
        self._heaps.clear()
