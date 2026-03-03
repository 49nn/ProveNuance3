"""
LogitMessagePassing — implementacja wzoru z project desc:

    Δs_{u→v}^(t) = p_u^(t) @ W+_τ  -  p_u^(t) @ W-_τ
    s_v^(t+1)    = s_v^(t) + Σ_{u∈N(v)} Δs_{u→v}^(t) + b_v

gdzie W-_τ ≥ 0 wymuszane przez softplus(W_neg_raw).

HeteroMessagePassingBank — rejestr jednego LogitMessagePassing per EdgeTypeSpec.
Zwraca słownik delta per typ docelowy ("fact" lub "c_{cluster_name}").
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import MessagePassing

from .graph_builder import EdgeTypeSpec


# ---------------------------------------------------------------------------
# LogitMessagePassing
# ---------------------------------------------------------------------------

class LogitMessagePassing(MessagePassing):
    """
    Message passing dla jednego typu krawędzi τ.

    Źródłowe logity → softmax → p_u → p_u @ (W+ - W-) → agregacja (sum) → Δs_v.
    W- = softplus(W_neg_raw) gwarantuje nieujemność bez nieciągłości gradientu.
    """

    def __init__(self, src_dim: int, dst_dim: int) -> None:
        super().__init__(aggr="add")
        self.src_dim = src_dim
        self.dst_dim = dst_dim

        self.W_pos = nn.Parameter(torch.empty(src_dim, dst_dim))
        self.W_neg_raw = nn.Parameter(torch.zeros(src_dim, dst_dim))

        nn.init.xavier_uniform_(self.W_pos)

    @property
    def W_neg(self) -> torch.Tensor:
        """W- ≥ 0 przez softplus."""
        return F.softplus(self.W_neg_raw)

    def forward(
        self,
        x_src: torch.Tensor,        # [N_src, src_dim]
        edge_index: torch.Tensor,   # [2, E]
        frozen_src: torch.BoolTensor,  # [N_src] — zamrożone węzły nadal propagują
    ) -> torch.Tensor:
        """
        Zwraca Δs_v [N_dst, dst_dim] — zagregowane wkłady do węzłów docelowych.

        Zamrożone węzły źródłowe nadal wysyłają wiadomości (ich logity są
        poprawnym rozkładem zaklampowanym); nie przyjmują tylko wkładów.
        """
        p_src = F.softmax(x_src, dim=-1)  # [N_src, src_dim]
        # propagate wywołuje message() i agreguje po dst
        return self.propagate(edge_index, x=p_src, size=(x_src.size(0), None))

    def message(self, x_j: torch.Tensor) -> torch.Tensor:
        """
        x_j: [E, src_dim] — p_u dla każdej krawędzi.
        Zwraca [E, dst_dim]: Δs = p_u @ W+ - p_u @ W-
        """
        return x_j @ self.W_pos - x_j @ self.W_neg


# ---------------------------------------------------------------------------
# HeteroMessagePassingBank
# ---------------------------------------------------------------------------

class HeteroMessagePassingBank(nn.Module):
    """
    Rejestr modułów LogitMessagePassing per EdgeTypeSpec.

    Klucz w nn.ModuleDict: "{src_type}__{relation}__{dst_type}"
    (dwukropek nie jest dozwolony w kluczach PyTorch ParameterDict).
    """

    def __init__(self, edge_type_specs: list[EdgeTypeSpec]) -> None:
        super().__init__()
        self.specs = edge_type_specs

        modules: dict[str, nn.Module] = {}
        for spec in edge_type_specs:
            key = self._key(spec)
            modules[key] = LogitMessagePassing(spec.src_dim, spec.dst_dim)

        self.mp_modules = nn.ModuleDict(modules)

    # ------------------------------------------------------------------

    @staticmethod
    def _key(spec: EdgeTypeSpec) -> str:
        return f"{spec.src_type}__{spec.relation}__{spec.dst_type}"

    def get_module(self, spec: EdgeTypeSpec) -> LogitMessagePassing:
        return self.mp_modules[self._key(spec)]  # type: ignore[return-value]

    # ------------------------------------------------------------------

    def forward(
        self,
        data: HeteroData,
        logits: dict[str, torch.Tensor],   # node_type -> logit tensor
        frozen: dict[str, torch.BoolTensor],  # node_type -> frozen mask
    ) -> dict[str, torch.Tensor]:
        """
        Oblicza i zwraca zagregowane Δs dla każdego docelowego typu węzłów.

        Args:
            logits:  node_type -> Tensor[N, dim]
            frozen:  node_type -> BoolTensor[N]

        Returns:
            delta: node_type -> Tensor[N, dim]  (tylko typy będące dst)
        """
        delta: dict[str, torch.Tensor] = {}

        for spec in self.specs:
            et = (spec.src_type, spec.relation, spec.dst_type)
            if et not in data.edge_types:
                continue

            edge_index = data[spec.src_type, spec.relation, spec.dst_type].edge_index
            if edge_index.numel() == 0:
                continue

            x_src = logits[spec.src_type]
            frozen_src = frozen.get(spec.src_type, torch.zeros(x_src.size(0), dtype=torch.bool))
            n_dst = logits[spec.dst_type].size(0)

            mp: LogitMessagePassing = self.get_module(spec)
            # propagate wymaga size=(N_src, N_dst) dla bipartite grafów
            p_src = F.softmax(x_src, dim=-1)
            delta_edge = mp.propagate(
                edge_index,
                x=p_src,
                size=(x_src.size(0), n_dst),
            )  # [N_dst, dst_dim]

            dst_type = spec.dst_type
            if dst_type in delta:
                delta[dst_type] = delta[dst_type] + delta_edge
            else:
                delta[dst_type] = delta_edge

        return delta

    def compute_per_edge_deltas(
        self,
        data: HeteroData,
        logits: dict[str, torch.Tensor],
    ) -> list[tuple[EdgeTypeSpec, torch.Tensor, torch.Tensor]]:
        """
        Oblicza Δs per krawędź (nie agreguje).
        Używane przez NeuralTracer podczas inference.

        Zwraca listę (spec, src_indices [E], delta_per_edge [E, dst_dim]).
        """
        result = []

        for spec in self.specs:
            et = (spec.src_type, spec.relation, spec.dst_type)
            if et not in data.edge_types:
                continue

            edge_index = data[spec.src_type, spec.relation, spec.dst_type].edge_index
            if edge_index.numel() == 0:
                continue

            x_src = logits[spec.src_type]
            p_src = F.softmax(x_src, dim=-1)  # [N_src, src_dim]

            mp: LogitMessagePassing = self.get_module(spec)
            src_indices = edge_index[0]              # [E]
            p_u = p_src[src_indices]                 # [E, src_dim]
            delta_per_edge = p_u @ mp.W_pos - p_u @ mp.W_neg  # [E, dst_dim]

            result.append((spec, src_indices, delta_per_edge.detach()))

        return result
