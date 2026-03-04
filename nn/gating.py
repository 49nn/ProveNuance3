"""
Exception gating for default rules with ab_* exceptions.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch_geometric.data import HeteroData

from data_model.rule import LiteralType

from .graph_builder import GraphNodeIndex


@dataclass
class GateSpec:
    """One gate mapping: default rule -> exception cluster -> destination node type."""

    default_rule_id: str
    exception_cluster_type: str
    exc_dim: int
    dst_type: str
    dst_dim: int


class ExceptionGate(nn.Module):
    """
    g = sigmoid(p_exc @ u)
    m' = (1 - g) * m
    """

    def __init__(self, exc_dim: int) -> None:
        super().__init__()
        self.u = nn.Parameter(torch.zeros(exc_dim))
        nn.init.normal_(self.u, std=0.01)

    def forward(self, p_exc: torch.Tensor, m_rule: torch.Tensor) -> torch.Tensor:
        g = torch.sigmoid(p_exc @ self.u).unsqueeze(-1)
        return (1.0 - g) * m_rule


class ExceptionGateBank(nn.Module):
    """Registry of exception gates built from learned rules."""

    def __init__(self, gate_specs: list[GateSpec]) -> None:
        super().__init__()
        self.specs = gate_specs
        self.gates = nn.ModuleDict(
            {spec.default_rule_id: ExceptionGate(spec.exc_dim) for spec in gate_specs}
        )

    def apply_gates(
        self,
        data: HeteroData,
        delta: dict[str, torch.Tensor],
        logits: dict[str, torch.Tensor],
        node_index: GraphNodeIndex,
    ) -> dict[str, torch.Tensor]:
        """
        Apply gates in-place to destination deltas.
        """
        if not self.specs:
            return delta

        import torch.nn.functional as F

        for spec in self.specs:
            gate: ExceptionGate = self.gates[spec.default_rule_id]  # type: ignore[assignment]

            exc_type = spec.exception_cluster_type
            if exc_type not in logits:
                continue

            p_exc = F.softmax(logits[exc_type], dim=-1)

            dst_type = spec.dst_type
            if dst_type not in delta:
                continue

            m_rule = delta[dst_type]
            p_exc_for_dst = self._project_exception_to_dst(
                data=data,
                p_exc=p_exc,
                exc_type=exc_type,
                dst_type=dst_type,
                n_dst=m_rule.size(0),
                node_index=node_index,
            )
            delta[dst_type] = gate(p_exc_for_dst, m_rule)

        return delta

    @staticmethod
    def _project_exception_to_dst(
        data: HeteroData,
        p_exc: torch.Tensor,
        exc_type: str,
        dst_type: str,
        n_dst: int,
        node_index: GraphNodeIndex,
    ) -> torch.Tensor:
        """
        Project exception probabilities to destination nodes.

        Priority:
          1) dst='fact': by edge (exc_type, role_of, fact), averaged per fact
          2) dst='c_*': by shared entity_id between source and target cluster nodes
          3) fallback: global mean
        """
        if p_exc.numel() == 0:
            return torch.zeros(n_dst, 0)

        global_mean = p_exc.mean(dim=0, keepdim=True).expand(n_dst, -1).clone()

        if dst_type == "fact":
            etype = (exc_type, "role_of", "fact")
            if etype not in data.edge_types:
                return global_mean

            edge_index = data[exc_type, "role_of", "fact"].edge_index
            if edge_index.numel() == 0:
                return global_mean

            src = edge_index[0]
            dst = edge_index[1]
            out = torch.zeros_like(global_mean)
            counts = torch.zeros(n_dst, dtype=p_exc.dtype, device=p_exc.device)

            out.index_add_(0, dst, p_exc[src])
            ones = torch.ones(dst.size(0), dtype=p_exc.dtype, device=p_exc.device)
            counts.index_add_(0, dst, ones)

            has = counts > 0
            if has.any():
                out[has] = out[has] / counts[has].unsqueeze(-1)
            if (~has).any():
                out[~has] = global_mean[~has]
            return out

        if exc_type.startswith("c_") and dst_type.startswith("c_"):
            exc_name = exc_type[2:]
            dst_name = dst_type[2:]
            exc_idx_to_entity = node_index.idx_to_cluster_node.get(exc_name, {})
            dst_entity_to_idx = node_index.cluster_node_to_idx.get(dst_name, {})
            out = global_mean

            for exc_idx, entity_id in exc_idx_to_entity.items():
                dst_idx = dst_entity_to_idx.get(entity_id)
                if dst_idx is None:
                    continue
                if exc_idx >= p_exc.size(0) or dst_idx >= out.size(0):
                    continue
                out[dst_idx] = p_exc[exc_idx]
            return out

        return global_mean

    @classmethod
    def from_rules(
        cls,
        rules: list,
        cluster_type_dims: dict[str, int] | None = None,
        fact_dim: int = 3,
    ) -> ExceptionGateBank:
        """
        Build gate specs from learned rules that contain NAF literals with ab_*.
        """
        dims = cluster_type_dims or {}
        specs: list[GateSpec] = []
        seen: set[tuple[str, str, str]] = set()

        exception_rules: dict[str, list] = {}
        for rule in rules:
            exception_rules.setdefault(rule.head.predicate, []).append(rule)

        for rule in rules:
            if not getattr(rule.metadata, "learned", False):
                continue

            naf_ab = [
                lit
                for lit in rule.body
                if lit.literal_type == LiteralType.naf and lit.predicate.startswith("ab_")
            ]
            if not naf_ab:
                continue

            head_pred = rule.head.predicate
            if head_pred in dims:
                dst_type = f"c_{head_pred}"
                dst_dim = dims[head_pred]
            else:
                dst_type = "fact"
                dst_dim = fact_dim

            for naf_lit in naf_ab:
                for exc_rule in exception_rules.get(naf_lit.predicate, []):
                    for body_lit in exc_rule.body:
                        if body_lit.literal_type != LiteralType.pos:
                            continue
                        exc_cluster = body_lit.predicate
                        if exc_cluster not in dims:
                            continue

                        exc_type = f"c_{exc_cluster}"
                        key = (rule.rule_id, exc_type, dst_type)
                        if key in seen:
                            continue
                        seen.add(key)

                        specs.append(
                            GateSpec(
                                default_rule_id=rule.rule_id,
                                exception_cluster_type=exc_type,
                                exc_dim=dims[exc_cluster],
                                dst_type=dst_type,
                                dst_dim=dst_dim,
                            )
                        )

        return cls(specs)
