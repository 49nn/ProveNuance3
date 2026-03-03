"""
Bramkowanie wyjątków (exception gating).

Dla reguły default R z wyjątkiem ab_*:

    m_R^(t)   = p_u^(t) @ W_R          (wiadomość reguły — z LogitMessagePassing)
    g^(t)     = σ( p_exc^(t) @ u )      (bramka; u ∈ R^{exc_dim})
    m'_R^(t)  = (1 − g^(t)) ⊙ m_R^(t)  (wyhamowana wiadomość)

Bramka gasi wpływ reguły domyślnej gdy predykat wyjątku jest aktywny,
zamiast "walczyć" w logitach z wieloma źródłami.

Identyfikacja wyjątków: reguły z body[].literal_type == "naf" i predykatem "ab_*".
Obecnie (wszystkie reguły hard) ExceptionGateBank jest pusty — kod gotowy
gdy pojawią się nauczone reguły z wyjątkami.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch_geometric.data import HeteroData

from .graph_builder import GraphNodeIndex


# ---------------------------------------------------------------------------
# Specyfikacja jednej bramki
# ---------------------------------------------------------------------------

@dataclass
class GateSpec:
    """Jedna para (reguła domyślna, predykat wyjątku)."""
    default_rule_id: str
    exception_cluster_type: str   # np. "c_digital_consent" — typ węzła klastra-wyjątku
    exc_dim: int                  # rozmiar domeny klastra-wyjątku
    dst_type: str                 # typ węzła celu (zwykle "fact")
    dst_dim: int                  # rozmiar domeny celu


# ---------------------------------------------------------------------------
# ExceptionGate — jedna bramka
# ---------------------------------------------------------------------------

class ExceptionGate(nn.Module):
    """
    g^(t) = σ(p_exc^(t) @ u)
    m'_R  = (1 − g) ⊙ m_R

    u ∈ R^{exc_dim} — nauczony wektor bramkowania.
    """

    def __init__(self, exc_dim: int) -> None:
        super().__init__()
        self.u = nn.Parameter(torch.zeros(exc_dim))
        nn.init.normal_(self.u, std=0.01)

    def forward(
        self,
        p_exc: torch.Tensor,  # [N_dst, exc_dim] — softmax rozkład węzła-wyjątku
        m_rule: torch.Tensor, # [N_dst, dst_dim] — wiadomość reguły przed bramkowaniem
    ) -> torch.Tensor:
        """Zwraca (1 − g) ⊙ m_rule. Kształt jak m_rule."""
        g = torch.sigmoid(p_exc @ self.u).unsqueeze(-1)  # [N_dst, 1]
        return (1.0 - g) * m_rule


# ---------------------------------------------------------------------------
# ExceptionGateBank — rejestr bramek
# ---------------------------------------------------------------------------

class ExceptionGateBank(nn.Module):
    """
    Rejestr bramek; po jednej per GateSpec.

    apply_gates() moduluje delta contributions dla reguł domyślnych
    z powiązanymi bramkami. Jeśli bank jest pusty (brak learned rules
    z wyjątkami), zwraca delta bez zmian.
    """

    def __init__(self, gate_specs: list[GateSpec]) -> None:
        super().__init__()
        self.specs = gate_specs

        gates: dict[str, nn.Module] = {}
        for spec in gate_specs:
            gates[spec.default_rule_id] = ExceptionGate(spec.exc_dim)
        self.gates = nn.ModuleDict(gates)

    # ------------------------------------------------------------------

    def apply_gates(
        self,
        data: HeteroData,
        delta: dict[str, torch.Tensor],   # node_type -> Δs tensor (IN-PLACE modulated)
        logits: dict[str, torch.Tensor],  # node_type -> logit tensor (read-only)
        node_index: GraphNodeIndex,
    ) -> dict[str, torch.Tensor]:
        """
        Dla każdej bramki: znajdź węzły-wyjątki, oblicz g, wyhamuj delta.

        Modifies delta in-place i zwraca je (dla czytelności łańcucha wywołań).
        Jeśli bank jest pusty — bezkosztowy no-op.
        """
        if not self.specs:
            return delta

        import torch.nn.functional as F  # lokalny import — metoda wywoływana rzadko

        for spec in self.specs:
            gate: ExceptionGate = self.gates[spec.default_rule_id]  # type: ignore

            exc_type = spec.exception_cluster_type
            if exc_type not in logits:
                continue

            logits_exc = logits[exc_type]            # [N_exc, exc_dim]
            p_exc = F.softmax(logits_exc, dim=-1)    # [N_exc, exc_dim]

            dst_type = spec.dst_type
            if dst_type not in delta:
                continue

            m_rule = delta[dst_type]  # [N_dst, dst_dim]

            # Bramka działa na węzłach dst powiązanych z węzłami-wyjątkami.
            # Prostym przybliżeniem jest zastosowanie bramki globally:
            # p_exc agregujemy do jednej wartości (mean) i używamy jako skalara.
            # Pełna implementacja per-węzeł wymaga edge_index gate→dst.
            p_exc_mean = p_exc.mean(dim=0, keepdim=True)  # [1, exc_dim]
            p_exc_broadcast = p_exc_mean.expand(m_rule.size(0), -1)  # [N_dst, exc_dim]

            delta[dst_type] = gate(p_exc_broadcast, m_rule)

        return delta

    @classmethod
    def from_rules(cls, rules: list) -> ExceptionGateBank:
        """
        Buduje bank bramek na podstawie listy Rule.
        Tworzy GateSpec dla reguł z body[i].literal_type=='naf'
        i predykatem zaczynającym się na 'ab_'.

        Obecnie wszystkie reguły mają learned=False → pusta lista.
        """
        specs: list[GateSpec] = []
        # TODO: gdy będą learned rules z wyjątkami, wypełnić specs
        return cls(specs)
