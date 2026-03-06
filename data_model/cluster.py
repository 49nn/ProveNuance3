"""
Lekkie typy domenowe dla schematów i stanów klastrów.

Ten moduł nie zależy od torch ani warstwy NN, więc może być używany
przez extractor, verifier, DB i explainer bez ładowania stosu PyTorch.
"""
from __future__ import annotations

from dataclasses import dataclass

from data_model.common import Span


@dataclass
class ClusterSchema:
    """Definicja klastra unarnego z aktywnej ontologii."""

    cluster_id: int
    name: str
    entity_type: str
    domain: list[str]
    entity_role: str | None = None
    value_role: str | None = None

    @property
    def dim(self) -> int:
        return len(self.domain)

    @property
    def resolved_entity_role(self) -> str:
        return (self.entity_role or self.entity_type).upper()

    @property
    def resolved_value_role(self) -> str:
        return (self.value_role or "VALUE").upper()


@dataclass
class ClusterStateRow:
    """Stan klastra dla jednej encji w jednym przypadku."""

    entity_id: str
    cluster_name: str
    logits: list[float]
    is_clamped: bool
    clamp_hard: bool
    clamp_source: str
    source_span: Span | None = None
