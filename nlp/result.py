"""
ExtractionResult — wynik ekstrakcji z jednego tekstu.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from data_model.entity import Entity
from data_model.fact import Fact
from nn.graph_builder import ClusterStateRow


@dataclass
class ExtractionResult:
    """
    Wynik ekstrakcji reguło-bazowej z tekstu.

    entities:        encje rozpoznane w tekście (explicit + implicit CUST1/STORE1 + DATE)
    facts:           reifikowane fakty (status=observed, truth=T/1.0)
    cluster_states:  stany klastrów (is_clamped=True, clamp_source='text')
    source_id:       identyfikator źródłowy tekstu
    """

    entities: list[Entity] = field(default_factory=list)
    facts: list[Fact] = field(default_factory=list)
    cluster_states: list[ClusterStateRow] = field(default_factory=list)
    source_id: str = "text"

    def summary(self) -> str:
        return (
            f"ExtractionResult: {len(self.entities)} entities, "
            f"{len(self.facts)} facts "
            f"({', '.join(sorted({f.predicate for f in self.facts}))}), "
            f"{len(self.cluster_states)} cluster_states"
        )
