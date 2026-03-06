"""
PipelineResult — wynik jednego przebiegu propose-verify.

Zawiera wszystkie fakty po obu fazach (NN + SV), zaktualizowane stany
klastrów oraz proof DAG. Niezależny od szczegółów NN i SV.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from data_model.cluster import ClusterStateRow
from data_model.fact import Fact, FactStatus
from sv.types import GroundAtom, ProofNode


@dataclass
class PipelineResult:
    """
    Wynik pełnego przebiegu propose-verify.

    facts:          wszystkie fakty po NN + SV
                    (observed | inferred_candidate | proved | rejected)
                    + new_facts dołączone na końcu
    cluster_states: zaktualizowane logity klastrów (wyjście NN)
    new_facts:      fakty derywowane przez reguły SV (podzbiór facts)
    proof_nodes:    proof DAG: GroundAtom → ProofNode
    """

    facts: list[Fact] = field(default_factory=list)
    cluster_states: list[ClusterStateRow] = field(default_factory=list)
    new_facts: list[Fact] = field(default_factory=list)
    proof_nodes: dict[GroundAtom, ProofNode] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Wygodne widoki
    # ------------------------------------------------------------------

    @property
    def proved(self) -> list[Fact]:
        """Fakty ze statusem proved (zweryfikowane symboliczne)."""
        return [f for f in self.facts if f.status == FactStatus.proved]

    @property
    def inferred(self) -> list[Fact]:
        """Fakty ze statusem inferred_candidate (tylko neuronowe, bez dowodu)."""
        return [f for f in self.facts if f.status == FactStatus.inferred_candidate]

    @property
    def observed(self) -> list[Fact]:
        """Fakty ze statusem observed (wejściowe, niezmodyfikowane)."""
        return [f for f in self.facts if f.status == FactStatus.observed]

    def summary(self) -> str:
        """Krótkie podsumowanie wyniku pipeline."""
        return (
            f"PipelineResult: {len(self.facts)} facts total "
            f"(observed={len(self.observed)}, "
            f"proved={len(self.proved)}, "
            f"inferred={len(self.inferred)}, "
            f"new={len(self.new_facts)}), "
            f"cluster_states={len(self.cluster_states)}"
        )
