"""
PipelineResult - wynik jednego przebiegu propose-verify.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from data_model.cluster import ClusterStateRow
from data_model.fact import Fact, FactStatus
from sv.types import CandidateFeedback, GroundAtom, ProofNode


@dataclass
class PipelineResult:
    """
    Wynik pelnego przebiegu propose-verify.

    facts: wszystkie fakty po NN + SV
    cluster_states: zaktualizowane logity klastrow
    new_facts: fakty derywowane przez reguly SV
    proof_nodes: proof DAG
    candidate_feedback: dyskretny feedback verifiera dla candidate facts
    rounds: liczba wykonanych rund propose-verify
    """

    facts: list[Fact] = field(default_factory=list)
    cluster_states: list[ClusterStateRow] = field(default_factory=list)
    new_facts: list[Fact] = field(default_factory=list)
    proof_nodes: dict[GroundAtom, ProofNode] = field(default_factory=dict)
    derived_atoms: frozenset[GroundAtom] = field(default_factory=frozenset)
    candidate_feedback: list[CandidateFeedback] = field(default_factory=list)
    rounds: int = 1

    @property
    def proved(self) -> list[Fact]:
        return [f for f in self.facts if f.status == FactStatus.proved]

    @property
    def inferred(self) -> list[Fact]:
        return [f for f in self.facts if f.status == FactStatus.inferred_candidate]

    @property
    def observed(self) -> list[Fact]:
        return [f for f in self.facts if f.status == FactStatus.observed]

    def summary(self) -> str:
        blocked = sum(1 for item in self.candidate_feedback if item.outcome == "blocked")
        return (
            f"PipelineResult: {len(self.facts)} facts total "
            f"(observed={len(self.observed)}, "
            f"proved={len(self.proved)}, "
            f"inferred={len(self.inferred)}, "
            f"new={len(self.new_facts)}), "
            f"cluster_states={len(self.cluster_states)}, "
            f"feedback={len(self.candidate_feedback)}, "
            f"blocked={blocked}, "
            f"rounds={self.rounds}"
        )
