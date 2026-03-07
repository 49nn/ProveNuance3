"""
Typy wewnetrzne Symbolic Verifier.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, NamedTuple

from data_model.fact import Fact


class GroundAtom(NamedTuple):
    """Uziemiony atom predykatu: predykat + posortowane pary (ROLA, wartosc)."""

    predicate: str
    bindings: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class GroundRule:
    """Uziemiona instancja reguly uzywana przy ekstrakcji proweniencji."""

    rule_id: str
    stratum: int
    head: GroundAtom
    pos_body: tuple[GroundAtom, ...]
    neg_body: tuple[GroundAtom, ...]
    substitution: tuple[tuple[str, str], ...]


@dataclass
class ProofNode:
    """Jeden krok dowodu dla atomu."""

    atom: GroundAtom
    rule_id: str | None
    substitution: dict[str, str]
    pos_used: tuple[GroundAtom, ...]
    neg_checked: tuple[GroundAtom, ...]


@dataclass
class CandidateFeedback:
    """Dyskretny feedback verifiera dla candidate fact."""

    fact_id: str
    predicate: str
    outcome: Literal["proved", "blocked", "not_proved", "unknown"]
    atom: GroundAtom | None = None
    violated_naf: tuple[GroundAtom, ...] = field(default_factory=tuple)
    missing_pos_body: tuple[GroundAtom, ...] = field(default_factory=tuple)
    supporting_rule_ids: tuple[str, ...] = field(default_factory=tuple)


@dataclass
class VerifyResult:
    """Wynik weryfikacji symbolicznej."""

    updated_facts: list[Fact]
    new_facts: list[Fact]
    derived_atoms: frozenset[GroundAtom]
    proof_nodes: dict[GroundAtom, ProofNode]
    candidate_feedback: list[CandidateFeedback] = field(default_factory=list)
