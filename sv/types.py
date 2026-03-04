"""
Typy wewnętrzne Symbolic Verifier.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

from data_model.fact import Fact


class GroundAtom(NamedTuple):
    """Uziemiony atom predykatu: predykat + posortowane pary (ROLA, wartość)."""
    predicate: str                          # lowercase, np. "order_placed"
    bindings: tuple[tuple[str, str], ...]  # posortowane (ROLA, wartość_clingo)


@dataclass(frozen=True)
class GroundRule:
    """Uziemiona instancja reguły — używana wyłącznie do ekstrakcji proweniencji."""
    rule_id:      str
    stratum:      int
    head:         GroundAtom
    pos_body:     tuple[GroundAtom, ...]
    neg_body:     tuple[GroundAtom, ...]       # atomy NAF (muszą NIE być w modelu)
    substitution: tuple[tuple[str, str], ...]  # posortowane (zmienna, wartość)


@dataclass
class ProofNode:
    """Jeden krok dowodu: atom + reguła która go wyprowadziła."""
    atom:         GroundAtom
    rule_id:      str | None                   # None = fakt bazowy (observed)
    substitution: dict[str, str]
    pos_used:     tuple[GroundAtom, ...]
    neg_checked:  tuple[GroundAtom, ...]       # NAF: te atomy nie były w modelu


@dataclass
class VerifyResult:
    """Wynik weryfikacji symbolicznej."""
    updated_facts:  list[Fact]                  # status zaktualizowany (proved / inferred_candidate)
    new_facts:      list[Fact]                  # fakty derywowane przez reguły, nieobecne w input
    derived_atoms:  frozenset[GroundAtom]       # cały model stabilny
    proof_nodes:    dict[GroundAtom, ProofNode] # proweniencja dla derywowanych atomów
