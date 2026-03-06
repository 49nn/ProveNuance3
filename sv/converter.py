"""
Konwersja między typami danych projektu a reprezentacją Clingo LP.

Odpowiada za:
  - Fact / ClusterStateRow → LP string (fakty bazowe dla programu Clingo)
  - clingo.Symbol → GroundAtom (model stabilny → reprezentacja role-based)
  - Normalizację entity_id do poprawnych atomów Clingo
  - Odwrotne mapowanie clingo_id → oryginalny entity_id
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import clingo

from data_model.cluster import ClusterSchema, ClusterStateRow
from data_model.fact import Fact
from sv._utils import to_clingo_id
from sv.types import GroundAtom

if TYPE_CHECKING:
    pass


class IdRegistry:
    """
    Rejestr odwrotnego mapowania clingo_id → oryginalny entity_id.
    Niezbędny do odtworzenia oryginalnych ID w ProofRun.
    """

    def __init__(self) -> None:
        self._orig: dict[str, str] = {}   # clingo_id → original

    def register(self, original: str) -> str:
        """Zwraca clingo_id i zapamiętuje mapowanie."""
        cid = to_clingo_id(original)
        self._orig.setdefault(cid, original)
        return cid

    def original(self, cid: str) -> str:
        """Zwraca oryginalny ID; jeśli nieznany — zwraca cid bez zmian."""
        return self._orig.get(cid, cid)

    def mapping(self) -> dict[str, str]:
        return dict(self._orig)


# ---------------------------------------------------------------------------
# Fact → LP string
# ---------------------------------------------------------------------------

def fact_to_lp(
    fact: Fact,
    registry: IdRegistry,
    predicate_positions: dict[str, list[str]] | None = None,
    truth_threshold: float = 0.5,
) -> str | None:
    """
    Konwertuje fakt do Clingo LP string, np. 'order_placed(c1,o1,d1).'.

    Zwraca None gdy:
      - fakt ma truth.value != "T"
      - fakt ma truth.confidence < truth_threshold (jeśli confidence jest ustawione)
      - predykat nieznany w predicate_positions

    Kolejność argumentów: positional wg predicate_positions.
    """
    if fact.truth.value != "T":
        return None
    if fact.truth.confidence is not None and fact.truth.confidence < truth_threshold:
        return None

    positions = predicate_positions or {}
    pred = fact.predicate.lower()
    roles = positions.get(pred)
    if roles is None:
        return None

    role_map: dict[str, str] = {
        arg.role.upper(): arg.entity_id or arg.literal_value  # type: ignore[arg-type]
        for arg in fact.args
    }
    ordered = [registry.register(role_map[r]) for r in roles if r in role_map]
    if len(ordered) != len(roles):
        return None
    return f"{pred}({','.join(ordered)})."


# ---------------------------------------------------------------------------
# ClusterStateRow → LP string
# ---------------------------------------------------------------------------

def cluster_to_lp(
    state: ClusterStateRow,
    schema: ClusterSchema,
    registry: IdRegistry,
    cluster_roles: dict[str, tuple[str, str]] | None = None,
) -> str | None:
    """
    Konwertuje stan klastra do LP string, np. 'customer_type(c1,consumer).'.

    Top-1 wartość wyznaczana przez argmax(logits); jeśli logits puste — None.
    """
    if not state.logits:
        return None

    top_idx = max(range(len(state.logits)), key=lambda i: state.logits[i])
    value = schema.domain[top_idx].lower()

    roles = cluster_roles or {}
    entity_role, value_role = roles.get(
        state.cluster_name,
        (schema.resolved_entity_role, schema.resolved_value_role),
    )

    eid = registry.register(state.entity_id)
    val = registry.register(value)
    return f"{state.cluster_name}({eid},{val})."


# ---------------------------------------------------------------------------
# LP string → GroundAtom (dla atomów bazowych)
# ---------------------------------------------------------------------------

def lp_to_atom(
    lp_fact: str,
    registry: IdRegistry,
    predicate_positions: dict[str, list[str]] | None = None,
    cluster_roles: dict[str, tuple[str, str]] | None = None,
) -> GroundAtom | None:
    """
    Parsuje LP string 'pred(a,b,c).' → GroundAtom z role-based bindings.
    Używany do budowania zbioru base_atoms przed solve().
    """
    lp = lp_fact.rstrip(". \t\n")
    if "(" not in lp:
        # atom bez argumentów, np. 'prepaid(card)' albo 'some_flag.'
        pred = lp.strip().lower()
        return GroundAtom(pred, ())

    pred_part, args_part = lp.split("(", 1)
    pred = pred_part.strip().lower()
    args_part = args_part.rstrip(")")
    args = [a.strip() for a in args_part.split(",")]

    positions = predicate_positions or {}
    roles_n = positions.get(pred)
    if roles_n and len(roles_n) == len(args):
        bindings = tuple(sorted(zip(roles_n, args)))
        return GroundAtom(pred, bindings)

    c_roles = cluster_roles or {}
    if pred in c_roles and len(args) == 2:
        entity_role, value_role = c_roles[pred]
        bindings = tuple(sorted([(entity_role, args[0]), (value_role, args[1])]))
        return GroundAtom(pred, bindings)

    # Predykat nieznany (np. pomocniczy) — bindings bez ról
    bindings = tuple(sorted(enumerate(args)))  # type: ignore[arg-type]
    return GroundAtom(pred, tuple((str(i), v) for i, v in enumerate(args)))


# ---------------------------------------------------------------------------
# clingo.Symbol → GroundAtom
# ---------------------------------------------------------------------------

def symbol_to_atom(
    sym: clingo.Symbol,
    predicate_positions: dict[str, list[str]] | None = None,
    cluster_roles: dict[str, tuple[str, str]] | None = None,
) -> GroundAtom:
    """
    Konwertuje clingo.Symbol (z modelu stabilnego) na GroundAtom z role-based bindings.
    Dla predykatów nieznanych używa indeksów numerycznych jako nazw ról.
    """
    pred = sym.name
    args = [str(a) for a in sym.arguments]

    positions = predicate_positions or {}
    roles_n = positions.get(pred)
    if roles_n and len(roles_n) == len(args):
        bindings = tuple(sorted(zip(roles_n, args)))
        return GroundAtom(pred, bindings)

    c_roles = cluster_roles or {}
    if pred in c_roles and len(args) == 2:
        entity_role, value_role = c_roles[pred]
        bindings = tuple(sorted([(entity_role, args[0]), (value_role, args[1])]))
        return GroundAtom(pred, bindings)

    # Predykat pomocniczy / derywowany — bindings bez znanych ról
    return GroundAtom(pred, tuple((str(i), v) for i, v in enumerate(args)))
