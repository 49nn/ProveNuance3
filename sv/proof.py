"""
Ekstrakcja proweniencji (proof DAG) po rozwiązaniu przez Clingo.

Zawiera:
  1. Backtracking grounder — używany WYŁĄCZNIE do budowy proweniencji,
     nie do rozwiązywania (to robi Clingo).
  2. extract_proof_dag() — dla każdego derywowanego atomu znajduje regułę
     i podstawienie, które go wyprowadziły.
  3. build_proof_run() — buduje serializowalny ProofRun gotowy do zapisu w DB.
"""
from __future__ import annotations

import re
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterator

from data_model.common import ConstTerm, VarTerm
from data_model.rule import LiteralType, Rule, RuleBodyLiteral, RuleHead
from sv.types import GroundAtom, GroundRule, ProofNode


# ---------------------------------------------------------------------------
# Normalizacja (lokalna kopia — unikamy cyklicznego importu runner ↔ proof)
# ---------------------------------------------------------------------------

def _to_cid(s: str) -> str:
    safe = re.sub(r"[^a-z0-9_]", "_", s.lower())
    return ("e_" + safe) if (not safe or safe[0].isdigit()) else safe


_ARG_ROLE_RE = re.compile(r"^ARG(\d+)$", re.IGNORECASE)


def _resolve_role_name(
    role: str,
    predicate: str,
    predicate_positions: dict[str, list[str]] | None,
) -> str:
    role_upper = role.upper()
    m = _ARG_ROLE_RE.match(role_upper)
    if m is None or not predicate_positions:
        return role_upper

    positions = predicate_positions.get(predicate)
    if not positions:
        return role_upper

    idx = int(m.group(1))
    if idx >= len(positions):
        return role_upper
    return positions[idx].upper()


# ---------------------------------------------------------------------------
# Backtracking grounder
# ---------------------------------------------------------------------------

def _match_literal(
    lit: RuleBodyLiteral,
    atom: GroundAtom,
    subst: dict[str, str],
    predicate_positions: dict[str, list[str]] | None = None,
) -> dict[str, str] | None:
    """
    Próbuje rozszerzyć podstawienie `subst` tak, by literał `lit` pasował do `atom`.

    Zwraca nowe (głębsze) podstawienie lub None gdy pasowanie niemożliwe.
    - VarTerm("_") → wildcard, zawsze pasuje, nic nie binduje
    - VarTerm(x)   → binduje lub sprawdza spójność
    - ConstTerm(c) → porównanie z wartością w atomie
    """
    if lit.predicate != atom.predicate:
        return None

    atom_lookup = dict(atom.bindings)
    new_subst = dict(subst)

    for arg in lit.args:
        role = _resolve_role_name(arg.role, lit.predicate, predicate_positions)
        atom_val = atom_lookup.get(role)
        if atom_val is None:
            return None

        term = arg.term
        if isinstance(term, VarTerm):
            if term.var == "_":
                continue  # wildcard
            existing = new_subst.get(term.var)
            if existing is None:
                new_subst[term.var] = atom_val
            elif existing != atom_val:
                return None  # konflikt
        else:  # ConstTerm
            if atom_val != _to_cid(term.const):
                return None

    return new_subst


def _apply_to_head(
    head: RuleHead,
    subst: dict[str, str],
    predicate_positions: dict[str, list[str]] | None = None,
) -> GroundAtom | None:
    """Podstawia zmienne w głowie reguły → GroundAtom."""
    bindings: list[tuple[str, str]] = []
    for arg in head.args:
        role = _resolve_role_name(arg.role, head.predicate, predicate_positions)
        term = arg.term
        if isinstance(term, VarTerm):
            if term.var == "_":
                return None  # głowa nie może mieć wildcard
            val = subst.get(term.var)
            if val is None:
                return None
        else:
            val = _to_cid(term.const)
        bindings.append((role, val))
    return GroundAtom(head.predicate, tuple(sorted(bindings)))


def _apply_to_literal(
    lit: RuleBodyLiteral,
    subst: dict[str, str],
    predicate_positions: dict[str, list[str]] | None = None,
) -> GroundAtom:
    """
    Podstawia zmienne w literale ciała → GroundAtom.
    Wildcardowe role są pomijane (partial atom — do sprawdzenia any_match w NAF).
    """
    bindings: list[tuple[str, str]] = []
    for arg in lit.args:
        role = _resolve_role_name(arg.role, lit.predicate, predicate_positions)
        term = arg.term
        if isinstance(term, VarTerm):
            if term.var == "_":
                continue  # wildcard → pomiń w bindingach
            val = subst.get(term.var, "_unbound_")
        else:
            val = _to_cid(term.const)
        bindings.append((role, val))
    return GroundAtom(lit.predicate, tuple(sorted(bindings)))


def _match_body(
    pos_lits: list[RuleBodyLiteral],
    pred_index: dict[str, list[GroundAtom]],
    subst: dict[str, str],
    matched: list[GroundAtom],
    predicate_positions: dict[str, list[str]] | None = None,
) -> Iterator[tuple[dict[str, str], list[GroundAtom]]]:
    """
    Backtracking: dopasowuje kolejne literały pozytywne do atomów z indeksu.
    Zwraca (podstawienie, lista faktycznie dopasowanych atomów).
    Faktycznie dopasowane atomy są pełne (zawierają wszystkie bindingi),
    w przeciwieństwie do "partial atoms" tworzonych przez _apply_to_literal.
    """
    if not pos_lits:
        yield subst, matched
        return
    lit, rest = pos_lits[0], pos_lits[1:]
    for atom in pred_index.get(lit.predicate, []):
        new_subst = _match_literal(lit, atom, subst, predicate_positions)
        if new_subst is not None:
            yield from _match_body(
                rest,
                pred_index,
                new_subst,
                matched + [atom],
                predicate_positions,
            )


def ground_rule(
    rule: Rule,
    atoms: set[GroundAtom],
    predicate_positions: dict[str, list[str]] | None = None,
) -> Iterator[GroundRule]:
    """
    Generuje wszystkie poprawne uziemienia reguły dla danego zbioru atomów.
    Używane TYLKO do ekstrakcji proweniencji (nie do rozwiązywania).
    """
    pred_index: dict[str, list[GroundAtom]] = defaultdict(list)
    for atom in atoms:
        pred_index[atom.predicate].append(atom)

    pos_lits = [lit for lit in rule.body if lit.literal_type == LiteralType.pos]
    naf_lits = [lit for lit in rule.body if lit.literal_type == LiteralType.naf]

    for subst, matched_atoms in _match_body(
        pos_lits,
        pred_index,
        {},
        [],
        predicate_positions,
    ):
        head = _apply_to_head(rule.head, subst, predicate_positions)
        if head is None:
            continue
        # pos_body: faktycznie dopasowane atomy (pełne bindingi, bez partial)
        pos_body = tuple(matched_atoms)
        # neg_body: partial atoms (wildcardy pominięte — do any_match)
        neg_body = tuple(
            _apply_to_literal(lit, subst, predicate_positions)
            for lit in naf_lits
        )
        yield GroundRule(
            rule_id=rule.rule_id,
            stratum=rule.metadata.stratum,
            head=head,
            pos_body=pos_body,
            neg_body=neg_body,
            substitution=tuple(sorted(subst.items())),
        )


# ---------------------------------------------------------------------------
# Sprawdzenie NAF dla partial atoms (z wildcardami)
# ---------------------------------------------------------------------------

def _any_match(partial: GroundAtom, model: set[GroundAtom]) -> bool:
    """
    Zwraca True gdy jakikolwiek atom w modelu pasuje do partial (superset bindingów).
    Używane do NAF: `not p(X,_)` → sprawdź czy NIE istnieje żaden p(X,*) w modelu.
    """
    needed = set(partial.bindings)
    for atom in model:
        if atom.predicate == partial.predicate:
            if needed.issubset(atom.bindings):
                return True
    return False


# ---------------------------------------------------------------------------
# Ekstrakcja proof DAG z modelu
# ---------------------------------------------------------------------------

def extract_proof_dag(
    derived_atoms: set[GroundAtom],
    base_atoms: set[GroundAtom],
    rules: list[Rule],
    id_map: dict[str, str],
    predicate_positions: dict[str, list[str]] | None = None,
) -> dict[GroundAtom, ProofNode]:
    """
    Dla każdego atomu derywowanego przez reguły (nie bazowego) szuka
    reguły + podstawienia które go wyprowadziły.

    Algorytm: iteracja po regułach, grounding na zbiorze modelu,
    pierwsza reguła której głowa pasuje do docelowego atomu i wszystkie
    warunki spełnione → ProofNode.

    id_map: clingo_id → oryginalny entity_id (do provenancji).
    """
    proofs: dict[GroundAtom, ProofNode] = {}

    # Atomy bazowe mają ProofNode z rule_id=None
    for atom in base_atoms:
        proofs[atom] = ProofNode(
            atom=atom,
            rule_id=None,
            substitution={},
            pos_used=(),
            neg_checked=(),
        )

    # Atomy derywowane
    to_explain = derived_atoms - base_atoms
    for atom in to_explain:
        node = _find_proof_node(atom, derived_atoms, rules, predicate_positions)
        if node is not None:
            proofs[atom] = node

    return proofs


def _find_proof_node(
    target: GroundAtom,
    model: set[GroundAtom],
    rules: list[Rule],
    predicate_positions: dict[str, list[str]] | None = None,
) -> ProofNode | None:
    """Szuka pierwszej reguły która mogła wyprowadzić target."""
    for rule in rules:
        if rule.head.predicate != target.predicate:
            continue
        for gr in ground_rule(rule, model, predicate_positions):
            if gr.head != target:
                continue
            # Sprawdź warunki pozytywne
            if not all(a in model for a in gr.pos_body):
                continue
            # Sprawdź warunki NAF (partial match)
            if any(_any_match(a, model) for a in gr.neg_body):
                continue
            return ProofNode(
                atom=target,
                rule_id=gr.rule_id,
                substitution=dict(gr.substitution),
                pos_used=gr.pos_body,
                neg_checked=gr.neg_body,
            )
    return None


# ---------------------------------------------------------------------------
# Serializowalny ProofRun (mapuje na proof_runs + proof_steps w DB)
# ---------------------------------------------------------------------------

@dataclass
class ProofStep:
    step_order:     int
    rule_id:        str | None
    rule_text:      str
    substitution:   dict[str, str]
    used_fact_ids:  list[str]       # fact_id lub "" dla atomów bez fact_id


@dataclass
class ProofRun:
    proof_id:    str
    result:      str                    # "proved" | "not_proved" | "unknown"
    proof_dag:   list[dict]             # JSON-serializable lista kroków
    steps:       list[ProofStep]


def build_proof_run(
    proofs: dict[GroundAtom, ProofNode],
    query_atoms: list[GroundAtom],
    base_atom_to_fact_id: dict[GroundAtom, str],
    id_map: dict[str, str],
    rules_index: dict[str, Rule] | None = None,
) -> ProofRun:
    """
    Buduje serializowalny ProofRun przez DFS od query_atoms.

    proofs:               wynik extract_proof_dag()
    query_atoms:          atomy które chcemy udowodnić (cel zapytania)
    base_atom_to_fact_id: GroundAtom → fact_id dla faktów wejściowych
    id_map:               clingo_id → oryginalny entity_id
    rules_index:          rule_id → Rule (do generowania rule_text)
    """
    proof_id = str(uuid.uuid4())
    visited: set[GroundAtom] = set()
    ordered: list[GroundAtom] = []

    def dfs(atom: GroundAtom) -> None:
        if atom in visited:
            return
        visited.add(atom)
        node = proofs.get(atom)
        if node and node.pos_used:
            for dep in node.pos_used:
                dfs(dep)
        ordered.append(atom)

    for qa in query_atoms:
        dfs(qa)

    steps: list[ProofStep] = []
    dag_entries: list[dict] = []

    for i, atom in enumerate(ordered):
        node = proofs.get(atom)
        if node is None:
            continue
        rule_text = ""
        if node.rule_id and rules_index:
            r = rules_index.get(node.rule_id)
            if r:
                from sv.runner import rule_to_lp
                rule_text = rule_to_lp(r)

        used_fact_ids = [
            base_atom_to_fact_id.get(a, "") for a in node.pos_used
        ]
        # Odtwórz oryginalne ID w podstawieniu
        orig_subst = {k: id_map.get(v, v) for k, v in node.substitution.items()}

        steps.append(ProofStep(
            step_order=i,
            rule_id=node.rule_id,
            rule_text=rule_text,
            substitution=orig_subst,
            used_fact_ids=used_fact_ids,
        ))
        dag_entries.append({
            "step": i,
            "atom": f"{atom.predicate}({','.join(v for _, v in atom.bindings)})",
            "rule_id": node.rule_id,
            "depends_on": [
                f"{a.predicate}({','.join(v for _, v in a.bindings)})"
                for a in node.pos_used
            ],
            "naf_checked": [
                f"{a.predicate}({','.join(v for _, v in a.bindings)})"
                for a in node.neg_checked
            ],
        })

    proved_any = any(a in proofs for a in query_atoms)
    result = "proved" if proved_any else "not_proved"

    return ProofRun(
        proof_id=proof_id,
        result=result,
        proof_dag=dag_entries,
        steps=steps,
    )
