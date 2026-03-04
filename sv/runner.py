"""
Clingo runner: składa program LP i zwraca model stabilny.

Generuje tekst LP z Pydantic Rule obiektów (nie potrzebuje clingo_text z DB).
"""
from __future__ import annotations

import clingo

from data_model.common import ConstTerm, Term, VarTerm
from data_model.rule import LiteralType, Rule, RuleBodyLiteral, RuleHead


# ---------------------------------------------------------------------------
# Generowanie tekstu LP z Rule (Pydantic)
# ---------------------------------------------------------------------------

def _term_to_clingo(term: Term) -> str:
    """VarTerm → 'X'  |  ConstTerm → 'const' (lowercase)."""
    if isinstance(term, VarTerm):
        return term.var
    # ConstTerm
    return to_clingo_id(term.const)


def _head_to_lp(head: RuleHead) -> str:
    if not head.args:
        return head.predicate
    args_str = ",".join(_term_to_clingo(a.term) for a in head.args)
    return f"{head.predicate}({args_str})"


def _literal_to_lp(lit: RuleBodyLiteral) -> str:
    if not lit.args:
        atom = lit.predicate
    else:
        args_str = ",".join(_term_to_clingo(a.term) for a in lit.args)
        atom = f"{lit.predicate}({args_str})"
    return f"not {atom}" if lit.literal_type == LiteralType.naf else atom


def _safe_vars(rule: Rule) -> set[str]:
    """Zmienne pojawiające się w co najmniej jednym pozytywnym literale ciała."""
    safe: set[str] = set()
    for lit in rule.body:
        if lit.literal_type == LiteralType.pos:
            for arg in lit.args:
                if isinstance(arg.term, VarTerm) and arg.term.var != "_":
                    safe.add(arg.term.var)
    return safe


def _all_named_vars(rule: Rule) -> set[str]:
    """Wszystkie nazwane zmienne (nie-wildcard) w głowie i ciele."""
    vs: set[str] = set()
    for arg in rule.head.args:
        if isinstance(arg.term, VarTerm) and arg.term.var != "_":
            vs.add(arg.term.var)
    for lit in rule.body:
        for arg in lit.args:
            if isinstance(arg.term, VarTerm) and arg.term.var != "_":
                vs.add(arg.term.var)
    return vs


# Nazwa predykatu domeny (wewnętrzna, nie wpływa na logikę)
_DOMAIN = "_sv_domain_"


def rule_to_lp(rule: Rule) -> str:
    """
    Konwertuje Rule Pydantic → Clingo LP string.

    Zmienne "unsafe" (pojawiające się tylko w NAF lub głowie, bez pozytywnego
    literału) są zabezpieczane przez literał domenowy `_sv_domain_(V)`.
    Wartości domeny generowane są przez build_program() z faktów bazowych.

    Przykład:
      contract_formed(O) :- order_accepted(store,O,_).
      prepaid(card).
      ab_refund_hold(O) :- _sv_domain_(O), not returned_or_proof(O).
      can_withdraw(C,O) :- customer_type(C,consumer), ..., not ab_withdraw(C,O).
    """
    head_str = _head_to_lp(rule.head)
    if not rule.body:
        return f"{head_str}."

    unsafe = _all_named_vars(rule) - _safe_vars(rule)
    body_parts = [_literal_to_lp(lit) for lit in rule.body]
    # Wstaw literały domenowe na początku ciała
    domain_lits = [f"{_DOMAIN}({v})" for v in sorted(unsafe)]
    all_body = domain_lits + body_parts
    return f"{head_str} :- {', '.join(all_body)}."


# ---------------------------------------------------------------------------
# Pomocnicza normalizacja (duplikat z converter.py — unikamy cyklicznych importów)
# ---------------------------------------------------------------------------

import re as _re

def to_clingo_id(s: str) -> str:
    safe = _re.sub(r"[^a-z0-9_]", "_", s.lower())
    if not safe or safe[0].isdigit():
        safe = "e_" + safe
    return safe


# ---------------------------------------------------------------------------
# Budowanie programu LP i wywołanie Clingo
# ---------------------------------------------------------------------------

def _domain_facts(base_lp_facts: list[str]) -> list[str]:
    """
    Ekstrahuje wszystkie wartości atomów z faktów bazowych LP i generuje fakty domenowe.
    Np. 'order_placed(c1,o1,d1).' → '_sv_domain_(c1). _sv_domain_(o1). _sv_domain_(d1).'
    Dzięki temu reguły z unsafe variables mogą używać _sv_domain_(V) jako literału domeny.
    """
    import re as _re_local
    entities: set[str] = set()
    for fact in base_lp_facts:
        if "(" not in fact:
            continue
        _, args_part = fact.split("(", 1)
        args_part = args_part.rstrip(". \t)\n")
        for arg in args_part.split(","):
            arg = arg.strip()
            # Pomijamy zmienne (uppercase) i wildcards
            if arg and arg[0].islower() and arg[0] != "_":
                entities.add(arg)
    return [f"{_DOMAIN}({e})." for e in sorted(entities)]


def build_program(
    rules: list[Rule],
    base_lp_facts: list[str],
) -> str:
    """
    Składa kompletny program LP:
      - fakty bazowe (z konwertera)
      - fakty domenowe _sv_domain_ (dla unsafe variables w regułach)
      - reguły (generowane z Rule obiektów)
    """
    lines: list[str] = list(base_lp_facts)
    lines += _domain_facts(base_lp_facts)
    for rule in rules:
        lines.append(rule_to_lp(rule))
    return "\n".join(lines)


def solve(program: str) -> frozenset[clingo.Symbol]:
    """
    Uruchamia Clingo i zwraca model stabilny jako frozenset[Symbol].

    Program Horn+NAF ze stratyfikowaną negacją → zawsze jeden stabilny model.
    `--warn=none` tłumi ostrzeżenia o predykatach niezdefiniowanych w input
    (np. computed predicates jak within_14_days dostarczane zewnętrznie).
    """
    ctl = clingo.Control(["--warn=none"])
    ctl.add("base", [], program)
    ctl.ground([("base", [])])

    result: list[clingo.Symbol] = []
    with ctl.solve(yield_=True) as handle:  # type: ignore[union-attr]
        for model in handle:
            result = list(model.symbols(shown=True))

    return frozenset(result)
