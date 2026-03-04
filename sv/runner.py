"""
Clingo runner: składa program LP i zwraca model stabilny.

Generuje tekst LP z Pydantic Rule obiektów (nie potrzebuje clingo_text z DB).
"""
from __future__ import annotations

import clingo
from datetime import date
import re

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


def _parse_lp_fact(lp_fact: str) -> tuple[str, list[str]] | None:
    """
    Parse 'pred(a,b,c).' -> ('pred', ['a','b','c']).
    """
    text = lp_fact.strip().rstrip(".")
    if not text:
        return None
    if "(" not in text:
        return text, []
    pred, tail = text.split("(", 1)
    args_raw = tail.rstrip(")")
    args = [a.strip() for a in args_raw.split(",")] if args_raw else []
    return pred.strip(), args


_DATE_TOKEN_RE = re.compile(
    r"^(?:e_|d_)?(?P<y>\d{4})_(?P<m>\d{2})_(?P<d>\d{2})(?:_[0-2]\d_[0-5]\d(?:_[0-5]\d)?)?$"
)


def _parse_date_token(token: str) -> date | None:
    """
    Obsługuje np. e_2026_02_10, d_2026_02_10, 2026_02_10.
    """
    m = _DATE_TOKEN_RE.match(token)
    if not m:
        return None
    try:
        return date(int(m.group("y")), int(m.group("m")), int(m.group("d")))
    except ValueError:
        return None


def _suffix_digits(token: str) -> str:
    m = re.search(r"(\d+)$", token)
    return m.group(1) if m else ""


def _extract_computed_facts(base_lp_facts: list[str]) -> list[str]:
    """
    Buduje fakty pomocnicze wymagane przez reguły (external facts layer):
      - within_14_days(DS, DD)
      - paid_within_48h(O, D0)
      - coupon_not_expired(Cpn, today)
      - meets_min_basket(Cpn, O)  (gdy istnieją order_amount + coupon_min_basket)
      - order_contains(O, P)      (heurystyka nazewnicza)
      - account_of_customer(C, A) (heurystyka nazewnicza)
    """
    parsed = [p for p in (_parse_lp_fact(f) for f in base_lp_facts) if p is not None]

    by_pred: dict[str, list[list[str]]] = {}
    for pred, args in parsed:
        by_pred.setdefault(pred, []).append(args)

    derived: set[str] = set()

    # within_14_days(DS, DD)
    date_tokens: dict[str, date] = {}
    for _, args in parsed:
        for arg in args:
            parsed_date = _parse_date_token(arg)
            if parsed_date is not None:
                date_tokens[arg] = parsed_date

    for ds_token, ds_date in date_tokens.items():
        for dd_token, dd_date in date_tokens.items():
            delta = (ds_date - dd_date).days
            if 0 <= delta <= 14:
                derived.add(f"within_14_days({ds_token},{dd_token}).")

    # paid_within_48h(O, D0) from order_placed(_,O,D0) + payment_made(O,_,DPAY,_)
    order_placed = by_pred.get("order_placed", [])
    payment_made = by_pred.get("payment_made", [])
    for op in order_placed:
        if len(op) < 3:
            continue
        order_token = op[1]
        d0_token = op[2]
        d0 = _parse_date_token(d0_token)
        if d0 is None:
            continue
        for pm in payment_made:
            if len(pm) < 3:
                continue
            if pm[0] != order_token:
                continue
            dpay = _parse_date_token(pm[2])
            if dpay is None:
                continue
            delta = (dpay - d0).days
            if 0 <= delta <= 2:
                derived.add(f"paid_within_48h({order_token},{d0_token}).")
                break

    # coupon_not_expired(Cpn, today) for coupons seen in coupon_applied facts
    for args in by_pred.get("coupon_applied", []):
        if len(args) >= 3:
            cpn = args[2]
            derived.add(f"coupon_not_expired({cpn},today).")

    # meets_min_basket(Cpn, O) when both base facts exist
    # order_amount(O, Amount), coupon_min_basket(Cpn, MinAmount)
    order_amount: dict[str, float] = {}
    coupon_min: dict[str, float] = {}

    def _parse_number_token(token: str) -> float | None:
        token = token.strip()
        if token.startswith("e_"):
            token = token[2:]
        # Heurystyka: ostatni underscore traktujemy jako separator dziesiętny.
        if "_" in token and token.replace("_", "").isdigit():
            left, right = token.rsplit("_", 1)
            if right.isdigit():
                candidate = f"{left}.{right}"
            else:
                candidate = token
        else:
            candidate = token
        candidate = candidate.replace("_", "")
        try:
            return float(candidate)
        except ValueError:
            return None

    for args in by_pred.get("order_amount", []):
        if len(args) >= 2:
            val = _parse_number_token(args[1])
            if val is not None:
                order_amount[args[0]] = val
    for args in by_pred.get("coupon_min_basket", []):
        if len(args) >= 2:
            val = _parse_number_token(args[1])
            if val is not None:
                coupon_min[args[0]] = val

    for cpn, min_amount in coupon_min.items():
        for order_token, amount in order_amount.items():
            if amount >= min_amount:
                derived.add(f"meets_min_basket({cpn},{order_token}).")

    # order_contains(O, P): heurystyka po nazwach identyfikatorów.
    orders: set[str] = set()
    for pred, args in parsed:
        if pred in {
            "order_placed",
            "order_accepted",
            "delivered",
            "withdrawal_statement_submitted",
            "returned",
            "coupon_applied",
            "payment_selected",
            "payment_made",
            "chargeback_opened",
        }:
            if len(args) >= 2:
                # W większości tych predykatów ORDER jest na pozycji 1, ale
                # payment_selected/payment_made mają ORDER na pozycji 0.
                if pred in {"payment_selected", "payment_made"}:
                    orders.add(args[0])
                else:
                    orders.add(args[1])

    products: set[str] = set()
    for pred, args in parsed:
        if pred == "product_type" and args:
            products.add(args[0])
    for token in date_tokens:
        # Ignoruj daty.
        products.discard(token)

    for product in products:
        if product.startswith("prod_"):
            rest = product[5:]
            if rest in orders:
                derived.add(f"order_contains({rest},{product}).")
        digits = _suffix_digits(product)
        if digits:
            candidate = f"o{digits}"
            if candidate in orders:
                derived.add(f"order_contains({candidate},{product}).")

    # account_of_customer(C, A): heurystyka po końcówce numerycznej.
    customers: set[str] = set()
    accounts: set[str] = set()
    for pred, args in parsed:
        if pred == "order_placed" and args:
            customers.add(args[0])
        if pred == "account_status" and args:
            accounts.add(args[0])
        if pred == "account_blocked" and len(args) >= 2:
            accounts.add(args[1])

    for customer in customers:
        c_digits = _suffix_digits(customer)
        if not c_digits:
            continue
        for account in accounts:
            if _suffix_digits(account) == c_digits:
                derived.add(f"account_of_customer({customer},{account}).")

    return sorted(derived)


def build_program(
    rules: list[Rule],
    base_lp_facts: list[str],
) -> str:
    """
    Składa kompletny program LP:
      - fakty bazowe (z konwertera)
      - fakty obliczane (external facts layer)
      - fakty domenowe _sv_domain_ (dla unsafe variables w regułach)
      - reguły (generowane z Rule obiektów)
    """
    base_unique = sorted(set(base_lp_facts))
    computed = _extract_computed_facts(base_unique)
    all_facts = sorted(set(base_unique + computed))

    lines: list[str] = list(all_facts)
    lines += _domain_facts(all_facts)
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
