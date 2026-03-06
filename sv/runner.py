"""
Clingo runner: builds an LP program and returns a stable model.
"""
from __future__ import annotations

import logging
import re
from datetime import date

import clingo

from data_model.common import ConstTerm, Term, VarTerm
from data_model.rule import LiteralType, Rule, RuleBodyLiteral, RuleHead
from sv._utils import to_clingo_id

log = logging.getLogger(__name__)


def _term_to_clingo(term: Term) -> str:
    if isinstance(term, VarTerm):
        return term.var
    return to_clingo_id(term.const)


def _head_to_lp(head: RuleHead) -> str:
    if not head.args:
        return head.predicate
    args_str = ",".join(_term_to_clingo(arg.term) for arg in head.args)
    return f"{head.predicate}({args_str})"


def _literal_to_lp(lit: RuleBodyLiteral) -> str:
    if not lit.args:
        atom = lit.predicate
    else:
        args_str = ",".join(_term_to_clingo(arg.term) for arg in lit.args)
        atom = f"{lit.predicate}({args_str})"
    return f"not {atom}" if lit.literal_type == LiteralType.naf else atom


def _safe_vars(rule: Rule) -> set[str]:
    safe: set[str] = set()
    for lit in rule.body:
        if lit.literal_type != LiteralType.pos:
            continue
        for arg in lit.args:
            if isinstance(arg.term, VarTerm) and arg.term.var != "_":
                safe.add(arg.term.var)
    return safe


def _all_named_vars(rule: Rule) -> set[str]:
    vars_: set[str] = set()
    for arg in rule.head.args:
        if isinstance(arg.term, VarTerm) and arg.term.var != "_":
            vars_.add(arg.term.var)
    for lit in rule.body:
        for arg in lit.args:
            if isinstance(arg.term, VarTerm) and arg.term.var != "_":
                vars_.add(arg.term.var)
    return vars_


_DOMAIN = "_sv_domain_"


def _parse_number_token(token: str) -> float | None:
    """
    Parse a number encoded as an LP token.
    Example: 'e_199_99' -> 199.99
    """
    token = token.strip()
    if token.startswith("e_"):
        token = token[2:]
    if "_" in token and token.replace("_", "").isdigit():
        left, right = token.rsplit("_", 1)
        candidate = f"{left}.{right}" if right.isdigit() else token
    else:
        candidate = token
    candidate = candidate.replace("_", "")
    try:
        return float(candidate)
    except ValueError:
        return None


def rule_to_lp(rule: Rule) -> str:
    """
    Convert a Pydantic Rule object into Clingo LP syntax.
    """
    head_str = _head_to_lp(rule.head)
    if not rule.body:
        return f"{head_str}."

    unsafe = _all_named_vars(rule) - _safe_vars(rule)
    body_parts = [_literal_to_lp(lit) for lit in rule.body]
    domain_lits = [f"{_DOMAIN}({var})" for var in sorted(unsafe)]
    return f"{head_str} :- {', '.join(domain_lits + body_parts)}."


def _domain_facts(base_lp_facts: list[str]) -> list[str]:
    entities: set[str] = set()
    for fact in base_lp_facts:
        if "(" not in fact:
            continue
        _, args_part = fact.split("(", 1)
        args_part = args_part.rstrip(". \t)\n")
        for arg in args_part.split(","):
            arg = arg.strip()
            if arg and arg[0].islower() and arg[0] != "_":
                entities.add(arg)
    return [f"{_DOMAIN}({entity})." for entity in sorted(entities)]


def _parse_lp_fact(lp_fact: str) -> tuple[str, list[str]] | None:
    text = lp_fact.strip().rstrip(".")
    if not text:
        return None
    if "(" not in text:
        return text, []
    pred, tail = text.split("(", 1)
    args_raw = tail.rstrip(")")
    args = [arg.strip() for arg in args_raw.split(",")] if args_raw else []
    return pred.strip(), args


_DATE_TOKEN_RE = re.compile(
    r"^(?:e_|d_)?(?P<y>\d{4})_(?P<m>\d{2})_(?P<d>\d{2})(?:_[0-2]\d_[0-5]\d(?:_[0-5]\d)?)?$"
)


def _parse_date_token(token: str) -> date | None:
    match = _DATE_TOKEN_RE.match(token)
    if not match:
        return None
    try:
        return date(int(match.group("y")), int(match.group("m")), int(match.group("d")))
    except ValueError:
        return None


def _extract_computed_facts(base_lp_facts: list[str]) -> list[str]:
    """
    Build helper facts required by rules:
      - within_14_days(DS, DD)
      - paid_within_48h(O, D0)
      - coupon_not_expired(Cpn, today)
      - meets_min_basket(Cpn, O)
    """
    parsed = [item for item in (_parse_lp_fact(fact) for fact in base_lp_facts) if item is not None]

    by_pred: dict[str, list[list[str]]] = {}
    for pred, args in parsed:
        by_pred.setdefault(pred, []).append(args)

    derived: set[str] = set()

    date_tokens: dict[str, date] = {}
    for _, args in parsed:
        for arg in args:
            parsed_date = _parse_date_token(arg)
            if parsed_date is not None:
                date_tokens[arg] = parsed_date

    sorted_dates = sorted(date_tokens.items(), key=lambda item: item[1])
    for index, (from_token, from_date) in enumerate(sorted_dates):
        for to_token, to_date in (item for item in sorted_dates[index:]):
            delta = (to_date - from_date).days
            if delta > 14:
                break
            derived.add(f"within_14_days({from_token},{to_token}).")
            if delta > 0:
                derived.add(f"within_14_days({to_token},{from_token}).")

    for order_args in by_pred.get("order_placed", []):
        if len(order_args) < 3:
            continue
        order_token = order_args[1]
        order_date_token = order_args[2]
        order_date = _parse_date_token(order_date_token)
        if order_date is None:
            continue

        for payment_args in by_pred.get("payment_made", []):
            if len(payment_args) < 3 or payment_args[0] != order_token:
                continue
            payment_date = _parse_date_token(payment_args[2])
            if payment_date is None:
                continue
            delta = (payment_date - order_date).days
            if 0 <= delta <= 2:
                derived.add(f"paid_within_48h({order_token},{order_date_token}).")
                break

    for args in by_pred.get("coupon_applied", []):
        if len(args) >= 3:
            derived.add(f"coupon_not_expired({args[2]},today).")

    order_amount: dict[str, float] = {}
    coupon_min: dict[str, float] = {}
    for args in by_pred.get("order_amount", []):
        if len(args) >= 2:
            value = _parse_number_token(args[1])
            if value is not None:
                order_amount[args[0]] = value
    for args in by_pred.get("coupon_min_basket", []):
        if len(args) >= 2:
            value = _parse_number_token(args[1])
            if value is not None:
                coupon_min[args[0]] = value

    for coupon, min_amount in coupon_min.items():
        for order_token, amount in order_amount.items():
            if amount >= min_amount:
                derived.add(f"meets_min_basket({coupon},{order_token}).")

    return sorted(derived)


def build_program(
    rules: list[Rule],
    base_lp_facts: list[str],
) -> str:
    """
    Assemble a full LP program from base facts, helper facts, domain facts, and rules.
    """
    all_facts = sorted(set(base_lp_facts) | set(_extract_computed_facts(base_lp_facts)))

    lines: list[str] = list(all_facts)
    lines += _domain_facts(all_facts)
    for rule in rules:
        lines.append(rule_to_lp(rule))
    return "\n".join(lines)


_CLINGO_TIMEOUT_SECONDS = 60.0


def solve(program: str) -> frozenset[clingo.Symbol]:
    """
    Run Clingo and return the stable model as frozenset[Symbol].
    """
    log.debug("Clingo solve: %d znakow programu", len(program))
    ctl = clingo.Control(["--warn=none"])
    ctl.add("base", [], program)
    ctl.ground([("base", [])])

    result: frozenset[clingo.Symbol] = frozenset()
    result_holder = {"value": result}
    with ctl.solve(
        async_=True,
        on_model=lambda model: _capture_model(model, result_holder),
    ) as handle:
        if not handle.wait(_CLINGO_TIMEOUT_SECONDS):
            handle.cancel()
            raise TimeoutError(
                f"Clingo solve przekroczyl limit {_CLINGO_TIMEOUT_SECONDS}s"
            )
        handle.get()
        result = result_holder["value"]

    log.debug("Clingo solve: %d atomow w modelu", len(result))
    return result


def _capture_model(
    model: clingo.Model,
    result_holder: dict[str, frozenset[clingo.Symbol]],
) -> None:
    result_holder["value"] = frozenset(model.symbols(shown=True))
