"""
Ładowanie reguł Horn+NAF z tabeli rules.
"""
from __future__ import annotations

import psycopg

from data_model.common import ConstTerm, VarTerm
from data_model.rule import (
    LiteralType,
    Rule,
    RuleArg,
    RuleBodyLiteral,
    RuleHead,
    RuleLanguage,
    RuleMetadata,
)


def _parse_term(t: dict) -> VarTerm | ConstTerm:
    if "var" in t:
        return VarTerm(var=t["var"])
    return ConstTerm(const=t["const"])


def _parse_rule_arg(a: dict) -> RuleArg:
    return RuleArg(
        role=a["role"],
        term=_parse_term(a["term"]),
        type_hint=a.get("type_hint"),
    )


def _parse_rule_head(h: dict) -> RuleHead:
    return RuleHead(
        predicate=h["predicate"],
        args=[_parse_rule_arg(a) for a in h.get("args", [])],
    )


def _parse_body_literal(b: dict) -> RuleBodyLiteral:
    return RuleBodyLiteral(
        literal_type=LiteralType(b["literal_type"]),
        predicate=b["predicate"],
        args=[_parse_rule_arg(a) for a in b.get("args", [])],
    )


def load_rules(conn: psycopg.Connection, enabled_only: bool = True) -> list[Rule]:
    where = "WHERE r.enabled = TRUE" if enabled_only else ""
    with conn.cursor() as cur:
        cur.execute(f"""
            SELECT
                r.rule_id,
                r.language,
                r.head,
                r.body,
                r.stratum,
                r.learned,
                r.weight,
                r.support,
                r.precision_est,
                r.last_validated_at,
                r.constraints
            FROM rules r
            {where}
            ORDER BY r.stratum, r.id
        """)
        rows = cur.fetchall()

    result: list[Rule] = []
    for row in rows:
        (
            rule_id, language, head_json, body_json,
            stratum, learned, weight, support,
            precision_est, last_validated_at, constraints,
        ) = row
        result.append(Rule(
            rule_id=rule_id,
            language=RuleLanguage(language),
            head=_parse_rule_head(head_json),
            body=[_parse_body_literal(b) for b in body_json],
            metadata=RuleMetadata(
                stratum=stratum,
                learned=learned,
                weight=weight,
                support=support,
                precision_est=precision_est,
                last_validated_at=last_validated_at,
                constraints=list(constraints) if constraints else [],
            ),
        ))
    return result
