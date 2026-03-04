"""
Ładowanie reguł Horn+NAF z tabeli rules.
"""
from __future__ import annotations

import psycopg
from psycopg.types.json import Jsonb

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


def _ensure_rule_module(conn: psycopg.Connection, module_name: str) -> int:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO rule_modules(name, description)
            VALUES (%s, %s)
            ON CONFLICT (name) DO UPDATE SET name = EXCLUDED.name
            RETURNING id
            """,
            (module_name, f"Auto-learned rules from {module_name}"),
        )
        return int(cur.fetchone()[0])  # type: ignore[index]


def upsert_learned_rules(
    conn: psycopg.Connection,
    rules: list[Rule],
    module_name: str = "learned_nn",
) -> None:
    """
    Save extracted learned rules into rules table (learned=true).
    """
    if not rules:
        return

    module_id = _ensure_rule_module(conn, module_name)
    with conn.cursor() as cur:
        for rule in rules:
            cur.execute(
                """
                INSERT INTO rules (
                    rule_id, module_id, language, head, body,
                    stratum, learned, weight, support, precision_est,
                    last_validated_at, constraints, enabled
                )
                VALUES (
                    %s, %s, %s::rule_language, %s, %s,
                    %s, %s, %s, %s, %s,
                    %s, %s, TRUE
                )
                ON CONFLICT (rule_id) DO UPDATE SET
                    module_id          = EXCLUDED.module_id,
                    language           = EXCLUDED.language,
                    head               = EXCLUDED.head,
                    body               = EXCLUDED.body,
                    stratum            = EXCLUDED.stratum,
                    learned            = EXCLUDED.learned,
                    weight             = EXCLUDED.weight,
                    support            = EXCLUDED.support,
                    precision_est      = EXCLUDED.precision_est,
                    last_validated_at  = EXCLUDED.last_validated_at,
                    constraints        = EXCLUDED.constraints,
                    enabled            = TRUE,
                    updated_at         = now()
                """,
                (
                    rule.rule_id,
                    module_id,
                    rule.language.value,
                    Jsonb(rule.head.model_dump(mode="python")),
                    Jsonb([lit.model_dump(mode="python") for lit in rule.body]),
                    rule.metadata.stratum,
                    True,
                    rule.metadata.weight,
                    rule.metadata.support,
                    rule.metadata.precision_est,
                    rule.metadata.last_validated_at,
                    list(rule.metadata.constraints) if rule.metadata.constraints else None,
                ),
            )
