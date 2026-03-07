"""
Case-level aggregate loaders.
"""
from __future__ import annotations

import psycopg

from data_model.cluster import ClusterStateRow
from data_model.entity import Entity
from data_model.fact import Fact, FactStatus
from data_model.rule import Rule

from .cluster_repo import load_cluster_states_for_case
from .entity_repo import load_entities_for_case
from .fact_repo import load_facts_for_case
from .rule_repo import load_rules


def resolve_case_id_int(conn: psycopg.Connection, case_id: str) -> int:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id
            FROM cases
            WHERE case_id = %s
            """,
            (case_id,),
        )
        row = cur.fetchone()
    if row is None:
        raise ValueError(f"Case not found: {case_id}")
    return int(row[0])


def load_case(
    conn: psycopg.Connection,
    case_id: str,
) -> tuple[list[Entity], list[Fact], list[Rule], list[ClusterStateRow]]:
    # Validate case_id first to fail fast with a clear message.
    resolve_case_id_int(conn, case_id)

    entities = load_entities_for_case(conn, case_id)
    facts = [
        fact
        for fact in load_facts_for_case(conn, case_id)
        if fact.status == FactStatus.observed
    ]
    rules = load_rules(conn, enabled_only=True)
    states = load_cluster_states_for_case(conn, case_id)

    return (entities, facts, rules, states)
