"""
CRUD for cluster_states (entity x cluster x case).
"""
from __future__ import annotations

import psycopg

from nn.graph_builder import ClusterStateRow


def resolve_existing_entity_ids(
    conn: psycopg.Connection,
    entity_ids: set[str],
) -> set[str]:
    """Zwraca podzbiór entity_ids które faktycznie istnieją w tabeli entities."""
    if not entity_ids:
        return set()
    with conn.cursor() as cur:
        cur.execute(
            "SELECT entity_id FROM entities WHERE entity_id = ANY(%s)",
            (list(entity_ids),),
        )
        return {row[0] for row in cur.fetchall()}


def _resolve_entity_db_ids(
    conn: psycopg.Connection,
    entity_ids: list[str],
) -> dict[str, int]:
    if not entity_ids:
        return {}
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT entity_id, id
            FROM entities
            WHERE entity_id = ANY(%s)
            """,
            (entity_ids,),
        )
        return {eid: db_id for eid, db_id in cur.fetchall()}


def _resolve_cluster_db_ids(
    conn: psycopg.Connection,
    cluster_names: list[str],
) -> dict[str, int]:
    if not cluster_names:
        return {}
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT name, id
            FROM cluster_definitions
            WHERE name = ANY(%s)
            """,
            (cluster_names,),
        )
        return {name: db_id for name, db_id in cur.fetchall()}


def upsert_cluster_states(
    conn: psycopg.Connection,
    states: list[ClusterStateRow],
    case_id_int: int,
) -> None:
    if not states:
        return

    entity_ids = sorted({s.entity_id for s in states})
    cluster_names = sorted({s.cluster_name for s in states})

    entity_map = _resolve_entity_db_ids(conn, entity_ids)
    cluster_map = _resolve_cluster_db_ids(conn, cluster_names)

    missing_entities = [eid for eid in entity_ids if eid not in entity_map]
    if missing_entities:
        raise ValueError(f"Unknown entities in cluster_states: {missing_entities}")

    missing_clusters = [name for name in cluster_names if name not in cluster_map]
    if missing_clusters:
        raise ValueError(f"Unknown clusters in cluster_states: {missing_clusters}")

    with conn.cursor() as cur:
        for state in states:
            cur.execute(
                """
                INSERT INTO cluster_states
                    (entity_id, cluster_id, case_id, logits, is_clamped, clamp_hard, clamp_source)
                VALUES (%s, %s, %s, %s, %s, %s, %s::clamp_source_t)
                ON CONFLICT (entity_id, cluster_id, case_id) DO UPDATE SET
                    logits       = EXCLUDED.logits,
                    is_clamped   = EXCLUDED.is_clamped,
                    clamp_hard   = EXCLUDED.clamp_hard,
                    clamp_source = EXCLUDED.clamp_source,
                    updated_at   = now()
                """,
                (
                    entity_map[state.entity_id],
                    cluster_map[state.cluster_name],
                    case_id_int,
                    state.logits,
                    state.is_clamped,
                    state.clamp_hard,
                    state.clamp_source,
                ),
            )


def load_cluster_states_for_case(
    conn: psycopg.Connection,
    case_id: str,
) -> list[ClusterStateRow]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                e.entity_id,
                cd.name,
                cs.logits,
                cs.is_clamped,
                cs.clamp_hard,
                cs.clamp_source
            FROM cluster_states cs
            JOIN entities e ON e.id = cs.entity_id
            JOIN cluster_definitions cd ON cd.id = cs.cluster_id
            JOIN cases c ON c.id = cs.case_id
            WHERE c.case_id = %s
            ORDER BY cs.id
            """,
            (case_id,),
        )
        rows = cur.fetchall()

    return [
        ClusterStateRow(
            entity_id=row[0],
            cluster_name=row[1],
            logits=list(row[2]) if row[2] is not None else [],
            is_clamped=bool(row[3]),
            clamp_hard=bool(row[4]),
            clamp_source=str(row[5]),
        )
        for row in rows
    ]
