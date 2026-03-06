"""
Ładowanie ClusterSchema z tabel ontologicznych.
"""
from __future__ import annotations

import psycopg

from nn.graph_builder import ClusterSchema


def load_cluster_schemas(conn: psycopg.Connection) -> list[ClusterSchema]:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT
                cd.id,
                cd.name,
                et.name AS entity_type,
                ARRAY_AGG(cdv.value ORDER BY cdv.position) AS domain
            FROM cluster_definitions cd
            JOIN entity_types et ON et.id = cd.entity_type_id
            JOIN cluster_domain_values cdv ON cdv.cluster_id = cd.id
            GROUP BY cd.id, cd.name, et.name
            ORDER BY cd.id
        """)
        rows = cur.fetchall()
    return [
        ClusterSchema(
            cluster_id=row[0],
            name=row[1],
            entity_type=row[2],
            domain=list(row[3]),
        )
        for row in rows
    ]


def load_predicate_positions(conn: psycopg.Connection) -> dict[str, list[str]]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT DISTINCT
                lower(pd.name) AS predicate_name,
                pr.role_name,
                pr.position
            FROM predicate_definitions pd
            JOIN predicate_roles pr ON pr.predicate_id = pd.id
            ORDER BY lower(pd.name), pr.position
            """
        )
        rows = cur.fetchall()

    positions: dict[str, list[str]] = {}
    for predicate_name, role_name, _position in rows:
        positions.setdefault(str(predicate_name), []).append(str(role_name).upper())
    return positions
