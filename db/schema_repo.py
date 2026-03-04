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
