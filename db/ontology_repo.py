"""
Zapis ontologii wyekstrahowanej przez LLM do tabel ontologicznych.
"""
from __future__ import annotations

import json

import psycopg
from psycopg.types.json import Jsonb

from nlp.ontology_builder import OntologyResult, clingo_to_head_body


def save_ontology(conn: psycopg.Connection, result: OntologyResult) -> None:
    """
    Zapisuje OntologyResult do tabel ontologicznych.

    Idempotentne — bezpieczne wielokrotne wywołanie (ON CONFLICT DO UPDATE).
    Kolejność: entity_types → predicate_definitions+roles → cluster_definitions+domain
               → rule_modules → rules → sources.

    Wywołujący jest odpowiedzialny za commit.
    """
    _save_entity_types(conn, result)
    _save_predicates(conn, result)
    _save_clusters(conn, result)
    _save_rules(conn, result)
    _save_source(conn, result.source_id)


# ---------------------------------------------------------------------------
# Entity types
# ---------------------------------------------------------------------------

def _save_entity_types(conn: psycopg.Connection, result: OntologyResult) -> None:
    with conn.cursor() as cur:
        for et in result.entity_types:
            cur.execute(
                """
                INSERT INTO entity_types (name, description, source_span_text)
                VALUES (%s, %s, %s)
                ON CONFLICT (name) DO UPDATE SET
                    description      = EXCLUDED.description,
                    source_span_text = EXCLUDED.source_span_text
                """,
                (et.name, et.description, et.source_span_text),
            )


# ---------------------------------------------------------------------------
# Predicates + roles
# ---------------------------------------------------------------------------

def _save_predicates(conn: psycopg.Connection, result: OntologyResult) -> None:
    with conn.cursor() as cur:
        for pred in result.predicates:
            cur.execute(
                """
                INSERT INTO predicate_definitions (name, description, source_span_text)
                VALUES (%s, %s, %s)
                ON CONFLICT (name) DO UPDATE SET
                    description      = EXCLUDED.description,
                    source_span_text = EXCLUDED.source_span_text
                RETURNING id
                """,
                (pred.name, pred.description, pred.source_span_text),
            )
            pred_id = cur.fetchone()[0]  # type: ignore[index]

            # Usuń istniejące role i wstaw nowe (pełna podmiana)
            cur.execute(
                "DELETE FROM predicate_roles WHERE predicate_id = %s",
                (pred_id,),
            )

            for role in pred.roles:
                # Rozwiąż entity_type_id (NULL dla literałów)
                entity_type_id: int | None = None
                if role.entity_type:
                    cur.execute(
                        "SELECT id FROM entity_types WHERE name = %s",
                        (role.entity_type,),
                    )
                    row = cur.fetchone()
                    entity_type_id = row[0] if row else None  # type: ignore[index]

                cur.execute(
                    """
                    INSERT INTO predicate_roles (predicate_id, position, role_name, entity_type_id)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (predicate_id, position) DO UPDATE SET
                        role_name      = EXCLUDED.role_name,
                        entity_type_id = EXCLUDED.entity_type_id
                    """,
                    (pred_id, role.position, role.role, entity_type_id),
                )


# ---------------------------------------------------------------------------
# Clusters + domain values
# ---------------------------------------------------------------------------

def _save_clusters(conn: psycopg.Connection, result: OntologyResult) -> None:
    with conn.cursor() as cur:
        for cl in result.clusters:
            # Rozwiąż entity_type_id — klaster wymaga NOT NULL
            cur.execute(
                "SELECT id FROM entity_types WHERE name = %s",
                (cl.entity_type,),
            )
            row = cur.fetchone()
            if row is None:
                # entity_type nieznany — pomiń klaster (nie ma sensu bez właściciela)
                continue
            entity_type_id = row[0]  # type: ignore[index]

            cur.execute(
                """
                INSERT INTO cluster_definitions (name, entity_type_id, description, source_span_text)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (name) DO UPDATE SET
                    entity_type_id   = EXCLUDED.entity_type_id,
                    description      = EXCLUDED.description,
                    source_span_text = EXCLUDED.source_span_text
                RETURNING id
                """,
                (cl.name, entity_type_id, cl.description, cl.source_span_text),
            )
            cluster_id = cur.fetchone()[0]  # type: ignore[index]

            # Usuń stare wartości domeny i wstaw nowe
            cur.execute(
                "DELETE FROM cluster_domain_values WHERE cluster_id = %s",
                (cluster_id,),
            )
            for pos, value in enumerate(cl.domain):
                cur.execute(
                    """
                    INSERT INTO cluster_domain_values (cluster_id, position, value)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (cluster_id, position) DO UPDATE SET value = EXCLUDED.value
                    """,
                    (cluster_id, pos, value),
                )


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------

def _save_rules(conn: psycopg.Connection, result: OntologyResult) -> None:
    with conn.cursor() as cur:
        for rule in result.rules:
            module_id = _ensure_rule_module(cur, rule.module)

            try:
                head, body = clingo_to_head_body(rule.clingo_text)
            except Exception:
                # Jeśli parser nie radzi sobie z tym clingo_text — pomiń regułę
                continue

            cur.execute(
                """
                INSERT INTO rules (
                    rule_id, module_id, language, head, body,
                    clingo_text, stratum, learned, enabled, source_span_text
                )
                VALUES (
                    %s, %s, 'horn_naf_stratified'::rule_language, %s, %s,
                    %s, %s, FALSE, TRUE, %s
                )
                ON CONFLICT (rule_id) DO UPDATE SET
                    module_id        = EXCLUDED.module_id,
                    head             = EXCLUDED.head,
                    body             = EXCLUDED.body,
                    clingo_text      = EXCLUDED.clingo_text,
                    stratum          = EXCLUDED.stratum,
                    source_span_text = EXCLUDED.source_span_text,
                    enabled          = TRUE,
                    updated_at       = now()
                """,
                (
                    rule.rule_id,
                    module_id,
                    Jsonb(head),
                    Jsonb(body),
                    rule.clingo_text,
                    rule.stratum,
                    rule.source_span_text,
                ),
            )


def _ensure_rule_module(cur: psycopg.Cursor, module_name: str) -> int:
    cur.execute(
        """
        INSERT INTO rule_modules (name, description)
        VALUES (%s, %s)
        ON CONFLICT (name) DO UPDATE SET name = EXCLUDED.name
        RETURNING id
        """,
        (module_name, f"Module: {module_name}"),
    )
    return int(cur.fetchone()[0])  # type: ignore[index]


# ---------------------------------------------------------------------------
# Source
# ---------------------------------------------------------------------------

def _save_source(conn: psycopg.Connection, source_id: str) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO sources (source_id, title, source_type, source_rank)
            VALUES (%s, %s, 'regulation', 0)
            ON CONFLICT (source_id) DO NOTHING
            """,
            (source_id, source_id),
        )
