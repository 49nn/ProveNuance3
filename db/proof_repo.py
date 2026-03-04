"""
Persistence helpers for proof_runs and proof_steps.
"""
from __future__ import annotations

import re
import uuid
from typing import Any

import psycopg
from psycopg.types.json import Jsonb


def _to_clingo_id(s: str) -> str:
    safe = re.sub(r"[^a-z0-9_]", "_", s.lower())
    if not safe or safe[0].isdigit():
        return f"e_{safe}"
    return safe


def _atom_key(predicate: str, bindings: tuple[tuple[str, str], ...]) -> tuple[str, tuple[tuple[str, str], ...]]:
    return (predicate.lower(), tuple(sorted((str(r).upper(), str(v)) for r, v in bindings)))


def _atom_to_str(atom: Any) -> str:
    args = ",".join(v for _, v in atom.bindings)
    if args:
        return f"{atom.predicate}({args})"
    return str(atom.predicate)


def _load_fact_atom_index(
    conn: psycopg.Connection,
    case_id_int: int,
) -> dict[tuple[str, tuple[tuple[str, str], ...]], str]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                f.fact_id,
                f.predicate,
                fa.role,
                fa.entity_id,
                fa.literal_value
            FROM facts f
            LEFT JOIN fact_args fa ON fa.fact_id = f.id
            WHERE f.case_id = %s
            ORDER BY f.id, fa.position
            """,
            (case_id_int,),
        )
        rows = cur.fetchall()

    facts_map: dict[str, tuple[str, list[tuple[str, str]]]] = {}
    for fact_id, predicate, role, entity_id, literal_value in rows:
        pred = str(predicate).lower()
        if fact_id not in facts_map:
            facts_map[fact_id] = (pred, [])
        if role is not None:
            raw_value = entity_id if entity_id is not None else literal_value
            if raw_value is None:
                continue
            facts_map[fact_id][1].append((str(role).upper(), _to_clingo_id(str(raw_value))))

    index: dict[tuple[str, tuple[tuple[str, str], ...]], str] = {}
    for fact_id, (pred, role_vals) in facts_map.items():
        index[(pred, tuple(sorted(role_vals)))] = str(fact_id)
    return index


def save_proof_run(
    conn: psycopg.Connection,
    proof_nodes: dict[Any, Any],
    query: str,
    result: str,
    case_id_int: int,
) -> str:
    proof_id = str(uuid.uuid4())
    fact_atom_index = _load_fact_atom_index(conn, case_id_int)

    ordered_atoms = sorted(proof_nodes.keys(), key=lambda a: _atom_to_str(a))
    proof_dag: dict[str, dict[str, object]] = {}
    step_rows: list[tuple[int, str | None, str, Jsonb, list[str]]] = []

    for i, atom in enumerate(ordered_atoms):
        node = proof_nodes[atom]
        atom_str = _atom_to_str(atom)

        used_fact_ids: list[str] = []
        for dep in node.pos_used:
            dep_key = _atom_key(dep.predicate, dep.bindings)
            dep_fact_id = fact_atom_index.get(dep_key)
            if dep_fact_id:
                used_fact_ids.append(dep_fact_id)

        proof_dag[atom_str] = {
            "rule_id": node.rule_id,
            "body_atoms": [_atom_to_str(dep) for dep in node.pos_used],
            "naf_atoms": [_atom_to_str(dep) for dep in node.neg_checked],
            "substitution": dict(node.substitution),
            "status": "base" if node.rule_id is None else "derived",
        }

        step_rows.append(
            (
                i,
                node.rule_id,
                "",
                Jsonb(dict(node.substitution)),
                used_fact_ids,
            )
        )

    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO proof_runs (proof_id, case_id, query, result, proof_dag)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id
            """,
            (proof_id, case_id_int, query, result, Jsonb(proof_dag)),
        )
        run_id = int(cur.fetchone()[0])  # type: ignore[index]

        for step_order, rule_id, rule_text, substitution, used_fact_ids in step_rows:
            cur.execute(
                """
                INSERT INTO proof_steps
                    (run_id, step_order, rule_id, rule_text, substitution, used_fact_ids)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (
                    run_id,
                    step_order,
                    rule_id,
                    rule_text,
                    substitution,
                    used_fact_ids,
                ),
            )

    return proof_id
