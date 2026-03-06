"""
Persistence helpers for self-training rounds and pseudo-labels.
"""
from __future__ import annotations

from dataclasses import asdict

import psycopg
from psycopg.types.json import Jsonb

from data_model.cluster import ClusterStateRow
from data_model.common import Span
from data_model.fact import Fact
from data_model.self_training import (
    CaseSplit,
    PseudoClusterLabel,
    PseudoFactLabel,
    RoundStatus,
    SelfTrainingRound,
)

from .case_repo import resolve_case_id_int


def _serialize_cluster_state(state: ClusterStateRow) -> dict[str, object]:
    payload = asdict(state)
    if state.source_span is not None:
        payload["source_span"] = state.source_span.model_dump(mode="json")
    return payload


def _deserialize_cluster_state(payload: dict[str, object]) -> ClusterStateRow:
    row = dict(payload)
    source_span = row.get("source_span")
    if source_span is not None:
        row["source_span"] = Span.model_validate(source_span)
    return ClusterStateRow(**row)


def upsert_round(
    conn: psycopg.Connection,
    round_info: SelfTrainingRound,
) -> int:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO self_training_rounds (
                round_id,
                parent_round_id,
                status,
                teacher_module,
                fact_conf_threshold,
                cluster_top1_threshold,
                cluster_margin_threshold,
                notes,
                promoted_at
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (round_id) DO UPDATE SET
                parent_round_id = EXCLUDED.parent_round_id,
                status = EXCLUDED.status,
                teacher_module = EXCLUDED.teacher_module,
                fact_conf_threshold = EXCLUDED.fact_conf_threshold,
                cluster_top1_threshold = EXCLUDED.cluster_top1_threshold,
                cluster_margin_threshold = EXCLUDED.cluster_margin_threshold,
                notes = EXCLUDED.notes,
                promoted_at = EXCLUDED.promoted_at
            RETURNING id
            """,
            (
                round_info.round_id,
                round_info.parent_round_id,
                round_info.status,
                round_info.teacher_module,
                round_info.fact_conf_threshold,
                round_info.cluster_top1_threshold,
                round_info.cluster_margin_threshold,
                round_info.notes,
                round_info.promoted_at,
            ),
        )
        return int(cur.fetchone()[0])  # type: ignore[index]


def load_round(
    conn: psycopg.Connection,
    round_id: str,
) -> SelfTrainingRound | None:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                round_id,
                parent_round_id,
                status,
                teacher_module,
                fact_conf_threshold,
                cluster_top1_threshold,
                cluster_margin_threshold,
                notes,
                created_at,
                promoted_at
            FROM self_training_rounds
            WHERE round_id = %s
            """,
            (round_id,),
        )
        row = cur.fetchone()
    if row is None:
        return None
    return SelfTrainingRound(
        round_id=row[0],
        parent_round_id=row[1],
        status=row[2],
        teacher_module=row[3],
        fact_conf_threshold=row[4],
        cluster_top1_threshold=row[5],
        cluster_margin_threshold=row[6],
        notes=row[7],
        created_at=row[8],
        promoted_at=row[9],
    )


def set_round_status(
    conn: psycopg.Connection,
    round_id: str,
    status: RoundStatus,
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE self_training_rounds
            SET status = %s
            WHERE round_id = %s
            """,
            (status, round_id),
        )


def promote_round(
    conn: psycopg.Connection,
    round_id: str,
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE self_training_rounds
            SET status = 'promoted',
                promoted_at = now()
            WHERE round_id = %s
            """,
            (round_id,),
        )
        cur.execute(
            """
            UPDATE pseudo_fact_labels pf
            SET promoted = TRUE
            FROM self_training_rounds r
            WHERE pf.round_id = r.id
              AND r.round_id = %s
              AND pf.accepted = TRUE
            """,
            (round_id,),
        )
        cur.execute(
            """
            UPDATE pseudo_cluster_labels pc
            SET promoted = TRUE
            FROM self_training_rounds r
            WHERE pc.round_id = r.id
              AND r.round_id = %s
              AND pc.accepted = TRUE
            """,
            (round_id,),
        )


def list_round_ids(
    conn: psycopg.Connection,
    promoted_only: bool = False,
) -> list[str]:
    where = "WHERE status = 'promoted'" if promoted_only else ""
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT round_id
            FROM self_training_rounds
            {where}
            ORDER BY created_at, round_id
            """
        )
        return [str(row[0]) for row in cur.fetchall()]


def list_case_ids_by_split(
    conn: psycopg.Connection,
    split: CaseSplit,
) -> list[str]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT case_id
            FROM cases
            WHERE dataset_split = %s
            ORDER BY id
            """,
            (split,),
        )
        return [str(row[0]) for row in cur.fetchall()]


def assign_case_split(
    conn: psycopg.Connection,
    split: CaseSplit,
    case_ids: list[str] | None = None,
    pattern: str | None = None,
    all_cases: bool = False,
) -> int:
    if all_cases:
        sql = "UPDATE cases SET dataset_split = %s"
        params: tuple[object, ...] = (split,)
    elif case_ids:
        sql = "UPDATE cases SET dataset_split = %s WHERE case_id = ANY(%s)"
        params = (split, case_ids)
    elif pattern:
        sql = "UPDATE cases SET dataset_split = %s WHERE case_id LIKE %s"
        params = (split, pattern)
    else:
        raise ValueError("assign_case_split requires case_ids, pattern, or all_cases=True")

    with conn.cursor() as cur:
        cur.execute(sql, params)
        return int(cur.rowcount)


def _round_id_map(
    conn: psycopg.Connection,
    round_ids: list[str],
) -> dict[str, int]:
    if not round_ids:
        return {}
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT round_id, id
            FROM self_training_rounds
            WHERE round_id = ANY(%s)
            """,
            (round_ids,),
        )
        return {str(round_id): int(db_id) for round_id, db_id in cur.fetchall()}


def save_pseudo_fact_labels(
    conn: psycopg.Connection,
    labels: list[PseudoFactLabel],
) -> None:
    if not labels:
        return

    round_map = _round_id_map(conn, list({label.round_id for label in labels}))
    case_map = {label.case_id: resolve_case_id_int(conn, label.case_id) for label in labels}

    with conn.cursor() as cur:
        for label in labels:
            round_db_id = round_map.get(label.round_id)
            if round_db_id is None:
                raise ValueError(f"Unknown self-training round: {label.round_id}")
            cur.execute(
                """
                INSERT INTO pseudo_fact_labels (
                    round_id,
                    case_id,
                    fact_key,
                    fact_json,
                    truth_confidence,
                    proof_id,
                    accepted,
                    rejection_reason,
                    promoted
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (round_id, case_id, fact_key) DO UPDATE SET
                    fact_json = EXCLUDED.fact_json,
                    truth_confidence = EXCLUDED.truth_confidence,
                    proof_id = EXCLUDED.proof_id,
                    accepted = EXCLUDED.accepted,
                    rejection_reason = EXCLUDED.rejection_reason,
                    promoted = EXCLUDED.promoted
                """,
                (
                    round_db_id,
                    case_map[label.case_id],
                    label.fact_key,
                    Jsonb(label.fact.model_dump(mode="json")),
                    label.truth_confidence,
                    label.proof_id,
                    label.accepted,
                    label.rejection_reason,
                    label.promoted,
                ),
            )


def save_pseudo_cluster_labels(
    conn: psycopg.Connection,
    labels: list[PseudoClusterLabel],
) -> None:
    if not labels:
        return

    round_map = _round_id_map(conn, list({label.round_id for label in labels}))
    case_map = {label.case_id: resolve_case_id_int(conn, label.case_id) for label in labels}

    with conn.cursor() as cur:
        for label in labels:
            round_db_id = round_map.get(label.round_id)
            if round_db_id is None:
                raise ValueError(f"Unknown self-training round: {label.round_id}")
            cur.execute(
                """
                INSERT INTO pseudo_cluster_labels (
                    round_id,
                    case_id,
                    entity_id,
                    cluster_name,
                    value,
                    state_json,
                    top1_confidence,
                    margin,
                    accepted,
                    rejection_reason,
                    promoted
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (round_id, case_id, entity_id, cluster_name) DO UPDATE SET
                    value = EXCLUDED.value,
                    state_json = EXCLUDED.state_json,
                    top1_confidence = EXCLUDED.top1_confidence,
                    margin = EXCLUDED.margin,
                    accepted = EXCLUDED.accepted,
                    rejection_reason = EXCLUDED.rejection_reason,
                    promoted = EXCLUDED.promoted
                """,
                (
                    round_db_id,
                    case_map[label.case_id],
                    label.entity_id,
                    label.cluster_name,
                    label.value,
                    Jsonb(_serialize_cluster_state(label.state)),
                    label.top1_confidence,
                    label.margin,
                    label.accepted,
                    label.rejection_reason,
                    label.promoted,
                ),
            )


def _round_filter_sql(
    round_ids: list[str] | None,
    promoted_only: bool,
) -> tuple[str, tuple[object, ...]]:
    if round_ids:
        return "AND r.round_id = ANY(%s)", (round_ids,)
    if promoted_only:
        return "AND r.status = 'promoted'", ()
    return "", ()


def load_pseudo_fact_labels_for_case(
    conn: psycopg.Connection,
    case_id: str,
    round_ids: list[str] | None = None,
    promoted_only: bool = False,
) -> list[PseudoFactLabel]:
    extra_where, extra_params = _round_filter_sql(round_ids, promoted_only)
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT
                r.round_id,
                c.case_id,
                pf.fact_key,
                pf.fact_json,
                pf.truth_confidence,
                pf.proof_id,
                pf.accepted,
                pf.rejection_reason,
                pf.promoted
            FROM pseudo_fact_labels pf
            JOIN self_training_rounds r ON r.id = pf.round_id
            JOIN cases c ON c.id = pf.case_id
            WHERE c.case_id = %s
              AND pf.accepted = TRUE
              {extra_where}
            ORDER BY pf.truth_confidence DESC, r.created_at DESC, pf.id DESC
            """,
            (case_id, *extra_params),
        )
        rows = cur.fetchall()

    labels: list[PseudoFactLabel] = []
    for row in rows:
        labels.append(PseudoFactLabel(
            round_id=row[0],
            case_id=row[1],
            fact_key=row[2],
            fact=Fact.model_validate(row[3]),
            truth_confidence=row[4],
            proof_id=row[5],
            accepted=bool(row[6]),
            rejection_reason=row[7],
            promoted=bool(row[8]),
        ))
    return labels


def load_pseudo_cluster_labels_for_case(
    conn: psycopg.Connection,
    case_id: str,
    round_ids: list[str] | None = None,
    promoted_only: bool = False,
) -> list[PseudoClusterLabel]:
    extra_where, extra_params = _round_filter_sql(round_ids, promoted_only)
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT
                r.round_id,
                c.case_id,
                pc.entity_id,
                pc.cluster_name,
                pc.value,
                pc.state_json,
                pc.top1_confidence,
                pc.margin,
                pc.accepted,
                pc.rejection_reason,
                pc.promoted
            FROM pseudo_cluster_labels pc
            JOIN self_training_rounds r ON r.id = pc.round_id
            JOIN cases c ON c.id = pc.case_id
            WHERE c.case_id = %s
              AND pc.accepted = TRUE
              {extra_where}
            ORDER BY pc.margin DESC, pc.top1_confidence DESC, r.created_at DESC, pc.id DESC
            """,
            (case_id, *extra_params),
        )
        rows = cur.fetchall()

    labels: list[PseudoClusterLabel] = []
    for row in rows:
        labels.append(PseudoClusterLabel(
            round_id=row[0],
            case_id=row[1],
            entity_id=row[2],
            cluster_name=row[3],
            value=row[4],
            state=_deserialize_cluster_state(row[5]),
            top1_confidence=row[6],
            margin=row[7],
            accepted=bool(row[8]),
            rejection_reason=row[9],
            promoted=bool(row[10]),
        ))
    return labels


def list_cases_with_pseudo_labels(
    conn: psycopg.Connection,
    round_ids: list[str] | None = None,
    promoted_only: bool = False,
) -> list[str]:
    extra_where, extra_params = _round_filter_sql(round_ids, promoted_only)
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT DISTINCT c.case_id
            FROM (
                SELECT case_id, round_id FROM pseudo_fact_labels WHERE accepted = TRUE
                UNION
                SELECT case_id, round_id FROM pseudo_cluster_labels WHERE accepted = TRUE
            ) pl
            JOIN self_training_rounds r ON r.id = pl.round_id
            JOIN cases c ON c.id = pl.case_id
            WHERE 1 = 1
              {extra_where}
            ORDER BY c.case_id
            """,
            extra_params,
        )
        return [str(row[0]) for row in cur.fetchall()]


__all__ = [
    "assign_case_split",
    "list_case_ids_by_split",
    "list_cases_with_pseudo_labels",
    "list_round_ids",
    "load_pseudo_cluster_labels_for_case",
    "load_pseudo_fact_labels_for_case",
    "load_round",
    "promote_round",
    "save_pseudo_cluster_labels",
    "save_pseudo_fact_labels",
    "set_round_status",
    "upsert_round",
]
