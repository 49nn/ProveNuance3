"""
CRUD dla faktów reifikowanych (Fact, FactArgs, NeuralTrace).
"""
from __future__ import annotations

import psycopg
from psycopg.types.json import Jsonb

from data_model.common import RoleArg, Span, TruthDistribution
from data_model.fact import Fact, FactProvenance, FactSource, FactStatus, FactTime, NeuralTraceItem


def upsert_fact(conn: psycopg.Connection, fact: Fact, case_id_int: int) -> None:
    truth = fact.truth
    truth_logits_param = Jsonb(truth.logits) if truth.logits is not None else None

    source = fact.source
    source_spans_param = (
        Jsonb([{"start": s.start, "end": s.end} for s in source.spans])
        if source is not None
        else None
    )

    time = fact.time
    provenance = fact.provenance

    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO facts (
                fact_id, predicate, arity, status,
                truth_domain, truth_value, truth_confidence, truth_logits,
                event_time, valid_from, valid_to,
                source_id, source_spans, source_extractor, source_confidence,
                proof_id, constraints_tags, case_id
            ) VALUES (
                %s, %s, %s, %s::fact_status,
                %s, %s::truth_value, %s, %s,
                %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s, %s
            )
            ON CONFLICT (fact_id) DO UPDATE SET
                status           = EXCLUDED.status,
                truth_value      = EXCLUDED.truth_value,
                truth_confidence = EXCLUDED.truth_confidence,
                truth_logits     = EXCLUDED.truth_logits,
                proof_id         = EXCLUDED.proof_id
            RETURNING id
            """,
            (
                fact.fact_id, fact.predicate, fact.arity, fact.status.value,
                list(truth.domain), truth.value, truth.confidence, truth_logits_param,
                time.event_time if time else None,
                time.valid_from if time else None,
                time.valid_to if time else None,
                source.source_id if source else None,
                source_spans_param,
                source.extractor if source else None,
                source.confidence if source else None,
                provenance.proof_id if provenance else None,
                list(fact.constraints_tags) if fact.constraints_tags else None,
                case_id_int,
            ),
        )
        fact_db_id: int = cur.fetchone()[0]  # type: ignore[index]

        # Idempotent: delete+reinsert args and neural trace
        cur.execute("DELETE FROM fact_args WHERE fact_id = %s", (fact_db_id,))
        for i, arg in enumerate(fact.args):
            cur.execute(
                """
                INSERT INTO fact_args (fact_id, position, role, entity_id, literal_value, type_hint)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (fact_db_id, i, arg.role, arg.entity_id, arg.literal_value, arg.type_hint),
            )

        cur.execute("DELETE FROM fact_neural_trace WHERE fact_id = %s", (fact_db_id,))
        if provenance and provenance.neural_trace:
            for trace in provenance.neural_trace:
                cur.execute(
                    """
                    INSERT INTO fact_neural_trace
                        (fact_id, step, edge_type, from_fact_id, from_cluster_id, delta_logits)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (
                        fact_db_id, trace.step, trace.edge_type,
                        trace.from_fact_id, trace.from_cluster_id,
                        Jsonb(trace.delta_logits),
                    ),
                )


def attach_proof_run_to_facts(
    conn: psycopg.Connection,
    fact_ids: list[str],
    proof_id: str,
    case_id_int: int,
) -> None:
    unique_fact_ids = list(dict.fromkeys(fid for fid in fact_ids if fid))
    if not unique_fact_ids:
        return

    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE facts
            SET proof_id = %s
            WHERE case_id = %s
              AND fact_id = ANY(%s)
            """,
            (proof_id, case_id_int, unique_fact_ids),
        )


def load_facts_for_case(conn: psycopg.Connection, case_id: str) -> list[Fact]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                f.id, f.fact_id, f.predicate, f.arity, f.status,
                f.truth_domain, f.truth_value, f.truth_confidence, f.truth_logits,
                f.event_time, f.valid_from, f.valid_to,
                f.source_id, f.source_spans, f.source_extractor, f.source_confidence,
                f.proof_id, f.constraints_tags
            FROM facts f
            JOIN cases c ON c.id = f.case_id
            WHERE c.case_id = %s
            ORDER BY f.id
            """,
            (case_id,),
        )
        fact_rows = cur.fetchall()

    if not fact_rows:
        return []

    fact_db_ids = [r[0] for r in fact_rows]

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT fact_id, position, role, entity_id, literal_value, type_hint
            FROM fact_args
            WHERE fact_id = ANY(%s)
            ORDER BY fact_id, position
            """,
            (fact_db_ids,),
        )
        args_by_fact: dict[int, list[tuple]] = {}
        for fid, pos, role, eid, lval, type_hint in cur.fetchall():
            args_by_fact.setdefault(fid, []).append((pos, role, eid, lval, type_hint))

        cur.execute(
            """
            SELECT fact_id, step, edge_type, from_fact_id, from_cluster_id, delta_logits
            FROM fact_neural_trace
            WHERE fact_id = ANY(%s)
            ORDER BY fact_id, step
            """,
            (fact_db_ids,),
        )
        traces_by_fact: dict[int, list[tuple]] = {}
        for fid, step, edge_type, from_fact_id, from_cluster_id, delta_logits in cur.fetchall():
            traces_by_fact.setdefault(fid, []).append(
                (step, edge_type, from_fact_id, from_cluster_id, delta_logits)
            )

    facts: list[Fact] = []
    for row in fact_rows:
        (
            db_id, fact_id, predicate, arity, status,
            truth_domain, truth_value, truth_confidence, truth_logits,
            event_time, valid_from, valid_to,
            source_id, source_spans, source_extractor, source_confidence,
            proof_id, constraints_tags,
        ) = row

        truth = TruthDistribution(
            domain=list(truth_domain) if truth_domain else ["T", "F", "U"],
            value=truth_value,
            confidence=truth_confidence,
            logits=truth_logits,  # psycopg3 auto-parses JSONB → dict
        )

        raw_args = sorted(args_by_fact.get(db_id, []), key=lambda x: x[0])
        args = [
            RoleArg(role=role, entity_id=eid, literal_value=lval, type_hint=type_hint)
            for _, role, eid, lval, type_hint in raw_args
        ]

        source: FactSource | None = None
        if source_id:
            spans = [Span(**s) for s in (source_spans or [])]
            source = FactSource(
                source_id=source_id, spans=spans,
                extractor=source_extractor, confidence=source_confidence,
            )

        time: FactTime | None = None
        if event_time or valid_from or valid_to:
            time = FactTime(event_time=event_time, valid_from=valid_from, valid_to=valid_to)

        traces = [
            NeuralTraceItem(
                step=step, edge_type=edge_type,
                from_fact_id=from_fact_id, from_cluster_id=from_cluster_id,
                delta_logits=delta_logits,
            )
            for step, edge_type, from_fact_id, from_cluster_id, delta_logits
            in traces_by_fact.get(db_id, [])
        ]
        provenance = FactProvenance(proof_id=proof_id, neural_trace=traces)

        facts.append(Fact(
            fact_id=fact_id,
            predicate=predicate,
            arity=arity,
            args=args,
            truth=truth,
            time=time,
            status=FactStatus(status),
            source=source,
            constraints_tags=list(constraints_tags) if constraints_tags else [],
            provenance=provenance,
        ))

    return facts
