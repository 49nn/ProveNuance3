"""
CRUD dla encji (Entity, EntityLinking, MemorySlotEntry).
"""
from __future__ import annotations

import psycopg
from psycopg.types.json import Jsonb

from data_model.common import ProvenanceItem, Span
from data_model.entity import Entity, EntityLinking, MemorySlotEntry


def _ensure_entity_type(conn: psycopg.Connection, entity_type: str) -> int:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO entity_types(name)
            VALUES (%s)
            ON CONFLICT (name) DO UPDATE SET name = EXCLUDED.name
            RETURNING id
            """,
            (entity_type,),
        )
        return cur.fetchone()[0]  # type: ignore[index]


def upsert_entity(conn: psycopg.Connection, entity: Entity) -> None:
    entity_type_id = _ensure_entity_type(conn, entity.type)

    blocking_keys = entity.linking.blocking_keys or None if entity.linking else None
    last_linked_from = entity.linking.last_linked_from or None if entity.linking else None

    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO entities
                (entity_id, entity_type_id, canonical_name, embedding_ref,
                 blocking_keys, last_linked_from, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (entity_id) DO UPDATE SET
                canonical_name   = EXCLUDED.canonical_name,
                embedding_ref    = EXCLUDED.embedding_ref,
                blocking_keys    = EXCLUDED.blocking_keys,
                last_linked_from = EXCLUDED.last_linked_from,
                updated_at       = now()
            RETURNING id, (xmax = 0) AS inserted
            """,
            (
                entity.entity_id,
                entity_type_id,
                entity.canonical_name,
                entity.embedding_ref,
                blocking_keys,
                last_linked_from,
                entity.created_at,
            ),
        )
        db_id, is_new = cur.fetchone()  # type: ignore[misc]

        # Aliases – has PK(entity_id, alias), safe to re-run
        for alias in entity.aliases:
            cur.execute(
                "INSERT INTO entity_aliases(entity_id, alias) VALUES (%s, %s) ON CONFLICT DO NOTHING",
                (db_id, alias),
            )

        # Provenance – only on initial insert to avoid duplicates
        if is_new:
            for prov in entity.provenance:
                span_start = prov.span.start if prov.span else None
                span_end = prov.span.end if prov.span else None
                spans_param = (
                    Jsonb([{"start": s.start, "end": s.end} for s in prov.spans])
                    if prov.spans
                    else None
                )
                cur.execute(
                    """
                    INSERT INTO entity_provenance
                        (entity_id, source_id, span_start, span_end, spans, extractor, confidence, note)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (db_id, prov.source_id, span_start, span_end, spans_param,
                     prov.extractor, prov.confidence, prov.note),
                )

        # Memory slots – each call appends new versions (version = max+1)
        for slot_name, entries in entity.memory_slots.items():
            for entry in entries:
                cur.execute(
                    """
                    INSERT INTO entity_slots
                        (entity_id, slot_name, value, normalized,
                         valid_from, valid_to, confidence, source_rank, version)
                    VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s,
                        COALESCE(
                            (SELECT MAX(version) + 1
                             FROM entity_slots
                             WHERE entity_id = %s AND slot_name = %s),
                            1
                        )
                    )
                    RETURNING id
                    """,
                    (
                        db_id, slot_name,
                        Jsonb(entry.value),
                        Jsonb(entry.normalized) if entry.normalized is not None else None,
                        entry.valid_from, entry.valid_to,
                        entry.confidence, entry.source_rank,
                        db_id, slot_name,
                    ),
                )
                slot_id = cur.fetchone()[0]  # type: ignore[index]
                for prov in entry.provenance:
                    span_start = prov.span.start if prov.span else None
                    span_end = prov.span.end if prov.span else None
                    spans_param = (
                        Jsonb([{"start": s.start, "end": s.end} for s in prov.spans])
                        if prov.spans
                        else None
                    )
                    cur.execute(
                        """
                        INSERT INTO entity_slot_provenance
                            (slot_id, source_id, span_start, span_end, spans,
                             extractor, confidence, note)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (slot_id, prov.source_id, span_start, span_end, spans_param,
                         prov.extractor, prov.confidence, prov.note),
                    )


def _load_entity_record(
    conn: psycopg.Connection,
    entity_id: str,
) -> Entity | None:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT e.id, e.entity_id, et.name, e.canonical_name, e.embedding_ref,
                   e.blocking_keys, e.last_linked_from, e.created_at, e.updated_at
            FROM entities e
            JOIN entity_types et ON et.id = e.entity_type_id
            WHERE e.entity_id = %s
            """,
            (entity_id,),
        )
        row = cur.fetchone()
        if row is None:
            return None

        db_id, eid, etype, canonical, emb_ref, blocking_keys, last_linked, created_at, updated_at = row

        # Aliases
        cur.execute("SELECT alias FROM entity_aliases WHERE entity_id = %s", (db_id,))
        aliases = [r[0] for r in cur.fetchall()]

        # Entity-level provenance
        cur.execute(
            """
            SELECT source_id, span_start, span_end, spans, extractor, confidence, note
            FROM entity_provenance WHERE entity_id = %s
            """,
            (db_id,),
        )
        provenance: list[ProvenanceItem] = []
        for src_id, span_start, span_end, spans_json, extractor, confidence, note in cur.fetchall():
            span = Span(start=span_start, end=span_end) if span_start is not None else None
            spans = [Span(**s) for s in spans_json] if spans_json else None
            provenance.append(ProvenanceItem(
                source_id=src_id, span=span, spans=spans,
                extractor=extractor, confidence=confidence, note=note,
            ))

        # Slots + provenance in two queries
        cur.execute(
            """
            SELECT id, slot_name, value, normalized, valid_from, valid_to,
                   confidence, source_rank, version
            FROM entity_slots
            WHERE entity_id = %s
            ORDER BY slot_name, version
            """,
            (db_id,),
        )
        slot_rows = cur.fetchall()

        slot_prov_map: dict[int, list[ProvenanceItem]] = {}
        if slot_rows:
            slot_ids = [r[0] for r in slot_rows]
            cur.execute(
                """
                SELECT slot_id, source_id, span_start, span_end, spans, extractor, confidence, note
                FROM entity_slot_provenance
                WHERE slot_id = ANY(%s)
                ORDER BY slot_id
                """,
                (slot_ids,),
            )
            for slot_id, src_id, span_start, span_end, spans_json, extractor, confidence, note in cur.fetchall():
                span = Span(start=span_start, end=span_end) if span_start is not None else None
                spans = [Span(**s) for s in spans_json] if spans_json else None
                slot_prov_map.setdefault(slot_id, []).append(ProvenanceItem(
                    source_id=src_id, span=span, spans=spans,
                    extractor=extractor, confidence=confidence, note=note,
                ))

        memory_slots: dict[str, list[MemorySlotEntry]] = {}
        for s_id, s_name, value, normalized, valid_from, valid_to, confidence, source_rank, _ in slot_rows:
            entry = MemorySlotEntry(
                value=value,
                normalized=normalized,
                valid_from=valid_from,
                valid_to=valid_to,
                confidence=confidence,
                source_rank=source_rank,
                provenance=slot_prov_map.get(s_id, []),
            )
            memory_slots.setdefault(s_name, []).append(entry)

        linking = None
        if blocking_keys or last_linked:
            linking = EntityLinking(
                blocking_keys=list(blocking_keys) if blocking_keys else [],
                last_linked_from=list(last_linked) if last_linked else [],
            )

        return Entity(
            entity_id=eid,
            type=etype,
            canonical_name=canonical,
            aliases=aliases,
            embedding_ref=emb_ref,
            created_at=created_at,
            updated_at=updated_at,
            provenance=provenance,
            memory_slots=memory_slots,
            linking=linking,
        )


def load_entities_for_case(
    conn: psycopg.Connection,
    case_id: str,
) -> list[Entity]:
    """Zwraca encje referencjonowane przez fakty lub cluster_states danego case."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT DISTINCT fa.entity_id
            FROM fact_args fa
            JOIN facts f ON f.id = fa.fact_id
            JOIN cases c ON c.id = f.case_id
            WHERE c.case_id = %s AND fa.entity_id IS NOT NULL

            UNION

            SELECT DISTINCT e.entity_id
            FROM cluster_states cs
            JOIN entities e ON e.id = cs.entity_id
            JOIN cases c ON c.id = cs.case_id
            WHERE c.case_id = %s
            """,
            (case_id, case_id),
        )
        entity_ids = [row[0] for row in cur.fetchall()]

    entities: list[Entity] = []
    for eid in entity_ids:
        entity = _load_entity_record(conn, eid)
        if entity is not None:
            entities.append(entity)
    return entities


def load_entities_by_ids(
    conn: psycopg.Connection,
    entity_ids: list[str],
) -> list[Entity]:
    unique_ids = list(dict.fromkeys(eid for eid in entity_ids if eid))
    entities: list[Entity] = []
    for eid in unique_ids:
        entity = _load_entity_record(conn, eid)
        if entity is not None:
            entities.append(entity)
    return entities


def _entity_match_score(
    candidate: Entity,
    existing_canonical: str,
    existing_aliases: list[str],
    existing_blocking: list[str],
) -> int:
    score = 0
    if existing_canonical.lower() == candidate.canonical_name.lower():
        score += 100

    candidate_aliases = {a.lower() for a in candidate.aliases}
    existing_aliases_l = {a.lower() for a in existing_aliases}
    overlap_alias = candidate_aliases & existing_aliases_l
    score += 20 * len(overlap_alias)

    cand_bk = set(candidate.linking.blocking_keys if candidate.linking else [])
    overlap_bk = cand_bk & set(existing_blocking or [])
    score += 10 * len(overlap_bk)
    return score


def link_or_upsert_entity(conn: psycopg.Connection, entity: Entity) -> str:
    """
    Deterministic match/create:
      - score by canonical_name, aliases, blocking_keys
      - tie-break by lexicographic entity_id
      - if score <= 0: create/upsert by own entity_id
      - if score > 0: link to best existing entity_id and upsert merged data
    """
    candidate_aliases = [entity.canonical_name] + list(entity.aliases)
    candidate_blocking = entity.linking.blocking_keys if entity.linking else []

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                e.entity_id,
                e.canonical_name,
                COALESCE(e.blocking_keys, '{}') AS blocking_keys,
                COALESCE(ARRAY_AGG(a.alias) FILTER (WHERE a.alias IS NOT NULL), '{}') AS aliases
            FROM entities e
            LEFT JOIN entity_aliases a ON a.entity_id = e.id
            WHERE
                lower(e.canonical_name) = lower(%s)
                OR EXISTS (
                    SELECT 1
                    FROM entity_aliases a2
                    WHERE a2.entity_id = e.id
                      AND lower(a2.alias) = ANY(%s)
                )
                OR (
                    %s::text[] IS NOT NULL
                    AND e.blocking_keys && %s::text[]
                )
            GROUP BY e.id, e.entity_id, e.canonical_name, e.blocking_keys
            """,
            (
                entity.canonical_name,
                [a.lower() for a in candidate_aliases] or [""],
                candidate_blocking if candidate_blocking else None,
                candidate_blocking if candidate_blocking else None,
            ),
        )
        rows = cur.fetchall()

    best_entity_id: str | None = None
    best_score = 0
    for eid, canonical, blocking_keys, aliases in rows:
        score = _entity_match_score(
            candidate=entity,
            existing_canonical=str(canonical),
            existing_aliases=list(aliases or []),
            existing_blocking=list(blocking_keys or []),
        )
        if score > best_score:
            best_score = score
            best_entity_id = str(eid)
        elif score == best_score and score > 0 and best_entity_id is not None:
            best_entity_id = min(best_entity_id, str(eid))

    if best_entity_id is None or best_score <= 0:
        upsert_entity(conn, entity)
        return entity.entity_id

    linked = entity.model_copy(update={"entity_id": best_entity_id})
    upsert_entity(conn, linked)
    return best_entity_id


def resolve_slot_conflicts(
    conn: psycopg.Connection,
    entity_id: str,
) -> dict[str, MemorySlotEntry]:
    """
    Resolve slot conflicts deterministically:
      1) source_rank desc
      2) confidence desc
      3) valid_from desc
      4) stable JSON value representation asc (tie-break)
    """
    entity = _load_entity_record(conn, entity_id)
    if entity is None:
        return {}

    resolved: dict[str, MemorySlotEntry] = {}
    for slot_name, entries in entity.memory_slots.items():
        if not entries:
            continue

        def key(e: MemorySlotEntry) -> tuple[float, float, float, str]:
            sr = float(e.source_rank) if e.source_rank is not None else -1.0
            cf = float(e.confidence) if e.confidence is not None else -1.0
            vf = e.valid_from.timestamp() if e.valid_from is not None else float("-inf")
            stable_val = str(e.normalized if e.normalized is not None else e.value)
            return (sr, cf, vf, stable_val)

        resolved[slot_name] = max(entries, key=key)

    return resolved
