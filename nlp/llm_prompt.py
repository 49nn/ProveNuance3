"""
Prompt builder and response parser for the LLM extractor.
"""
from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timedelta
from typing import Any

from data_model.common import ProvenanceItem, RoleArg, Span, TruthDistribution
from data_model.entity import Entity
from data_model.fact import Fact, FactSource, FactStatus
from nn.graph_builder import ClusterSchema, ClusterStateRow

from .result import ExtractionResult

IMPLICIT_CUSTOMER = "CUST1"
IMPLICIT_STORE = "STORE1"

_TRUTH_OBSERVED = TruthDistribution(domain=["T", "F", "U"], value="T", confidence=1.0)

_SYNTHETIC_PREFIXES: dict[str, str] = {
    "DEL_": "DELIVERY",
    "STMT_": "STATEMENT",
    "RET_": "RETURN_SHIPMENT",
    "PAY_": "PAYMENT",
    "PROD_": "PRODUCT",
}

_DIRECT_ENTITY_ROLES = {
    "ACCOUNT",
    "CHARGEBACK",
    "COMPLAINT",
    "COUPON",
    "CUSTOMER",
    "DATE",
    "DELIVERY",
    "ORDER",
    "PAYMENT",
    "PRODUCT",
    "RETURN_SHIPMENT",
    "STATEMENT",
    "STORE",
}

_CLAMP_M = 10.0


def build_system_prompt(
    schemas: list[ClusterSchema],
    predicate_positions: dict[str, list[str]],
) -> str:
    lines: list[str] = [
        "Jestes precyzyjnym ekstraktorem faktow prawnych z polskich opisow spraw e-commerce.",
        "Na podstawie tekstu zwroc ustrukturyzowane dane: encje, fakty i wlasciwosci encji (klastry).",
        "",
        "## ENCJE",
        "Kopiuj entity_id dokladnie z tekstu. Nie normalizuj i nie upraszczaj identyfikatorow.",
        "Dopuszczalne sa nowe formaty, np. O183a, PAY183a, STMT11, PROD11, A205, CB20.",
        "Polskie daty konwertuj do DATE entity_id w formacie D_YYYY-MM-DD.",
        "",
        "Zawsze tworz:",
        f"  - {IMPLICIT_CUSTOMER} (type: CUSTOMER, canonical_name: Klient)",
        f"  - {IMPLICIT_STORE} (type: STORE, canonical_name: Sklep)",
        "",
        "Kazda encja uzyta w facts[] lub cluster_states[] musi tez wystapic w entities[].",
        "",
        "## FAKTY",
        "Wyciagaj tylko fakty, ktore sa wprost wspomniane w tekscie.",
        "Pomin fakt, jesli nie masz pewnosci. role = nazwa roli pozycyjnej.",
        "Dla rol opcjonalnych (DATE, COUPON, COMPLAINT, ACCOUNT, CHARGEBACK):",
        "  pomin argument, jesli nie mozesz ustalic encji - NIE blokuj calego faktu.",
        "",
    ]

    lines.append("Dostepne predykaty (UPPERCASE) i ich role pozycyjne:")
    for pred, roles in predicate_positions.items():
        pred_upper = pred.upper()
        role_str = ", ".join(role.upper() for role in roles)
        lines.append(f"  - {pred_upper}({role_str})")
    lines.append("")

    lines.extend([
        "Rola REASON przyjmuje literal_value (nie entity_id).",
        "Rola RESULT przyjmuje literal_value (nie entity_id).",
        "",
        "## KLASTRY",
        "Dla kazdej encji rozpoznaj jej wlasciwosci klastrowe.",
        "Kazda encja moze miec co najwyzej jedna wartosc dla danego klastra.",
        "",
    ])

    for schema in schemas:
        domain_str = " | ".join(schema.domain)
        lines.append(
            f"  - {schema.name} (entity_type: {schema.entity_type}, "
            f"entity_role: {schema.resolved_entity_role}, value_role: {schema.resolved_value_role}): "
            f"{domain_str}"
        )

    lines.extend([
        "",
        "## WAZNE ZASADY",
        "1. Jeden fakt = jedno zdarzenie. Nie duplikuj faktow.",
        "2. entity_id musi byc dokladnie takim ID jak w tekscie, np. O183a albo PAY11.",
        "3. Daty polskie konwertuj do D_2026-MM-DD.",
        "4. Literaly (REASON, RESULT, AMOUNT, PAYMENT_METHOD) zapisuj jako literal_value, nie entity_id.",
        "5. Klaster: maksymalnie 1 wartosc na pare (entity_id, cluster_name).",
        "6. Jezeli encja nie jest wymieniona wprost, uzyj tylko CUST1 lub STORE1.",
        "7. Nie wymyslaj nowych ID typu DEL_O123 albo PROD_O123, jesli tekst zawiera wlasne ID.",
        "8. Kazda encja referencjonowana przez fakt lub klaster musi istniec w entities[].",
        "9. Dla kazdego faktu podaj span_start/span_end i span_text.",
        "10. Dla kazdej encji i klastra podaj span_text.",
    ])

    return "\n".join(lines)


def build_response_schema() -> dict[str, Any]:
    role_arg_schema = {
        "type": "OBJECT",
        "properties": {
            "role": {"type": "STRING"},
            "entity_id": {"type": "STRING"},
            "literal_value": {"type": "STRING"},
        },
        "required": ["role"],
    }

    entity_schema = {
        "type": "OBJECT",
        "properties": {
            "entity_id": {"type": "STRING"},
            "type": {"type": "STRING"},
            "canonical_name": {"type": "STRING"},
            "span_text": {"type": "STRING"},
        },
        "required": ["entity_id", "type", "canonical_name"],
    }

    fact_schema = {
        "type": "OBJECT",
        "properties": {
            "predicate": {"type": "STRING"},
            "args": {"type": "ARRAY", "items": role_arg_schema},
            "span_start": {"type": "INTEGER"},
            "span_end": {"type": "INTEGER"},
            "span_text": {"type": "STRING"},
        },
        "required": ["predicate", "args"],
    }

    cluster_state_schema = {
        "type": "OBJECT",
        "properties": {
            "entity_id": {"type": "STRING"},
            "cluster_name": {"type": "STRING"},
            "value": {"type": "STRING"},
            "span_text": {"type": "STRING"},
        },
        "required": ["entity_id", "cluster_name", "value"],
    }

    return {
        "type": "OBJECT",
        "properties": {
            "entities": {"type": "ARRAY", "items": entity_schema},
            "facts": {"type": "ARRAY", "items": fact_schema},
            "cluster_states": {"type": "ARRAY", "items": cluster_state_schema},
        },
        "required": ["entities", "facts", "cluster_states"],
    }


def parse_llm_response(
    data: dict[str, Any],
    source_id: str,
    year: int,
    schemas: list[ClusterSchema],
    text: str | None = None,
) -> ExtractionResult:
    schema_map = {s.name: s for s in schemas}

    entities: list[Entity] = []
    seen_ids: set[str] = set()

    def _add_entity(
        eid: str,
        etype: str,
        name: str,
        marker: str,
        span_text: str | None = None,
    ) -> None:
        if not eid or eid in seen_ids:
            return
        seen_ids.add(eid)
        provenance = (
            [ProvenanceItem(source_id=source_id, span=Span(text=span_text), extractor="LLMExtractor")]
            if span_text else []
        )
        entities.append(Entity(
            entity_id=eid,
            type=etype,
            canonical_name=name,
            created_at=_stable_timestamp(year, source_id, "entity", etype, eid, marker),
            provenance=provenance,
        ))

    _add_entity(IMPLICIT_CUSTOMER, "CUSTOMER", "Klient", "implicit_customer")
    _add_entity(IMPLICIT_STORE, "STORE", "Sklep", "implicit_store")

    for item in data.get("entities", []):
        eid = str(item.get("entity_id", "")).strip()
        etype = str(item.get("type", "")).strip().upper()
        name = str(item.get("canonical_name", eid)).strip() or eid
        span_text = str(item.get("span_text", "")).strip() or None
        if eid and etype:
            _add_entity(eid, etype, name, "llm", span_text=span_text)

    facts: list[Fact] = []
    seen_fact_keys: set[tuple[str, ...]] = set()

    for item in data.get("facts", []):
        predicate = str(item.get("predicate", "")).strip().upper()
        if not predicate:
            continue

        args: list[RoleArg] = []
        for raw_arg in item.get("args", []):
            role = str(raw_arg.get("role", "")).strip().upper()
            entity_id = raw_arg.get("entity_id")
            literal_value = raw_arg.get("literal_value")
            entity_id = str(entity_id).strip() if entity_id is not None else None
            literal_value = str(literal_value).strip() if literal_value is not None else None
            if not role:
                continue
            if not entity_id:
                entity_id = None
            if not literal_value:
                literal_value = None
            if entity_id is None and literal_value is None:
                continue
            args.append(RoleArg(role=role, entity_id=entity_id, literal_value=literal_value))

        if not args:
            continue

        fact_key = (predicate,) + tuple(
            sorted(f"{arg.role}:{arg.entity_id or arg.literal_value or ''}" for arg in args)
        )
        if fact_key in seen_fact_keys:
            continue
        seen_fact_keys.add(fact_key)

        span_start = int(item["span_start"]) if item.get("span_start") is not None else None
        span_end = int(item["span_end"]) if item.get("span_end") is not None else None
        llm_text = str(item.get("span_text", "")).strip() or None
        computed_text = llm_text
        if computed_text is None and text is not None and span_start is not None and span_end is not None:
            computed_text = text[span_start:span_end]
        span = Span(start=span_start, end=span_end, text=computed_text)

        seed = f"{source_id}|{predicate}|{span.start or 0}:{span.end or 0}|{'|'.join(fact_key[1:])}"
        fact = Fact(
            fact_id=str(uuid.uuid5(uuid.NAMESPACE_URL, seed)),
            predicate=predicate,
            arity=len(args),
            args=args,
            truth=_TRUTH_OBSERVED,
            status=FactStatus.observed,
            source=FactSource(
                source_id=source_id,
                spans=[span],
                extractor="LLMExtractor",
                confidence=1.0,
            ),
        )
        facts.append(fact)

        for arg in args:
            if arg.entity_id and arg.entity_id not in seen_ids:
                entity_type = _infer_entity_type_from_role(arg.role, arg.entity_id, schemas)
                if entity_type is not None:
                    _add_entity(arg.entity_id, entity_type, arg.entity_id, f"fact:{predicate}")

    cluster_states: list[ClusterStateRow] = []
    seen_clusters: set[tuple[str, str]] = set()

    for item in data.get("cluster_states", []):
        eid = str(item.get("entity_id", "")).strip()
        cluster_name = str(item.get("cluster_name", "")).strip()
        value = str(item.get("value", "")).strip().upper()
        if not eid or not cluster_name or not value:
            continue

        schema = schema_map.get(cluster_name)
        if schema is None or value not in schema.domain:
            continue

        cluster_key = (eid, cluster_name)
        if cluster_key in seen_clusters:
            continue
        seen_clusters.add(cluster_key)

        span_text = str(item.get("span_text", "")).strip() or None
        logits = [_CLAMP_M if domain_value == value else -_CLAMP_M for domain_value in schema.domain]
        cluster_states.append(ClusterStateRow(
            entity_id=eid,
            cluster_name=cluster_name,
            logits=logits,
            is_clamped=True,
            clamp_hard=True,
            clamp_source="text",
            source_span=Span(text=span_text) if span_text else None,
        ))

        if eid not in seen_ids:
            _add_entity(eid, schema.entity_type, eid, f"cluster:{cluster_name}", span_text=span_text)

    return ExtractionResult(
        entities=entities,
        facts=facts,
        cluster_states=cluster_states,
        source_id=source_id,
    )


def build_correction_prompt(
    original_text: str,
    conflicts: list[str],
) -> str:
    conflict_block = "\n".join(f"  - {conflict}" for conflict in conflicts)
    return (
        "Poprzednia ekstrakcja zawierala sprzecznosci logiczne:\n"
        f"{conflict_block}\n\n"
        "Popraw ekstrakcje eliminujac powyzsze sprzecznosci. "
        "Jesli nie mozesz rozstrzygnac, pomin watpliwy fakt lub wartosc klastra.\n\n"
        f"Oryginalny tekst:\n{original_text}"
    )


def _stable_timestamp(year: int, *parts: object) -> datetime:
    seed = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    seconds = int(digest[:10], 16) % (365 * 24 * 60 * 60)
    base = datetime(year, 1, 1, 0, 0, 0)
    return base + timedelta(seconds=seconds)


def _infer_synthetic_type(entity_id: str) -> str | None:
    for prefix, entity_type in _SYNTHETIC_PREFIXES.items():
        if entity_id.startswith(prefix):
            return entity_type
    return None


def _infer_entity_type_from_role(
    role: str,
    entity_id: str,
    schemas: list[ClusterSchema],
) -> str | None:
    synthetic = _infer_synthetic_type(entity_id)
    if synthetic is not None:
        return synthetic

    role_upper = role.upper()
    if role_upper in _DIRECT_ENTITY_ROLES:
        return role_upper

    for schema in schemas:
        if schema.resolved_entity_role == role_upper:
            return schema.entity_type
    return None
