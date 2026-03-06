"""
Strict alignment of extracted facts to the active ontology.
"""
from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timedelta

from data_model.common import RoleArg, Span, TruthDistribution
from data_model.entity import Entity
from data_model.fact import Fact, FactSource, FactStatus

from .result import ExtractionResult

_TRUTH_OBSERVED = TruthDistribution(domain=["T", "F", "U"], value="T", confidence=1.0)

_IMPLICIT_BY_ROLE: dict[str, str] = {
    "CUSTOMER": "CUST1",
    "STORE": "STORE1",
}

_SYNTHETIC_ENTITY_TYPES: dict[str, str] = {
    "DEL_": "DELIVERY",
    "STMT_": "STATEMENT",
    "RET_": "RETURN_SHIPMENT",
    "PAY_": "PAYMENT",
    "PROD_": "PRODUCT",
}


def align_extraction_to_ontology(
    result: ExtractionResult,
    predicate_positions: dict[str, list[str]] | None,
    cluster_schemas: list,
    year: int,
) -> ExtractionResult:
    if not predicate_positions:
        return result

    normalized_positions = {
        _norm_name(predicate): [role.upper() for role in roles]
        for predicate, roles in predicate_positions.items()
    }
    if not normalized_positions:
        return result

    allowed_clusters = {getattr(schema, "name") for schema in cluster_schemas}
    aligned_facts: list[Fact] = []
    seen: set[tuple[str, tuple[tuple[str, str], ...]]] = set()

    for fact in result.facts:
        projected = _project_fact(
            source_fact=fact,
            target_predicate=_norm_name(fact.predicate),
            predicate_positions=normalized_positions,
            year=year,
        )
        if projected is not None:
            _append_fact(aligned_facts, seen, projected)

    filtered_cluster_states = [
        cluster_state
        for cluster_state in result.cluster_states
        if cluster_state.cluster_name in allowed_clusters
    ]
    entities = _ensure_entities(result.entities, aligned_facts, result.source_id, year)
    return ExtractionResult(
        entities=entities,
        facts=aligned_facts,
        cluster_states=filtered_cluster_states,
        source_id=result.source_id,
    )


def _project_fact(
    source_fact: Fact,
    target_predicate: str,
    predicate_positions: dict[str, list[str]],
    year: int,
) -> Fact | None:
    target_roles = predicate_positions.get(target_predicate)
    if not target_roles:
        return None

    arg_map = {arg.role.upper(): arg for arg in source_fact.args}
    bindings: list[RoleArg] = []
    for role in target_roles:
        binding = _resolve_role_binding(role, arg_map)
        if binding is None:
            return None
        bindings.append(RoleArg(role=role, **binding))

    return _make_fact(
        source_id=source_fact.source.source_id if source_fact.source else "text",
        predicate=target_predicate.upper(),
        args=bindings,
        span=_first_span(source_fact),
        extractor=_extractor_name(source_fact.source, "OntologyAligner"),
        year=year,
    )


def _resolve_role_binding(role: str, arg_map: dict[str, RoleArg]) -> dict[str, str] | None:
    direct = arg_map.get(role)
    if direct is not None:
        return _to_binding(direct)

    if role in _IMPLICIT_BY_ROLE:
        return {"entity_id": _IMPLICIT_BY_ROLE[role]}
    return None


def _to_binding(arg: RoleArg) -> dict[str, str]:
    if arg.entity_id is not None:
        return {"entity_id": arg.entity_id}
    return {"literal_value": arg.literal_value or ""}


def _make_fact(
    source_id: str,
    predicate: str,
    args: list[RoleArg],
    span: Span,
    extractor: str,
    year: int,
) -> Fact:
    args_key = "|".join(
        sorted(f"{arg.role}:{arg.entity_id or arg.literal_value or ''}" for arg in args)
    )
    seed = f"{source_id}|{predicate}|{span.start or 0}:{span.end or 0}|{args_key}"
    return Fact(
        fact_id=str(uuid.uuid5(uuid.NAMESPACE_URL, seed)),
        predicate=predicate,
        arity=len(args),
        args=args,
        truth=_TRUTH_OBSERVED,
        status=FactStatus.observed,
        source=FactSource(
            source_id=source_id,
            spans=[span],
            extractor=extractor,
            confidence=1.0,
        ),
    )


def _append_fact(
    facts: list[Fact],
    seen: set[tuple[str, tuple[tuple[str, str], ...]]],
    fact: Fact,
) -> None:
    key = (
        fact.predicate.upper(),
        tuple(
            sorted(
                (arg.role.upper(), arg.entity_id or arg.literal_value or "")
                for arg in fact.args
            )
        ),
    )
    if key in seen:
        return
    seen.add(key)
    facts.append(fact)


def _ensure_entities(
    entities: list[Entity],
    facts: list[Fact],
    source_id: str,
    year: int,
) -> list[Entity]:
    out = list(entities)
    seen = {entity.entity_id for entity in entities}

    for fact in facts:
        for arg in fact.args:
            if not arg.entity_id or arg.entity_id in seen:
                continue
            entity_type = _synthetic_entity_type(arg.entity_id)
            if entity_type is None:
                continue
            seen.add(arg.entity_id)
            out.append(Entity(
                entity_id=arg.entity_id,
                type=entity_type,
                canonical_name=arg.entity_id,
                created_at=_stable_timestamp(year, source_id, "entity", entity_type, arg.entity_id),
                provenance=[],
            ))
    return out


def _synthetic_entity_type(entity_id: str) -> str | None:
    for prefix, entity_type in _SYNTHETIC_ENTITY_TYPES.items():
        if entity_id.startswith(prefix):
            return entity_type
    return None


def _stable_timestamp(year: int, *parts: object) -> datetime:
    seed = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    seconds = int(digest[:10], 16) % (365 * 24 * 60 * 60)
    return datetime(year, 1, 1) + timedelta(seconds=seconds)


def _extractor_name(source: FactSource | None, suffix: str) -> str:
    if source is None or not source.extractor:
        return suffix
    return f"{source.extractor}+{suffix}"


def _first_span(fact: Fact) -> Span:
    if fact.source and fact.source.spans:
        return fact.source.spans[0]
    return Span()


def _norm_name(name: str) -> str:
    return name.strip().lower()
