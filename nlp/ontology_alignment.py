"""
Utilities for aligning extracted facts to the currently active ontology.
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

_ROLE_ALIASES: dict[str, tuple[str, ...]] = {
    "METHOD": ("PAYMENT_METHOD",),
    "PAYMENT_METHOD": ("METHOD",),
    "TYPE": ("VALUE",),
    "VALUE": ("TYPE",),
}

_FACT_ALIASES: dict[str, tuple[str, ...]] = {
    "ORDER_ACCEPTED": (
        "ORDER_CONFIRMED_BY_STORE",
        "STORE_CONFIRMED_ORDER",
        "ORDER_CONFIRMED",
    ),
    "DELIVERED": ("ORDER_DELIVERED",),
    "WITHDRAWAL_STATEMENT_SUBMITTED": (
        "WITHDRAWAL_SUBMITTED",
        "CUSTOMER_SUBMITTED_WITHDRAWAL",
    ),
    "REFUND_ISSUED": ("REFUND_MADE", "STORE_REFUNDED_ORDER"),
    "RETURNED": ("ORDER_RETURNED",),
    "RETURN_PROOF_PROVIDED": ("RETURN_SHIPMENT_PROOF_PROVIDED",),
    "COMPLAINT_SUBMITTED": ("CUSTOMER_SUBMITTED_COMPLAINT",),
    "COMPLAINT_RESPONSE_SENT": ("STORE_RESPONDED_TO_COMPLAINT",),
    "ACCOUNT_BLOCKED": ("ACCOUNT_IS_BLOCKED",),
    "CHARGEBACK_OPENED": ("CHARGEBACK_WAS_OPENED",),
}

_CLUSTER_VALUE_ALIASES: dict[tuple[str, str], tuple[str, ...]] = {
    ("customer_type", "CONSUMER"): ("CUSTOMER_IS_CONSUMER",),
    ("customer_type", "BUSINESS"): ("CUSTOMER_IS_BUSINESS",),
    ("order_status", "ACCEPTED"): ("ORDER_CONFIRMED_BY_STORE",),
    ("order_status", "PAID"): ("ORDER_IS_PAID",),
    ("order_status", "DELIVERED"): ("ORDER_DELIVERED",),
    ("defective", "YES"): ("PRODUCT_IS_DEFECTIVE", "ORDER_IS_DEFECTIVE"),
    ("store_pays_return", "YES"): (
        "STORE_AGREED_TO_COVER_RETURN_COST",
        "STORE_COVERS_RETURN_COST",
    ),
    ("digital_consent", "YES"): ("DIGITAL_CONSENT_WAS_GIVEN",),
    ("download_started_flag", "YES"): ("DOWNLOAD_STARTED", "DIGITAL_DOWNLOAD_STARTED"),
    ("account_status", "BLOCKED"): ("ACCOUNT_IS_BLOCKED",),
    ("coupon_stackable", "NO"): ("COUPON_IS_NOT_STACKABLE",),
    ("password_shared", "YES"): ("PASSWORD_WAS_SHARED",),
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

    schema_map = {getattr(schema, "name"): schema for schema in cluster_schemas}

    aligned_facts: list[Fact] = []
    seen: set[tuple[str, tuple[tuple[str, str], ...]]] = set()

    for fact in result.facts:
        direct = _project_fact(
            source_fact=fact,
            target_predicate=_norm_name(fact.predicate),
            predicate_positions=normalized_positions,
            year=year,
        )
        if direct is not None:
            _append_fact(aligned_facts, seen, direct)

        for alias in _FACT_ALIASES.get(fact.predicate.upper(), ()):
            projected = _project_fact(
                source_fact=fact,
                target_predicate=_norm_name(alias),
                predicate_positions=normalized_positions,
                year=year,
            )
            if projected is not None:
                _append_fact(aligned_facts, seen, projected)

    for cluster_state in result.cluster_states:
        schema = schema_map.get(cluster_state.cluster_name)
        if schema is None:
            continue
        cluster_value = _cluster_value(cluster_state.logits, list(schema.domain))
        cluster_aliases = _CLUSTER_VALUE_ALIASES.get(
            (_norm_name(cluster_state.cluster_name), cluster_value),
            (),
        )
        for alias in cluster_aliases:
            fact = _fact_from_cluster_alias(
                source_id=result.source_id,
                cluster_state=cluster_state,
                target_predicate=_norm_name(alias),
                predicate_positions=normalized_positions,
                year=year,
            )
            if fact is not None:
                _append_fact(aligned_facts, seen, fact)

    entities = _ensure_entities(result.entities, aligned_facts, result.source_id, year)
    return ExtractionResult(
        entities=entities,
        facts=aligned_facts,
        cluster_states=result.cluster_states,
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

    arg_map = {
        arg.role.upper(): arg
        for arg in source_fact.args
    }
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


def _fact_from_cluster_alias(
    source_id: str,
    cluster_state,
    target_predicate: str,
    predicate_positions: dict[str, list[str]],
    year: int,
) -> Fact | None:
    target_roles = predicate_positions.get(target_predicate)
    if not target_roles:
        return None

    context = {
        "ORDER": cluster_state.entity_id if cluster_state.cluster_name != "coupon_stackable" else None,
        "ACCOUNT": cluster_state.entity_id if cluster_state.cluster_name == "account_status" else None,
        "COUPON": cluster_state.entity_id if cluster_state.cluster_name == "coupon_stackable" else None,
        "CUSTOMER": cluster_state.entity_id if cluster_state.cluster_name == "customer_type" else None,
        "PRODUCT": cluster_state.entity_id if cluster_state.cluster_name == "product_type" else None,
    }
    bindings: list[RoleArg] = []
    for role in target_roles:
        binding = _resolve_context_role_binding(role, context)
        if binding is None:
            return None
        bindings.append(RoleArg(role=role, **binding))

    return _make_fact(
        source_id=source_id,
        predicate=target_predicate.upper(),
        args=bindings,
        span=cluster_state.source_span or Span(),
        extractor="OntologyAligner",
        year=year,
    )


def _resolve_role_binding(role: str, arg_map: dict[str, RoleArg]) -> dict[str, str] | None:
    direct = arg_map.get(role)
    if direct is not None:
        return _to_binding(direct)

    for alias in _ROLE_ALIASES.get(role, ()):
        aliased = arg_map.get(alias)
        if aliased is not None:
            return _to_binding(aliased)

    if role in _IMPLICIT_BY_ROLE:
        return {"entity_id": _IMPLICIT_BY_ROLE[role]}

    order_arg = arg_map.get("ORDER")
    if order_arg and order_arg.entity_id:
        if role == "PAYMENT":
            return {"entity_id": f"PAY_{order_arg.entity_id}"}
        if role == "DELIVERY":
            return {"entity_id": f"DEL_{order_arg.entity_id}"}
        if role == "STATEMENT":
            return {"entity_id": f"STMT_{order_arg.entity_id}"}
        if role == "RETURN_SHIPMENT":
            return {"entity_id": f"RET_{order_arg.entity_id}"}
        if role == "PRODUCT":
            product_arg = arg_map.get("PRODUCT")
            return {"entity_id": product_arg.entity_id} if product_arg and product_arg.entity_id else {
                "entity_id": f"PROD_{order_arg.entity_id}"
            }

    return None


def _resolve_context_role_binding(
    role: str,
    context: dict[str, str | None],
) -> dict[str, str] | None:
    value = context.get(role)
    if value:
        return {"entity_id": value}

    if role in _IMPLICIT_BY_ROLE:
        return {"entity_id": _IMPLICIT_BY_ROLE[role]}

    order_id = context.get("ORDER")
    if order_id:
        if role == "PRODUCT":
            return {"entity_id": f"PROD_{order_id}"}
        if role == "PAYMENT":
            return {"entity_id": f"PAY_{order_id}"}
        if role == "DELIVERY":
            return {"entity_id": f"DEL_{order_id}"}
        if role == "STATEMENT":
            return {"entity_id": f"STMT_{order_id}"}
        if role == "RETURN_SHIPMENT":
            return {"entity_id": f"RET_{order_id}"}

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
    if key not in seen:
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
            etype = _synthetic_entity_type(arg.entity_id)
            if etype is None:
                continue
            seen.add(arg.entity_id)
            out.append(
                Entity(
                    entity_id=arg.entity_id,
                    type=etype,
                    canonical_name=arg.entity_id,
                    created_at=_stable_timestamp(year, source_id, "entity", etype, arg.entity_id),
                    provenance=[],
                )
            )

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


def _cluster_value(logits: list[float], domain: list[str]) -> str:
    if not logits or not domain:
        return ""
    best_idx = max(range(len(logits)), key=logits.__getitem__)
    if best_idx >= len(domain):
        return ""
    return str(domain[best_idx]).upper()


def _norm_name(name: str) -> str:
    return name.strip().lower()
