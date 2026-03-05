"""
llm_prompt.py — Budowanie promptów dla Gemini i parsowanie odpowiedzi.

Eksportuje:
    build_system_prompt(schemas, predicate_positions) → str
    build_response_schema() → dict       (JSON Schema dla Gemini structured output)
    parse_llm_response(data, source_id, year, schemas) → ExtractionResult
    build_correction_prompt(original_text, conflicts) → str
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

# ---------------------------------------------------------------------------
# Stałe
# ---------------------------------------------------------------------------

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

_CLAMP_M = 10.0


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def build_system_prompt(
    schemas: list[ClusterSchema],
    predicate_positions: dict[str, list[str]],
) -> str:
    """
    Buduje system prompt dla LLM na podstawie ontologii projektu.
    Prompt jest deterministyczny — przy tych samych schemas zwraca ten sam string.
    """
    lines: list[str] = [
        "Jesteś precyzyjnym ekstrakatorem faktów prawnych z polskich opisów spraw e-commerce.",
        "Na podstawie tekstu zwróć ustrukturyzowane dane: encje, fakty i właściwości encji (klastry).",
        "",
        "## ENCJE",
        "Rozpoznawaj identyfikatory encji według wzorców (rozróżniaj wielkie litery):",
        "  - ORDER:      O\\d+     (np. O100, O25)",
        "  - ACCOUNT:    A\\d+     (np. A5, A123)",
        "  - CHARGEBACK: CB\\d+    (np. CB20) — uwaga: CB przed C",
        "  - COMPLAINT:  C\\d{2,}  (np. C99, C900) — min. 2 cyfry",
        "  - PRODUCT:    P\\d+     (np. P3)",
        "  - DATE:       D_YYYY-MM-DD (np. D_2026-02-10) — konwertuj polskie daty",
        "",
        "Zawsze twórz:",
        f"  - {IMPLICIT_CUSTOMER} (type: CUSTOMER, canonical_name: Klient)",
        f"  - {IMPLICIT_STORE}  (type: STORE, canonical_name: Sklep)",
        "",
        "Encje syntetyczne (twórz gdy pojawiają się w faktach):",
        "  - DEL_{{ORDER}}  (type: DELIVERY)    — dla faktu DELIVERED",
        "  - STMT_{{ORDER}} (type: STATEMENT)   — dla WITHDRAWAL_STATEMENT_SUBMITTED",
        "  - RET_{{ORDER}}  (type: RETURN_SHIPMENT) — dla RETURNED",
        "  - PAY_{{ORDER}}  (type: PAYMENT)     — dla PAYMENT_MADE / REFUND_ISSUED",
        "",
        "## FAKTY",
        "Wyciągaj tylko fakty, które są wprost wspomniane w tekście.",
        "Pomiń fakt jeśli nie masz pewności. role = nazwa roli pozycyjnej.",
        "Dla ról opcjonalnych (DATE, COUPON, COMPLAINT, ACCOUNT, CHARGEBACK):",
        "  pomiń argument jeśli nie możesz ustalić encji — NIE blokuj całego faktu.",
        "",
    ]

    # Predykaty z rolami
    lines.append("Dostępne predykaty (UPPERCASE) i ich role pozycyjne (? = opcjonalna):")
    for pred, roles in predicate_positions.items():
        pred_upper = pred.upper()
        role_str = ", ".join(roles)
        lines.append(f"  - {pred_upper}({role_str})")
    lines.append("")

    lines += [
        "Rola REASON przyjmuje literal_value (nie entity_id):",
        "  - ORDER_CANCELLED.REASON → literal_value: NONPAYMENT | ACCOUNT_BLOCKED | OTHER",
        "  - ACCOUNT_BLOCKED.REASON → literal_value: FRAUD_SUSPECT | PREV_NONPAYMENT",
        "  - CHARGEBACK_RESOLVED.RESULT → WON_BY_CUSTOMER lub WON_BY_STORE",
        "",
        "## KLASTRY (właściwości encji)",
        "Dla każdej encji rozpoznaj jej właściwości klastrowe.",
        "Podaj entity_id encji właściciela i nazwę wartości z listy dozwolonych.",
        "Każda encja może mieć co najwyżej JEDNĄ wartość dla danego klastra.",
        "",
    ]

    # Klastry z domenami
    for schema in schemas:
        domain_str = " | ".join(schema.domain)
        lines.append(
            f"  - {schema.name} (entity_type: {schema.entity_type}): {domain_str}"
        )

    lines += [
        "",
        "## WAŻNE ZASADY",
        "1. Jeden fakt = jedno zdarzenie. Nie duplikuj faktów.",
        "2. entity_id MUSI być dokładnie takim IDem jak w tekście (O100, nie 'zamówienie').",
        "3. Daty polskie ('10 marca') → D_2026-MM-DD (rok bieżący: 2026).",
        "4. Literały (REASON, RESULT) → literal_value, nie entity_id.",
        "5. Klaster: maksymalnie 1 wartość na parę (entity_id, cluster_name).",
        "6. Jeśli encja nie jest wymieniona wprost — użyj CUST1 lub STORE1.",
        "7. Dla każdego faktu podaj span_start/span_end (pozycje bajtów) i span_text"
        " (dosłowny cytat z tekstu potwierdzający fakt).",
        "8. Dla każdej encji i klastra podaj span_text: dosłowny cytat fragmentu tekstu,"
        " w którym ta encja/właściwość została zidentyfikowana.",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# JSON Schema dla Gemini structured output
# ---------------------------------------------------------------------------

def build_response_schema() -> dict[str, Any]:
    """
    JSON Schema (Gemini format) dla structured output.
    Gemini używa wielkich liter dla typów: STRING, INTEGER, OBJECT, ARRAY.
    """
    role_arg_schema = {
        "type": "OBJECT",
        "properties": {
            "role":          {"type": "STRING"},
            "entity_id":     {"type": "STRING"},
            "literal_value": {"type": "STRING"},
        },
        "required": ["role"],
    }

    entity_schema = {
        "type": "OBJECT",
        "properties": {
            "entity_id":      {"type": "STRING"},
            "type":           {"type": "STRING"},
            "canonical_name": {"type": "STRING"},
            "span_text":      {"type": "STRING"},
        },
        "required": ["entity_id", "type", "canonical_name"],
    }

    fact_schema = {
        "type": "OBJECT",
        "properties": {
            "predicate":  {"type": "STRING"},
            "args":       {"type": "ARRAY", "items": role_arg_schema},
            "span_start": {"type": "INTEGER"},
            "span_end":   {"type": "INTEGER"},
            "span_text":  {"type": "STRING"},
        },
        "required": ["predicate", "args"],
    }

    cluster_state_schema = {
        "type": "OBJECT",
        "properties": {
            "entity_id":    {"type": "STRING"},
            "cluster_name": {"type": "STRING"},
            "value":        {"type": "STRING"},
            "span_text":    {"type": "STRING"},
        },
        "required": ["entity_id", "cluster_name", "value"],
    }

    return {
        "type": "OBJECT",
        "properties": {
            "entities":       {"type": "ARRAY", "items": entity_schema},
            "facts":          {"type": "ARRAY", "items": fact_schema},
            "cluster_states": {"type": "ARRAY", "items": cluster_state_schema},
        },
        "required": ["entities", "facts", "cluster_states"],
    }


# ---------------------------------------------------------------------------
# Parser odpowiedzi LLM → ExtractionResult
# ---------------------------------------------------------------------------

def parse_llm_response(
    data: dict[str, Any],
    source_id: str,
    year: int,
    schemas: list[ClusterSchema],
    text: str | None = None,
) -> ExtractionResult:
    """
    Parsuje surowy dict z Gemini (po json.loads) → ExtractionResult.

    Zawsze dodaje CUST1 i STORE1 jeśli nie zostały podane przez LLM.
    Encje syntetyczne (DEL_*, STMT_*, RET_*, PAY_*) są tworzone automatycznie
    na podstawie entity_id w args faktów.
    """
    schema_map = {s.name: s for s in schemas}

    # ── Encje ────────────────────────────────────────────────────────────────
    entities: list[Entity] = []
    seen_ids: set[str] = set()

    def _add_entity(
        eid: str, etype: str, name: str, marker: str, span_text: str | None = None
    ) -> None:
        if eid not in seen_ids:
            seen_ids.add(eid)
            prov = (
                [ProvenanceItem(source_id=source_id, span=Span(text=span_text), extractor="LLMExtractor")]
                if span_text else []
            )
            entities.append(Entity(
                entity_id=eid,
                type=etype,
                canonical_name=name,
                created_at=_stable_timestamp(year, source_id, "entity", etype, eid, marker),
                provenance=prov,
            ))

    # Zawsze: CUST1 + STORE1 (bez spanu — nie pochodzą z tekstu)
    _add_entity(IMPLICIT_CUSTOMER, "CUSTOMER", "Klient", "implicit_customer")
    _add_entity(IMPLICIT_STORE, "STORE", "Sklep", "implicit_store")

    for e in data.get("entities", []):
        eid = str(e.get("entity_id", "")).strip()
        etype = str(e.get("type", "")).strip().upper()
        name = str(e.get("canonical_name", eid)).strip()
        span_text = str(e["span_text"]).strip() if e.get("span_text") else None
        if eid and etype:
            _add_entity(eid, etype, name, "llm", span_text=span_text)

    # ── Fakty ─────────────────────────────────────────────────────────────────
    facts: list[Fact] = []
    seen_fact_keys: set[tuple[str, ...]] = set()

    for f in data.get("facts", []):
        predicate = str(f.get("predicate", "")).strip().upper()
        if not predicate:
            continue

        args_raw = f.get("args", [])
        args: list[RoleArg] = []
        for a in args_raw:
            role = str(a.get("role", "")).strip().upper()
            entity_id = a.get("entity_id") or None
            literal_value = a.get("literal_value") or None
            if not role:
                continue
            # Normalizuj entity_id: usuń białe znaki, upewnij się że niepuste
            if entity_id is not None:
                entity_id = str(entity_id).strip() or None
            if literal_value is not None:
                literal_value = str(literal_value).strip() or None
            if entity_id is None and literal_value is None:
                continue
            args.append(RoleArg(
                role=role,
                entity_id=entity_id,
                literal_value=literal_value,
            ))

        if not args:
            continue

        # Deduplication
        fact_key = (predicate,) + tuple(
            sorted(f"{a.role}:{a.entity_id or a.literal_value or ''}" for a in args)
        )
        if fact_key in seen_fact_keys:
            continue
        seen_fact_keys.add(fact_key)

        span_start = int(f["span_start"]) if f.get("span_start") is not None else None
        span_end   = int(f["span_end"])   if f.get("span_end")   is not None else None
        llm_text   = str(f["span_text"]).strip() if f.get("span_text") else None
        computed_text = (
            llm_text
            or (text[span_start:span_end] if text and span_start is not None and span_end is not None else None)
        )
        span = Span(start=span_start, end=span_end, text=computed_text)

        fact_seed = f"{source_id}|{predicate}|{span.start or 0}:{span.end or 0}|{'|'.join(fact_key[1:])}"
        fact = Fact(
            fact_id=str(uuid.uuid5(uuid.NAMESPACE_URL, fact_seed)),
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

        # Syntetyczne encje z args
        for a in args:
            if a.entity_id and a.entity_id not in seen_ids:
                eid = a.entity_id
                etype = _infer_synthetic_type(eid)
                if etype:
                    _add_entity(eid, etype, eid, "synthetic")

    # ── Klastry ───────────────────────────────────────────────────────────────
    cluster_states: list[ClusterStateRow] = []
    seen_clusters: set[tuple[str, str]] = set()

    for cs in data.get("cluster_states", []):
        eid = str(cs.get("entity_id", "")).strip()
        cname = str(cs.get("cluster_name", "")).strip()
        value = str(cs.get("value", "")).strip().upper()
        if not (eid and cname and value):
            continue

        schema = schema_map.get(cname)
        if schema is None:
            continue  # nieznany klaster — pomiń

        if value not in schema.domain:
            continue  # nieznana wartość — pomiń

        key = (eid, cname)
        if key in seen_clusters:
            continue  # duplikat — weź pierwszą wartość
        seen_clusters.add(key)

        span_text = str(cs["span_text"]).strip() if cs.get("span_text") else None

        # Hard clamp: +M dla dopasowanej wartości, -M dla pozostałych
        logits = [
            _CLAMP_M if d == value else -_CLAMP_M
            for d in schema.domain
        ]
        cluster_states.append(ClusterStateRow(
            entity_id=eid,
            cluster_name=cname,
            logits=logits,
            is_clamped=True,
            clamp_hard=True,
            clamp_source="text",
            source_span=Span(text=span_text) if span_text else None,
        ))

    return ExtractionResult(
        entities=entities,
        facts=facts,
        cluster_states=cluster_states,
        source_id=source_id,
    )


# ---------------------------------------------------------------------------
# Correction prompt (feedback loop po odrzuceniu przez SV)
# ---------------------------------------------------------------------------

def build_correction_prompt(
    original_text: str,
    conflicts: list[str],
) -> str:
    """
    Buduje prompt korekcyjny gdy SV wykryło sprzeczności w poprzedniej ekstrakcji.

    conflicts: lista opisów konfliktów po polsku, np.:
        ["CLUSTER customer_type: encja CUST1 ma 2 wartości: CONSUMER, BUSINESS",
         "FAKT ORDER_PLACED(CUST1, O100): duplikat z różnymi datami"]
    """
    conflict_block = "\n".join(f"  - {c}" for c in conflicts)
    return (
        "Poprzednia ekstrakcja zawierała sprzeczności logiczne:\n"
        f"{conflict_block}\n\n"
        "Popraw ekstrakcję eliminując powyższe sprzeczności. "
        "Jeśli nie możesz rozstrzygnąć — pomiń wątpliwy fakt lub wartość klastra.\n\n"
        f"Oryginalny tekst:\n{original_text}"
    )


# ---------------------------------------------------------------------------
# Pomocnicze
# ---------------------------------------------------------------------------

def _stable_timestamp(year: int, *parts: object) -> datetime:
    seed = "|".join(str(p) for p in parts)
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    seconds = int(digest[:10], 16) % (365 * 24 * 60 * 60)
    base = datetime(year, 1, 1, 0, 0, 0)
    return base + timedelta(seconds=seconds)


def _infer_synthetic_type(entity_id: str) -> str | None:
    for prefix, etype in _SYNTHETIC_PREFIXES.items():
        if entity_id.startswith(prefix):
            return etype
    return None
