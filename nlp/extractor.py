"""
TextExtractor — reguło-bazowy ekstraktor faktów z tekstu polskiego.

Nie wymaga modeli spaCy — używa wyłącznie regex.
Pokrywa 18 case'ów testowych z docs/test use case.md.

Przepływ:
  1. Znajdź encje (O[0-9]+, A[0-9]+, CB[0-9]+, C[0-9]{2,}, P[0-9]+) i kody kuponów.
  2. Znajdź daty ("10 marca" → "D_2026-03-10").
  3. Dla każdego FactRule.trigger → FactRule.role_sources → Fact (observed).
  4. Dla każdego ClusterRule.pattern → ClusterStateRow (is_clamped=True).
"""
from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timedelta
from typing import Any

from data_model.common import ProvenanceItem, RoleArg, Span, TruthDistribution
from data_model.entity import Entity
from data_model.fact import Fact, FactProvenance, FactSource, FactStatus
from nn.graph_builder import ClusterSchema, ClusterStateRow

from .patterns import (
    CLUSTER_RULES,
    COUPON_CODE_RE,
    COUPON_CONTEXT_RE,
    DATE_RE,
    ENTITY_ID_PATTERNS,
    FACT_RULES,
    NON_COUPON_WORDS,
    PL_MONTHS,
    ClusterRule,
    FactRule,
)
from .result import ExtractionResult

# ---------------------------------------------------------------------------
# Stałe
# ---------------------------------------------------------------------------

IMPLICIT_CUSTOMER = "CUST1"
IMPLICIT_STORE = "STORE1"

_TRUTH_OBSERVED = TruthDistribution(domain=["T", "F", "U"], value="T", confidence=1.0)

# Mapowanie specyfikatora AUTO_ → prefiks entity_id
_AUTO_PREFIX: dict[str, str] = {
    "AUTO_DELIVERY_ORDER":   "DEL_",
    "AUTO_STATEMENT_ORDER":  "STMT_",
    "AUTO_RETURN_ORDER":     "RET_",
    "AUTO_PAYMENT_ORDER":    "PAY_",
}

# Role opcjonalne — brak nie anuluje faktu
_OPTIONAL_SOURCES = frozenset({
    "DATE", "COUPON", "COMPLAINT", "ACCOUNT", "CHARGEBACK",
    "AUTO_DELIVERY_ORDER", "AUTO_STATEMENT_ORDER",
    "AUTO_RETURN_ORDER", "AUTO_PAYMENT_ORDER",
})


# ---------------------------------------------------------------------------
# TextExtractor
# ---------------------------------------------------------------------------

class TextExtractor:
    """
    Reguło-bazowy ekstraktor dla języka polskiego.

    Parametry:
        cluster_schemas: lista schematów klastrów (z DB / seed_ontology)
        year:            rok domyślny dla dat (np. "1 lutego" → 2026-02-01)
    """

    def __init__(
        self,
        cluster_schemas: list[ClusterSchema],
        year: int = 2026,
    ) -> None:
        self.schemas: dict[str, ClusterSchema] = {s.name: s for s in cluster_schemas}
        self.year = year

    # ------------------------------------------------------------------
    # Publiczne API
    # ------------------------------------------------------------------

    def extract(self, text: str, source_id: str = "text") -> ExtractionResult:
        """
        Ekstrahuje encje, fakty i stany klastrów z tekstu.

        Zwraca ExtractionResult z:
          - entities:       encje explicit + implicit (CUST1, STORE1) + DATE
          - facts:          reifikowane fakty (status=observed)
          - cluster_states: stany klastrów z clamp_source='text'
        """
        # 1. Pozycje encji: {entity_type → [(entity_id, char_pos), ...]}
        entity_pos = self._find_entity_positions(text)

        # 2. Pozycje dat: [(date_id, char_pos), ...]
        date_pos = self._find_date_positions(text)

        # 3. Kody kuponów
        entity_pos["COUPON"] = self._find_coupon_positions(text)

        # 4. Encje
        entities = self._build_entities(entity_pos, date_pos, source_id)

        # 5. Fakty
        facts = self._extract_facts(text, entity_pos, date_pos, source_id)

        # 6. Klastry
        cluster_states = self._extract_cluster_states(text, entity_pos)

        # Dodaj syntetyczne encje wygenerowane przez fakty i klastry
        entities = self._add_synthetic_entities(entities, facts, cluster_states, source_id)

        return ExtractionResult(
            entities=entities,
            facts=facts,
            cluster_states=cluster_states,
            source_id=source_id,
        )

    # ------------------------------------------------------------------
    # Znajdowanie pozycji encji i dat
    # ------------------------------------------------------------------

    def _find_entity_positions(
        self, text: str
    ) -> dict[str, list[tuple[str, int]]]:
        result: dict[str, list[tuple[str, int]]] = {}
        for entity_type, pattern in ENTITY_ID_PATTERNS:
            positions = [(m.group(), m.start()) for m in pattern.finditer(text)]
            if positions:
                result[entity_type] = positions
        return result

    def _find_date_positions(self, text: str) -> list[tuple[str, int]]:
        dates: list[tuple[str, int]] = []
        for m in DATE_RE.finditer(text):
            day = int(m.group(1))
            month_key = m.group(2).lower()
            month = PL_MONTHS.get(month_key, 0)
            if month == 0:
                continue
            date_id = f"D_{self.year}-{month:02d}-{day:02d}"
            dates.append((date_id, m.start()))
        return dates

    def _find_coupon_positions(self, text: str) -> list[tuple[str, int]]:
        """
        Krok 1: Znajdź kody kuponów w kontekście słów 'kupon'/'kod'.
        Krok 2: Wyszukaj wszystkie wystąpienia tych kodów w tekście.
        """
        # Kody zidentyfikowane przez kontekst
        known_codes: set[str] = set()
        for m in COUPON_CONTEXT_RE.finditer(text):
            raw = m.group(1)
            for part in raw.split():
                code = part.strip().upper()
                if code and code not in NON_COUPON_WORDS:
                    known_codes.add(code)

        if not known_codes:
            return []

        # Wszystkie wystąpienia tych kodów w tekście
        positions: list[tuple[str, int]] = []
        for m in COUPON_CODE_RE.finditer(text):
            code = m.group(1).upper()
            if code in known_codes:
                positions.append((code, m.start()))

        return positions

    # ------------------------------------------------------------------
    # Budowanie Entity obiektów
    # ------------------------------------------------------------------

    def _build_entities(
        self,
        entity_pos: dict[str, list[tuple[str, int]]],
        date_pos: list[tuple[str, int]],
        source_id: str,
    ) -> list[Entity]:
        seen: set[str] = set()
        entities: list[Entity] = []

        def add(eid: str, etype: str, name: str, marker: str) -> None:
            if eid not in seen:
                seen.add(eid)
                entities.append(Entity(
                    entity_id=eid,
                    type=etype,
                    canonical_name=name,
                    created_at=self._stable_timestamp(
                        source_id,
                        "entity",
                        etype,
                        eid,
                        marker,
                    ),
                ))

        # Zawsze: implicit customer + store
        add(IMPLICIT_CUSTOMER, "CUSTOMER", "Klient", "implicit_customer")
        add(IMPLICIT_STORE, "STORE", "Sklep", "implicit_store")

        # Explicit entities z tekstu
        for etype, positions in entity_pos.items():
            seen_local: set[str] = set()
            for eid, pos in positions:
                if eid not in seen_local:
                    seen_local.add(eid)
                    add(eid, etype, f"{etype} {eid}", f"pos:{pos}")

        # DATE entities
        seen_dates: set[str] = set()
        for date_id, pos in date_pos:
            if date_id not in seen_dates:
                seen_dates.add(date_id)
                date_str = date_id[2:]  # usuń "D_"
                add(date_id, "DATE", date_str, f"pos:{pos}")

        return entities

    def _add_synthetic_entities(
        self,
        entities: list[Entity],
        facts: list[Fact],
        cluster_states: list,
        source_id: str,
    ) -> list[Entity]:
        """
        Dodaje encje syntetyczne (DEL_*, STMT_*, RET_*, PAY_*, PROD_*)
        wygenerowane przez fakty i cluster_states.
        """
        _PREFIXES = ("DEL_", "STMT_", "RET_", "PAY_", "PROD_")
        seen = {e.entity_id for e in entities}

        def _add(eid: str, ref_id: str) -> None:
            if eid in seen or not any(eid.startswith(p) for p in _PREFIXES):
                return
            etype = (
                "DELIVERY" if eid.startswith("DEL_") else
                "STATEMENT" if eid.startswith("STMT_") else
                "RETURN_SHIPMENT" if eid.startswith("RET_") else
                "PAYMENT" if eid.startswith("PAY_") else
                "PRODUCT"
            )
            entities.append(Entity(
                entity_id=eid,
                type=etype,
                canonical_name=eid,
                created_at=self._stable_timestamp(
                    source_id, "synthetic_entity", etype, eid, ref_id,
                ),
            ))
            seen.add(eid)

        for fact in facts:
            for arg in fact.args:
                if arg.entity_id:
                    _add(arg.entity_id, fact.fact_id)

        for cs in cluster_states:
            _add(cs.entity_id, cs.cluster_name)

        return entities

    # ------------------------------------------------------------------
    # Ekstrakcja faktów
    # ------------------------------------------------------------------

    def _extract_facts(
        self,
        text: str,
        entity_pos: dict[str, list[tuple[str, int]]],
        date_pos: list[tuple[str, int]],
        source_id: str,
    ) -> list[Fact]:
        facts: list[Fact] = []
        dedup: set[tuple[str, ...]] = set()

        for rule in FACT_RULES:
            for m in rule.trigger.finditer(text):
                trigger_pos = m.start()
                span = Span(start=m.start(), end=m.end())

                if rule.predicate == "COUPON_APPLIED":
                    # Specjalny przypadek: jedna fact per kupon w oknie
                    new_facts = self._resolve_coupon_applied(
                        rule, trigger_pos, entity_pos, date_pos, source_id, span, dedup, text
                    )
                    facts.extend(new_facts)
                else:
                    args = self._resolve_roles(rule, trigger_pos, entity_pos, date_pos, text)
                    if args is None:
                        continue
                    fact = self._make_fact(rule.predicate, args, source_id, span)
                    key = self._fact_key(fact)
                    if key not in dedup:
                        dedup.add(key)
                        facts.append(fact)

        return facts

    def _resolve_coupon_applied(
        self,
        rule: FactRule,
        pos: int,
        entity_pos: dict[str, list[tuple[str, int]]],
        date_pos: list[tuple[str, int]],
        source_id: str,
        span: Span,
        dedup: set[tuple[str, ...]],
        text: str,
    ) -> list[Fact]:
        """Generuje jedną COUPON_APPLIED per kupon w oknie ±500 znaków."""
        coupon_all = entity_pos.get("COUPON", [])
        nearby = [eid for eid, cp in coupon_all if abs(cp - pos) <= 500]

        if not nearby:
            nearby_eid = self._closest(pos, coupon_all)
            if nearby_eid:
                nearby = [nearby_eid]

        result: list[Fact] = []
        for coupon_id in dict.fromkeys(nearby):   # unikalne, zachowując kolejność
            modified = dict(entity_pos)
            modified["COUPON"] = [(coupon_id, pos)]
            args = self._resolve_roles(rule, pos, modified, date_pos, text)
            if args is None:
                continue
            fact = self._make_fact(rule.predicate, args, source_id, span)
            key = self._fact_key(fact)
            if key not in dedup:
                dedup.add(key)
                result.append(fact)

        return result

    def _resolve_roles(
        self,
        rule: FactRule,
        pos: int,
        entity_pos: dict[str, list[tuple[str, int]]],
        date_pos: list[tuple[str, int]],
        text: str,
    ) -> list[RoleArg] | None:
        """
        Dla każdego (role, source_spec) zwraca listę RoleArg.
        Zwraca None jeśli brakuje wymaganego argumentu.
        """
        args: list[RoleArg] = []
        for role, source in rule.role_sources:
            resolved = self._resolve_one(source, pos, entity_pos, date_pos, text)
            if resolved is None:
                if source in _OPTIONAL_SOURCES:
                    continue    # pole opcjonalne — pomiń
                return None     # pole wymagane — fakt niemożliwy
            args.append(RoleArg(role=role, **resolved))

        return args if args else None

    def _resolve_one(
        self,
        source: str,
        pos: int,
        entity_pos: dict[str, list[tuple[str, int]]],
        date_pos: list[tuple[str, int]],
        text: str,
    ) -> dict[str, Any] | None:
        if source == "IMPLICIT_CUSTOMER":
            return {"entity_id": IMPLICIT_CUSTOMER}
        if source == "IMPLICIT_STORE":
            return {"entity_id": IMPLICIT_STORE}
        if source.startswith("literal:"):
            return {"literal_value": source[8:]}
        if source == "DATE":
            # Preferuj datę z tego samego zdania co trigger
            eid = self._closest_in_sentence(pos, date_pos, text)
            return {"entity_id": eid} if eid else None
        if source in ("ORDER", "ACCOUNT", "CHARGEBACK", "COMPLAINT", "COUPON", "PRODUCT"):
            eid = self._closest(pos, entity_pos.get(source, []))
            return {"entity_id": eid} if eid else None
        if source in _AUTO_PREFIX:
            prefix = _AUTO_PREFIX[source]
            order_id = self._closest(pos, entity_pos.get("ORDER", []))
            if order_id is None:
                return None
            return {"entity_id": f"{prefix}{order_id}"}
        return None

    # ------------------------------------------------------------------
    # Ekstrakcja klastrów
    # ------------------------------------------------------------------

    def _extract_cluster_states(
        self,
        text: str,
        entity_pos: dict[str, list[tuple[str, int]]],
    ) -> list[ClusterStateRow]:
        states: list[ClusterStateRow] = []
        seen: set[tuple[str, str, str]] = set()

        for rule in CLUSTER_RULES:
            schema = self.schemas.get(rule.cluster_name)
            if schema is None or rule.value not in schema.domain:
                continue

            for m in rule.pattern.finditer(text):
                eid = self._resolve_cluster_entity(rule, m.start(), entity_pos)
                if eid is None:
                    continue

                key = (eid, rule.cluster_name, rule.value)
                if key in seen:
                    continue
                seen.add(key)

                # Hard clamp: clamp_value=M, rest=-M
                M = 10.0
                logits = [-M] * schema.dim
                logits[schema.domain.index(rule.value)] = M

                states.append(ClusterStateRow(
                    entity_id=eid,
                    cluster_name=rule.cluster_name,
                    logits=logits,
                    is_clamped=True,
                    clamp_hard=True,
                    clamp_source="text",
                ))

        return states

    def _resolve_cluster_entity(
        self,
        rule: ClusterRule,
        pos: int,
        entity_pos: dict[str, list[tuple[str, int]]],
    ) -> str | None:
        etype = rule.entity_type
        if etype == "CUSTOMER":
            return IMPLICIT_CUSTOMER
        if etype == "PRODUCT":
            # Preferuj explicit PRODUCT, fallback na ORDER (pragmatyczne uproszczenie)
            eid = self._closest(pos, entity_pos.get("PRODUCT", []))
            if eid:
                return eid
            order_id = self._closest(pos, entity_pos.get("ORDER", []))
            return f"PROD_{order_id}" if order_id else None
        return self._closest(pos, entity_pos.get(etype, []))

    # ------------------------------------------------------------------
    # Pomocnicze
    # ------------------------------------------------------------------

    @staticmethod
    def _closest(pos: int, positions: list[tuple[str, int]]) -> str | None:
        if not positions:
            return None
        return min(positions, key=lambda x: abs(x[1] - pos))[0]

    @staticmethod
    def _closest_in_sentence(
        pos: int,
        positions: list[tuple[str, int]],
        text: str,
    ) -> str | None:
        """
        Preferuje encję w tej samej zdaniu co trigger (pos).
        Zdanie = fragment między poprzednim a następnym '.' / '?' / '!'.
        Fallback: globalne _closest.
        """
        if not positions:
            return None

        # Granice zdania zawierającego pos
        sent_start = max(text.rfind(".", 0, pos), text.rfind("\n", 0, pos)) + 1
        sent_end_dot = text.find(".", pos)
        sent_end = sent_end_dot if sent_end_dot != -1 else len(text)

        in_sentence = [
            (eid, p) for eid, p in positions
            if sent_start <= p <= sent_end
        ]
        if in_sentence:
            return min(in_sentence, key=lambda x: abs(x[1] - pos))[0]

        # Fallback: globalne minimum
        return min(positions, key=lambda x: abs(x[1] - pos))[0]

    def _make_fact(
        self,
        predicate: str,
        args: list[RoleArg],
        source_id: str,
        span: Span,
    ) -> Fact:
        args_key = "|".join(
            sorted(f"{a.role}:{a.entity_id or a.literal_value or ''}" for a in args)
        )
        fact_seed = f"{source_id}|{predicate}|{span.start}:{span.end}|{args_key}"
        return Fact(
            fact_id=str(uuid.uuid5(uuid.NAMESPACE_URL, fact_seed)),
            predicate=predicate,
            arity=len(args),
            args=args,
            truth=_TRUTH_OBSERVED,
            status=FactStatus.observed,
            source=FactSource(
                source_id=source_id,
                spans=[span],
                extractor="TextExtractor",
                confidence=1.0,
            ),
        )

    def _stable_timestamp(self, *parts: object) -> datetime:
        """
        Deterministyczny timestamp pochodny od source_id / pozycji / typu.
        """
        seed = "|".join(str(p) for p in parts)
        digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
        seconds = int(digest[:10], 16) % (365 * 24 * 60 * 60)
        base = datetime(self.year, 1, 1, 0, 0, 0)
        return base + timedelta(seconds=seconds)

    @staticmethod
    def _fact_key(fact: Fact) -> tuple[str, ...]:
        return (fact.predicate,) + tuple(
            sorted(
                (a.role, a.entity_id or a.literal_value or "")
                for a in fact.args
            )
        )
