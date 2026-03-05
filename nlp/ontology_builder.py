"""
ontology_builder.py — Budowanie promptów i parsowanie odpowiedzi dla generowania
ontologii z tekstu regulaminu przez LLM (Gemini structured output).

Eksportuje:
    build_ontology_prompt(regulatory_text) → str
    build_ontology_schema() → dict
    parse_ontology_response(data, source_id) → OntologyResult
    OntologyResult, EntityTypeSpec, PredicateSpec, PredicateRoleSpec,
    ClusterSpec, RuleSpec
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Dataclassy
# ---------------------------------------------------------------------------

@dataclass
class EntityTypeSpec:
    name: str
    description: str
    source_span_text: str | None = None


@dataclass
class PredicateRoleSpec:
    position: int
    role: str
    entity_type: str | None  # None = literał (REASON, RESULT, AMOUNT…)


@dataclass
class PredicateSpec:
    name: str
    description: str
    roles: list[PredicateRoleSpec] = field(default_factory=list)
    source_span_text: str | None = None


@dataclass
class ClusterSpec:
    name: str
    entity_type: str
    domain: list[str] = field(default_factory=list)
    description: str = ""
    source_span_text: str | None = None


@dataclass
class RuleSpec:
    rule_id: str
    module: str
    clingo_text: str
    stratum: int = 0
    source_span_text: str | None = None


@dataclass
class OntologyResult:
    entity_types: list[EntityTypeSpec] = field(default_factory=list)
    predicates: list[PredicateSpec] = field(default_factory=list)
    clusters: list[ClusterSpec] = field(default_factory=list)
    rules: list[RuleSpec] = field(default_factory=list)
    source_id: str = "regulation"

    def summary(self) -> str:
        return (
            f"entity_types={len(self.entity_types)} "
            f"predicates={len(self.predicates)} "
            f"clusters={len(self.clusters)} "
            f"rules={len(self.rules)}"
        )


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

_FEW_SHOT_RULES = """\
Przykłady par (przepis → clingo_text):

  Przepis: "Umowa zostaje zawarta z chwilą potwierdzenia przyjęcia Zamówienia."
  clingo_text: contract_formed(O) :- order_accepted(_,O,_).
  stratum: 0

  Przepis: "Konsument może odstąpić w ciągu 14 dni od doręczenia, chyba że towar jest cyfrowy i pobieranie zostało rozpoczęte."
  clingo_text: can_withdraw(C,O) :- customer_type(C,consumer), delivered(O,_,DD), withdrawal_statement_submitted(C,O,_,DS), within_14_days(DS,DD), not ab_withdraw(C,O).
  stratum: 2

  Przepis: "Wyjątek: treść cyfrowa + zgoda + pobieranie."
  clingo_text: ab_withdraw(C,O) :- product_type(P,digital), order_contains(O,P), digital_consent(O,yes), download_started_flag(O,yes).
  stratum: 1

Konwencje Clingo:
- Zmienne: wielkie litery (O, C, D) lub podkreślnik _ (wildcard)
- Stałe: małe litery (consumer, yes, no, card)
- NAF: not predykat(...)
- Stratum 0 = fakty i reguły bez NAF; 1 = reguły z NAF nad stratum 0; 2 = NAF nad stratum 1
- rule_id: "{moduł}.{nazwa_predykatu_głowy}" (snake_case)
"""


def build_ontology_prompt(regulatory_text: str) -> str:
    return "\n".join([
        "Jesteś ekspertem od modelowania ontologii dla systemów prawnych e-commerce.",
        "Na podstawie poniższego regulaminu wyciągnij pełną ontologię.",
        "",
        "## TYPY ENCJI",
        "Rozpoznaj wszystkie typy obiektów domenowych (rzeczowniki pełniące role).",
        "Przykłady: ORDER, CUSTOMER, PRODUCT, PAYMENT, DELIVERY, COUPON, ACCOUNT, CHARGEBACK, COMPLAINT, DATE.",
        "Dla każdego podaj name (UPPER_SNAKE_CASE), description i source_span_text.",
        "",
        "## PREDYKATY N-ARNE (zdarzenia)",
        "Każde zdarzenie opisane w regulaminie = jeden predykat.",
        "Przykłady: ORDER_PLACED, PAYMENT_MADE, DELIVERED, COMPLAINT_SUBMITTED.",
        "Role pozycyjne: CUSTOMER, ORDER, DATE, STORE, itp.",
        "Literały (wartości inline, nie encje): REASON, RESULT, AMOUNT, PAYMENT_METHOD → entity_type: null.",
        "Dla każdego podaj name, description, roles (position, role, entity_type) i source_span_text.",
        "",
        "## KLASTRY UNARNE (właściwości dyskretne)",
        "Enumy/kategorie przypisane do encji — każdy klaster = zmienna dyskretna z softmax.",
        "Przykłady: customer_type(CUSTOMER) ∈ {CONSUMER, BUSINESS}",
        "           payment_method(ORDER) ∈ {CARD, TRANSFER, BLIK, COD}",
        "Dla każdego podaj name (snake_case), entity_type, domain (lista wartości UPPER_CASE),",
        "description i source_span_text (cytat wyliczający wartości).",
        "",
        "## REGUŁY HORN + NAF",
        _FEW_SHOT_RULES,
        "Dla każdej reguły podaj rule_id, module (snake_case), clingo_text, stratum i source_span_text.",
        "",
        "## WAŻNE ZASADY",
        "1. Dla KAŻDEGO elementu podaj source_span_text: dosłowny, minimalny cytat z regulaminu",
        "   (jedno zdanie lub klauzula), który uzasadnia ten element.",
        "2. Nazwy predykatów i entity_type: UPPER_SNAKE_CASE.",
        "3. Nazwy klastrów: lower_snake_case.",
        "4. Nie dodawaj elementów spoza tekstu regulaminu.",
        "5. Jeśli reguła ma wyjątek (NAF), wyciągnij też regułę definiującą wyjątek.",
        "",
        "## REGULAMIN",
        regulatory_text,
    ])


# ---------------------------------------------------------------------------
# JSON Schema dla Gemini structured output
# ---------------------------------------------------------------------------

def build_ontology_schema() -> dict[str, Any]:
    role_schema = {
        "type": "OBJECT",
        "properties": {
            "position":    {"type": "INTEGER"},
            "role":        {"type": "STRING"},
            "entity_type": {"type": "STRING"},
        },
        "required": ["position", "role"],
    }

    entity_type_schema = {
        "type": "OBJECT",
        "properties": {
            "name":             {"type": "STRING"},
            "description":      {"type": "STRING"},
            "source_span_text": {"type": "STRING"},
        },
        "required": ["name", "description"],
    }

    predicate_schema = {
        "type": "OBJECT",
        "properties": {
            "name":             {"type": "STRING"},
            "description":      {"type": "STRING"},
            "roles":            {"type": "ARRAY", "items": role_schema},
            "source_span_text": {"type": "STRING"},
        },
        "required": ["name", "description", "roles"],
    }

    cluster_schema = {
        "type": "OBJECT",
        "properties": {
            "name":             {"type": "STRING"},
            "entity_type":      {"type": "STRING"},
            "domain":           {"type": "ARRAY", "items": {"type": "STRING"}},
            "description":      {"type": "STRING"},
            "source_span_text": {"type": "STRING"},
        },
        "required": ["name", "entity_type", "domain"],
    }

    rule_schema = {
        "type": "OBJECT",
        "properties": {
            "rule_id":          {"type": "STRING"},
            "module":           {"type": "STRING"},
            "clingo_text":      {"type": "STRING"},
            "stratum":          {"type": "INTEGER"},
            "source_span_text": {"type": "STRING"},
        },
        "required": ["rule_id", "module", "clingo_text"],
    }

    return {
        "type": "OBJECT",
        "properties": {
            "entity_types": {"type": "ARRAY", "items": entity_type_schema},
            "predicates":   {"type": "ARRAY", "items": predicate_schema},
            "clusters":     {"type": "ARRAY", "items": cluster_schema},
            "rules":        {"type": "ARRAY", "items": rule_schema},
        },
        "required": ["entity_types", "predicates", "clusters", "rules"],
    }


# ---------------------------------------------------------------------------
# Parser odpowiedzi LLM
# ---------------------------------------------------------------------------

def parse_ontology_response(data: dict[str, Any], source_id: str) -> OntologyResult:
    """
    Parsuje surowy dict z Gemini → OntologyResult.
    Toleruje brakujące pola opcjonalne.
    """
    entity_types: list[EntityTypeSpec] = []
    seen_et: set[str] = set()
    for e in data.get("entity_types", []):
        name = str(e.get("name", "")).strip().upper()
        if not name or name in seen_et:
            continue
        seen_et.add(name)
        entity_types.append(EntityTypeSpec(
            name=name,
            description=str(e.get("description", "")).strip(),
            source_span_text=_opt_str(e, "source_span_text"),
        ))

    predicates: list[PredicateSpec] = []
    seen_pred: set[str] = set()
    for p in data.get("predicates", []):
        name = str(p.get("name", "")).strip().upper()
        if not name or name in seen_pred:
            continue
        seen_pred.add(name)
        roles = [
            PredicateRoleSpec(
                position=int(r.get("position", i)),
                role=str(r.get("role", f"ARG{i}")).strip().upper(),
                entity_type=str(r["entity_type"]).strip().upper() if r.get("entity_type") else None,
            )
            for i, r in enumerate(p.get("roles", []))
        ]
        predicates.append(PredicateSpec(
            name=name,
            description=str(p.get("description", "")).strip(),
            roles=roles,
            source_span_text=_opt_str(p, "source_span_text"),
        ))

    clusters: list[ClusterSpec] = []
    seen_cl: set[str] = set()
    for c in data.get("clusters", []):
        name = str(c.get("name", "")).strip().lower()
        entity_type = str(c.get("entity_type", "")).strip().upper()
        if not name or not entity_type or name in seen_cl:
            continue
        domain = [str(v).strip().upper() for v in c.get("domain", []) if str(v).strip()]
        if not domain:
            continue
        seen_cl.add(name)
        clusters.append(ClusterSpec(
            name=name,
            entity_type=entity_type,
            domain=domain,
            description=str(c.get("description", "")).strip(),
            source_span_text=_opt_str(c, "source_span_text"),
        ))

    rules: list[RuleSpec] = []
    seen_rid: set[str] = set()
    for r in data.get("rules", []):
        rule_id = str(r.get("rule_id", "")).strip()
        module = str(r.get("module", "unknown")).strip().lower()
        clingo_text = str(r.get("clingo_text", "")).strip()
        if not rule_id or not clingo_text or rule_id in seen_rid:
            continue
        seen_rid.add(rule_id)
        rules.append(RuleSpec(
            rule_id=rule_id,
            module=module,
            clingo_text=clingo_text,
            stratum=int(r.get("stratum", 0)),
            source_span_text=_opt_str(r, "source_span_text"),
        ))

    return OntologyResult(
        entity_types=entity_types,
        predicates=predicates,
        clusters=clusters,
        rules=rules,
        source_id=source_id,
    )


# ---------------------------------------------------------------------------
# Parser clingo_text → head/body JSONB
# ---------------------------------------------------------------------------

def clingo_to_head_body(clingo_text: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """
    Parsuje uproszczony clingo_text do head i body JSONB.

    Obsługuje:
      pred(A, B) :- p1(A), not p2(B), p3(const).
      fact_head(X).

    Zmienne: [A-Z][A-Za-z0-9_]* lub _
    Stałe: [a-z][A-Za-z0-9_]* lub cytowane

    Zwraca (head_dict, body_list) zgodne ze schematem rule_repo.py.
    """
    text = clingo_text.strip().rstrip(".")

    if " :- " in text:
        head_part, body_part = text.split(" :- ", 1)
    else:
        head_part = text
        body_part = ""

    head = _parse_atom_to_head(head_part.strip())
    body = _parse_body(body_part.strip()) if body_part else []

    return head, body


def _parse_atom_to_head(atom_str: str) -> dict[str, Any]:
    pred, args = _split_pred_args(atom_str)
    return {
        "predicate": pred,
        "args": [
            {"role": f"ARG{i}", "term": _term_dict(a)}
            for i, a in enumerate(args)
        ],
    }


def _parse_body(body_str: str) -> list[dict[str, Any]]:
    literals = _split_top_level(body_str)
    result = []
    for lit in literals:
        lit = lit.strip()
        if not lit:
            continue
        naf = lit.startswith("not ")
        if naf:
            lit = lit[4:].strip()
        pred, args = _split_pred_args(lit)
        result.append({
            "literal_type": "naf" if naf else "pos",
            "predicate": pred,
            "args": [
                {"role": f"ARG{i}", "term": _term_dict(a)}
                for i, a in enumerate(args)
            ],
        })
    return result


def _split_pred_args(atom_str: str) -> tuple[str, list[str]]:
    m = re.match(r"(\w+)\((.*)\)$", atom_str.strip(), re.DOTALL)
    if not m:
        return atom_str.strip(), []
    pred = m.group(1)
    args_str = m.group(2)
    return pred, _split_top_level(args_str)


def _split_top_level(s: str) -> list[str]:
    """Dzieli string po przecinkach, ignorując przecinki wewnątrz nawiasów."""
    parts: list[str] = []
    depth = 0
    current: list[str] = []
    for ch in s:
        if ch == "(" :
            depth += 1
            current.append(ch)
        elif ch == ")":
            depth -= 1
            current.append(ch)
        elif ch == "," and depth == 0:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
    if current:
        parts.append("".join(current).strip())
    return [p for p in parts if p]


def _term_dict(token: str) -> dict[str, str]:
    t = token.strip()
    if t == "_" or (t and t[0].isupper()):
        return {"var": t}
    return {"const": t}


# ---------------------------------------------------------------------------
# Pomocnicze
# ---------------------------------------------------------------------------

def _opt_str(d: dict[str, Any], key: str) -> str | None:
    v = d.get(key)
    if v is None:
        return None
    s = str(v).strip()
    return s if s else None
