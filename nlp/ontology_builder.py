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

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

_PROMPT_TEMPLATE_PATH = Path(__file__).with_name("gen_ontology_prompt_template.txt")


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
    entity_role: str
    value_role: str
    domain: list[str] = field(default_factory=list)
    description: str = ""
    source_span_text: str | None = None


@dataclass
class RuleSpec:
    rule_id: str
    module: str
    head: dict[str, Any]
    body: list[dict[str, Any]] = field(default_factory=list)
    clingo_text: str | None = None
    stratum: int = 0
    source_span_text: str | None = None


@dataclass
class OntologyResult:
    entity_types: list[EntityTypeSpec] = field(default_factory=list)
    predicates: list[PredicateSpec] = field(default_factory=list)
    clusters: list[ClusterSpec] = field(default_factory=list)
    rules: list[RuleSpec] = field(default_factory=list)
    source_id: str = "regulation"
    validation_errors: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"entity_types={len(self.entity_types)} "
            f"predicates={len(self.predicates)} "
            f"clusters={len(self.clusters)} "
            f"rules={len(self.rules)}"
        )


# ---------------------------------------------------------------------------
# JSON Schema dla Gemini structured output
# ---------------------------------------------------------------------------

def build_ontology_schema() -> dict[str, Any]:
    term_schema = {
        "type": "OBJECT",
        "properties": {
            "var": {"type": "STRING"},
            "const": {"type": "STRING"},
        },
    }

    rule_arg_schema = {
        "type": "OBJECT",
        "properties": {
            "role": {"type": "STRING"},
            "term": term_schema,
        },
        "required": ["role", "term"],
    }

    rule_head_schema = {
        "type": "OBJECT",
        "properties": {
            "predicate": {"type": "STRING"},
            "args": {"type": "ARRAY", "items": rule_arg_schema},
        },
        "required": ["predicate", "args"],
    }

    rule_body_literal_schema = {
        "type": "OBJECT",
        "properties": {
            "literal_type": {"type": "STRING"},
            "predicate": {"type": "STRING"},
            "args": {"type": "ARRAY", "items": rule_arg_schema},
        },
        "required": ["literal_type", "predicate", "args"],
    }

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
            "entity_role":      {"type": "STRING"},
            "value_role":       {"type": "STRING"},
            "domain":           {"type": "ARRAY", "items": {"type": "STRING"}},
            "description":      {"type": "STRING"},
            "source_span_text": {"type": "STRING"},
        },
        "required": ["name", "entity_type", "entity_role", "value_role", "domain"],
    }

    rule_schema = {
        "type": "OBJECT",
        "properties": {
            "rule_id":          {"type": "STRING"},
            "module":           {"type": "STRING"},
            "head":             rule_head_schema,
            "body":             {"type": "ARRAY", "items": rule_body_literal_schema},
            "clingo_text":      {"type": "STRING"},
            "stratum":          {"type": "INTEGER"},
            "source_span_text": {"type": "STRING"},
        },
        "required": ["rule_id", "module", "head", "body"],
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
        entity_role = str(c.get("entity_role", "")).strip().upper()
        value_role = str(c.get("value_role", "")).strip().upper()
        if not name or not entity_type or not entity_role or not value_role or name in seen_cl:
            continue
        domain = [str(v).strip().upper() for v in c.get("domain", []) if str(v).strip()]
        if not domain:
            continue
        seen_cl.add(name)
        clusters.append(ClusterSpec(
            name=name,
            entity_type=entity_type,
            entity_role=entity_role,
            value_role=value_role,
            domain=domain,
            description=str(c.get("description", "")).strip(),
            source_span_text=_opt_str(c, "source_span_text"),
        ))

    rules: list[RuleSpec] = []
    seen_rid: set[str] = set()
    for r in data.get("rules", []):
        rule_id = str(r.get("rule_id", "")).strip()
        module = str(r.get("module", "unknown")).strip().lower()
        head = _normalize_rule_head(r.get("head"))
        body = _normalize_rule_body(r.get("body"))
        if not rule_id or head is None or body is None or rule_id in seen_rid:
            continue
        seen_rid.add(rule_id)
        clingo_text = str(r.get("clingo_text", "")).strip() or head_body_to_clingo(head, body)
        rules.append(RuleSpec(
            rule_id=rule_id,
            module=module,
            head=head,
            body=body,
            clingo_text=clingo_text,
            stratum=int(r.get("stratum", 0)),
            source_span_text=_opt_str(r, "source_span_text"),
        ))

    rules, validation_errors = _validate_rules(rules, predicates)

    return OntologyResult(
        entity_types=entity_types,
        predicates=predicates,
        clusters=clusters,
        rules=rules,
        source_id=source_id,
        validation_errors=validation_errors,
    )

def _validate_rules(
    rules: list[RuleSpec],
    predicates: list[PredicateSpec],
) -> tuple[list[RuleSpec], list[str]]:
    """
    Filtruje reguły z błędami strukturalnymi.
    Zwraca (valid_rules, error_descriptions).

    Sprawdza:
      1. Predykat głowy musi być zdefiniowany w predicates.
      2. Arność głowy musi zgadzać się z liczbą ról predykatu.
      3. Każda zmienna w głowie musi wystąpić w co najmniej jednym pozytywnym literale ciała
         (zakaz unsafe variables).
    """
    pred_arity: dict[str, int] = {p.name.lower(): len(p.roles) for p in predicates}
    valid: list[RuleSpec] = []
    errors: list[str] = []

    for rule in rules:
        head_pred = rule.head.get("predicate", "").lower()
        head_args = rule.head.get("args", [])

        # 1. Predykat głowy musi być zdefiniowany
        if head_pred not in pred_arity:
            errors.append(
                f"REGULA {rule.rule_id}: predykat glowy '{head_pred}' nie jest zdefiniowany "
                f"w sekcji predykatow — dodaj go najpierw do ## PREDYKATY"
            )
            continue

        # 2. Arność musi zgadzać się z definicją
        expected = pred_arity[head_pred]
        if len(head_args) != expected:
            errors.append(
                f"REGULA {rule.rule_id}: predykat '{head_pred}' ma {expected} rol(e) "
                f"w definicji, ale glowa reguly ma {len(head_args)} argumentow — wyrownaj arnosc"
            )
            continue

        # 3. Zakaz unsafe variables: zbierz zmienne z głowy
        head_vars: set[str] = {
            arg["term"]["var"]
            for arg in head_args
            if isinstance(arg.get("term"), dict)
            and "var" in arg["term"]
            and arg["term"]["var"] != "_"
        }

        # Zmienne dostępne z pozytywnych literałów ciała
        pos_body_vars: set[str] = set()
        for lit in rule.body:
            if lit.get("literal_type") == "pos":
                for arg in lit.get("args", []):
                    term = arg.get("term", {})
                    if "var" in term and term["var"] != "_":
                        pos_body_vars.add(term["var"])

        unsafe = head_vars - pos_body_vars
        if unsafe:
            errors.append(
                f"REGULA {rule.rule_id}: zmienne {sorted(unsafe)} w glowie nie sa "
                f"bindowane przez zadne pozytywne literaly ciala (unsafe variables) — "
                f"usun je z glowy lub dodaj literaly wiazace"
            )
            continue

        valid.append(rule)

    return valid, errors


def build_ontology_correction_prompt(
    original_text: str,
    errors: list[str],
) -> str:
    """
    Buduje prompt korekcyjny dla LLM gdy walidator odrzucil niepoprawne reguly.
    """
    error_block = "\n".join(f"  - {e}" for e in errors)
    return (
        "Poprzednia odpowiedz zawierala niepoprawne reguly, ktore zostaly odrzucone:\n"
        f"{error_block}\n\n"
        "Popraw ontologie eliminujac powyzsze bledy:\n"
        "- Kazda regula z niezdefiniowanym predykatem glowy: dodaj predykat do sekcji predicates\n"
        "- Kazda regula z niezgodna arnoscia: wyrownaj liczbe argumentow glowy do liczby rol predykatu\n"
        "- Kazda regula z unsafe variables: usun zmienna z glowy lub dodaj literaly wiazace do body\n\n"
        "Nie zmieniaj poprawnych elementow ontologii. Zwroc pelna poprawiona ontologie.\n\n"
        f"Oryginalny regulamin:\n{original_text}"
    )


def head_body_to_clingo(
    head: dict[str, Any],
    body: list[dict[str, Any]],
) -> str:
    head_text = _rule_atom_to_clingo(head)
    if not body:
        return f"{head_text}."

    body_text = ", ".join(_rule_literal_to_clingo(literal) for literal in body)
    return f"{head_text} :- {body_text}."


def _normalize_rule_head(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    predicate = str(raw.get("predicate", "")).strip().lower()
    args_raw = raw.get("args")
    if not predicate or not isinstance(args_raw, list):
        return None

    args: list[dict[str, Any]] = []
    for arg in args_raw:
        normalized = _normalize_rule_arg(arg)
        if normalized is None:
            return None
        args.append(normalized)
    return {"predicate": predicate, "args": args}


def _normalize_rule_body(raw: Any) -> list[dict[str, Any]] | None:
    if raw is None:
        return []
    if not isinstance(raw, list):
        return None

    body: list[dict[str, Any]] = []
    for literal in raw:
        if not isinstance(literal, dict):
            return None
        literal_type = str(literal.get("literal_type", "")).strip().lower()
        predicate = str(literal.get("predicate", "")).strip().lower()
        args_raw = literal.get("args")
        if literal_type not in {"pos", "naf"} or not predicate or not isinstance(args_raw, list):
            return None

        args: list[dict[str, Any]] = []
        for arg in args_raw:
            normalized = _normalize_rule_arg(arg)
            if normalized is None:
                return None
            args.append(normalized)

        body.append({
            "literal_type": literal_type,
            "predicate": predicate,
            "args": args,
        })
    return body


def _normalize_rule_arg(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    role = str(raw.get("role", "")).strip().upper()
    term = _normalize_term(raw.get("term"))
    if not role or term is None:
        return None
    return {"role": role, "term": term}


def _normalize_term(raw: Any) -> dict[str, str] | None:
    if not isinstance(raw, dict):
        return None
    var = raw.get("var")
    const = raw.get("const")
    if bool(var) == bool(const):
        return None
    if var:
        return {"var": str(var).strip()}
    return {"const": str(const).strip().lower()}


def _rule_atom_to_clingo(atom: dict[str, Any]) -> str:
    predicate = str(atom.get("predicate", "")).strip().lower()
    args = atom.get("args", [])
    if not args:
        return predicate
    rendered = ",".join(_term_to_clingo(arg["term"]) for arg in args)
    return f"{predicate}({rendered})"


def _rule_literal_to_clingo(literal: dict[str, Any]) -> str:
    atom = _rule_atom_to_clingo(literal)
    if literal.get("literal_type") == "naf":
        return f"not {atom}"
    return atom


def _term_to_clingo(term: dict[str, str]) -> str:
    if "var" in term:
        return term["var"]
    return term["const"].lower()


# ---------------------------------------------------------------------------
# Pomocnicze
# ---------------------------------------------------------------------------

def _opt_str(d: dict[str, Any], key: str) -> str | None:
    v = d.get(key)
    if v is None:
        return None
    s = str(v).strip()
    return s if s else None


@lru_cache(maxsize=1)
def _load_ontology_prompt_template() -> str:
    return _PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8")


def build_ontology_prompt(regulatory_text: str) -> str:
    """
    Render ontology generation prompt from the external template file.
    """
    template = _load_ontology_prompt_template()
    return template.replace("{{REGULATORY_TEXT}}", regulatory_text)
