#!/usr/bin/env python3
"""pn3train - self-training CLI for ProveNuance3."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import replace
from pathlib import Path

from rich.console import Console
from rich.table import Table
from runtime_env import load_project_env

from data_model.cluster import ClusterSchema, ClusterStateRow
from data_model.common import Span
from data_model.fact import Fact, FactStatus
from data_model.self_training import (
    CaseSplit,
    PseudoClusterLabel,
    PseudoFactLabel,
    SelfTrainingRound,
)
from db import connect, load_predicate_positions, load_rules
from db.entity_repo import load_entities_by_ids
from db.self_training_repo import (
    assign_case_split,
    list_case_ids_by_split,
    list_cases_with_pseudo_labels,
    list_round_ids,
    load_pseudo_cluster_labels_for_case,
    load_pseudo_fact_labels_for_case,
    load_round,
    promote_round,
    save_pseudo_cluster_labels,
    save_pseudo_fact_labels,
    set_round_status,
    upsert_round,
)
from sv.types import CandidateFeedback
from sv.types import GroundAtom

console = Console()
load_project_env()

_QUERY_RE = re.compile(r"^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\((.*)\)\s*$")
_EVAL_LABELS: tuple[str, ...] = ("proved", "not_proved", "blocked", "unknown")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def _fact_key(fact: Fact) -> str:
    parts = [fact.predicate.upper()]
    arg_parts = sorted(
        (
            arg.role.upper(),
            arg.entity_id or "",
            arg.literal_value or "",
        )
        for arg in fact.args
    )
    for role, entity_id, literal_value in arg_parts:
        value = entity_id if entity_id else f"${literal_value}"
        parts.append(f"{role}={value}")
    return "|".join(parts)


def _softmax_stats(logits: list[float], dim: int) -> tuple[int, float, float]:
    if dim <= 0:
        raise ValueError("dim must be > 0")
    values = logits[:dim]
    if not values:
        raise ValueError("logits are empty")
    max_logit = max(values)
    exps = [math.exp(v - max_logit) for v in values]
    total = sum(exps) or 1.0
    probs = [v / total for v in exps]
    order = sorted(range(len(probs)), key=lambda idx: probs[idx], reverse=True)
    top_idx = order[0]
    top1 = probs[top_idx]
    top2 = probs[order[1]] if len(order) > 1 else 0.0
    return top_idx, top1, top1 - top2


def _parse_query_atom(
    query: str,
    predicate_positions: dict[str, list[str]] | None = None,
) -> GroundAtom:
    from sv._utils import to_clingo_id

    q = query.strip()
    m = _QUERY_RE.match(q)
    if not m:
        return GroundAtom(q.lower(), ())
    pred = m.group(1).strip().lower()
    args_raw = [a.strip() for a in m.group(2).split(",") if a.strip()]
    roles = (predicate_positions or {}).get(pred, [])
    bindings = tuple(
        (
            roles[i].upper() if i < len(roles) else str(i),
            to_clingo_id(a),
        )
        for i, a in enumerate(args_raw)
    )
    return GroundAtom(pred, tuple(sorted(bindings)))


def _precision_recall_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def _load_eval_queries(
    conn,
    case_ids: list[str],
) -> dict[str, list[tuple[int, str, str]]]:
    query_map = {case_id: [] for case_id in case_ids}
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT c.case_id, cq.id, cq.query, cq.expected_result
            FROM cases c
            LEFT JOIN case_queries cq ON cq.case_id = c.id
            WHERE c.case_id = ANY(%s)
            ORDER BY c.id, cq.id
            """,
            (case_ids,),
        )
        for case_id, query_id, query_text, expected in cur.fetchall():
            if query_id is None:
                continue
            query_map[str(case_id)].append((int(query_id), str(query_text), str(expected)))
    return query_map


def _normalize_expected_result(value: str, *, field_name: str = "expected_result") -> str:
    normalized = value.strip().lower()
    if normalized not in _EVAL_LABELS:
        raise ValueError(
            f"{field_name} must be one of {', '.join(_EVAL_LABELS)}; got: {value}"
        )
    return normalized


def _normalize_case_query_payload(
    raw: dict[str, object],
    *,
    row_ref: str,
) -> dict[str, str]:
    case_id = str(raw.get("case_id", "")).strip()
    query = str(raw.get("query", "")).strip()
    expected_raw = str(raw.get("expected_result", "")).strip()
    notes = str(raw.get("notes", "")).strip() or ""

    if not case_id and not query and not expected_raw and not notes:
        return {}
    if not case_id:
        raise ValueError(f"{row_ref}: missing case_id")
    if not query and not expected_raw:
        return {}
    if not query:
        raise ValueError(f"{row_ref}: missing query")
    if not expected_raw:
        raise ValueError(f"{row_ref}: missing expected_result")

    return {
        "case_id": case_id,
        "query": query,
        "expected_result": _normalize_expected_result(expected_raw, field_name=f"{row_ref}.expected_result"),
        "notes": notes,
    }


def _load_case_queries_from_file(path: Path) -> list[dict[str, str]]:
    suffix = path.suffix.lower()
    rows: list[dict[str, str]] = []

    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = set(reader.fieldnames or [])
            required = {"case_id", "query", "expected_result"}
            if not required.issubset(fieldnames):
                raise ValueError(
                    f"{path}: CSV must contain headers: {', '.join(sorted(required))}"
                )
            for idx, raw in enumerate(reader, start=2):
                payload = _normalize_case_query_payload(dict(raw), row_ref=f"{path}:{idx}")
                if payload:
                    rows.append(payload)
    elif suffix in {".jsonl", ".ndjson"}:
        for idx, raw_line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
            line = raw_line.strip()
            if not line:
                continue
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError(f"{path}:{idx}: expected JSON object")
            payload = _normalize_case_query_payload(item, row_ref=f"{path}:{idx}")
            if payload:
                rows.append(payload)
    else:
        raise ValueError(f"Unsupported case query format: {path.suffix} (expected .csv or .jsonl)")

    return rows


def _load_case_query_catalog(
    conn,
    *,
    case_ids: list[str] | None = None,
    split: str | None = None,
    all_cases: bool = False,
) -> list[dict[str, str]]:
    if not all_cases and not case_ids and not split:
        raise ValueError("Select cases via --case, --split, or --all")

    where = ""
    params: tuple[object, ...] = ()
    if all_cases:
        pass
    elif case_ids:
        where = "WHERE c.case_id = ANY(%s)"
        params = (case_ids,)
    else:
        where = "WHERE c.dataset_split = %s"
        params = (split,)

    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT
                c.case_id,
                c.dataset_split,
                COALESCE(c.title, ''),
                s.source_id,
                COALESCE(s.content, '')
            FROM cases c
            JOIN sources s ON s.id = c.source_id
            {where}
            ORDER BY c.id
            """,
            params,
        )
        return [
            {
                "case_id": str(case_id),
                "dataset_split": str(dataset_split or ""),
                "title": str(title or ""),
                "source_id": str(source_id or ""),
                "source_content": str(content or ""),
            }
            for case_id, dataset_split, title, source_id, content in cur.fetchall()
        ]


def _load_case_query_counts(
    conn,
    case_ids: list[str],
) -> dict[str, int]:
    if not case_ids:
        return {}
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT c.case_id, COUNT(cq.id)
            FROM cases c
            LEFT JOIN case_queries cq ON cq.case_id = c.id
            WHERE c.case_id = ANY(%s)
            GROUP BY c.case_id
            """,
            (case_ids,),
        )
        return {
            str(case_id): int(count)
            for case_id, count in cur.fetchall()
        }


def _text_excerpt(text: str, limit: int = 240) -> str:
    flattened = " ".join(text.split())
    if len(flattened) <= limit:
        return flattened
    return flattened[: max(0, limit - 3)].rstrip() + "..."


def _write_case_query_template(
    path: Path,
    rows: list[dict[str, str]],
    *,
    include_content: bool,
) -> None:
    suffix = path.suffix.lower()
    template_rows: list[dict[str, str]] = []
    for row in rows:
        item = {
            "case_id": row["case_id"],
            "dataset_split": row["dataset_split"],
            "title": row["title"],
            "source_id": row["source_id"],
            "source_excerpt": _text_excerpt(row["source_content"]),
            "query": "",
            "expected_result": "",
            "notes": "",
        }
        if include_content:
            item["source_content"] = row["source_content"]
        template_rows.append(item)

    if suffix == ".csv":
        fieldnames = [
            "case_id",
            "dataset_split",
            "title",
            "source_id",
            "source_excerpt",
        ]
        if include_content:
            fieldnames.append("source_content")
        fieldnames.extend(["query", "expected_result", "notes"])
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(template_rows)
        return

    if suffix in {".jsonl", ".ndjson"}:
        lines = [json.dumps(row, ensure_ascii=False) for row in template_rows]
        path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        return

    raise ValueError(f"Unsupported template format: {path.suffix} (expected .csv or .jsonl)")


def _write_case_query_records(path: Path, rows: list[dict[str, str]]) -> None:
    if path.suffix.lower() not in {".jsonl", ".ndjson"}:
        raise ValueError("draft-case-queries output must be .jsonl or .ndjson")
    lines = [json.dumps(row, ensure_ascii=False) for row in rows]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _build_eval_metrics(
    details: list[dict[str, object]],
) -> tuple[dict[str, dict[str, int]], dict[str, dict[str, float | int]], dict[str, float]]:
    confusion = {
        expected: {got: 0 for got in _EVAL_LABELS}
        for expected in _EVAL_LABELS
    }
    for item in details:
        expected = str(item["expected"])
        got = str(item["got"])
        if expected in confusion and got in confusion[expected]:
            confusion[expected][got] += 1

    per_label: dict[str, dict[str, float | int]] = {}
    macro_precision = 0.0
    macro_recall = 0.0
    macro_f1 = 0.0
    weighted_f1 = 0.0
    total_support = 0

    for label in _EVAL_LABELS:
        tp = confusion[label][label]
        fp = sum(confusion[expected][label] for expected in _EVAL_LABELS if expected != label)
        fn = sum(confusion[label][got] for got in _EVAL_LABELS if got != label)
        support = sum(confusion[label].values())
        precision, recall, f1 = _precision_recall_f1(tp, fp, fn)
        per_label[label] = {
            "support": support,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        macro_precision += precision
        macro_recall += recall
        macro_f1 += f1
        weighted_f1 += f1 * support
        total_support += support

    n_labels = len(_EVAL_LABELS) or 1
    summary = {
        "macro_precision": macro_precision / n_labels,
        "macro_recall": macro_recall / n_labels,
        "macro_f1": macro_f1 / n_labels,
        "weighted_f1": (weighted_f1 / total_support) if total_support > 0 else 0.0,
    }
    return confusion, per_label, summary


def _cluster_value(schema: ClusterSchema, state: ClusterStateRow) -> str:
    top_idx, _, _ = _softmax_stats(state.logits, schema.dim)
    return schema.domain[top_idx].upper()


def _serialize_round_record(round_info: SelfTrainingRound) -> str:
    return json.dumps(
        {"record_type": "round", "payload": round_info.model_dump(mode="json")},
        ensure_ascii=False,
    )


def _serialize_fact_label(label: PseudoFactLabel) -> str:
    return json.dumps(
        {"record_type": "pseudo_fact", "payload": label.model_dump(mode="json")},
        ensure_ascii=False,
    )


def _serialize_cluster_label(label: PseudoClusterLabel) -> str:
    payload = label.model_dump(mode="json", exclude={"state"})
    payload["state"] = {
        "entity_id": label.state.entity_id,
        "cluster_name": label.state.cluster_name,
        "logits": label.state.logits,
        "is_clamped": label.state.is_clamped,
        "clamp_hard": label.state.clamp_hard,
        "clamp_source": label.state.clamp_source,
        "source_span": (
            label.state.source_span.model_dump(mode="json")
            if label.state.source_span is not None else None
        ),
    }
    return json.dumps(
        {"record_type": "pseudo_cluster", "payload": payload},
        ensure_ascii=False,
    )


def _parse_export_file(path: Path) -> tuple[SelfTrainingRound, list[PseudoFactLabel], list[PseudoClusterLabel]]:
    round_info: SelfTrainingRound | None = None
    fact_labels: list[PseudoFactLabel] = []
    cluster_labels: list[PseudoClusterLabel] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        item = json.loads(line)
        record_type = item.get("record_type")
        payload = item.get("payload")
        if record_type == "round":
            round_info = SelfTrainingRound.model_validate(payload)
        elif record_type == "pseudo_fact":
            fact_labels.append(PseudoFactLabel.model_validate(payload))
        elif record_type == "pseudo_cluster":
            row = dict(payload)
            state_payload = row.pop("state")
            if state_payload.get("source_span") is not None:
                state_payload["source_span"] = Span.model_validate(state_payload["source_span"])
            state = ClusterStateRow(**state_payload)
            cluster_labels.append(PseudoClusterLabel.model_validate({**row, "state": state}))
        else:
            raise ValueError(f"Unknown record_type in {path}: {record_type}")
    if round_info is None:
        raise ValueError(f"Missing round metadata in {path}")
    return round_info, fact_labels, cluster_labels


def _term_preview(term) -> str:
    var = getattr(term, "var", None)
    if var is not None:
        return str(var)
    const = getattr(term, "const", None)
    if const is not None:
        return str(const)
    return "?"


def _literal_preview(literal) -> str:
    args = ", ".join(
        f"{arg.role}={_term_preview(arg.term)}"
        for arg in literal.args
    )
    return f"{literal.predicate}({args})"


def _rule_preview(rule) -> str:
    head = _literal_preview(rule.head)
    if not rule.body:
        return f"{head}."
    body = ", ".join(_literal_preview(literal) for literal in rule.body)
    return f"{head} :- {body}."


def _print_extracted_rules_preview(rules, *, title: str, limit: int = 20) -> None:
    if not rules:
        console.print("[dim]No extracted rules above current threshold.[/dim]")
        return

    table = Table(title=title, header_style="bold cyan")
    table.add_column("weight", justify="right", style="dim")
    table.add_column("rule_id", style="bold yellow")
    table.add_column("rule")

    for rule in rules[:limit]:
        weight = float(rule.metadata.weight or 0.0)
        table.add_row(
            f"{weight:.4f}",
            rule.rule_id,
            _rule_preview(rule),
        )

    console.print(table)
    if len(rules) > limit:
        console.print(f"[dim]showing {limit}/{len(rules)} extracted rules[/dim]")


def _collect_case_pseudo_labels(
    round_id: str,
    case_id: str,
    gold_facts: list[Fact],
    gold_states: list[ClusterStateRow],
    result_facts: list[Fact],
    candidate_feedback: list[CandidateFeedback],
    result_states: list[ClusterStateRow],
    schemas: list[ClusterSchema],
    fact_conf_threshold: float,
    cluster_top1_threshold: float,
    cluster_margin_threshold: float,
) -> tuple[list[PseudoFactLabel], list[PseudoClusterLabel]]:
    gold_fact_keys = {_fact_key(fact) for fact in gold_facts}
    schema_by_name = {schema.name: schema for schema in schemas}

    gold_state_lookup: dict[tuple[str, str], tuple[ClusterStateRow, str]] = {}
    for state in gold_states:
        schema = schema_by_name.get(state.cluster_name)
        if schema is None or not state.logits:
            continue
        gold_state_lookup[(state.entity_id, state.cluster_name)] = (
            state,
            _cluster_value(schema, state),
        )

    pseudo_facts: list[PseudoFactLabel] = []
    seen_fact_keys: set[str] = set()
    fact_by_id = {fact.fact_id: fact for fact in result_facts}
    for fact in result_facts:
        confidence = float(fact.truth.confidence if fact.truth.confidence is not None else 0.0)
        if fact.status.value != "proved":
            continue
        if (fact.truth.value or "").upper() != "T":
            continue
        if confidence < fact_conf_threshold:
            continue
        fact_key = _fact_key(fact)
        if fact_key in gold_fact_keys:
            continue
        seen_fact_keys.add(fact_key)
        pseudo_facts.append(PseudoFactLabel(
            round_id=round_id,
            case_id=case_id,
            fact_key=fact_key,
            fact=fact,
            truth_confidence=confidence,
            proof_id=fact.provenance.proof_id if fact.provenance is not None else None,
        ))

    for item in candidate_feedback:
        if item.outcome != "blocked":
            continue
        fact = fact_by_id.get(item.fact_id)
        if fact is None:
            continue
        if fact.status != FactStatus.rejected:
            continue
        if (fact.truth.value or "").upper() != "F":
            continue
        confidence = float(fact.truth.confidence if fact.truth.confidence is not None else 0.0)
        if confidence < fact_conf_threshold:
            continue
        fact_key = _fact_key(fact)
        if fact_key in gold_fact_keys or fact_key in seen_fact_keys:
            continue
        seen_fact_keys.add(fact_key)
        pseudo_facts.append(PseudoFactLabel(
            round_id=round_id,
            case_id=case_id,
            fact_key=fact_key,
            fact=fact,
            truth_confidence=confidence,
            proof_id=fact.provenance.proof_id if fact.provenance is not None else None,
        ))

    pseudo_clusters: list[PseudoClusterLabel] = []
    for state in result_states:
        schema = schema_by_name.get(state.cluster_name)
        if schema is None or not state.logits:
            continue
        top_idx, top1, margin = _softmax_stats(state.logits, schema.dim)
        if top1 < cluster_top1_threshold or margin < cluster_margin_threshold:
            continue
        if state.is_clamped and state.clamp_hard and state.clamp_source in {"text", "manual"}:
            continue

        value = schema.domain[top_idx].upper()
        gold_match = gold_state_lookup.get((state.entity_id, state.cluster_name))
        if gold_match is not None:
            gold_state, gold_value = gold_match
            if gold_state.is_clamped and gold_state.clamp_hard and gold_state.clamp_source in {"text", "manual"}:
                if gold_value != value:
                    continue
                continue

        pseudo_clusters.append(PseudoClusterLabel(
            round_id=round_id,
            case_id=case_id,
            entity_id=state.entity_id,
            cluster_name=state.cluster_name,
            value=value,
            state=ClusterStateRow(
                entity_id=state.entity_id,
                cluster_name=state.cluster_name,
                logits=state.logits,
                is_clamped=True,
                clamp_hard=False,
                clamp_source="memory",
                source_span=state.source_span,
            ),
            top1_confidence=top1,
            margin=margin,
        ))

    return pseudo_facts, pseudo_clusters


def _pseudo_fact_label_stats(labels: list[PseudoFactLabel]) -> dict[str, int]:
    counts = {"total": 0, "t": 0, "f": 0, "u": 0}
    for label in labels:
        counts["total"] += 1
        value = (label.fact.truth.value or "U").upper()
        if value == "T":
            counts["t"] += 1
        elif value == "F":
            counts["f"] += 1
        else:
            counts["u"] += 1
    return counts


def _merge_pseudo_overlay(
    conn,
    case_id: str,
    entities,
    facts: list[Fact],
    states: list[ClusterStateRow],
    schemas: list[ClusterSchema],
    round_ids: list[str] | None,
    promoted_only: bool,
) -> tuple[list, list[Fact], list[ClusterStateRow]]:
    pseudo_facts = load_pseudo_fact_labels_for_case(
        conn,
        case_id=case_id,
        round_ids=round_ids,
        promoted_only=promoted_only,
    )
    pseudo_clusters = load_pseudo_cluster_labels_for_case(
        conn,
        case_id=case_id,
        round_ids=round_ids,
        promoted_only=promoted_only,
    )
    if not pseudo_facts and not pseudo_clusters:
        return entities, facts, states

    entity_map = {entity.entity_id: entity for entity in entities}
    needed_entity_ids = {
        arg.entity_id
        for label in pseudo_facts
        for arg in label.fact.args
        if arg.entity_id is not None
    }
    needed_entity_ids |= {label.entity_id for label in pseudo_clusters}
    missing_entity_ids = sorted(eid for eid in needed_entity_ids if eid not in entity_map)
    if missing_entity_ids:
        for entity in load_entities_by_ids(conn, missing_entity_ids):
            entity_map.setdefault(entity.entity_id, entity)

    merged_facts = list(facts)
    existing_fact_keys = {_fact_key(fact) for fact in merged_facts}
    best_pseudo_fact: dict[str, PseudoFactLabel] = {}
    for label in pseudo_facts:
        current = best_pseudo_fact.get(label.fact_key)
        if current is None or label.truth_confidence > current.truth_confidence:
            best_pseudo_fact[label.fact_key] = label
    for fact_key, label in best_pseudo_fact.items():
        if fact_key in existing_fact_keys:
            continue
        merged_facts.append(label.fact)
        existing_fact_keys.add(fact_key)

    schema_by_name = {schema.name: schema for schema in schemas}
    merged_states: dict[tuple[str, str], ClusterStateRow] = {
        (state.entity_id, state.cluster_name): state for state in states
    }
    best_pseudo_cluster: dict[tuple[str, str], PseudoClusterLabel] = {}
    for label in pseudo_clusters:
        key = (label.entity_id, label.cluster_name)
        current = best_pseudo_cluster.get(key)
        if current is None or label.margin > current.margin:
            best_pseudo_cluster[key] = label

    for key, label in best_pseudo_cluster.items():
        existing = merged_states.get(key)
        schema = schema_by_name.get(label.cluster_name)
        if schema is None:
            continue
        if existing is not None and existing.is_clamped and existing.clamp_hard and existing.clamp_source in {"text", "manual"}:
            if _cluster_value(schema, existing) != label.value:
                continue
            continue
        merged_states[key] = label.state

    return (
        list(entity_map.values()),
        merged_facts,
        list(merged_states.values()),
    )


def _build_pseudo_cluster_key_set(
    conn,
    case_id: str,
    round_ids: list[str] | None,
    promoted_only: bool,
) -> set[tuple[str, str]]:
    labels = load_pseudo_cluster_labels_for_case(
        conn,
        case_id=case_id,
        round_ids=round_ids,
        promoted_only=promoted_only,
    )
    return {
        (label.entity_id, label.cluster_name)
        for label in labels
        if label.accepted
    }


def _attach_mask_weights(
    data,
    node_index,
    schemas: list[ClusterSchema],
    pseudo_cluster_keys: set[tuple[str, str]],
    pseudo_cluster_weight: float,
) -> None:
    import torch

    for schema in schemas:
        node_type = f"c_{schema.name}"
        cluster_map = node_index.cluster_node_to_idx.get(schema.name, {})
        weights = torch.ones(len(cluster_map), dtype=torch.float32)
        for entity_id, idx in cluster_map.items():
            if (entity_id, schema.name) in pseudo_cluster_keys:
                weights[idx] = float(pseudo_cluster_weight)
        if node_type in data.node_types:
            data[node_type].mask_weight = weights


def _attach_fact_supervision(
    data,
    node_index,
    facts: list[Fact],
    pseudo_fact_weight: float,
) -> None:
    import torch

    if "fact" not in data.node_types:
        return

    n_facts = len(node_index.fact_node_to_idx)
    targets = torch.full((n_facts,), -1, dtype=torch.long)
    weights = torch.zeros(n_facts, dtype=torch.float32)
    truth_to_idx = {"T": 0, "F": 1, "U": 2}

    for fact in facts:
        if fact.status not in {FactStatus.proved, FactStatus.rejected}:
            continue
        idx = node_index.fact_node_to_idx.get(fact.fact_id)
        if idx is None:
            continue
        truth_value = (fact.truth.value or "").upper()
        target_idx = truth_to_idx.get(truth_value)
        if target_idx is None:
            continue
        confidence = float(fact.truth.confidence if fact.truth.confidence is not None else 1.0)
        targets[idx] = target_idx
        weights[idx] = float(pseudo_fact_weight) * confidence
        data["fact"].is_clamped[idx] = False
        data["fact"].clamp_hard[idx] = False
        data["fact"].x[idx].zero_()

    data["fact"].supervision_target = targets
    data["fact"].supervision_weight = weights


def _fact_supervision_stats(data) -> dict[str, float]:
    if "fact" not in data.node_types:
        return {"labeled": 0.0, "t": 0.0, "f": 0.0, "u": 0.0, "weight_sum": 0.0}

    targets = data["fact"].get("supervision_target")
    weights = data["fact"].get("supervision_weight")
    if targets is None or targets.numel() == 0:
        return {"labeled": 0.0, "t": 0.0, "f": 0.0, "u": 0.0, "weight_sum": 0.0}

    labeled = 0
    counts = {0: 0, 1: 0, 2: 0}
    weight_sum = 0.0

    weight_list = weights.tolist() if weights is not None and weights.numel() == targets.numel() else None
    for idx, target in enumerate(targets.tolist()):
        if target < 0:
            continue
        labeled += 1
        counts[target] = counts.get(target, 0) + 1
        if weight_list is not None:
            weight_sum += float(weight_list[idx])

    return {
        "labeled": float(labeled),
        "t": float(counts.get(0, 0)),
        "f": float(counts.get(1, 0)),
        "u": float(counts.get(2, 0)),
        "weight_sum": float(weight_sum),
    }


def cmd_set_split(args: argparse.Namespace) -> None:
    with connect() as conn:
        updated = assign_case_split(
            conn,
            split=args.split,
            case_ids=args.case,
            pattern=args.pattern,
            all_cases=bool(args.all),
        )
        conn.commit()
    console.print(f"[bold green]set-split completed[/bold green]: split={args.split}, updated={updated}")


def cmd_export_case_query_template(args: argparse.Namespace) -> None:
    output_path = Path(args.output)
    with connect() as conn:
        rows = _load_case_query_catalog(
            conn,
            case_ids=args.case,
            split=args.split,
            all_cases=bool(args.all),
        )
    if not rows:
        console.print("[bold red]No cases selected for template export.[/bold red]")
        raise SystemExit(1)

    _write_case_query_template(
        output_path,
        rows,
        include_content=bool(args.include_content),
    )
    console.print(
        f"[bold green]export-case-query-template completed[/bold green]: "
        f"cases={len(rows)}, file={output_path}"
    )


def cmd_import_case_queries(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    if not input_path.is_file():
        console.print(f"[bold red]Plik nie istnieje:[/bold red] {input_path}")
        raise SystemExit(1)

    rows = _load_case_queries_from_file(input_path)
    if not rows:
        console.print("[bold red]No case queries found in input file.[/bold red]")
        raise SystemExit(1)

    unique_rows: list[dict[str, str]] = []
    seen_exact: set[tuple[str, str, str]] = set()
    expected_by_query: dict[tuple[str, str], str] = {}
    for row in rows:
        query_key = (row["case_id"], row["query"])
        existing_expected = expected_by_query.get(query_key)
        if existing_expected is not None and existing_expected != row["expected_result"]:
            raise ValueError(
                f"Conflicting expected_result for {row['case_id']} :: {row['query']}: "
                f"{existing_expected} vs {row['expected_result']}"
            )
        expected_by_query[query_key] = row["expected_result"]

        exact_key = (row["case_id"], row["query"], row["expected_result"])
        if exact_key in seen_exact:
            continue
        seen_exact.add(exact_key)
        unique_rows.append(row)

    case_ids = sorted({row["case_id"] for row in unique_rows})
    inserted = 0
    skipped = 0

    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT case_id, id
                FROM cases
                WHERE case_id = ANY(%s)
                """,
                (case_ids,),
            )
            case_id_map = {str(case_id): int(case_int_id) for case_id, case_int_id in cur.fetchall()}

            missing_cases = [case_id for case_id in case_ids if case_id not in case_id_map]
            if missing_cases:
                raise ValueError(
                    f"Unknown case_id in import: {', '.join(missing_cases[:10])}"
                )

            if args.replace:
                cur.execute(
                    """
                    DELETE FROM case_queries
                    WHERE case_id = ANY(%s)
                    """,
                    ([case_id_map[case_id] for case_id in case_ids],),
                )
            else:
                cur.execute(
                    """
                    SELECT c.case_id, cq.query, cq.expected_result
                    FROM case_queries cq
                    JOIN cases c ON c.id = cq.case_id
                    WHERE c.case_id = ANY(%s)
                    """,
                    (case_ids,),
                )
                existing_exact = {
                    (str(case_id), str(query), str(expected))
                    for case_id, query, expected in cur.fetchall()
                }
                existing_query_expected = {
                    (case_id, query): expected
                    for case_id, query, expected in existing_exact
                }
                for case_id, query, expected in seen_exact:
                    existing_expected = existing_query_expected.get((case_id, query))
                    if existing_expected is not None and existing_expected != expected:
                        raise ValueError(
                            f"Existing case_query conflict for {case_id} :: {query}: "
                            f"{existing_expected} vs {expected}. Use --replace to overwrite this case."
                        )

            rows_to_insert: list[tuple[int, str, str, str | None]] = []
            existing_exact = set() if args.replace else existing_exact
            for row in unique_rows:
                exact_key = (row["case_id"], row["query"], row["expected_result"])
                if exact_key in existing_exact:
                    skipped += 1
                    continue
                rows_to_insert.append(
                    (
                        case_id_map[row["case_id"]],
                        row["query"],
                        row["expected_result"],
                        row["notes"] or None,
                    )
                )

            if args.dry_run:
                inserted = len(rows_to_insert)
            else:
                if rows_to_insert:
                    cur.executemany(
                        """
                        INSERT INTO case_queries (case_id, query, expected_result, notes)
                        VALUES (%s, %s, %s, %s)
                        """,
                        rows_to_insert,
                    )
                conn.commit()
                inserted = len(rows_to_insert)

    mode = "dry-run" if args.dry_run else "completed"
    console.print(
        f"[bold green]import-case-queries {mode}[/bold green]: "
        f"inserted={inserted}, skipped={skipped}, cases={len(case_ids)}"
    )


def cmd_draft_case_queries(args: argparse.Namespace) -> None:
    from config import ProjectConfig
    from nlp.case_query_drafter import CaseQueryDrafter

    cfg = ProjectConfig.load()
    output_path = Path(args.output)

    with connect() as conn:
        predicate_positions = load_predicate_positions(conn)
        if not predicate_positions:
            console.print("[bold red]Brak aktywnej ontologii z predykatami.[/bold red]")
            raise SystemExit(1)

        preferred_predicates = sorted({
            rule.head.predicate.lower()
            for rule in load_rules(conn, enabled_only=True)
            if getattr(rule, "head", None) is not None and rule.head.predicate
        })

        rows = _load_case_query_catalog(
            conn,
            case_ids=args.case,
            split=args.split,
            all_cases=bool(args.all),
        )
        if not args.include_existing:
            query_counts = _load_case_query_counts(conn, [row["case_id"] for row in rows])
            rows = [row for row in rows if query_counts.get(row["case_id"], 0) == 0]

    if not rows:
        console.print("[bold red]No cases selected for draft-case-queries.[/bold red]")
        raise SystemExit(1)

    console.print(
        f"[dim]draft-case-queries: cases={len(rows)} | model={cfg.extractor.gemini_model} "
        f"| max_queries={args.max_queries}[/dim]"
    )

    drafter = CaseQueryDrafter(
        predicate_positions,
        cfg.extractor,
        year=cfg.year,
        preferred_predicates=preferred_predicates,
    )
    draft_rows: list[dict[str, str]] = []
    errors = 0
    try:
        for row in rows:
            case_id = row["case_id"]
            console.print(f"  [blue]START[/blue] {case_id} [dim](draft)[/dim]")
            try:
                drafts = drafter.draft(
                    case_id=case_id,
                    title=row["title"],
                    case_text=row["source_content"],
                    max_queries=args.max_queries,
                )
                if not drafts:
                    console.print(f"  [yellow]EMPTY[/yellow] {case_id}")
                    continue
                for item in drafts:
                    note_parts = [part for part in [item.get("notes", ""), item.get("rationale", "")] if part]
                    draft_rows.append({
                        "case_id": case_id,
                        "query": item["query"],
                        "expected_result": item["expected_result"],
                        "notes": " | ".join(note_parts),
                        "rationale": item.get("rationale", ""),
                        "dataset_split": row["dataset_split"],
                        "source_id": row["source_id"],
                        "draft_model": cfg.extractor.gemini_model,
                    })
                console.print(f"  [cyan]drafted[/cyan] {case_id}: queries={len(drafts)}")
            except Exception as exc:
                console.print(f"  [bold red]ERR[/bold red] {case_id} (draft): {exc}")
                errors += 1
                if not args.continue_on_error:
                    raise SystemExit(1) from exc
    finally:
        drafter.close()

    _write_case_query_records(output_path, draft_rows)
    console.print(
        f"[bold green]draft-case-queries completed[/bold green]: "
        f"drafts={len(draft_rows)}, cases={len(rows)}, errors={errors}, file={output_path}"
    )


def cmd_collect_pseudo_labels(args: argparse.Namespace) -> None:
    from db import DBSession
    from nn.config import NNConfig
    from pipeline.temporal_config import get_temporal_constraints
    from pipeline.runner import ProposeVerifyRunner

    round_info = SelfTrainingRound(
        round_id=args.round_id,
        parent_round_id=args.parent_round,
        status="draft",
        teacher_module=args.teacher_module,
        fact_conf_threshold=args.fact_conf_threshold,
        cluster_top1_threshold=args.cluster_top1_threshold,
        cluster_margin_threshold=args.cluster_margin_threshold,
        notes=args.notes,
    )
    output_path = Path(args.output or f"pseudo_{args.round_id}.jsonl")

    with DBSession.connect() as session:
        upsert_round(session.conn, round_info)
        schemas = session.load_cluster_schemas()
        predicate_positions = session.load_predicate_positions()
        temporal_constraints = get_temporal_constraints(predicate_positions)
        runner = ProposeVerifyRunner.from_schemas(
            schemas,
            config=replace(
                NNConfig(),
                candidate_fact_threshold=args.candidate_fact_threshold,
            ),
            predicate_positions=predicate_positions,
            temporal_constraints=temporal_constraints,
        )

        case_ids = list(dict.fromkeys(args.case or list_case_ids_by_split(session.conn, args.split)))
        if not case_ids:
            console.print("[bold red]No cases selected.[/bold red]")
            raise SystemExit(1)

        fact_labels: list[PseudoFactLabel] = []
        cluster_labels: list[PseudoClusterLabel] = []
        for case_id in case_ids:
            try:
                entities, facts, _rules, states = session.load_case(case_id)
            except ValueError:
                console.print(f"[yellow]Skipping missing case[/yellow]: {case_id}")
                continue
            rules = session.load_rules(
                enabled_only=True,
                include_learned_modules=(
                    [args.teacher_module]
                    if args.teacher_module and args.teacher_module != "all"
                    else None
                ),
            )

            result = runner.run(entities, facts, rules, states)
            case_fact_labels, case_cluster_labels = _collect_case_pseudo_labels(
                round_id=args.round_id,
                case_id=case_id,
                gold_facts=facts,
                gold_states=states,
                result_facts=result.facts,
                candidate_feedback=result.candidate_feedback,
                result_states=result.cluster_states,
                schemas=schemas,
                fact_conf_threshold=args.fact_conf_threshold,
                cluster_top1_threshold=args.cluster_top1_threshold,
                cluster_margin_threshold=args.cluster_margin_threshold,
            )
            fact_labels.extend(case_fact_labels)
            cluster_labels.extend(case_cluster_labels)
            case_fact_stats = _pseudo_fact_label_stats(case_fact_labels)
            console.print(
                f"  [cyan]collected[/cyan] {case_id}: "
                f"facts={len(case_fact_labels)} "
                f"(T={case_fact_stats['t']} F={case_fact_stats['f']} U={case_fact_stats['u']}) "
                f"clusters={len(case_cluster_labels)}"
            )

        if not args.dry_run:
            save_pseudo_fact_labels(session.conn, fact_labels)
            save_pseudo_cluster_labels(session.conn, cluster_labels)
            set_round_status(session.conn, args.round_id, "collected")
            session.conn.commit()

    exported_round = round_info.model_copy(update={
        "status": "draft" if args.dry_run else "collected",
    })
    lines = [_serialize_round_record(exported_round)]
    lines.extend(_serialize_fact_label(label) for label in fact_labels)
    lines.extend(_serialize_cluster_label(label) for label in cluster_labels)
    output_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    fact_stats = _pseudo_fact_label_stats(fact_labels)
    console.print(
        "[dim]pseudo facts[/dim] "
        f"total={fact_stats['total']} "
        f"(T={fact_stats['t']} F={fact_stats['f']} U={fact_stats['u']})"
    )
    console.print(
        f"[bold green]collect-pseudo-labels completed[/bold green]: "
        f"round={args.round_id}, facts={len(fact_labels)}, clusters={len(cluster_labels)}, "
        f"file={output_path}"
    )


def cmd_import_pseudo_labels(args: argparse.Namespace) -> None:
    from db import DBSession

    input_path = Path(args.input)
    if not input_path.is_file():
        console.print(f"[bold red]Plik nie istnieje:[/bold red] {input_path}")
        raise SystemExit(1)

    round_info, fact_labels, cluster_labels = _parse_export_file(input_path)
    round_info = round_info.model_copy(update={"status": "imported"})

    with DBSession.connect() as session:
        upsert_round(session.conn, round_info)
        schemas = session.load_cluster_schemas()
        schema_by_name = {schema.name: schema for schema in schemas}

        accepted_facts: list[PseudoFactLabel] = []
        accepted_clusters: list[PseudoClusterLabel] = []
        rejected = 0
        case_ids = sorted({label.case_id for label in fact_labels} | {label.case_id for label in cluster_labels})

        gold_facts_by_case: dict[str, set[str]] = {}
        gold_states_by_case: dict[str, dict[tuple[str, str], tuple[ClusterStateRow, str]]] = {}
        existing_case_ids: set[str] = set()
        for case_id in case_ids:
            try:
                _entities, facts, _rules, states = session.load_case(case_id)
            except ValueError:
                gold_facts_by_case[case_id] = set()
                gold_states_by_case[case_id] = {}
                continue
            existing_case_ids.add(case_id)
            gold_facts_by_case[case_id] = {_fact_key(fact) for fact in facts}
            state_lookup: dict[tuple[str, str], tuple[ClusterStateRow, str]] = {}
            for state in states:
                schema = schema_by_name.get(state.cluster_name)
                if schema is None or not state.logits:
                    continue
                state_lookup[(state.entity_id, state.cluster_name)] = (state, _cluster_value(schema, state))
            gold_states_by_case[case_id] = state_lookup

        for label in fact_labels:
            if label.case_id not in existing_case_ids:
                rejected += 1
                continue
            if label.fact_key in gold_facts_by_case.get(label.case_id, set()):
                rejected += 1
                accepted_facts.append(label.model_copy(update={
                    "accepted": False,
                    "rejection_reason": "duplicates_gold_fact",
                }))
            else:
                accepted_facts.append(label)

        for label in cluster_labels:
            if label.case_id not in existing_case_ids:
                rejected += 1
                continue
            state_lookup = gold_states_by_case.get(label.case_id, {})
            existing = state_lookup.get((label.entity_id, label.cluster_name))
            if existing is not None:
                gold_state, gold_value = existing
                if gold_state.is_clamped and gold_state.clamp_hard and gold_state.clamp_source in {"text", "manual"}:
                    if gold_value != label.value:
                        rejected += 1
                        accepted_clusters.append(label.model_copy(update={
                            "accepted": False,
                            "rejection_reason": "conflicts_text_clamp",
                        }))
                        continue
                    rejected += 1
                    accepted_clusters.append(label.model_copy(update={
                        "accepted": False,
                        "rejection_reason": "duplicates_text_clamp",
                    }))
                    continue
            accepted_clusters.append(label)

        save_pseudo_fact_labels(session.conn, accepted_facts)
        save_pseudo_cluster_labels(session.conn, accepted_clusters)
        set_round_status(session.conn, round_info.round_id, "imported")
        session.conn.commit()

    imported_facts = sum(1 for label in accepted_facts if label.accepted)
    imported_clusters = sum(1 for label in accepted_clusters if label.accepted)
    console.print(
        f"[bold green]import-pseudo-labels completed[/bold green]: "
        f"round={round_info.round_id}, facts={imported_facts}, clusters={imported_clusters}, rejected={rejected}"
    )


def cmd_promote_round(args: argparse.Namespace) -> None:
    with connect() as conn:
        round_info = load_round(conn, args.round_id)
        if round_info is None:
            console.print(f"[bold red]Unknown round:[/bold red] {args.round_id}")
            raise SystemExit(1)
        promote_round(conn, args.round_id)
        conn.commit()
    console.print(f"[bold green]promote-round completed[/bold green]: {args.round_id}")


def cmd_learn_rules(args: argparse.Namespace) -> None:
    from db import DBSession
    from nn import (
        EntityMemoryBiasEncoder,
        ExceptionGateBank,
        GraphBuilder,
        HeteroMessagePassingBank,
        NeuralProposer,
        ProposerTrainer,
        RuleExtractionConfig,
        SVFeedbackProvider,
        TrainingCase,
        extract_rules_from_mp_bank,
        fact_cluster_rule_signature,
    )
    from nn.clamp import apply_clamp
    from nn.config import NNConfig
    from nn.graph_builder import EdgeTypeSpec, supports_relation

    with DBSession.connect() as session:
        schemas = session.load_cluster_schemas()
        predicate_positions = session.load_predicate_positions()

        gold_case_ids: list[str]
        if args.case:
            gold_case_ids = list(dict.fromkeys(args.case))
        elif args.label_source == "pseudo":
            gold_case_ids = []
        else:
            gold_case_ids = list_case_ids_by_split(session.conn, args.gold_split)

        use_promoted_pseudo = not args.pseudo_round
        pseudo_round_ids = list(args.pseudo_round or [])
        if args.label_source in {"gold+pseudo", "pseudo"}:
            pseudo_case_ids = list_cases_with_pseudo_labels(
                session.conn,
                round_ids=pseudo_round_ids if pseudo_round_ids else None,
                promoted_only=use_promoted_pseudo,
            )
        else:
            pseudo_case_ids = []

        if args.label_source == "gold":
            case_ids = gold_case_ids
        elif args.label_source == "pseudo":
            case_ids = pseudo_case_ids
        else:
            case_ids = list(dict.fromkeys(gold_case_ids + pseudo_case_ids))

        if not case_ids:
            console.print("[bold red]No training cases available.[/bold red]")
            raise SystemExit(1)

        config = replace(
            NNConfig(),
            max_epochs=args.epochs,
            lambda_fact_sup=args.fact_supervision_weight,
        )

        role_specs = [
            EdgeTypeSpec(
                src_type=f"c_{schema.name}",
                relation="role_of",
                dst_type="fact",
                src_dim=schema.dim,
                dst_dim=GraphBuilder.FACT_DIM,
            )
            for schema in schemas
        ]
        implies_specs = [
            EdgeTypeSpec(
                src_type=f"c_{src.name}",
                relation="implies",
                dst_type=f"c_{dst.name}",
                src_dim=src.dim,
                dst_dim=dst.dim,
            )
            for src in schemas
            for dst in schemas
            if src.name != dst.name and src.entity_type == dst.entity_type
        ]
        cluster_names = {schema.name.lower() for schema in schemas}
        supports_specs = [
            EdgeTypeSpec(
                src_type="fact",
                relation=supports_relation(predicate, role),
                dst_type=f"c_{dst.name}",
                src_dim=GraphBuilder.FACT_DIM,
                dst_dim=dst.dim,
            )
            for predicate, roles in sorted(predicate_positions.items())
            if predicate.lower() not in cluster_names
            if not predicate.lower().startswith("_sv_")
            if not predicate.lower().startswith("ab_")
            for role in roles
            for dst in schemas
        ]

        import torch as _torch

        _torch.manual_seed(args.seed)
        mp_bank = HeteroMessagePassingBank(role_specs + implies_specs + supports_specs)
        gate_bank = ExceptionGateBank(gate_specs=[])
        cluster_type_dims = {schema.name: schema.dim for schema in schemas}
        proposer = NeuralProposer(config, mp_bank, gate_bank, cluster_type_dims)
        graph_builder = GraphBuilder(schemas)
        memory_encoder = EntityMemoryBiasEncoder(schemas, config)
        from pipeline.temporal_config import get_temporal_constraints
        from sv import SymbolicVerifier

        temporal_constraints = get_temporal_constraints(predicate_positions)
        _verifier = SymbolicVerifier(
            cluster_schemas=schemas,
            predicate_positions=predicate_positions,
            temporal_constraints=temporal_constraints,
        )

        def _sv_provider(facts, rules, cluster_states):
            result = _verifier.verify(facts, rules, cluster_states)
            return result.candidate_feedback

        trainer = ProposerTrainer(
            proposer=proposer,
            cluster_schemas=schemas,
            config=config,
            seed=args.seed,
            sv_provider=_sv_provider if config.sv_feedback_in_training else None,
        )

        train_cases: list[TrainingCase] = []
        active_cluster_pairs: set[tuple[str, str]] = set()
        active_support_pairs: set[tuple[str, str, str]] = set()
        loaded_case_ids: list[str] = []
        fact_sup_labeled = 0.0
        fact_sup_t = 0.0
        fact_sup_f = 0.0
        fact_sup_u = 0.0
        fact_sup_weight_sum = 0.0
        fact_sup_cases = 0

        for case_id in case_ids:
            try:
                entities, facts, rules, states = session.load_case(
                    case_id,
                    include_non_observed=True,
                )
            except ValueError:
                console.print(f"[yellow]Skipping missing case[/yellow]: {case_id}")
                continue

            pseudo_cluster_keys: set[tuple[str, str]] = set()
            if args.label_source in {"gold+pseudo", "pseudo"}:
                entities, facts, states = _merge_pseudo_overlay(
                    session.conn,
                    case_id=case_id,
                    entities=entities,
                    facts=facts,
                    states=states,
                    schemas=schemas,
                    round_ids=pseudo_round_ids if pseudo_round_ids else None,
                    promoted_only=use_promoted_pseudo,
                )
                pseudo_cluster_keys = _build_pseudo_cluster_key_set(
                    session.conn,
                    case_id=case_id,
                    round_ids=pseudo_round_ids if pseudo_round_ids else None,
                    promoted_only=use_promoted_pseudo,
                )

            data, node_index, _ = graph_builder.build(
                entities=entities,
                facts=facts,
                rules=[],
                cluster_states=states,
                memory_biases=None,
            )
            _attach_mask_weights(
                data,
                node_index,
                schemas,
                pseudo_cluster_keys,
                args.pseudo_cluster_weight,
            )
            _attach_fact_supervision(
                data,
                node_index,
                facts,
                args.pseudo_fact_weight,
            )
            fact_stats = _fact_supervision_stats(data)
            fact_sup_labeled += fact_stats["labeled"]
            fact_sup_t += fact_stats["t"]
            fact_sup_f += fact_stats["f"]
            fact_sup_u += fact_stats["u"]
            fact_sup_weight_sum += fact_stats["weight_sum"]
            if fact_stats["labeled"] > 0:
                fact_sup_cases += 1

            active_cluster_pairs |= _add_template_cluster_edges(data, node_index, schemas)
            active_support_pairs |= _add_template_fact_cluster_edges(data, node_index, facts, schemas)

            memory_biases = memory_encoder.compute_memory_bias(entities, node_index)
            for schema in schemas:
                node_type = f"c_{schema.name}"
                if schema.name in memory_biases:
                    data[node_type].memory_bias = memory_biases[schema.name]

            for node_type in data.node_types:
                x = data[node_type].x
                is_clamped = data[node_type].get("is_clamped")
                clamp_hard = data[node_type].get("clamp_hard")
                logits_out, _ = apply_clamp(x, is_clamped, clamp_hard, config)
                data[node_type].x = logits_out

            train_cases.append(TrainingCase(
                data=data,
                node_index=node_index,
                facts=facts,
                rules=rules,
                cluster_states=states,
            ))
            loaded_case_ids.append(case_id)

        if not train_cases:
            console.print("[bold red]No training cases available after overlay.[/bold red]")
            raise SystemExit(1)

        console.print(
            "[dim]fact supervision[/dim] "
            f"labeled={int(fact_sup_labeled)} "
            f"(T={int(fact_sup_t)} F={int(fact_sup_f)} U={int(fact_sup_u)}) "
            f"cases={fact_sup_cases}/{len(train_cases)} "
            f"weight_sum={fact_sup_weight_sum:.2f}"
        )

        epoch_metrics: list[dict[str, float]] = []
        for metrics in trainer.train_epochs(train_cases):
            epoch_metrics.append(metrics)
            if len(epoch_metrics) == len(train_cases):
                epoch = int(metrics.get("epoch", 0))
                total = sum(float(item.get("L_total", 0.0)) for item in epoch_metrics) / len(epoch_metrics)
                l_mask = sum(float(item.get("L_mask", 0.0)) for item in epoch_metrics) / len(epoch_metrics)
                l_fact = sum(float(item.get("L_fact_sup", 0.0)) for item in epoch_metrics) / len(epoch_metrics)
                l_sv = sum(float(item.get("L_sv_feedback", 0.0)) for item in epoch_metrics) / len(epoch_metrics)
                console.print(
                    f"[dim]epoch {epoch + 1}/{args.epochs}[/dim] "
                    f"L_total={total:.4f} "
                    f"L_mask={l_mask:.4f} "
                    f"L_fact_sup={l_fact:.4f} "
                    f"L_sv_feedback={l_sv:.4f}"
                )
                epoch_metrics = []

        extracted = extract_rules_from_mp_bank(
            mp_bank=proposer.mp_bank,
            cluster_schemas=schemas,
            config=RuleExtractionConfig(
                min_weight=args.min_weight,
                top_k_per_source_value=args.top_k,
                rule_id_prefix=args.rule_prefix,
            ),
            predicate_positions=predicate_positions,
        )

        filtered = [
            rule
            for rule in extracted
            if _keep_active_learned_rule(
                rule,
                schemas,
                active_cluster_pairs,
                active_support_pairs,
                fact_cluster_rule_signature,
            )
        ]

        if args.dry_run:
            console.print(
                f"[bold yellow]learn-rules dry-run[/bold yellow]: "
                f"cases={len(loaded_case_ids)}, extracted={len(filtered)}"
            )
            _print_extracted_rules_preview(
                filtered,
                title="Extracted Learned Rules (preview)",
            )
            return

        session.save_learned_rules(filtered, module_name=args.module)

    console.print(
        f"[bold green]learn-rules completed[/bold green]: "
        f"cases={len(loaded_case_ids)}, saved={len(filtered)}, module={args.module}"
    )
    _print_extracted_rules_preview(
        filtered,
        title="Extracted Learned Rules",
    )


def cmd_eval_round(args: argparse.Namespace) -> None:
    from db import DBSession
    from pipeline.runner import ProposeVerifyRunner

    with DBSession.connect() as session:
        schemas = session.load_cluster_schemas()
        predicate_positions = session.load_predicate_positions()
        runner = ProposeVerifyRunner.from_schemas(
            schemas,
            predicate_positions=predicate_positions,
        )
        rules = session.load_rules(
            enabled_only=True,
            include_learned_modules=[args.module] if args.module else None,
        )

        case_ids = list(dict.fromkeys(args.case or list_case_ids_by_split(session.conn, args.split)))
        if not case_ids:
            console.print("[bold red]No evaluation cases selected.[/bold red]")
            raise SystemExit(1)

        query_map = _load_eval_queries(session.conn, case_ids)
        missing_queries = [case_id for case_id in case_ids if not query_map.get(case_id)]
        if missing_queries:
            console.print(
                "[bold red]eval-round failed[/bold red]: "
                f"brak case_queries dla {len(missing_queries)} case(s): {', '.join(missing_queries[:10])}"
            )
            raise SystemExit(1)

        total_queries = 0
        correct_queries = 0
        per_case: list[dict[str, object]] = []
        details: list[dict[str, object]] = []

        for case_id in case_ids:
            try:
                entities, facts, _rules, states = session.load_case(case_id)
            except ValueError:
                console.print(f"[yellow]Skipping missing case[/yellow]: {case_id}")
                continue

            nn_facts, nn_states = runner.nn_inference.propose(entities, facts, rules, states)
            sv_result = runner.verifier.verify(nn_facts, rules, nn_states)
            queries = query_map[case_id]

            case_total = 0
            case_correct = 0
            for query_id, query_text, expected in queries:
                expected_str = str(expected)
                query_atom = _parse_query_atom(str(query_text), predicate_positions)
                got = runner.verifier.classify_query_atom(query_atom, sv_result, rules)
                ok = got == expected_str
                total_queries += 1
                case_total += 1
                if ok:
                    correct_queries += 1
                    case_correct += 1
                details.append({
                    "case_id": case_id,
                    "query_id": int(query_id),
                    "query": str(query_text),
                    "expected": expected_str,
                    "got": got,
                    "correct": ok,
                })

            accuracy = (case_correct / case_total) if case_total else None
            per_case.append({
                "case_id": case_id,
                "queries": case_total,
                "correct": case_correct,
                "accuracy": accuracy,
            })
            console.print(
                f"  [cyan]eval[/cyan] {case_id}: "
                f"{case_correct}/{case_total} correct"
            )

    if total_queries == 0:
        console.print("[bold red]eval-round failed[/bold red]: brak zapytań do oceny.")
        raise SystemExit(1)

    accuracy = (correct_queries / total_queries) if total_queries else 0.0
    confusion, per_label_metrics, metric_summary = _build_eval_metrics(details)
    report = {
        "split": args.split,
        "module": args.module,
        "cases": len(per_case),
        "queries": total_queries,
        "correct": correct_queries,
        "accuracy": accuracy,
        "labels": list(_EVAL_LABELS),
        "confusion_matrix": confusion,
        "per_label": per_label_metrics,
        **metric_summary,
        "per_case": per_case,
        "details": details if args.include_details else None,
    }
    if args.output_json:
        Path(args.output_json).write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    console.print(
        f"[bold green]eval-round completed[/bold green]: "
        f"accuracy={accuracy:.4f}, macro_f1={metric_summary['macro_f1']:.4f}, "
        f"correct={correct_queries}/{total_queries}"
    )


def _add_template_cluster_edges(data, node_index, schemas) -> set[tuple[str, str]]:
    import torch

    active_pairs: set[tuple[str, str]] = set()
    for src in schemas:
        src_type = f"c_{src.name}"
        src_map = node_index.cluster_node_to_idx.get(src.name, {})
        if not src_map:
            continue

        for dst in schemas:
            if src.name == dst.name or src.entity_type != dst.entity_type:
                continue
            dst_map = node_index.cluster_node_to_idx.get(dst.name, {})
            if not dst_map:
                continue

            pairs = sorted(
                (src_idx, dst_map[eid])
                for eid, src_idx in src_map.items()
                if eid in dst_map
            )
            if not pairs:
                pairs = sorted(set(
                    (si, di)
                    for si in src_map.values()
                    for di in dst_map.values()
                ))
            if not pairs:
                continue

            uniq_pairs = sorted(set(pairs))
            src_idx = [pair[0] for pair in uniq_pairs]
            dst_idx = [pair[1] for pair in uniq_pairs]
            data[src_type, "implies", f"c_{dst.name}"].edge_index = torch.tensor(
                [src_idx, dst_idx],
                dtype=torch.long,
            )
            active_pairs.add((src.name, dst.name))
    return active_pairs


def _add_template_fact_cluster_edges(data, node_index, facts, schemas) -> set[tuple[str, str, str]]:
    import torch

    edge_bucket: dict[tuple[str, str], set[tuple[int, int]]] = {}
    active_pairs: set[tuple[str, str, str]] = set()

    for fact in facts:
        fact_idx = node_index.fact_node_to_idx.get(fact.fact_id)
        if fact_idx is None:
            continue
        predicate = fact.predicate.lower()
        for arg in fact.args:
            if arg.entity_id is None:
                continue
            role = arg.role.upper()
            entity_id = arg.entity_id
            for schema in schemas:
                dst_map = node_index.cluster_node_to_idx.get(schema.name, {})
                dst_idx = dst_map.get(entity_id)
                if dst_idx is None:
                    continue
                relation = f"supports:{predicate}:{role}"
                edge_bucket.setdefault((relation, schema.name), set()).add((fact_idx, dst_idx))
                active_pairs.add((predicate, role, schema.name))

    for (relation, cluster_name), pairs in edge_bucket.items():
        ordered = sorted(pairs)
        if not ordered:
            continue
        src_idx = [src for src, _ in ordered]
        dst_idx = [dst for _, dst in ordered]
        data["fact", relation, f"c_{cluster_name}"].edge_index = torch.tensor(
            [src_idx, dst_idx],
            dtype=torch.long,
        )

    return active_pairs


def _keep_active_learned_rule(
    rule,
    schemas,
    active_cluster_pairs,
    active_support_pairs,
    fact_cluster_rule_signature,
) -> bool:
    if not rule.body:
        return False
    schema_names = {schema.name for schema in schemas}
    body_predicate = rule.body[0].predicate
    if body_predicate in schema_names:
        return (body_predicate, rule.head.predicate) in active_cluster_pairs

    signature = fact_cluster_rule_signature(rule, schemas)
    return signature in active_support_pairs


def main() -> None:
    parser = argparse.ArgumentParser(prog="pn3train", description="Self-training CLI for ProveNuance3")
    sub = parser.add_subparsers(dest="command")
    sub.required = True

    cmd = sub.add_parser("set-split", help="Assign cases to a dataset split")
    cmd.add_argument("split", choices=["train_gold", "train_unlabeled", "holdout"])
    target = cmd.add_mutually_exclusive_group(required=True)
    target.add_argument("--case", action="append", help="Case ID to update (repeatable)")
    target.add_argument("--pattern", help="SQL LIKE pattern for case_id, e.g. TC-%")
    target.add_argument("--all", action="store_true", help="Update all cases")
    cmd.set_defaults(func=cmd_set_split)

    cmd = sub.add_parser("export-case-query-template", help="Export CSV/JSONL template for manual case query annotation")
    target = cmd.add_mutually_exclusive_group(required=True)
    target.add_argument("--case", action="append", help="Case ID to export (repeatable)")
    target.add_argument("--split", choices=["train_gold", "train_unlabeled", "holdout"], help="Export all cases from a split")
    target.add_argument("--all", action="store_true", help="Export all cases")
    cmd.add_argument("--output", required=True, help="Template path (.csv or .jsonl)")
    cmd.add_argument("--include-content", action="store_true", help="Include full source_content in the template")
    cmd.set_defaults(func=cmd_export_case_query_template)

    cmd = sub.add_parser("import-case-queries", help="Import gold case queries from CSV/JSONL into current DB")
    cmd.add_argument("input", help="Path to .csv or .jsonl file with case queries")
    cmd.add_argument("--replace", action="store_true", help="Replace existing queries for affected cases before import")
    cmd.add_argument("--dry-run", action="store_true")
    cmd.set_defaults(func=cmd_import_case_queries)

    cmd = sub.add_parser("draft-case-queries", help="Draft reviewable case queries with LLM and export JSONL")
    target = cmd.add_mutually_exclusive_group(required=True)
    target.add_argument("--case", action="append", help="Case ID to draft (repeatable)")
    target.add_argument("--split", choices=["train_gold", "train_unlabeled", "holdout"], help="Draft for all cases in a split")
    target.add_argument("--all", action="store_true", help="Draft for all cases")
    cmd.add_argument("--output", required=True, help="Output JSONL path")
    cmd.add_argument("--max-queries", type=int, default=3, help="Maximum number of drafted queries per case")
    cmd.add_argument("--include-existing", action="store_true", help="Also draft for cases that already have case_queries")
    cmd.add_argument("--continue-on-error", action="store_true", help="Keep drafting remaining cases after an LLM/API error")
    cmd.set_defaults(func=cmd_draft_case_queries)

    cmd = sub.add_parser("collect-pseudo-labels", help="Run teacher inference and export pseudo-labels")
    cmd.add_argument("round_id", help="Round identifier, e.g. R1")
    cmd.add_argument("--split", choices=["train_gold", "train_unlabeled", "holdout"], default="train_unlabeled")
    cmd.add_argument("--case", action="append", help="Explicit case ID (repeatable)")
    cmd.add_argument("--parent-round", default=None, help="Parent round ID")
    cmd.add_argument("--teacher-module", default="learned_nn", help="Teacher rule module label")
    cmd.add_argument(
        "--candidate-fact-threshold",
        type=float,
        default=0.55,
        help="Teacher NN top-1 threshold for proposing candidate facts before verifier blocking",
    )
    cmd.add_argument("--fact-conf-threshold", type=float, default=0.95)
    cmd.add_argument("--cluster-top1-threshold", type=float, default=0.95)
    cmd.add_argument("--cluster-margin-threshold", type=float, default=0.80)
    cmd.add_argument("--notes", default=None)
    cmd.add_argument("--output", default=None, help="Export JSONL path")
    cmd.add_argument("--dry-run", action="store_true")
    cmd.set_defaults(func=cmd_collect_pseudo_labels)

    cmd = sub.add_parser("import-pseudo-labels", help="Import pseudo-labels JSONL into current DB")
    cmd.add_argument("input", help="Path to exported pseudo-label JSONL")
    cmd.set_defaults(func=cmd_import_pseudo_labels)

    cmd = sub.add_parser("promote-round", help="Mark a round as promoted for future training")
    cmd.add_argument("round_id", help="Round identifier")
    cmd.set_defaults(func=cmd_promote_round)

    cmd = sub.add_parser("learn-rules", help="Train learned rules on gold and/or pseudo overlays")
    cmd.add_argument("--gold-split", choices=["train_gold", "train_unlabeled", "holdout"], default="train_gold")
    cmd.add_argument("--case", action="append", help="Explicit gold case ID (repeatable)")
    cmd.add_argument("--label-source", choices=["gold", "gold+pseudo", "pseudo"], default="gold+pseudo")
    cmd.add_argument("--pseudo-round", action="append", help="Pseudo-label round to use (repeatable). Default: promoted rounds")
    cmd.add_argument("--epochs", type=int, default=20)
    cmd.add_argument("--min-weight", type=float, default=0.5)
    cmd.add_argument("--top-k", type=int, default=2)
    cmd.add_argument("--module", default="learned_nn")
    cmd.add_argument("--rule-prefix", default="learned.nn")
    cmd.add_argument("--seed", type=int, default=42)
    cmd.add_argument("--pseudo-cluster-weight", type=float, default=0.35)
    cmd.add_argument("--pseudo-fact-weight", type=float, default=1.0)
    cmd.add_argument("--fact-supervision-weight", type=float, default=1.0)
    cmd.add_argument("--dry-run", action="store_true")
    cmd.set_defaults(func=cmd_learn_rules)

    cmd = sub.add_parser("eval-round", help="Evaluate a learned module on a holdout split")
    cmd.add_argument("--split", choices=["train_gold", "train_unlabeled", "holdout"], default="holdout")
    cmd.add_argument("--case", action="append", help="Explicit case ID (repeatable)")
    cmd.add_argument("--module", default=None, help="Learned rule module to include; default uses all enabled rules")
    cmd.add_argument("--output-json", default=None, help="Optional JSON report path")
    cmd.add_argument("--include-details", action="store_true")
    cmd.set_defaults(func=cmd_eval_round)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
