#!/usr/bin/env python3
"""pn3 – ProveNuance3 command-line interface."""

import argparse
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
from dataclasses import replace

import psycopg2
from rich.console import Console
from rich.table import Table
from rich.text import Text
from runtime_env import load_project_env

# Force UTF-8 output so Rich unicode symbols work on Windows legacy consoles.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

console = Console()
load_project_env()

DSN = dict(
    host=os.getenv("PN3_HOST", "localhost"),
    port=int(os.getenv("PN3_PORT", "5432")),
    dbname=os.getenv("PN3_DB", "provenuance"),
    user=os.getenv("PN3_USER", "provenuance"),
    password=os.getenv("PN3_PASSWORD", "provenuance"),
)

_QUERY_RE = re.compile(r"^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\((.*)\)\s*$")


def connect() -> psycopg2.extensions.connection:
    return psycopg2.connect(**DSN)


# ---------------------------------------------------------------------------
# entities
# ---------------------------------------------------------------------------

def cmd_entities(_args: argparse.Namespace) -> None:
    with connect() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT
                e.entity_id,
                et.name            AS entity_type,
                e.canonical_name,
                array_length(e.blocking_keys, 1)  AS n_blocking,
                e.embedding IS NOT NULL            AS has_emb,
                e.created_at
            FROM entities e
            JOIN entity_types et ON et.id = e.entity_type_id
            ORDER BY e.id
        """)
        rows = cur.fetchall()

    table = Table(title="Entities", header_style="bold cyan", show_lines=False)
    table.add_column("entity_id",     style="bold yellow", no_wrap=True)
    table.add_column("type",          style="cyan")
    table.add_column("canonical_name")
    table.add_column("blocking_keys", justify="right", style="dim")
    table.add_column("embedding",     justify="center")
    table.add_column("created_at",    style="dim", no_wrap=True)

    for entity_id, etype, name, n_blocking, has_emb, created_at in rows:
        emb_cell = Text("✓", style="green") if has_emb else Text("–", style="dim")
        table.add_row(
            entity_id,
            etype,
            name,
            str(n_blocking) if n_blocking is not None else "0",
            emb_cell,
            str(created_at)[:19],
        )

    console.print(table)
    console.print(f"[dim]{len(rows)} row(s)[/dim]")


# ---------------------------------------------------------------------------
# facts
# ---------------------------------------------------------------------------

_STATUS_STYLE: dict[str, str] = {
    "observed":            "blue",
    "inferred_candidate":  "yellow",
    "proved":              "bold green",
    "rejected":            "bold red",
    "retracted":           "dim",
}


def cmd_facts(_args: argparse.Namespace) -> None:
    with connect() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT
                c.case_id,
                f.fact_id,
                f.predicate,
                f.arity,
                f.status,
                f.truth_value,
                f.truth_confidence,
                (
                    SELECT COUNT(*)
                    FROM fact_args fa
                    WHERE fa.fact_id = f.id
                ) AS n_args,
                (
                    SELECT string_agg(
                        fa.role || '=' || COALESCE(fa.entity_id, fa.literal_value),
                        ', '
                        ORDER BY fa.position
                    )
                    FROM fact_args fa
                    WHERE fa.fact_id = f.id
                ) AS args_list,
                f.source_id,
                f.source_extractor,
                f.source_spans,
                f.proof_id,
                (
                    SELECT pr.result
                    FROM proof_runs pr
                    WHERE pr.proof_id = f.proof_id
                    ORDER BY pr.id DESC
                    LIMIT 1
                ) AS proof_value,
                (
                    SELECT COUNT(*)
                    FROM fact_neural_trace fnt
                    WHERE fnt.fact_id = f.id
                ) AS n_trace,
                f.created_at,
                s.content AS source_content
            FROM facts f
            LEFT JOIN cases c ON c.id = f.case_id
            LEFT JOIN sources s ON s.source_id = f.source_id
            ORDER BY f.id
        """)
        rows = cur.fetchall()

    console.print("[bold cyan]Facts[/bold cyan]")
    if not rows:
        console.print("[dim]0 row(s)[/dim]")
        return

    for i, (
        case_id, fact_id, predicate, arity, status, truth_val, conf,
        n_args, args_list, source_id, source_extractor, source_spans_raw,
        proof_id, proof_value, n_trace, created_at, source_content,
    ) in enumerate(rows, start=1):
        s_style = _STATUS_STYLE.get(status, "")
        status_text = Text(status, style=s_style)
        conf_text = f"{conf:.2f}" if conf is not None else "-"

        header = Text(f"{i}. ", style="dim")
        header.append(fact_id, style="bold yellow")
        header.append(" ")
        header.append(predicate, style="cyan")
        console.print(header)
        console.print("  case:", case_id or "-", " status:", status_text, f" truth: {truth_val or '-'} ({conf_text})")
        console.print(
            "  arity:",
            str(arity) if arity is not None else "-",
            f" args: {n_args} [{args_list or '-'}] source: {source_id or '-'} extractor: {source_extractor or '-'}",
        )
        console.print(
            f"  proof: {proof_value or '-'} proof_id: {proof_id or '-'} "
            f"trace: {n_trace} created_at: {str(created_at)[:19]}"
        )
        # Pokaż fragment tekstu źródłowego jeśli dostępny
        span_text = _extract_span_text(source_content, source_spans_raw)
        if span_text:
            console.print(f"  [dim]source text:[/dim] [italic]{span_text}[/italic]")
        if i < len(rows):
            console.print()

    console.print(f"[dim]{len(rows)} row(s)[/dim]")


# ---------------------------------------------------------------------------
# rules
# ---------------------------------------------------------------------------

def cmd_rules(_args: argparse.Namespace) -> None:
    with connect() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT
                r.rule_id,
                rm.name        AS module,
                r.stratum,
                r.enabled,
                r.learned,
                r.weight,
                r.precision_est,
                r.support,
                r.created_at,
                r.source_span_text
            FROM rules r
            JOIN rule_modules rm ON rm.id = r.module_id
            ORDER BY r.stratum, r.id
        """)
        rows = cur.fetchall()

    console.print("[bold cyan]Rules[/bold cyan]")
    if not rows:
        console.print("[dim]0 row(s)[/dim]")
        return

    for i, (
        rule_id, module, stratum, enabled, learned, weight, precision, support, created_at,
        source_span_text,
    ) in enumerate(rows, start=1):
        header = Text(f"{i}. ", style="dim")
        header.append(rule_id, style="bold yellow")
        console.print(header)
        console.print(
            f"  module: {module} stratum: {stratum} "
            f"enabled: {'yes' if enabled else 'no'} "
            f"learned: {'yes' if learned else 'no'}"
        )
        console.print(
            f"  weight: {f'{weight:.3f}' if weight is not None else '-'} "
            f"precision: {f'{precision:.2f}' if precision is not None else '-'} "
            f"support: {str(support) if support is not None else '-'} "
            f"created_at: {str(created_at)[:19]}"
        )
        if source_span_text:
            console.print(f"  [dim]source text:[/dim] [italic]{source_span_text}[/italic]")
        if i < len(rows):
            console.print()

    console.print(f"[dim]{len(rows)} row(s)[/dim]")


# ---------------------------------------------------------------------------
# entity-types
# ---------------------------------------------------------------------------

def cmd_entity_types(_args: argparse.Namespace) -> None:
    with connect() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT et.id, et.name, et.description,
                   COUNT(e.id) AS n_entities
            FROM entity_types et
            LEFT JOIN entities e ON e.entity_type_id = et.id
            GROUP BY et.id
            ORDER BY et.id
        """)
        rows = cur.fetchall()

    table = Table(title="Entity Types", header_style="bold cyan", show_lines=False)
    table.add_column("id",          justify="right", style="dim", width=4)
    table.add_column("name",        style="bold yellow", no_wrap=True)
    table.add_column("description", style="dim")
    table.add_column("entities",    justify="right")

    for row_id, name, desc, n_ent in rows:
        table.add_row(str(row_id), name, desc or "–", str(n_ent))

    console.print(table)
    console.print(f"[dim]{len(rows)} row(s)[/dim]")


# ---------------------------------------------------------------------------
# predicates  (definitions + roles joined)
# ---------------------------------------------------------------------------

def cmd_predicates(_args: argparse.Namespace) -> None:
    with connect() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT
                pd.id,
                pd.name,
                pd.description,
                string_agg(
                    pr.position::text || ':' || pr.role_name ||
                    CASE WHEN et.name IS NOT NULL THEN '(' || et.name || ')' ELSE '' END,
                    '  ' ORDER BY pr.position
                ) AS roles
            FROM predicate_definitions pd
            LEFT JOIN predicate_roles pr ON pr.predicate_id = pd.id
            LEFT JOIN entity_types et    ON et.id = pr.entity_type_id
            GROUP BY pd.id
            ORDER BY pd.id
        """)
        rows = cur.fetchall()

    table = Table(title="Predicates", header_style="bold cyan", show_lines=False)
    table.add_column("id",          justify="right", style="dim", width=4)
    table.add_column("name",        style="bold yellow", no_wrap=True)
    table.add_column("description", style="dim")
    table.add_column("roles",       style="cyan")

    for row_id, name, desc, roles in rows:
        table.add_row(str(row_id), name, desc or "–", roles or "–")

    console.print(table)
    console.print(f"[dim]{len(rows)} row(s)[/dim]")


# ---------------------------------------------------------------------------
# clusters  (definitions + domain values joined)
# ---------------------------------------------------------------------------

def cmd_clusters(_args: argparse.Namespace) -> None:
    with connect() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT
                cd.id,
                cd.name,
                et.name                          AS entity_type,
                cd.description,
                string_agg(cdv.value, ' | ' ORDER BY cdv.position) AS domain
            FROM cluster_definitions cd
            JOIN entity_types et ON et.id = cd.entity_type_id
            LEFT JOIN cluster_domain_values cdv ON cdv.cluster_id = cd.id
            GROUP BY cd.id, et.name
            ORDER BY et.name, cd.name
        """)
        rows = cur.fetchall()

    table = Table(title="Clusters", header_style="bold cyan", show_lines=False)
    table.add_column("id",          justify="right", style="dim", width=4)
    table.add_column("name",        style="bold yellow", no_wrap=True)
    table.add_column("entity_type", style="cyan", no_wrap=True)
    table.add_column("description", style="dim")
    table.add_column("domain")

    for row_id, name, etype, desc, domain in rows:
        table.add_row(str(row_id), name, etype, desc or "–", domain or "–")

    console.print(table)
    console.print(f"[dim]{len(rows)} row(s)[/dim]")


# ---------------------------------------------------------------------------
# rule-modules
# ---------------------------------------------------------------------------

def cmd_rule_modules(_args: argparse.Namespace) -> None:
    with connect() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT
                rm.id,
                rm.name,
                rm.description,
                COUNT(r.id)                               AS n_rules,
                COUNT(r.id) FILTER (WHERE r.enabled)      AS n_enabled,
                COUNT(r.id) FILTER (WHERE r.learned)      AS n_learned
            FROM rule_modules rm
            LEFT JOIN rules r ON r.module_id = rm.id
            GROUP BY rm.id
            ORDER BY rm.id
        """)
        rows = cur.fetchall()

    table = Table(title="Rule Modules", header_style="bold cyan", show_lines=False)
    table.add_column("id",          justify="right", style="dim", width=4)
    table.add_column("name",        style="bold yellow", no_wrap=True)
    table.add_column("description", style="dim")
    table.add_column("rules",       justify="right")
    table.add_column("enabled",     justify="right")
    table.add_column("learned",     justify="right")

    for row_id, name, desc, n_rules, n_enabled, n_learned in rows:
        table.add_row(
            str(row_id), name, desc or "–",
            str(n_rules), str(n_enabled), str(n_learned),
        )

    console.print(table)
    console.print(f"[dim]{len(rows)} row(s)[/dim]")


# ---------------------------------------------------------------------------
# sources
# ---------------------------------------------------------------------------

def cmd_sources(_args: argparse.Namespace) -> None:
    with connect() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT
                s.source_id,
                s.source_type,
                s.source_rank,
                s.title,
                length(s.content)  AS content_len,
                s.created_at
            FROM sources s
            ORDER BY s.id
        """)
        rows = cur.fetchall()

    table = Table(title="Sources", header_style="bold cyan", show_lines=False)
    table.add_column("source_id",   style="bold yellow", no_wrap=True)
    table.add_column("type",        style="cyan", no_wrap=True)
    table.add_column("rank",        justify="right", style="dim")
    table.add_column("title")
    table.add_column("content_len", justify="right", style="dim")
    table.add_column("created_at",  style="dim", no_wrap=True)

    for source_id, stype, rank, title, clen, created_at in rows:
        table.add_row(
            source_id,
            stype or "–",
            str(rank),
            title or "–",
            str(clen) if clen is not None else "–",
            str(created_at)[:19],
        )

    console.print(table)
    console.print(f"[dim]{len(rows)} row(s)[/dim]")


# ---------------------------------------------------------------------------
# cases  (cases + case_queries joined)
# ---------------------------------------------------------------------------

def cmd_cases(_args: argparse.Namespace) -> None:
    with connect() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT
                c.case_id,
                s.source_id,
                c.title,
                c.created_at,
                cq.query,
                cq.expected_result,
                cq.notes
            FROM cases c
            JOIN sources s ON s.id = c.source_id
            LEFT JOIN case_queries cq ON cq.case_id = c.id
            ORDER BY c.id, cq.id
        """)
        rows = cur.fetchall()

    table = Table(title="Cases & Queries", header_style="bold cyan", show_lines=True)
    table.add_column("case_id",    style="bold yellow", no_wrap=True)
    table.add_column("source",     style="dim", no_wrap=True)
    table.add_column("title",      style="dim")
    table.add_column("query",      style="cyan")
    table.add_column("expected",   justify="center")
    table.add_column("notes",      style="dim")

    _RESULT_STYLE = {
        "proved":      "bold green",
        "not_proved":  "bold red",
        "blocked":     "yellow",
        "unknown":     "dim",
    }

    seen: set[str] = set()
    for case_id, source_id, title, created_at, query, expected, notes in rows:
        case_cell  = case_id  if case_id  not in seen else ""
        source_cell = source_id if case_id not in seen else ""
        title_cell = title    if case_id  not in seen else ""
        seen.add(case_id)
        exp_style = _RESULT_STYLE.get(expected or "", "")
        exp_text  = Text(expected or "–", style=exp_style)
        table.add_row(
            case_cell, source_cell, title_cell,
            query or "–", exp_text, notes or "–",
        )

    n_cases   = len({r[0] for r in rows})
    n_queries = sum(1 for r in rows if r[4])
    console.print(table)
    console.print(f"[dim]{n_cases} case(s), {n_queries} query/queries[/dim]")


# ---------------------------------------------------------------------------
# proof
# ---------------------------------------------------------------------------

def _coerce_json(value):
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return None
    return None


def _extract_span_text(content: str | None, spans_raw) -> str | None:
    """Wyciąga fragment tekstu ze źródła używając pierwszego spanu [{start,end}]."""
    if not content or not spans_raw:
        return None
    spans = spans_raw if isinstance(spans_raw, list) else _coerce_json(spans_raw)
    if not spans:
        return None
    first = spans[0]
    if not isinstance(first, dict):
        return None
    start = first.get("start")
    end = first.get("end")
    if start is None or end is None:
        return None
    text = content[start:end].strip()
    return text if text else None


def _dot_escape(value: str) -> str:
    # Keep backslash escapes like "\n" intact for Graphviz line breaks.
    return (
        value
        .replace('"', '\\"')
        .replace("\r\n", "\n")
        .replace("\r", "\n")
        .replace("\n", "\\n")
    )


def _build_proof_dot(
    proof_id: str,
    case_id: str,
    query: str,
    result: str,
    dag_map: dict,
) -> str:
    node_ids: dict[str, str] = {}
    node_order: list[str] = []

    def get_node_id(atom: str) -> str:
        if atom not in node_ids:
            node_ids[atom] = f"n{len(node_ids)}"
            node_order.append(atom)
        return node_ids[atom]

    # Register all nodes first (including body/naf leaves not present in dag keys).
    for atom, node in dag_map.items():
        get_node_id(str(atom))
        if isinstance(node, dict):
            for dep in node.get("body_atoms", []) or []:
                get_node_id(str(dep))
            for dep in node.get("naf_atoms", []) or []:
                get_node_id(str(dep))

    lines: list[str] = [
        "digraph ProofDAG {",
        "  rankdir=LR;",
        '  graph [fontname="Helvetica", fontsize=11, labeljust="l", labelloc="t"];',
        '  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10, color="#666666"];',
        '  edge [fontname="Helvetica", fontsize=9, color="#444444"];',
        (
            f'  label="Proof DAG\\nproof_id={_dot_escape(proof_id)}'
            f'\\ncase={_dot_escape(case_id)}'
            f'\\nquery={_dot_escape(query)}'
            f'\\nresult={_dot_escape(result)}";'
        ),
        "",
    ]

    for atom in node_order:
        node = dag_map.get(atom, {})
        status = str(node.get("status", "external")) if isinstance(node, dict) else "external"
        rule_id = str(node.get("rule_id") or "-") if isinstance(node, dict) else "-"

        if status == "derived":
            fill = "#E8F7E8"
        elif status == "base":
            fill = "#EFEFEF"
        else:
            fill = "#FFF8D9"

        label = f"{atom}\\nstatus={status}\\nrule={rule_id}"
        lines.append(
            f'  {get_node_id(atom)} [label="{_dot_escape(label)}", fillcolor="{fill}"];'
        )

    lines.append("")
    for atom, node in dag_map.items():
        if not isinstance(node, dict):
            continue
        dst = get_node_id(str(atom))
        for dep in node.get("body_atoms", []) or []:
            src = get_node_id(str(dep))
            lines.append(f"  {src} -> {dst};")
        for dep in node.get("naf_atoms", []) or []:
            src = get_node_id(str(dep))
            lines.append(f'  {src} -> {dst} [style="dashed", color="#AA3333", label="not"];')

    lines.append("}")
    return "\n".join(lines) + "\n"


def _resolve_graphviz_engine(engine: str) -> str | None:
    # 1) PATH lookup by command name (or explicit executable name).
    resolved = shutil.which(engine)
    if resolved:
        return resolved

    # 2) Explicit path supplied by user.
    engine_path = Path(engine)
    if engine_path.exists():
        return str(engine_path)

    # 3) Environment hints.
    for env_name in ("GRAPHVIZ_DOT", "DOT_BINARY"):
        env_path = os.getenv(env_name)
        if env_path and Path(env_path).exists():
            return env_path

    # 4) Typical Windows install paths.
    roots = [
        os.getenv("ProgramFiles"),
        os.getenv("ProgramFiles(x86)"),
        os.getenv("ChocolateyInstall"),
    ]
    candidates: list[Path] = []
    for root in roots:
        if not root:
            continue
        root_path = Path(root)
        candidates.extend([
            root_path / "Graphviz" / "bin" / f"{engine}.exe",
            root_path / "bin" / f"{engine}.exe",
        ])

    # Fallback hardcoded common locations.
    candidates.extend([
        Path(r"C:\Program Files\Graphviz\bin") / f"{engine}.exe",
        Path(r"C:\Program Files (x86)\Graphviz\bin") / f"{engine}.exe",
        Path(r"C:\ProgramData\chocolatey\bin") / f"{engine}.exe",
    ])

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    return None


def _build_case_network_dot(
    case_id: str,
    data,
    node_index,
    schemas,
    facts,
    include_entities: bool = True,
) -> str:
    schema_by_name = {s.name: s for s in schemas}
    fact_by_id = {f.fact_id: f for f in facts}

    dot_ids: dict[str, str] = {}

    def get_dot_id(key: str) -> str:
        if key not in dot_ids:
            dot_ids[key] = f"n{len(dot_ids)}"
        return dot_ids[key]

    lines: list[str] = [
        "digraph CaseNetwork {",
        "  rankdir=LR;",
        "  splines=true;",
        '  graph [fontname="Helvetica", fontsize=11, labeljust="l", labelloc="t"];',
        '  node [fontname="Helvetica", fontsize=10, color="#666666"];',
        '  edge [fontname="Helvetica", fontsize=9, color="#444444"];',
        f'  label="Case Network\\ncase_id={_dot_escape(case_id)}";',
        "",
    ]

    entity_ids: set[str] = set()
    for cmap in node_index.cluster_node_to_idx.values():
        entity_ids.update(cmap.keys())
    for fact in facts:
        for arg in fact.args:
            if arg.entity_id:
                entity_ids.add(arg.entity_id)

    if include_entities:
        for eid in sorted(entity_ids):
            nid = get_dot_id(f"entity:{eid}")
            label = f"ENTITY\\n{eid}"
            lines.append(
                f'  {nid} [shape=ellipse, style="filled", fillcolor="#EAF2FF", label="{_dot_escape(label)}"];'
            )

    for fid in sorted(node_index.fact_node_to_idx.keys()):
        fact = fact_by_id.get(fid)
        nid = get_dot_id(f"fact:{fid}")
        if fact is None:
            label = f"FACT\\n{fid}"
            status_val = "-"
        else:
            status_val = fact.status.value
            label = f"FACT\\n{fid}\\n{fact.predicate}\\nstatus={status_val}"
        fill = "#E8F7E8" if status_val == "proved" else "#FFF4DD"
        lines.append(
            f'  {nid} [shape=box, style="rounded,filled", fillcolor="{fill}", label="{_dot_escape(label)}"];'
        )

    for schema in schemas:
        cname = schema.name
        node_type = f"c_{cname}"
        idx_to_entity = node_index.idx_to_cluster_node.get(cname, {})
        if not idx_to_entity:
            continue

        logits = data[node_type].x
        is_clamped = data[node_type].get("is_clamped")
        clamp_hard = data[node_type].get("clamp_hard")

        for idx, eid in idx_to_entity.items():
            idx_int = int(idx)
            logit_row = logits[idx_int].tolist() if logits.size(0) > idx_int else []
            top_info = "-"
            if logit_row:
                top_idx = max(range(len(logit_row)), key=lambda i: float(logit_row[i]))
                top_val = schema.domain[top_idx] if top_idx < len(schema.domain) else str(top_idx)
                vals = [math.exp(float(v)) for v in logit_row]
                denom = sum(vals) or 1.0
                conf = vals[top_idx] / denom
                top_info = f"{top_val} ({conf:.2f})"

            clamped = False
            if is_clamped is not None and is_clamped.size(0) > idx_int:
                clamped = bool(is_clamped[idx_int].item())
            hard = False
            if clamp_hard is not None and clamp_hard.size(0) > idx_int:
                hard = bool(clamp_hard[idx_int].item())

            nid = get_dot_id(f"cluster:{cname}:{eid}")
            label = (
                f"CLUSTER\\n{cname}\\nentity={eid}\\n"
                f"top={top_info}\\nclamped={'yes' if clamped else 'no'}"
                f"{' hard' if hard else ''}"
            )
            fill = "#E6FBF2" if clamped else "#F0FFF8"
            lines.append(
                f'  {nid} [shape=component, style="filled", fillcolor="{fill}", label="{_dot_escape(label)}"];'
            )
            if include_entities:
                e_node = get_dot_id(f"entity:{eid}")
                lines.append(f'  {e_node} -> {nid} [color="#99AACC", label="state"];')

    for fact in facts:
        dst = get_dot_id(f"fact:{fact.fact_id}")
        for arg in fact.args:
            if include_entities and arg.entity_id:
                src = get_dot_id(f"entity:{arg.entity_id}")
                lines.append(
                    f'  {src} -> {dst} [color="#808080", label="{_dot_escape(arg.role)}"];'
                )

    for src_type, relation, dst_type in data.edge_types:
        edge_index = data[src_type, relation, dst_type].edge_index
        n_edges = edge_index.size(1) if edge_index is not None else 0
        for e in range(n_edges):
            src_idx = int(edge_index[0, e].item())
            dst_idx = int(edge_index[1, e].item())

            src_key: str | None = None
            dst_key: str | None = None

            if src_type == "fact":
                fid = node_index.idx_to_fact_node.get(src_idx)
                if fid is not None:
                    src_key = f"fact:{fid}"
            elif src_type.startswith("c_"):
                cname = src_type[2:]
                eid = node_index.idx_to_cluster_node.get(cname, {}).get(src_idx)
                if eid is not None:
                    src_key = f"cluster:{cname}:{eid}"

            if dst_type == "fact":
                fid = node_index.idx_to_fact_node.get(dst_idx)
                if fid is not None:
                    dst_key = f"fact:{fid}"
            elif dst_type.startswith("c_"):
                cname = dst_type[2:]
                eid = node_index.idx_to_cluster_node.get(cname, {}).get(dst_idx)
                if eid is not None:
                    dst_key = f"cluster:{cname}:{eid}"

            if src_key is None or dst_key is None:
                continue

            src = get_dot_id(src_key)
            dst = get_dot_id(dst_key)

            if relation == "implies":
                style = 'color="#1F77B4", penwidth=1.7'
            elif relation == "supports" or relation.startswith("supports:"):
                style = 'color="#A15C00", style="dashed"'
            else:
                style = 'color="#3A7A3A"'

            lines.append(
                f'  {src} -> {dst} [{style}, label="{_dot_escape(relation)}"];'
            )

    lines.append("}")
    return "\n".join(lines) + "\n"


def cmd_proof(args: argparse.Namespace) -> None:
    proof_id = args.proof_id

    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                pr.id,
                pr.proof_id,
                c.case_id,
                pr.query,
                pr.result,
                pr.created_at,
                pr.proof_dag
            FROM proof_runs pr
            JOIN cases c ON c.id = pr.case_id
            WHERE pr.proof_id = %s
            """,
            (proof_id,),
        )
        run_row = cur.fetchone()

        if run_row is None:
            console.print(f"[bold red]proof not found[/bold red]: {proof_id}")
            return

        run_id, run_proof_id, case_id, query, result, created_at, proof_dag = run_row

        cur.execute(
            """
            SELECT fact_id, predicate, status
            FROM facts
            WHERE proof_id = %s
            ORDER BY id
            """,
            (proof_id,),
        )
        fact_rows = cur.fetchall()

        cur.execute(
            """
            SELECT ps.step_order, ps.rule_id, ps.substitution, ps.used_fact_ids
            FROM proof_steps ps
            WHERE ps.run_id = %s
            ORDER BY ps.step_order
            """,
            (run_id,),
        )
        step_rows = cur.fetchall()

        cur.execute(
            """
            SELECT
                fact_id,
                predicate,
                outcome,
                atom_text,
                violated_naf,
                missing_pos_body,
                supporting_rule_ids
            FROM proof_candidate_feedback
            WHERE proof_id = %s
            ORDER BY id
            """,
            (proof_id,),
        )
        feedback_rows = cur.fetchall()

    dag_obj = _coerce_json(proof_dag)
    dag_map = dag_obj if isinstance(dag_obj, dict) else {}

    def _sub_key(value) -> str:
        obj = _coerce_json(value) or {}
        if not isinstance(obj, dict):
            return "-"
        return json.dumps(obj, sort_keys=True, ensure_ascii=False)

    # Best-effort map step -> (atom, status, source_span_text) using (rule_id, substitution).
    step_atom_index: dict[tuple[str, str], list[tuple[str, str, str | None]]] = {}
    for atom, node in dag_map.items():
        if not isinstance(node, dict):
            continue
        key = (
            str(node.get("rule_id") or ""),
            _sub_key(node.get("substitution")),
        )
        status = str(node.get("status") or "-")
        sst = node.get("source_span_text") or None
        step_atom_index.setdefault(key, []).append((str(atom), status, sst))

    console.print("[bold cyan]Proof Run[/bold cyan]")
    console.print(f"proof_id: {run_proof_id}")
    console.print(f"case_id: {case_id}")
    console.print(f"query: {query}")
    console.print(f"result: {result}")
    console.print(f"created_at: {str(created_at)[:19]}")
    console.print(f"steps: {len(step_rows)} linked_facts: {len(fact_rows)} feedback: {len(feedback_rows)}")

    console.print()
    console.print("[bold cyan]Linked Facts[/bold cyan]")
    if not fact_rows:
        console.print("[dim]-[/dim]")
    else:
        for i, (fact_id, predicate, status) in enumerate(fact_rows, start=1):
            status_text = Text(status, style=_STATUS_STYLE.get(status, ""))
            console.print(f"{i}. {fact_id} {predicate} ", status_text)

    console.print()
    console.print("[bold cyan]Verifier Feedback[/bold cyan]")
    if not feedback_rows:
        console.print("[dim]-[/dim]")
    else:
        for i, (fact_id, predicate, outcome, atom_text, violated_naf, missing_pos_body, supporting_rule_ids) in enumerate(feedback_rows, start=1):
            outcome_text = Text(outcome, style=_FEEDBACK_STYLE_RC.get(outcome, ""))
            console.print(f"{i}. {fact_id} {predicate} ", outcome_text, f" atom={atom_text or '-'}")
            if violated_naf:
                console.print(f"   violated_naf={list(violated_naf)}")
            if missing_pos_body:
                console.print(f"   missing_pos_body={list(missing_pos_body)}")
            if supporting_rule_ids:
                console.print(f"   supporting_rule_ids={list(supporting_rule_ids)}")

    console.print()
    console.print("[bold cyan]Proof Steps[/bold cyan]")
    if not step_rows:
        console.print("[dim]-[/dim]")
    else:
        for step_order, rule_id, substitution, used_fact_ids in step_rows:
            sub = _coerce_json(substitution) or {}
            used = list(used_fact_ids) if used_fact_ids else []
            step_key = (str(rule_id or ""), _sub_key(substitution))
            candidates = step_atom_index.get(step_key, [])
            atom = "-"
            atom_status = "-"
            source_span_text = None
            if candidates:
                atom, atom_status, source_span_text = candidates.pop(0)
            console.print(
                f"{step_order}. atom={atom} status={atom_status} rule={rule_id or '-'} "
                f"sub={sub if sub else '-'} "
                f"used_facts={used if used else '-'}"
            )
            if source_span_text:
                console.print(f"   [dim]source text:[/dim] [italic]{source_span_text}[/italic]")

    if args.dag:
        console.print()
        console.print("[bold cyan]Proof DAG[/bold cyan]")
        if dag_obj is None:
            console.print("[dim]-[/dim]")
        else:
            console.print(json.dumps(dag_obj, indent=2, ensure_ascii=False))


# ---------------------------------------------------------------------------
# proof-graph
# ---------------------------------------------------------------------------

def cmd_proof_graph(args: argparse.Namespace) -> None:
    proof_id = args.proof_id

    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                pr.proof_id,
                c.case_id,
                pr.query,
                pr.result,
                pr.proof_dag
            FROM proof_runs pr
            JOIN cases c ON c.id = pr.case_id
            WHERE pr.proof_id = %s
            """,
            (proof_id,),
        )
        row = cur.fetchone()

    if row is None:
        console.print(f"[bold red]proof not found[/bold red]: {proof_id}")
        return

    run_proof_id, case_id, query, result, proof_dag = row
    dag_obj = _coerce_json(proof_dag)
    if not isinstance(dag_obj, dict):
        console.print("[bold red]invalid proof_dag JSON[/bold red]")
        return

    out_base = Path(args.output) if args.output else Path(f"proof_{run_proof_id}")
    dot_path = out_base.with_suffix(".dot")
    dot_path.parent.mkdir(parents=True, exist_ok=True)

    dot_text = _build_proof_dot(
        proof_id=str(run_proof_id),
        case_id=str(case_id),
        query=str(query),
        result=str(result),
        dag_map=dag_obj,
    )
    dot_path.write_text(dot_text, encoding="utf-8")
    console.print(f"[bold green]DOT saved[/bold green]: {dot_path}")

    if args.format == "dot":
        return

    dot_bin = _resolve_graphviz_engine(args.engine)
    if dot_bin is None:
        console.print(
            f"[yellow]Graphviz binary '{args.engine}' not found.[/yellow] "
            f"Install Graphviz or use --format dot."
        )
        return

    out_path = out_base.with_suffix(f".{args.format}")
    try:
        subprocess.run(
            [dot_bin, f"-T{args.format}", str(dot_path), "-o", str(out_path)],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        err = (exc.stderr or exc.stdout or "").strip()
        console.print(f"[bold red]Graphviz render failed[/bold red]: {err or exc}")
        return

    console.print(f"[bold green]Diagram saved[/bold green]: {out_path}")


# ---------------------------------------------------------------------------
# network-graph
# ---------------------------------------------------------------------------

def cmd_network_graph(args: argparse.Namespace) -> None:
    from db import DBSession
    from nn.graph_builder import GraphBuilder

    case_id = args.case_id

    with DBSession.connect() as session:
        schemas = session.load_cluster_schemas()
        entities, facts, rules, cluster_states = session.load_case(case_id)
        data, node_index, _ = GraphBuilder(schemas).build(
            entities=entities,
            facts=facts,
            rules=rules,
            cluster_states=cluster_states,
            memory_biases=None,
        )

    out_base = Path(args.output) if args.output else Path(f"network_{case_id}")
    dot_path = out_base.with_suffix(".dot")
    dot_path.parent.mkdir(parents=True, exist_ok=True)

    dot_text = _build_case_network_dot(
        case_id=case_id,
        data=data,
        node_index=node_index,
        schemas=schemas,
        facts=facts,
        include_entities=not args.no_entities,
    )
    dot_path.write_text(dot_text, encoding="utf-8")
    console.print(f"[bold green]DOT saved[/bold green]: {dot_path}")

    if args.format == "dot":
        return

    dot_bin = _resolve_graphviz_engine(args.engine)
    if dot_bin is None:
        console.print(
            f"[yellow]Graphviz binary '{args.engine}' not found.[/yellow] "
            f"Install Graphviz or use --format dot."
        )
        return

    out_path = out_base.with_suffix(f".{args.format}")
    try:
        subprocess.run(
            [dot_bin, f"-T{args.format}", str(dot_path), "-o", str(out_path)],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        err = (exc.stderr or exc.stdout or "").strip()
        console.print(f"[bold red]Graphviz render failed[/bold red]: {err or exc}")
        return

    console.print(f"[bold green]Diagram saved[/bold green]: {out_path}")


# ---------------------------------------------------------------------------
# reset-state
# ---------------------------------------------------------------------------

def _collect_ontology_counts(conn) -> dict[str, int]:
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM entity_types")
        entity_types = int(cur.fetchone()[0])
        cur.execute("SELECT COUNT(*) FROM predicate_definitions")
        predicates = int(cur.fetchone()[0])
        cur.execute("SELECT COUNT(*) FROM cluster_definitions")
        clusters = int(cur.fetchone()[0])
        cur.execute("SELECT COUNT(*) FROM rules WHERE learned = FALSE")
        rules = int(cur.fetchone()[0])
        cur.execute("SELECT COUNT(*) FROM rule_modules")
        modules = int(cur.fetchone()[0])
        cur.execute("SELECT COUNT(*) FROM entities")
        entities = int(cur.fetchone()[0])
        cur.execute("SELECT COUNT(*) FROM facts")
        facts = int(cur.fetchone()[0])
        cur.execute("SELECT COUNT(*) FROM cases")
        cases = int(cur.fetchone()[0])
        cur.execute("SELECT COUNT(*) FROM sources")
        sources = int(cur.fetchone()[0])
    return {
        "entity_types": entity_types,
        "predicates": predicates,
        "clusters": clusters,
        "rules_static": rules,
        "rule_modules": modules,
        "entities": entities,
        "facts": facts,
        "cases": cases,
        "sources": sources,
    }


def _apply_ontology_reset(conn) -> dict[str, int]:
    """
    Pełne wyczyszczenie DB: wszystkie dane + ontologia.
    Kolejność uwzględnia FK (bez ON DELETE CASCADE).
    """
    result: dict[str, int] = {}
    with conn.cursor() as cur:
        cur.execute("DELETE FROM fact_neural_trace")
        result["fact_neural_trace"] = cur.rowcount
        cur.execute("DELETE FROM facts")
        result["facts"] = cur.rowcount
        cur.execute("DELETE FROM cluster_states")
        result["cluster_states"] = cur.rowcount
        cur.execute("DELETE FROM entities")
        result["entities"] = cur.rowcount
        cur.execute("DELETE FROM proof_runs")        # cascades proof_steps
        result["proof_runs"] = cur.rowcount
        cur.execute("DELETE FROM cases")             # cascades case_queries
        result["cases"] = cur.rowcount
        cur.execute("DELETE FROM sources")
        result["sources"] = cur.rowcount
        cur.execute("DELETE FROM cluster_definitions")   # cascades cluster_domain_values
        result["cluster_definitions"] = cur.rowcount
        cur.execute("DELETE FROM predicate_definitions") # cascades predicate_roles
        result["predicate_definitions"] = cur.rowcount
        cur.execute("DELETE FROM entity_types")
        result["entity_types"] = cur.rowcount
        cur.execute("DELETE FROM rules")
        result["rules"] = cur.rowcount
        cur.execute("DELETE FROM rule_modules")
        result["rule_modules"] = cur.rowcount
    return result


def _collect_reset_counts(
    conn, prune_cluster_states: bool, case_id: str | None = None
) -> dict[str, int]:
    with conn.cursor() as cur:
        if case_id is None:
            cur.execute("SELECT COUNT(*) FROM rules WHERE learned = TRUE")
            learned_rules = int(cur.fetchone()[0])

            cur.execute("SELECT COUNT(*) FROM proof_runs")
            proof_runs = int(cur.fetchone()[0])

            cur.execute("SELECT COUNT(*) FROM proof_steps")
            proof_steps = int(cur.fetchone()[0])

            cur.execute("SELECT COUNT(*) FROM facts WHERE status <> 'observed'")
            non_observed_facts = int(cur.fetchone()[0])

            cur.execute("SELECT COUNT(*) FROM fact_neural_trace")
            neural_traces = int(cur.fetchone()[0])

            cur.execute("SELECT COUNT(*) FROM facts WHERE proof_id IS NOT NULL")
            facts_with_proof = int(cur.fetchone()[0])

            cur.execute(
                """
                SELECT COUNT(*)
                FROM facts
                WHERE status = 'observed'
                  AND (
                        truth_value IS DISTINCT FROM 'T'::truth_value
                     OR truth_confidence IS DISTINCT FROM 1.0
                     OR truth_logits IS NOT NULL
                  )
                """
            )
            observed_needing_truth_reset = int(cur.fetchone()[0])

            prunable_cluster_states = 0
            if prune_cluster_states:
                cur.execute("SELECT COUNT(*) FROM cluster_states WHERE is_clamped = FALSE")
                prunable_cluster_states = int(cur.fetchone()[0])
        else:
            learned_rules = 0  # reguły są globalne — nie dotykamy przy reset per-case

            cur.execute(
                """
                SELECT COUNT(*) FROM proof_runs pr
                JOIN cases c ON c.id = pr.case_id
                WHERE c.case_id = %s
                """,
                (case_id,),
            )
            proof_runs = int(cur.fetchone()[0])

            cur.execute(
                """
                SELECT COUNT(*) FROM proof_steps ps
                JOIN proof_runs pr ON pr.id = ps.run_id
                JOIN cases c ON c.id = pr.case_id
                WHERE c.case_id = %s
                """,
                (case_id,),
            )
            proof_steps = int(cur.fetchone()[0])

            cur.execute(
                """
                SELECT COUNT(*) FROM facts f
                JOIN cases c ON c.id = f.case_id
                WHERE c.case_id = %s AND f.status <> 'observed'
                """,
                (case_id,),
            )
            non_observed_facts = int(cur.fetchone()[0])

            cur.execute(
                """
                SELECT COUNT(*) FROM fact_neural_trace fnt
                JOIN facts f ON f.id = fnt.fact_id
                JOIN cases c ON c.id = f.case_id
                WHERE c.case_id = %s
                """,
                (case_id,),
            )
            neural_traces = int(cur.fetchone()[0])

            cur.execute(
                """
                SELECT COUNT(*) FROM facts f
                JOIN cases c ON c.id = f.case_id
                WHERE c.case_id = %s AND f.proof_id IS NOT NULL
                """,
                (case_id,),
            )
            facts_with_proof = int(cur.fetchone()[0])

            cur.execute(
                """
                SELECT COUNT(*) FROM facts f
                JOIN cases c ON c.id = f.case_id
                WHERE c.case_id = %s
                  AND f.status = 'observed'
                  AND (
                        f.truth_value IS DISTINCT FROM 'T'::truth_value
                     OR f.truth_confidence IS DISTINCT FROM 1.0
                     OR f.truth_logits IS NOT NULL
                  )
                """,
                (case_id,),
            )
            observed_needing_truth_reset = int(cur.fetchone()[0])

            prunable_cluster_states = 0
            if prune_cluster_states:
                cur.execute(
                    """
                    SELECT COUNT(*) FROM cluster_states cs
                    JOIN cases c ON c.id = cs.case_id
                    WHERE c.case_id = %s AND cs.is_clamped = FALSE
                    """,
                    (case_id,),
                )
                prunable_cluster_states = int(cur.fetchone()[0])

    return {
        "learned_rules": learned_rules,
        "proof_runs": proof_runs,
        "proof_steps": proof_steps,
        "non_observed_facts": non_observed_facts,
        "neural_traces": neural_traces,
        "facts_with_proof": facts_with_proof,
        "observed_needing_truth_reset": observed_needing_truth_reset,
        "prunable_cluster_states": prunable_cluster_states,
    }


def _apply_reset(
    conn, prune_cluster_states: bool, case_id: str | None = None
) -> dict[str, int]:
    result: dict[str, int] = {}
    with conn.cursor() as cur:
        if case_id is None:
            cur.execute("DELETE FROM rules WHERE learned = TRUE")
            result["deleted_learned_rules"] = cur.rowcount

            cur.execute("DELETE FROM proof_runs")
            result["deleted_proof_runs"] = cur.rowcount

            cur.execute("DELETE FROM fact_neural_trace")
            result["deleted_neural_traces"] = cur.rowcount

            cur.execute("DELETE FROM facts WHERE status <> 'observed'")
            result["deleted_non_observed_facts"] = cur.rowcount

            cur.execute(
                """
                UPDATE facts
                SET
                    proof_id = NULL,
                    truth_value = 'T'::truth_value,
                    truth_confidence = 1.0,
                    truth_logits = NULL
                WHERE status = 'observed'
                  AND (
                        proof_id IS NOT NULL
                     OR truth_value IS DISTINCT FROM 'T'::truth_value
                     OR truth_confidence IS DISTINCT FROM 1.0
                     OR truth_logits IS NOT NULL
                  )
                """
            )
            result["reset_observed_facts"] = cur.rowcount

            if prune_cluster_states:
                cur.execute("DELETE FROM cluster_states WHERE is_clamped = FALSE")
                result["deleted_non_clamped_cluster_states"] = cur.rowcount
            else:
                result["deleted_non_clamped_cluster_states"] = 0

            cur.execute(
                """
                DELETE FROM rule_modules rm
                WHERE rm.name = 'learned_nn'
                  AND NOT EXISTS (
                        SELECT 1 FROM rules r WHERE r.module_id = rm.id
                  )
                """
            )
            result["deleted_empty_learned_modules"] = cur.rowcount
        else:
            result["deleted_learned_rules"] = 0  # reguły są globalne

            cur.execute(
                """
                DELETE FROM proof_runs
                WHERE case_id IN (SELECT id FROM cases WHERE case_id = %s)
                """,
                (case_id,),
            )
            result["deleted_proof_runs"] = cur.rowcount

            cur.execute(
                """
                DELETE FROM fact_neural_trace
                WHERE fact_id IN (
                    SELECT f.id FROM facts f
                    JOIN cases c ON c.id = f.case_id
                    WHERE c.case_id = %s
                )
                """,
                (case_id,),
            )
            result["deleted_neural_traces"] = cur.rowcount

            cur.execute(
                """
                DELETE FROM facts
                WHERE case_id IN (SELECT id FROM cases WHERE case_id = %s)
                  AND status <> 'observed'
                """,
                (case_id,),
            )
            result["deleted_non_observed_facts"] = cur.rowcount

            cur.execute(
                """
                UPDATE facts
                SET
                    proof_id = NULL,
                    truth_value = 'T'::truth_value,
                    truth_confidence = 1.0,
                    truth_logits = NULL
                WHERE case_id IN (SELECT id FROM cases WHERE case_id = %s)
                  AND status = 'observed'
                  AND (
                        proof_id IS NOT NULL
                     OR truth_value IS DISTINCT FROM 'T'::truth_value
                     OR truth_confidence IS DISTINCT FROM 1.0
                     OR truth_logits IS NOT NULL
                  )
                """,
                (case_id,),
            )
            result["reset_observed_facts"] = cur.rowcount

            if prune_cluster_states:
                cur.execute(
                    """
                    DELETE FROM cluster_states
                    WHERE case_id IN (SELECT id FROM cases WHERE case_id = %s)
                      AND is_clamped = FALSE
                    """,
                    (case_id,),
                )
                result["deleted_non_clamped_cluster_states"] = cur.rowcount
            else:
                result["deleted_non_clamped_cluster_states"] = 0

            result["deleted_empty_learned_modules"] = 0

    return result


def cmd_reset_state(args: argparse.Namespace) -> None:
    prune_cluster_states = not args.keep_cluster_states
    case_id: str | None = getattr(args, "case_id", None) or None
    reset_ontology: bool = getattr(args, "ontology", False)

    if reset_ontology and case_id:
        console.print("[bold red]--ontology nie może być używane razem z --case-id.[/bold red]")
        raise SystemExit(1)

    with connect() as conn:
        scope = f"[bold yellow]{case_id}[/bold yellow]" if case_id else "[bold yellow]ALL[/bold yellow]"

        if reset_ontology:
            onto_counts = _collect_ontology_counts(conn)
            console.print(f"[bold red]Reset Preview[/bold red]  scope={scope}  [bold red]+ ONTOLOGIA (PEŁNY WIPE)[/bold red]")
            console.print("[dim]Usuwa CAŁĄ ontologię + wszystkie dane (entities, facts, cases, sources).[/dim]")
            console.print(f"entity_types: {onto_counts['entity_types']}")
            console.print(f"predicates:   {onto_counts['predicates']}")
            console.print(f"clusters:     {onto_counts['clusters']}")
            console.print(f"rules(static):{onto_counts['rules_static']}")
            console.print(f"rule_modules: {onto_counts['rule_modules']}")
            console.print(f"entities:     {onto_counts['entities']}")
            console.print(f"facts:        {onto_counts['facts']}")
            console.print(f"cases:        {onto_counts['cases']}")
            console.print(f"sources:      {onto_counts['sources']}")
        else:
            counts = _collect_reset_counts(
                conn, prune_cluster_states=prune_cluster_states, case_id=case_id
            )
            console.print(f"[bold cyan]Reset Preview[/bold cyan]  scope={scope}")
            if case_id:
                console.print("[dim]Learned rules are global — skipped for per-case reset.[/dim]")
            else:
                console.print(f"learned_rules: {counts['learned_rules']}")
            console.print(f"proof_runs: {counts['proof_runs']}")
            console.print(f"proof_steps: {counts['proof_steps']}")
            console.print(f"facts_non_observed: {counts['non_observed_facts']}")
            console.print(f"fact_neural_trace: {counts['neural_traces']}")
            console.print(f"facts_with_proof_link: {counts['facts_with_proof']}")
            console.print(f"observed_facts_truth_reset: {counts['observed_needing_truth_reset']}")
            if prune_cluster_states:
                console.print(f"cluster_states_non_clamped: {counts['prunable_cluster_states']}")
            else:
                console.print("cluster_states_non_clamped: [dim]kept[/dim]")

        if not args.yes:
            console.print()
            console.print("[yellow]Dry-run only.[/yellow] Run again with [bold]--yes[/bold] to apply.")
            return

        try:
            if reset_ontology:
                applied_onto = _apply_ontology_reset(conn)
                conn.commit()
            else:
                applied = _apply_reset(
                    conn, prune_cluster_states=prune_cluster_states, case_id=case_id
                )
                conn.commit()
        except Exception:
            conn.rollback()
            raise

    console.print()
    if reset_ontology:
        console.print("[bold green]Ontology Reset Applied[/bold green]")
        console.print(f"deleted entity_types: {applied_onto['entity_types']}")
        console.print(f"deleted predicates:   {applied_onto['predicate_definitions']}")
        console.print(f"deleted clusters:     {applied_onto['cluster_definitions']}")
        console.print(f"deleted rules:        {applied_onto['rules']}")
        console.print(f"deleted rule_modules: {applied_onto['rule_modules']}")
        console.print(f"deleted entities:     {applied_onto['entities']}")
        console.print(f"deleted facts:        {applied_onto['facts']}")
        console.print(f"deleted cases:        {applied_onto['cases']}")
        console.print(f"deleted sources:      {applied_onto['sources']}")
    else:
        console.print("[bold green]Reset Applied[/bold green]")
        if not case_id:
            console.print(f"deleted learned rules: {applied['deleted_learned_rules']}")
        console.print(f"deleted proof runs: {applied['deleted_proof_runs']}")
        console.print(f"deleted neural traces: {applied['deleted_neural_traces']}")
        console.print(f"deleted non-observed facts: {applied['deleted_non_observed_facts']}")
        console.print(f"reset observed facts (proof/truth): {applied['reset_observed_facts']}")
        console.print(f"deleted non-clamped cluster_states: {applied['deleted_non_clamped_cluster_states']}")
        if not case_id:
            console.print(f"deleted empty learned modules: {applied['deleted_empty_learned_modules']}")


# ---------------------------------------------------------------------------
# llm-prompt
# ---------------------------------------------------------------------------

def cmd_llm_prompt(args: argparse.Namespace) -> None:
    """
    Podgląd promptu wysyłanego do Gemini bez wywoływania API.

    Tryby:
      pn3 llm-prompt                  — pokaż system prompt (z DB schemas)
      pn3 llm-prompt --text "..."     — pokaż system + user message
      pn3 llm-prompt --json-schema    — pokaż JSON Schema odpowiedzi
      pn3 llm-prompt --raw            — surowy tekst bez Rich formatowania
    """
    from rich.panel import Panel
    from rich.syntax import Syntax

    from config import ProjectConfig
    from db import DBSession
    from nlp.llm_prompt import build_response_schema, build_system_prompt
    cfg = ProjectConfig.load()

    with DBSession.connect() as session:
        schemas = session.load_cluster_schemas()
        predicate_positions = session.load_predicate_positions()

    system_prompt = build_system_prompt(schemas, predicate_positions)

    if args.raw:
        print(system_prompt)
        if args.text:
            print("\n--- USER MESSAGE ---\n")
            print(args.text)
        if args.json_schema:
            print("\n--- JSON SCHEMA ---\n")
            print(json.dumps(build_response_schema(), indent=2, ensure_ascii=False))
        return

    # Rich output
    console.print(Panel(
        system_prompt,
        title=f"[bold cyan]System Prompt[/bold cyan]  "
              f"[dim](model: {cfg.extractor.gemini_model})[/dim]",
        border_style="cyan",
        expand=True,
    ))

    if args.text:
        console.print()
        console.print(Panel(
            args.text,
            title="[bold yellow]User Message[/bold yellow]",
            border_style="yellow",
            expand=True,
        ))

    if args.json_schema:
        console.print()
        schema_json = json.dumps(build_response_schema(), indent=2, ensure_ascii=False)
        console.print(Panel(
            Syntax(schema_json, "json", theme="monokai", word_wrap=True),
            title="[bold green]Response JSON Schema[/bold green]",
            border_style="green",
            expand=True,
        ))

    console.print(
        f"\n[dim]Schemas loaded: {len(schemas)} clusters | "
        f"Predicates: {len(predicate_positions)}[/dim]"
    )


# ---------------------------------------------------------------------------
# ingest-text
# ---------------------------------------------------------------------------

def _ensure_case_exists(conn, case_id: str, source_id: str, title: str) -> None:
    """Tworzy wpis w sources + cases jeśli nie istnieje."""
    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM cases WHERE case_id = %s", (case_id,))
        if cur.fetchone() is not None:
            return
        cur.execute(
            """
            INSERT INTO sources (source_id, title, source_type, source_rank)
            VALUES (%s, %s, 'case_text', 10)
            ON CONFLICT (source_id) DO NOTHING
            """,
            (source_id, title),
        )
        cur.execute(
            """
            INSERT INTO cases (case_id, source_id, title)
            SELECT %s, id, %s FROM sources WHERE source_id = %s
            ON CONFLICT (case_id) DO NOTHING
            """,
            (case_id, title, source_id),
        )


def _mask_api_key(api_key: str) -> str:
    return f"...{api_key[-4:]}" if api_key else "BRAK"


def _assert_extraction_runtime_ready(
    cluster_schemas: list,
    predicate_positions: dict[str, list[str]],
    cfg,
) -> None:
    if not cluster_schemas:
        raise RuntimeError(
            "Brak aktywnej ontologii. Najpierw uruchom gen-ontology."
        )
    if not predicate_positions:
        raise RuntimeError(
            "Aktywna ontologia nie zawiera predykatów. Uruchom gen-ontology ponownie."
        )
    api_key = os.environ.get(cfg.extractor.api_key_env, "").strip()
    if not api_key:
        raise RuntimeError(
            f"Brak zmiennej środowiskowej {cfg.extractor.api_key_env}."
        )


def _extract_once(
    cluster_schemas: list,
    predicate_positions: dict[str, list[str]],
    cfg,
    text: str,
    source_id: str,
):
    from config import ExtractorConfig, ProjectConfig
    from nlp import get_extractor

    # Ingest uruchamia verifier jako osobny subprocess zaraz po ekstrakcji,
    # więc w subprocessie ekstraktora nie łączymy LLM i clingo w jednym procesie.
    if isinstance(cfg, ProjectConfig):
        runtime_cfg = replace(cfg, extractor=replace(cfg.extractor, sv_verification=False))
    elif isinstance(cfg, ExtractorConfig):
        runtime_cfg = replace(cfg, sv_verification=False)
    else:
        runtime_cfg = cfg

    extractor = get_extractor(
        cluster_schemas,
        runtime_cfg,
        predicate_positions=predicate_positions,
    )
    try:
        return extractor.extract(text, source_id=source_id)
    finally:
        close = getattr(extractor, "close", None)
        if callable(close):
            close()


def _serialize_extraction_result(result) -> str:
    cluster_states: list[dict[str, object]] = []
    for cs in result.cluster_states:
        payload = {
            "entity_id": cs.entity_id,
            "cluster_name": cs.cluster_name,
            "logits": cs.logits,
            "is_clamped": cs.is_clamped,
            "clamp_hard": cs.clamp_hard,
            "clamp_source": cs.clamp_source,
            "source_span": None,
        }
        if cs.source_span is not None:
            payload["source_span"] = cs.source_span.model_dump(mode="json")
        cluster_states.append(payload)

    return json.dumps(
        {
            "source_id": result.source_id,
            "entities": [entity.model_dump(mode="json") for entity in result.entities],
            "facts": [fact.model_dump(mode="json") for fact in result.facts],
            "cluster_states": cluster_states,
        },
        ensure_ascii=False,
    )


def _deserialize_extraction_result(payload: dict[str, object]):
    from data_model.cluster import ClusterStateRow
    from data_model.common import Span
    from data_model.entity import Entity
    from data_model.fact import Fact
    from nlp.result import ExtractionResult

    cluster_states: list[ClusterStateRow] = []
    for item in payload.get("cluster_states", []):
        row = dict(item)
        if row.get("source_span") is not None:
            row["source_span"] = Span.model_validate(row["source_span"])
        cluster_states.append(ClusterStateRow(**row))

    return ExtractionResult(
        entities=[Entity.model_validate(item) for item in payload.get("entities", [])],
        facts=[Fact.model_validate(item) for item in payload.get("facts", [])],
        cluster_states=cluster_states,
        source_id=str(payload.get("source_id", "text")),
    )


def _run_extract_subprocess(file_path: Path, source_id: str, timeout_s: int):
    subprocess_timeout = timeout_s + 5
    cmd = [
        sys.executable,
        "-m",
        "cli.pn3",
        "_extract-json",
        "--file",
        str(file_path),
        "--source-id",
        source_id,
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=subprocess_timeout,
            cwd=str(Path(__file__).resolve().parent.parent),
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(f"extract timeout po {timeout_s}s") from exc

    stdout = (proc.stdout or "").strip()
    stderr = (proc.stderr or "").strip()
    if proc.returncode != 0:
        detail = stderr or stdout or f"subprocess exit code {proc.returncode}"
        raise RuntimeError(detail)
    if not stdout:
        raise RuntimeError("Subprocess ekstrakcji zwrócił pusty output.")
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as exc:
        snippet = stdout[:300]
        raise RuntimeError(f"Nieprawidłowy JSON z subprocessu ekstrakcji: {snippet}") from exc
    return _deserialize_extraction_result(payload)


def _run_verify_subprocess(result, timeout_s: int):
    import tempfile

    subprocess_timeout = timeout_s + 5
    payload_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            suffix=".json",
            delete=False,
        ) as tmp:
            tmp.write(_serialize_extraction_result(result))
            payload_path = Path(tmp.name)

        cmd = [
            sys.executable,
            "-m",
            "cli.pn3",
            "_verify-json",
            "--input-json",
            str(payload_path),
        ]
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=subprocess_timeout,
            cwd=str(Path(__file__).resolve().parent.parent),
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(f"verify timeout po {timeout_s}s") from exc
    finally:
        if payload_path is not None and payload_path.exists():
            payload_path.unlink()

    stdout = (proc.stdout or "").strip()
    stderr = (proc.stderr or "").strip()
    if proc.returncode != 0:
        detail = stderr or stdout or f"subprocess exit code {proc.returncode}"
        raise RuntimeError(detail)
    if not stdout:
        raise RuntimeError("Subprocess weryfikacji zwrócił pusty output.")
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as exc:
        snippet = stdout[:300]
        raise RuntimeError(f"Nieprawidłowy JSON z subprocessu weryfikacji: {snippet}") from exc
    return _deserialize_extraction_result(payload)


def cmd_extract_json(args: argparse.Namespace) -> None:
    from config import ProjectConfig
    from db import DBSession

    try:
        file_path = Path(args.file)
        if not file_path.is_file():
            print(f"Plik nie istnieje: {file_path}", file=sys.stderr)
            raise SystemExit(1)

        text = file_path.read_text(encoding="utf-8").strip()
        if not text:
            print(f"Tekst jest pusty: {file_path}", file=sys.stderr)
            raise SystemExit(1)

        cfg = ProjectConfig.load()
        with DBSession.connect() as session:
            schemas = session.load_cluster_schemas()
            predicate_positions = session.load_predicate_positions()
            _assert_extraction_runtime_ready(schemas, predicate_positions, cfg)
            result = _extract_once(
                schemas,
                predicate_positions,
                cfg,
                text,
                args.source_id,
            )

        print(_serialize_extraction_result(result), flush=True)
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1) from None


def cmd_verify_json(args: argparse.Namespace) -> None:
    from db import DBSession
    from nlp.result import ExtractionResult
    from sv import SymbolicVerifier

    try:
        input_path = Path(args.input_json)
        if not input_path.is_file():
            print(f"Plik nie istnieje: {input_path}", file=sys.stderr)
            raise SystemExit(1)

        payload = json.loads(input_path.read_text(encoding="utf-8"))
        result: ExtractionResult = _deserialize_extraction_result(payload)

        with DBSession.connect() as session:
            schemas = session.load_cluster_schemas()
            predicate_positions = session.load_predicate_positions()

        verifier = SymbolicVerifier(
            cluster_schemas=schemas,
            predicate_positions=predicate_positions,
        )
        verify_result = verifier.verify(
            facts=result.facts,
            rules=[],
            cluster_states=result.cluster_states,
        )
        verified = ExtractionResult(
            entities=result.entities,
            facts=verify_result.updated_facts,
            cluster_states=result.cluster_states,
            source_id=result.source_id,
        )
        print(_serialize_extraction_result(verified), flush=True)
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1) from None


def cmd_ingest_text(args: argparse.Namespace) -> None:
    """
    Ekstrahuje fakty z tekstu i zapisuje do DB dla podanego case_id.

    Tryby:
      pn3 ingest-text TC-002 --text "Złożyłem zamówienie..."
      pn3 ingest-text TC-002 --file text_cases/TC-002.txt
      pn3 ingest-text TC-002 --file ... --create      # utwórz case jeśli nie istnieje
      pn3 ingest-text TC-002 --file ... --dry-run     # pokaż wynik bez zapisu
    """
    import tempfile
    from pathlib import Path

    from config import ProjectConfig
    from db import DBSession

    case_id: str = args.case_id

    cleanup_path: Path | None = None
    if args.file:
        input_path = Path(args.file)
        text = input_path.read_text(encoding="utf-8").strip()
    elif args.text:
        text = args.text.strip()
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            suffix=".txt",
            delete=False,
        ) as tmp:
            tmp.write(text)
            input_path = Path(tmp.name)
            cleanup_path = input_path
    else:
        console.print("[bold red]Podaj --text lub --file.[/bold red]")
        raise SystemExit(1)

    if not text:
        console.print("[bold red]Tekst jest pusty.[/bold red]")
        raise SystemExit(1)

    source_id = f"{case_id}-TEXT"
    title = args.title or case_id
    cfg = ProjectConfig.load()

    with DBSession.connect() as session:
        schemas = session.load_cluster_schemas()
        predicate_positions = session.load_predicate_positions()
        try:
            _assert_extraction_runtime_ready(schemas, predicate_positions, cfg)
        except Exception as exc:
            if cleanup_path is not None and cleanup_path.exists():
                cleanup_path.unlink()
            console.print(f"[bold red]Fail-fast[/bold red]: {exc}")
            raise SystemExit(1) from exc

        _api_key = os.environ.get(cfg.extractor.api_key_env, "").strip()
        _key_hint = _mask_api_key(_api_key)
        console.print(
            f"[dim]Ekstrakcja (llm, model={cfg.extractor.gemini_model}, key={_key_hint}) -> {case_id}...[/dim]"
        )
        console.print(
            f"[dim]Fail-fast: sprawdzam odpowiedz LLM (timeout {cfg.extractor.preflight_timeout_s}s)[/dim]"
        )
        console.print(f"  [blue]START[/blue] {case_id}")
        try:
            result = _run_extract_subprocess(
                input_path,
                source_id,
                cfg.extractor.preflight_timeout_s,
            )
        except Exception as exc:
            if cleanup_path is not None and cleanup_path.exists():
                cleanup_path.unlink()
            console.print(f"[bold red]Fail-fast[/bold red]: {exc}")
            raise SystemExit(1) from exc

        if cleanup_path is not None and cleanup_path.exists():
            cleanup_path.unlink()

        if cfg.extractor.sv_verification:
            console.print(
                f"[dim]Verifier: sprawdzam spojnosc ekstrakcji dla {case_id}...[/dim]"
            )
            try:
                result = _run_verify_subprocess(
                    result,
                    cfg.extractor.preflight_timeout_s,
                )
            except Exception as exc:
                console.print(f"[bold red]Verifier failed[/bold red]: {exc}")
                raise SystemExit(1) from exc

        if args.dry_run:
            console.print(
                f"[bold yellow]dry-run[/bold yellow] — wynik dla {case_id}:"
            )
            console.print(result.summary())
            return

        if args.create:
            with session.conn.transaction():
                _ensure_case_exists(session.conn, case_id, source_id, title)
            session.conn.commit()

        session.save_extraction_result(result, case_id=case_id, source_text=text)

    console.print(
        f"[bold green]ingest-text completed[/bold green]: {case_id}"
    )
    console.print(result.summary())


# ---------------------------------------------------------------------------
# ingest-folder
# ---------------------------------------------------------------------------

def cmd_ingest_folder(args: argparse.Namespace) -> None:
    """
    Ekstrahuje fakty ze wszystkich plików .txt w podanym folderze.
    Case ID = nazwa pliku bez rozszerzenia (np. TC-001.txt → TC-001).

    Tryby:
      pn3 ingest-folder text_cases/
      pn3 ingest-folder text_cases/ --dry-run
      pn3 ingest-folder text_cases/ --pattern "TC-*.txt"
      pn3 ingest-folder text_cases/ --workers 8
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from pathlib import Path

    from config import ProjectConfig
    from db import DBSession

    folder = Path(args.folder)
    if not folder.is_dir():
        console.print(f"[bold red]Folder nie istnieje:[/bold red] {folder}")
        raise SystemExit(1)

    pattern = args.pattern or "*.txt"
    files = sorted(folder.glob(pattern))
    if not files:
        console.print(f"[yellow]Brak plików pasujących do wzorca '{pattern}' w {folder}[/yellow]")
        return

    cfg = ProjectConfig.load()
    workers = getattr(args, "workers", 8) or 8

    _api_key = os.environ.get(cfg.extractor.api_key_env, "").strip()
    _key_hint = _mask_api_key(_api_key)
    backend_info = f"llm, model={cfg.extractor.gemini_model}, key={_key_hint}"
    console.print(
        f"[dim]Folder: {folder}  |  pliki: {len(files)}  |  "
        f"backend: {backend_info}  |  workers: {workers}[/dim]"
    )

    # ── Faza 1: wczytaj pliki i utwórz case'y w DB (sekwencyjnie) ─────────────
    tasks: list[tuple[str, str, str, Path]] = []  # (case_id, source_id, text, file_path)
    skipped = 0
    with DBSession.connect() as session:
        schemas = session.load_cluster_schemas()
        predicate_positions = session.load_predicate_positions()
        try:
            _assert_extraction_runtime_ready(schemas, predicate_positions, cfg)
        except Exception as exc:
            console.print(f"[bold red]Fail-fast[/bold red]: {exc}")
            raise SystemExit(1) from exc
        for f in files:
            case_id = f.stem
            source_id = f"{case_id}-TEXT"
            text = f.read_text(encoding="utf-8").strip()
            if not text:
                console.print(f"  [yellow]SKIP[/yellow] {f.name} — pusty plik")
                skipped += 1
                continue
            tasks.append((case_id, source_id, text, f))

    if not tasks:
        console.print("[yellow]Brak zadań do przetworzenia.[/yellow]")
        return

    # ── Faza 2: ekstrakcja równoległa ─────────────────────────────────────────
    # Każdy wątek tworzy własny ekstraktor – osobna sesja HTTP do LLM.
    console.print(
        f"[dim]Preflight: sprawdzam jedna realna ekstrakcje (timeout {cfg.extractor.preflight_timeout_s}s)[/dim]"
    )
    probe_case_id, probe_source_id, probe_text, probe_path = tasks[0]
    console.print(f"  [blue]START[/blue] {probe_case_id} [dim](preflight)[/dim]")
    try:
        probe_result = _run_extract_subprocess(
            probe_path,
            probe_source_id,
            cfg.extractor.preflight_timeout_s,
        )
    except Exception as exc:
        console.print(f"[bold red]Preflight failed[/bold red]: {exc}")
        raise SystemExit(1) from exc
    console.print(f"  [green]preflight OK[/green] {probe_case_id}")

    print_lock = threading.Lock()

    def _extract_one(task: tuple[str, str, str, Path]):
        case_id, source_id, text, file_path = task
        with print_lock:
            console.print(f"  [blue]START[/blue] {case_id}")
        result = _run_extract_subprocess(
            file_path,
            source_id,
            cfg.extractor.preflight_timeout_s,
        )
        return case_id, source_id, text, result

    results: list[tuple[str, str, str, object]] = [
        (probe_case_id, probe_source_id, probe_text, probe_result)
    ]
    errors_extract = 0

    remaining_tasks = tasks[1:]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_extract_one, t): t[0] for t in remaining_tasks}
        for fut in as_completed(futures):
            case_id = futures[fut]
            try:
                results.append(fut.result())
                with print_lock:
                    console.print(f"  [cyan]extracted[/cyan] {case_id}")
            except Exception as exc:
                with print_lock:
                    console.print(f"  [bold red]ERR[/bold red] {case_id} (extract): {exc}")
                errors_extract += 1

    if cfg.extractor.sv_verification:
        console.print("[dim]Verifier: sprawdzam spojnosc wyekstrahowanych wynikow...[/dim]")
        verified_results: list[tuple[str, str, str, object]] = []
        errors_verify = 0
        for case_id, sid, text, result in sorted(results, key=lambda r: r[0]):
            try:
                verified = _run_verify_subprocess(
                    result,
                    cfg.extractor.preflight_timeout_s,
                )
                console.print(f"  [magenta]verified[/magenta] {case_id}")
                verified_results.append((case_id, sid, text, verified))
            except Exception as exc:
                console.print(f"  [bold red]ERR[/bold red] {case_id} (verify): {exc}")
                errors_verify += 1
        results = verified_results
    else:
        errors_verify = 0

    if args.dry_run:
        for case_id, _sid, _txt, result in sorted(results, key=lambda r: r[0]):
            console.print(f"  [yellow]DRY[/yellow] {case_id}: {result.summary()}")
        console.print(
            f"\n[bold]ingest-folder dry-run[/bold]: "
            f"[cyan]{len(results)} extracted[/cyan]  "
            f"[yellow]{skipped} pominięto[/yellow]  [red]{errors_extract + errors_verify} błędów[/red]"
        )
        return

    # ── Faza 3: zapis do DB (sekwencyjnie) ────────────────────────────────────
    ok = errors_save = 0
    with DBSession.connect() as session:
        for case_id, _sid, text, result in sorted(results, key=lambda r: r[0]):
            try:
                with session.conn.transaction():
                    _ensure_case_exists(session.conn, case_id, f"{case_id}-TEXT", case_id)
                session.conn.commit()
                session.save_extraction_result(result, case_id=case_id, source_text=text)
                console.print(f"  [green]OK[/green]  {case_id}: {result.summary()}")
                ok += 1
            except Exception as exc:
                console.print(f"  [bold red]ERR[/bold red] {case_id} (save): {exc}")
                errors_save += 1

    console.print(
        f"\n[bold]ingest-folder zakończony[/bold]: "
        f"[green]{ok} OK[/green]  "
        f"[yellow]{skipped} pominięto[/yellow]  "
        f"[red]{errors_extract + errors_verify + errors_save} błędów[/red]"
    )


# ---------------------------------------------------------------------------
# run-case
# ---------------------------------------------------------------------------

_STATUS_STYLE_RC: dict[str, str] = {
    "observed":            "cyan",
    "inferred_candidate":  "yellow",
    "proved":              "bold green",
    "rejected":            "red",
    "retracted":           "dim",
}

_FEEDBACK_STYLE_RC: dict[str, str] = {
    "proved": "bold green",
    "blocked": "bold red",
    "not_proved": "yellow",
    "unknown": "dim",
}


def _print_run_case_facts(facts) -> None:
    console.print()
    console.print("[bold cyan]Facts[/bold cyan]")
    for i, f in enumerate(facts, start=1):
        args_str = ", ".join(
            a.entity_id if a.entity_id is not None else a.literal_value
            for a in f.args
        )
        truth_val = f.truth.value or "-"
        conf = f"{f.truth.confidence:.2f}" if f.truth.confidence is not None else "-"
        status_text = Text(f.status.value, style=_STATUS_STYLE_RC.get(f.status.value, ""))
        header = Text(f"{i}. ", style="dim")
        header.append(f.fact_id, style="bold yellow")
        header.append("  ")
        header.append(f.predicate, style="cyan")
        header.append(f"({args_str})")
        console.print(header)
        console.print(f"   status=", status_text, f"  truth={truth_val} ({conf})")
    console.print(f"[dim]{len(facts)} fact(s)[/dim]")


def _print_run_case_cluster_states(cluster_states, schemas) -> None:
    domain_by_name = {s.name: s.domain for s in schemas} if schemas else {}
    console.print()
    console.print("[bold cyan]Cluster States[/bold cyan]")
    for cs in cluster_states:
        domain = domain_by_name.get(cs.cluster_name, [])
        if domain and len(cs.logits) == len(domain):
            logit_str = "  ".join(f"{v}:{l:.2f}" for v, l in zip(domain, cs.logits))
        else:
            logit_str = "  ".join(f"{l:.2f}" for l in cs.logits)
        clamp_info = ""
        if cs.is_clamped:
            clamp_info = f"  [{'bold' if cs.clamp_hard else 'dim'}]clamped({cs.clamp_source})[/{'bold' if cs.clamp_hard else 'dim'}]"
        console.print(
            f"  [yellow]{cs.entity_id}[/yellow]  [cyan]{cs.cluster_name}[/cyan]"
            f"  [{logit_str}]{clamp_info}"
        )
    console.print(f"[dim]{len(cluster_states)} cluster_state(s)[/dim]")


def _ground_atom_to_text(atom, predicate_positions: dict[str, list[str]] | None = None) -> str:
    if atom is None:
        return "-"
    bindings = getattr(atom, "bindings", ()) or ()
    predicate = str(getattr(atom, "predicate", atom))
    if not bindings:
        return predicate

    ordered_values: list[str] = []
    binding_map = {str(role).upper(): str(value) for role, value in bindings}
    roles = (predicate_positions or {}).get(predicate.lower(), [])
    used_roles: set[str] = set()

    for role in roles:
        role_key = str(role).upper()
        value = binding_map.get(role_key)
        if value is None:
            continue
        ordered_values.append(value)
        used_roles.add(role_key)

    for role, value in bindings:
        role_key = str(role).upper()
        if role_key in used_roles:
            continue
        ordered_values.append(str(value))

    args = ", ".join(ordered_values)
    return f"{predicate}({args})" if args else predicate


def _parse_query_atom(query: str, predicate_positions: dict[str, list[str]] | None = None):
    from sv._utils import to_clingo_id
    from sv.types import GroundAtom

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


def _load_case_queries_for_case(conn, case_id: str) -> list[tuple[int, str, str | None]]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT cq.id, cq.query, cq.expected_result
            FROM case_queries cq
            JOIN cases c ON c.id = cq.case_id
            WHERE c.case_id = %s
            ORDER BY cq.id
            """,
            (case_id,),
        )
        return [
            (int(query_id), str(query_text), str(expected) if expected is not None else None)
            for query_id, query_text, expected in cur.fetchall()
        ]


def _print_run_case_feedback(
    feedback_items,
    predicate_positions: dict[str, list[str]] | None = None,
) -> None:
    console.print()
    console.print("[bold cyan]Verifier Feedback[/bold cyan]")
    if not feedback_items:
        console.print("[dim]0 feedback item(s)[/dim]")
        return

    for i, item in enumerate(feedback_items, start=1):
        outcome_text = Text(item.outcome, style=_FEEDBACK_STYLE_RC.get(item.outcome, ""))
        header = Text(f"{i}. ", style="dim")
        header.append(item.fact_id, style="bold yellow")
        header.append("  ")
        header.append(item.predicate, style="cyan")
        console.print(header)
        console.print(
            "   outcome=",
            outcome_text,
            f"  atom={_ground_atom_to_text(item.atom, predicate_positions)}",
        )
        if item.violated_naf:
            console.print(
                f"   violated_naf={[_ground_atom_to_text(atom, predicate_positions) for atom in item.violated_naf]}"
            )
        if item.missing_pos_body:
            console.print(
                f"   missing_pos_body={[_ground_atom_to_text(atom, predicate_positions) for atom in item.missing_pos_body]}"
            )
        if item.supporting_rule_ids:
            console.print(f"   supporting_rule_ids={list(item.supporting_rule_ids)}")
    console.print(f"[dim]{len(feedback_items)} feedback item(s)[/dim]")


def _print_run_case_query_feedback(query_feedback_rows) -> None:
    console.print()
    console.print("[bold cyan]Query Feedback[/bold cyan]")
    if not query_feedback_rows:
        console.print("[dim]0 query/queries[/dim]")
        return

    for i, row in enumerate(query_feedback_rows, start=1):
        outcome_text = Text(row["outcome"], style=_FEEDBACK_STYLE_RC.get(row["outcome"], ""))
        expected = row["expected"] or "-"
        console.print(f"{i}. qid={row['query_id']} query={row['query']}")
        console.print("   outcome=", outcome_text, f"  expected={expected}  atom={row['atom_text']}")
        if row["violated_naf"]:
            console.print(f"   violated_naf={row['violated_naf']}")
        if row["missing_pos_body"]:
            console.print(f"   missing_pos_body={row['missing_pos_body']}")
        if row["supporting_rule_ids"]:
            console.print(f"   supporting_rule_ids={row['supporting_rule_ids']}")
    console.print(f"[dim]{len(query_feedback_rows)} query feedback row(s)[/dim]")


def cmd_run_case(args: argparse.Namespace) -> None:
    from db import DBSession
    from pipeline.runner import ProposeVerifyRunner

    case_id = args.case_id
    query_feedback_rows: list[dict[str, object]] = []
    with DBSession.connect() as session:
        schemas = session.load_cluster_schemas()
        predicate_positions = session.load_predicate_positions()
        entities, facts, rules, cluster_states = session.load_case(case_id)

        runner = ProposeVerifyRunner.from_schemas(
            schemas,
            predicate_positions=predicate_positions,
        )
        result = runner.run(entities, facts, rules, cluster_states)
        proof_id = session.save_pipeline_result(result, case_id=case_id)
        case_queries = _load_case_queries_for_case(session.conn, case_id)
        for query_id, query_text, expected in case_queries:
            query_atom = _parse_query_atom(query_text, predicate_positions)
            feedback = runner.verifier.explain_query_atom(
                query_atom,
                derived_atoms=result.derived_atoms,
                proof_nodes=result.proof_nodes,
                rules=rules,
            )
            query_feedback_rows.append({
                "query_id": query_id,
                "query": query_text,
                "expected": expected,
                "outcome": feedback.outcome,
                "atom_text": _ground_atom_to_text(feedback.atom, predicate_positions),
                "violated_naf": [
                    _ground_atom_to_text(atom, predicate_positions)
                    for atom in feedback.violated_naf
                ],
                "missing_pos_body": [
                    _ground_atom_to_text(atom, predicate_positions)
                    for atom in feedback.missing_pos_body
                ],
                "supporting_rule_ids": list(feedback.supporting_rule_ids),
            })

    console.print(f"[bold green]run-case completed[/bold green]: {case_id}")
    if proof_id:
        console.print(f"proof_id: {proof_id}")
    console.print(result.summary())
    _print_run_case_facts(result.facts)
    _print_run_case_cluster_states(result.cluster_states, schemas)
    _print_run_case_feedback(result.candidate_feedback, predicate_positions)
    _print_run_case_query_feedback(query_feedback_rows)


def _load_case_ids(selected: list[str] | None) -> list[str]:
    if selected:
        return selected
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT case_id
            FROM cases
            ORDER BY id
            """
        )
        return [str(r[0]) for r in cur.fetchall()]


def _add_template_cluster_edges(data, node_index, schemas) -> set[tuple[str, str]]:
    import torch

    active_pairs: set[tuple[str, str]] = set()
    for src in schemas:
        src_type = f"c_{src.name}"
        src_map = node_index.cluster_node_to_idx.get(src.name, {})
        if not src_map:
            continue

        for dst in schemas:
            if src.name == dst.name:
                continue
            if src.entity_type != dst.entity_type:
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
                # No same-entity overlap — use cross-product of all nodes of the same entity_type
                pairs = sorted(set(
                    (si, di)
                    for si in src_map.values()
                    for di in dst_map.values()
                ))
            if not pairs:
                continue

            uniq_pairs = sorted(set(pairs))
            src_idx = [p[0] for p in uniq_pairs]
            dst_idx = [p[1] for p in uniq_pairs]
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
        extract_rules_from_mp_bank,
        fact_cluster_rule_signature,
    )
    from nn.clamp import apply_clamp
    from nn.config import NNConfig
    from nn.graph_builder import EdgeTypeSpec, supports_relation

    case_ids = _load_case_ids(args.case)
    if not case_ids:
        console.print("[bold red]No cases found.[/bold red]")
        return

    config = replace(NNConfig(), max_epochs=args.epochs)

    with DBSession.connect() as session:
        schemas = session.load_cluster_schemas()
        predicate_positions = session.load_predicate_positions()

        role_specs = [
            EdgeTypeSpec(
                src_type=f"c_{s.name}",
                relation="role_of",
                dst_type="fact",
                src_dim=s.dim,
                dst_dim=GraphBuilder.FACT_DIM,
            )
            for s in schemas
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
        cluster_type_dims = {s.name: s.dim for s in schemas}
        proposer = NeuralProposer(config, mp_bank, gate_bank, cluster_type_dims)

        graph_builder = GraphBuilder(schemas)
        memory_encoder = EntityMemoryBiasEncoder(schemas, config)
        trainer = ProposerTrainer(
            proposer=proposer,
            cluster_schemas=schemas,
            config=config,
            seed=args.seed,
        )

        train_cases: list[tuple[object, object]] = []
        active_cluster_pairs: set[tuple[str, str]] = set()
        active_support_pairs: set[tuple[str, str, str]] = set()
        loaded_case_ids: list[str] = []

        for case_id in case_ids:
            try:
                entities, facts, _rules, states = session.load_case(case_id)
            except ValueError:
                console.print(f"[yellow]Skipping missing case[/yellow]: {case_id}")
                continue

            data, node_index, _ = graph_builder.build(
                entities=entities,
                facts=facts,
                rules=[],
                cluster_states=states,
                memory_biases=None,
            )

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

            train_cases.append((data, node_index))
            loaded_case_ids.append(case_id)

        if not train_cases:
            console.print("[bold red]No training cases available.[/bold red]")
            return

        step_count = 0
        for metrics in trainer.train_epochs(train_cases):
            step_count += 1
            if step_count % len(train_cases) == 0:
                epoch = int(metrics.get("epoch", 0))
                total = float(metrics.get("L_total", 0.0))
                console.print(
                    f"[dim]epoch {epoch + 1}/{args.epochs}[/dim] "
                    f"L_total={total:.4f}"
                )

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
            if filtered:
                table = Table(title="Extracted Learned Rules (preview)", header_style="bold cyan")
                table.add_column("rule_id", style="bold yellow")
                table.add_column("weight", justify="right", style="dim")
                for rule in filtered[:20]:
                    w = rule.metadata.weight if rule.metadata.weight is not None else 0.0
                    table.add_row(rule.rule_id, f"{w:.4f}")
                console.print(table)
            return

        session.save_learned_rules(filtered, module_name=args.module)

    console.print(
        f"[bold green]learn-rules completed[/bold green]: "
        f"cases={len(loaded_case_ids)}, saved={len(filtered)}, module={args.module}"
    )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# explain
# ---------------------------------------------------------------------------

def _load_case_text_from_db(conn, case_id: str) -> str | None:
    """Zwraca content z tabeli sources powiązany z case_id (lub None)."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT s.content
            FROM sources s
            JOIN cases c ON c.source_id = s.id
            WHERE c.case_id = %s
            """,
            (case_id,),
        )
        row = cur.fetchone()
    return str(row[0]) if row and row[0] else None


def _load_proof_run_from_db(conn, case_id: str, proof_id: str | None):
    """
    Ładuje ProofRun z DB dla podanego case_id.
    Jeśli proof_id podane — ładuje konkretny; inaczej najnowszy.
    Zwraca sv.proof.ProofRun lub None.
    """
    from sv.proof import ProofRun, ProofStep

    with conn.cursor() as cur:
        if proof_id:
            cur.execute(
                """
                SELECT pr.id, pr.proof_id, pr.result, pr.proof_dag
                FROM proof_runs pr
                JOIN cases c ON c.id = pr.case_id
                WHERE c.case_id = %s AND pr.proof_id = %s
                """,
                (case_id, proof_id),
            )
        else:
            cur.execute(
                """
                SELECT pr.id, pr.proof_id, pr.result, pr.proof_dag
                FROM proof_runs pr
                JOIN cases c ON c.id = pr.case_id
                WHERE c.case_id = %s
                ORDER BY pr.id DESC
                LIMIT 1
                """,
                (case_id,),
            )
        run_row = cur.fetchone()

    if run_row is None:
        return None

    run_id, run_proof_id, result, proof_dag_raw = run_row
    dag = _coerce_json(proof_dag_raw)
    if isinstance(dag, dict):
        # Zachowaj atom jako klucz "atom" w każdym wpisie
        dag_list = [{"atom": k, **v} for k, v in dag.items()] if dag else []
    elif isinstance(dag, list):
        dag_list = dag
    else:
        dag_list = []

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT ps.step_order, ps.rule_id, ps.substitution, ps.used_fact_ids
            FROM proof_steps ps
            WHERE ps.run_id = %s
            ORDER BY ps.step_order
            """,
            (run_id,),
        )
        step_rows = cur.fetchall()

    steps = []
    for step_order, rule_id, substitution, used_fact_ids in step_rows:
        sub = _coerce_json(substitution) or {}
        used = list(used_fact_ids) if used_fact_ids else []
        steps.append(ProofStep(
            step_order=int(step_order),
            rule_id=rule_id,
            rule_text="",
            substitution=sub if isinstance(sub, dict) else {},
            used_fact_ids=used,
        ))

    return ProofRun(
        proof_id=str(run_proof_id),
        result=str(result),
        proof_dag=dag_list,
        steps=steps,
    )


def cmd_explain(args: argparse.Namespace) -> None:
    """
    Wysyła dane sprawy do Gemini i wyświetla wyjaśnienie w języku naturalnym.

    Tryby:
      pn3 explain TC-001                      — wywołaj Gemini, wydrukuj odpowiedź
      pn3 explain TC-001 --dry-run            — podgląd promptu (bez API)
      pn3 explain TC-001 --proof-id UUID      — użyj konkretnego proof run
      pn3 explain TC-001 --file text.txt      — fallback tekstu gdy nie ma w DB
      pn3 explain TC-001 --output result.txt  — zapisz wyjaśnienie do pliku
    """
    from rich.panel import Panel

    from config import ProjectConfig
    from db import DBSession
    from explainer import LLMExplainer

    case_id: str = args.case_id
    cfg = ProjectConfig.load()

    with DBSession.connect() as session:
        schemas = session.load_cluster_schemas()
        entities, facts, _rules, cluster_states = session.load_case(case_id)

        case_text = _load_case_text_from_db(session.conn, case_id)
        if not case_text and args.file:
            case_text = Path(args.file).read_text(encoding="utf-8")
        if not case_text:
            console.print(
                f"[yellow]Brak tekstu sprawy dla {case_id}.[/yellow] "
                "Podaj --file lub najpierw uruchom ingest-text."
            )
            case_text = ""

        proof_run = _load_proof_run_from_db(session.conn, case_id, args.proof_id)

    if not facts:
        console.print(f"[bold red]Brak faktów dla case {case_id}.[/bold red] Uruchom najpierw run-case.")
        return

    # Wyciągnij neural_trace z proweniencji faktów (załadowanej z DB przez fact_repo)
    neural_trace = {
        f.fact_id: f.provenance.neural_trace
        for f in facts
        if f.provenance and f.provenance.neural_trace
    } or None

    explainer = LLMExplainer(cfg.explainer)

    if args.dry_run:
        req = explainer.preview_request(
            case_text=case_text,
            facts=facts,
            proof_run=proof_run,
            cluster_states=cluster_states,
            cluster_schemas=schemas,
            entities=entities,
            neural_trace=neural_trace,
        )
        if args.raw:
            print("=== SYSTEM PROMPT ===")
            print(req["system_prompt"])
            print("\n=== USER MESSAGE ===")
            print(req["user_message"])
        else:
            console.print(Panel(
                req["system_prompt"],
                title=f"[bold cyan]System Prompt[/bold cyan]  "
                      f"[dim](model: {req['model']}, lang: {req['language']})[/dim]",
                border_style="cyan",
                expand=True,
            ))
            console.print()
            console.print(Panel(
                req["user_message"],
                title="[bold yellow]User Message[/bold yellow]",
                border_style="yellow",
                expand=True,
            ))
            nn_trace_count = sum(len(v) for v in (neural_trace or {}).values())
            console.print(
                f"\n[dim]Fakty: {len(facts)} | "
                f"Klastry: {len(cluster_states)} | "
                f"Proof: {'tak' if proof_run else 'brak'} | "
                f"Neural trace: {nn_trace_count} wpisów[/dim]"
            )
        return

    _api_key = os.environ.get(cfg.explainer.api_key_env, "")
    _key_hint = f"…{_api_key[-4:]}" if _api_key else "BRAK"
    console.print(f"[dim]Generuję wyjaśnienie dla {case_id} ({cfg.explainer.gemini_model}, key={_key_hint})...[/dim]")
    explanation = explainer.explain(
        case_text=case_text,
        facts=facts,
        proof_run=proof_run,
        cluster_states=cluster_states,
        cluster_schemas=schemas,
        entities=entities,
        neural_trace=neural_trace,
    )

    if args.output:
        Path(args.output).write_text(explanation, encoding="utf-8")
        console.print(f"[bold green]Zapisano:[/bold green] {args.output}")
    elif args.raw:
        print(explanation)
    else:
        console.print(Panel(
            explanation,
            title=f"[bold cyan]Wyjaśnienie: {case_id}[/bold cyan]",
            border_style="cyan",
            expand=True,
        ))

# ---------------------------------------------------------------------------
# gen-ontology
# ---------------------------------------------------------------------------

def cmd_gen_ontology(args: argparse.Namespace) -> None:
    """
    Generuje ontologię z tekstu regulaminu przez Gemini i ładuje do DB.

    Tryby:
      pn3 gen-ontology --file regulamin.txt [--source-id REG-001] [--dry-run] [--raw]
      pn3 gen-ontology --text "§1. ..."
    """
    import json as _json
    from pathlib import Path as _Path
    from rich.panel import Panel
    from rich.syntax import Syntax
    from rich.table import Table as RichTable

    from config import ProjectConfig
    from db import DBSession
    from nlp.ontology_builder import (
        build_ontology_correction_prompt,
        build_ontology_prompt,
        build_ontology_schema,
        parse_ontology_response,
    )

    if args.file:
        text = _Path(args.file).read_text(encoding="utf-8").strip()
    elif args.text:
        text = args.text.strip()
    else:
        console.print("[bold red]Podaj --text lub --file.[/bold red]")
        raise SystemExit(1)

    if not text:
        console.print("[bold red]Tekst jest pusty.[/bold red]")
        raise SystemExit(1)

    source_id: str = args.source_id or "regulation"
    cfg = ProjectConfig.load()

    _api_key = os.environ.get(cfg.extractor.api_key_env, "")
    if not _api_key:
        console.print(
            f"[bold red]Brak klucza API Gemini.[/bold red] "
            f"Ustaw zmienną środowiskową: {cfg.extractor.api_key_env}"
        )
        raise SystemExit(1)

    _key_hint = f"…{_api_key[-4:]}"
    console.print(
        f"[dim]gen-ontology replace (model={cfg.extractor.gemini_model}, key={_key_hint}) "
        f"source_id={source_id}...[/dim]"
    )

    # Wywołaj Gemini
    try:
        from google import genai
    except ImportError:
        console.print(
            "[bold red]Brak pakietu google-genai.[/bold red] "
            "Zainstaluj: pip install google-genai"
        )
        raise SystemExit(1)

    client = genai.Client(api_key=_api_key)
    schema = build_ontology_schema()
    _gen_cfg = {
        "temperature": 0.0,
        "response_mime_type": "application/json",
        "response_schema": schema,
    }

    _max_retries: int = getattr(cfg.extractor, "max_retries", 2)
    current_prompt: str = build_ontology_prompt(text)
    result = None

    for attempt in range(_max_retries + 1):
        is_correction = attempt > 0
        attempt_label = (
            f"[dim]korekta {attempt}/{_max_retries}[/dim]"
            if is_correction
            else "[dim]attempt 1[/dim]"
        )
        console.print(f"  {attempt_label} — wywolanie Gemini ({cfg.extractor.gemini_model})...")

        response = client.models.generate_content(
            model=cfg.extractor.gemini_model,
            contents=current_prompt,
            config=_gen_cfg,
        )
        raw: dict = _json.loads(response.text)
        result = parse_ontology_response(raw, source_id)

        n_err = len(result.validation_errors)
        if not result.validation_errors:
            console.print(
                f"  [green]OK[/green] — {result.summary()}"
                + (f" (po {attempt} korekcie)" if attempt > 0 else "")
            )
            break

        if attempt < _max_retries:
            console.print(
                f"  [yellow]Odrzucono {n_err} regul(e) — wysylam korekce ({attempt + 1}/{_max_retries})...[/yellow]"
            )
            for err in result.validation_errors:
                console.print(f"    [dim]• {err}[/dim]")
            current_prompt = build_ontology_correction_prompt(text, result.validation_errors)
        else:
            console.print(
                f"  [bold red]Wyczerpano retries — zapisuje {len(result.rules)} poprawnych regul "
                f"(odrzucono {n_err}).[/bold red]"
            )
            for err in result.validation_errors:
                console.print(f"    [dim]• {err}[/dim]")

    if args.raw:
        print(_json.dumps(raw, indent=2, ensure_ascii=False))
        if args.dry_run:
            return

    if args.dry_run:
        console.print(
            f"[bold yellow]dry-run[/bold yellow] — {result.summary()}"
        )
        _print_ontology_tables(result)
        return

    with DBSession.connect() as session:
        session.save_ontology(result)

    console.print(
        f"[bold green]gen-ontology completed[/bold green]: {result.summary()} "
        f"(active ontology replaced)"
    )
    _print_ontology_tables(result)


def _print_ontology_tables(result) -> None:
    from rich.table import Table as RichTable

    if result.entity_types:
        t = RichTable(title="Entity Types", header_style="bold cyan", show_lines=False)
        t.add_column("name", style="bold yellow", no_wrap=True)
        t.add_column("source_span", style="dim")
        for et in result.entity_types:
            t.add_row(et.name, (et.source_span_text or "")[:80])
        console.print(t)

    if result.predicates:
        t = RichTable(title="Predicates", header_style="bold cyan", show_lines=False)
        t.add_column("name", style="bold yellow", no_wrap=True)
        t.add_column("roles", style="cyan")
        t.add_column("source_span", style="dim")
        for pred in result.predicates:
            roles_str = ", ".join(
                f"{r.position}:{r.role}({'?' if r.entity_type is None else r.entity_type})"
                for r in pred.roles
            )
            t.add_row(pred.name, roles_str, (pred.source_span_text or "")[:60])
        console.print(t)

    if result.clusters:
        t = RichTable(title="Clusters", header_style="bold cyan", show_lines=False)
        t.add_column("name", style="bold yellow", no_wrap=True)
        t.add_column("entity_type", style="cyan", no_wrap=True)
        t.add_column("domain")
        t.add_column("source_span", style="dim")
        for cl in result.clusters:
            t.add_row(
                cl.name, cl.entity_type,
                " | ".join(cl.domain),
                (cl.source_span_text or "")[:60],
            )
        console.print(t)

    if result.rules:
        t = RichTable(title="Rules", header_style="bold cyan", show_lines=False)
        t.add_column("rule_id", style="bold yellow", no_wrap=True)
        t.add_column("module", style="cyan", no_wrap=True)
        t.add_column("stratum", justify="right", style="dim")
        t.add_column("clingo_text")
        for rule in result.rules:
            t.add_row(
                rule.rule_id, rule.module, str(rule.stratum),
                rule.clingo_text[:80],
            )
        console.print(t)


def cmd_eval(args: argparse.Namespace) -> None:
    """
    Run evaluation pipeline and save JSON/CSV reports.
    Wraps eval/run_eval.py to keep a single CLI entrypoint.
    """
    script_path = Path(__file__).resolve().parent.parent / "eval" / "run_eval.py"
    if not script_path.exists():
        console.print(f"[bold red]Evaluation script not found:[/bold red] {script_path}")
        raise SystemExit(1)

    cmd: list[str] = [
        sys.executable,
        str(script_path),
        "--replay",
        str(args.replay),
        "--output-json",
        args.output_json,
    ]
    if args.output_csv:
        cmd.extend(["--output-csv", args.output_csv])
    if args.include_details:
        cmd.append("--include-details")
    if args.case:
        for case_id in args.case:
            cmd.extend(["--case", case_id])

    console.print(f"[dim]Running eval: {' '.join(cmd)}[/dim]")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.stdout:
        print(proc.stdout, end="" if proc.stdout.endswith("\n") else "\n")
    if proc.returncode != 0:
        if proc.stderr:
            print(proc.stderr, end="" if proc.stderr.endswith("\n") else "\n", file=sys.stderr)
        raise SystemExit(proc.returncode)


_COMMANDS = {
    "_extract-json": (cmd_extract_json, argparse.SUPPRESS),
    "_verify-json": (cmd_verify_json, argparse.SUPPRESS),
    "entity-types":  (cmd_entity_types,  "List entity type definitions"),
    "predicates":    (cmd_predicates,    "List predicate definitions with roles"),
    "clusters":      (cmd_clusters,      "List cluster definitions with domain values"),
    "rule-modules":  (cmd_rule_modules,  "List rule modules with rule counts"),
    "sources":       (cmd_sources,       "List source documents"),
    "cases":         (cmd_cases,         "List test cases and their queries"),
    "entities":      (cmd_entities,      "List entity instances"),
    "facts":         (cmd_facts,         "List facts"),
    "rules":         (cmd_rules,         "List rules"),
    "proof":         (cmd_proof,         "Show one proof run by proof_id"),
    "proof-graph":   (cmd_proof_graph,   "Export one proof run as Graphviz diagram"),
    "network-graph": (cmd_network_graph, "Export full case network graph (entities/facts/clusters/edges)"),
    "reset-state":   (cmd_reset_state,   "Reset runtime artifacts (learned rules, inferred facts, proofs, traces)"),
    "gen-ontology":  (cmd_gen_ontology,  "Generate ontology from regulatory text using LLM and save to DB"),
    "ingest-text":   (cmd_ingest_text,   "Extract facts from text and save to DB for a given case_id"),
    "ingest-folder": (cmd_ingest_folder, "Extract facts from all .txt files in a folder (case_id = filename)"),
    "run-case":      (cmd_run_case,      "Run full pipeline for one case_id and save results"),
    "learn-rules":   (cmd_learn_rules,   "Train NN proposer and persist extracted learned rules"),
    "eval":          (cmd_eval,          "Run evaluation metrics and save JSON/CSV report"),
    "llm-prompt":    (cmd_llm_prompt,    "Preview LLM system prompt and JSON schema (no API call)"),
    "explain":       (cmd_explain,       "Explain case results in natural language using Gemini"),
}


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="pn3",
        description="ProveNuance3 – CLI for inspecting the database",
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True
    for name, (_, help_text) in _COMMANDS.items():
        cmd_parser = sub.add_parser(name, help=help_text)
        if name == "_extract-json":
            cmd_parser.add_argument(
                "--file",
                required=True,
                metavar="PATH",
                help=argparse.SUPPRESS,
            )
            cmd_parser.add_argument(
                "--source-id",
                required=True,
                metavar="ID",
                help=argparse.SUPPRESS,
            )
        elif name == "_verify-json":
            cmd_parser.add_argument(
                "--input-json",
                required=True,
                metavar="PATH",
                help=argparse.SUPPRESS,
            )
        elif name == "run-case":
            cmd_parser.add_argument("case_id", help="Case ID, e.g. TC-001")
        elif name == "proof":
            cmd_parser.add_argument("proof_id", help="Proof run ID (facts.proof_id / proof_runs.proof_id)")
            cmd_parser.add_argument(
                "--dag",
                action="store_true",
                help="Print full proof_dag JSON.",
            )
        elif name == "proof-graph":
            cmd_parser.add_argument("proof_id", help="Proof run ID (facts.proof_id / proof_runs.proof_id)")
            cmd_parser.add_argument(
                "--output",
                help="Output file base path (without extension). Default: proof_<proof_id> in current dir.",
            )
            cmd_parser.add_argument(
                "--format",
                choices=("dot", "svg", "png", "pdf"),
                default="svg",
                help="Output format (default: svg).",
            )
            cmd_parser.add_argument(
                "--engine",
                default="dot",
                help="Graphviz engine binary name (default: dot).",
            )
        elif name == "network-graph":
            cmd_parser.add_argument("case_id", help="Case ID, e.g. TC-001")
            cmd_parser.add_argument(
                "--output",
                help="Output file base path (without extension). Default: network_<case_id> in current dir.",
            )
            cmd_parser.add_argument(
                "--format",
                choices=("dot", "svg", "png", "pdf"),
                default="svg",
                help="Output format (default: svg).",
            )
            cmd_parser.add_argument(
                "--engine",
                default="dot",
                help="Graphviz engine binary name (default: dot).",
            )
            cmd_parser.add_argument(
                "--no-entities",
                action="store_true",
                help="Hide ENTITY nodes and only render fact/cluster network.",
            )
        elif name == "reset-state":
            cmd_parser.add_argument(
                "--yes",
                action="store_true",
                help="Apply reset (without this flag command runs in preview mode).",
            )
            cmd_parser.add_argument(
                "--keep-cluster-states",
                action="store_true",
                help="Do not delete non-clamped cluster_states rows.",
            )
            cmd_parser.add_argument(
                "--case-id",
                default=None,
                metavar="CASE_ID",
                help="Limit reset to a single case (e.g. TC-010). Learned rules are never deleted in per-case mode.",
            )
            cmd_parser.add_argument(
                "--ontology",
                action="store_true",
                help=(
                    "FULL wipe: usuwa całą ontologię (entity_types, predicates, clusters, rules) "
                    "oraz wszystkie zależne dane (entities, facts, cases, sources). "
                    "Używaj po gen-ontology gdy chcesz przeładować regulamin od zera. "
                    "Niezgodne z --case-id."
                ),
            )
        elif name == "learn-rules":
            cmd_parser.add_argument(
                "--case",
                action="append",
                help="Case ID to use for training (repeatable). Default: all cases.",
            )
            cmd_parser.add_argument(
                "--epochs",
                type=int,
                default=20,
                help="Training epochs (default: 20).",
            )
            cmd_parser.add_argument(
                "--min-weight",
                type=float,
                default=0.5,
                help="Min extracted rule weight (default: 0.5).",
            )
            cmd_parser.add_argument(
                "--top-k",
                type=int,
                default=2,
                help="Top-k targets per source value during extraction (default: 2).",
            )
            cmd_parser.add_argument(
                "--module",
                default="learned_nn",
                help="Target rule_modules.name for saved learned rules (default: learned_nn).",
            )
            cmd_parser.add_argument(
                "--rule-prefix",
                default="learned.nn",
                help="Prefix for generated rule_id (default: learned.nn).",
            )
            cmd_parser.add_argument(
                "--seed",
                type=int,
                default=42,
                help="Training RNG seed (default: 42).",
            )
            cmd_parser.add_argument(
                "--dry-run",
                action="store_true",
                help="Train and extract, but do not save rules to DB.",
            )
        elif name == "eval":
            cmd_parser.add_argument(
                "--case",
                action="append",
                help="Case ID to evaluate (repeatable). Default: all cases.",
            )
            cmd_parser.add_argument(
                "--replay",
                type=int,
                default=1,
                help="How many times to replay each case for stability check (default: 1).",
            )
            cmd_parser.add_argument(
                "--output-json",
                default="eval_report.json",
                metavar="PATH",
                help="Output JSON report path (default: eval_report.json).",
            )
            cmd_parser.add_argument(
                "--output-csv",
                default=None,
                metavar="PATH",
                help="Optional CSV path with per-query details.",
            )
            cmd_parser.add_argument(
                "--include-details",
                action="store_true",
                help="Include per-query details in JSON output.",
            )
        elif name == "gen-ontology":
            group = cmd_parser.add_mutually_exclusive_group(required=True)
            group.add_argument(
                "--file",
                metavar="PATH",
                help="Ścieżka do pliku .txt z tekstem regulaminu.",
            )
            group.add_argument(
                "--text",
                metavar="TEXT",
                help="Tekst regulaminu (inline).",
            )
            cmd_parser.add_argument(
                "--source-id",
                metavar="ID",
                default=None,
                help="Identyfikator regulaminu w DB (domyślnie: 'regulation').",
            )
            cmd_parser.add_argument(
                "--dry-run",
                action="store_true",
                help="Pokaż wynik bez zapisu do DB.",
            )
            cmd_parser.add_argument(
                "--raw",
                action="store_true",
                help="Wydrukuj surowy JSON z Gemini.",
            )
        elif name == "ingest-text":
            cmd_parser.add_argument("case_id", help="Case ID, np. TC-002")
            group = cmd_parser.add_mutually_exclusive_group(required=True)
            group.add_argument(
                "--text",
                metavar="TEXT",
                help="Tekst sprawy (inline).",
            )
            group.add_argument(
                "--file",
                metavar="PATH",
                help="Ścieżka do pliku .txt z tekstem sprawy.",
            )
            cmd_parser.add_argument(
                "--title",
                metavar="TITLE",
                default=None,
                help="Tytuł case'u (domyślnie: case_id).",
            )
            cmd_parser.add_argument(
                "--create",
                action="store_true",
                help="Utwórz case w DB jeśli nie istnieje.",
            )
            cmd_parser.add_argument(
                "--dry-run",
                action="store_true",
                help="Pokaż wynik ekstrakcji bez zapisu do DB.",
            )
            cmd_parser.add_argument(
                "--backend",
                choices=["llm"],
                default=None,
                help="Backend ekstrakcji: tylko llm (Gemini).",
            )
        elif name == "ingest-folder":
            cmd_parser.add_argument("folder", help="Ścieżka do folderu z plikami .txt")
            cmd_parser.add_argument(
                "--pattern",
                default="*.txt",
                metavar="GLOB",
                help="Wzorzec glob plików (domyślnie: *.txt).",
            )
            cmd_parser.add_argument(
                "--backend",
                choices=["llm"],
                default=None,
                help="Backend ekstrakcji: tylko llm (Gemini).",
            )
            cmd_parser.add_argument(
                "--dry-run",
                action="store_true",
                help="Pokaż wynik ekstrakcji bez zapisu do DB.",
            )
            cmd_parser.add_argument(
                "--workers",
                type=int,
                default=8,
                metavar="N",
                help="Liczba równoległych wątków ekstrakcji (domyślnie: 8).",
            )
        elif name == "llm-prompt":
            cmd_parser.add_argument(
                "--text",
                metavar="TEXT",
                default=None,
                help="Pokaż również user message dla podanego tekstu.",
            )
            cmd_parser.add_argument(
                "--json-schema",
                action="store_true",
                help="Pokaż JSON Schema odpowiedzi Gemini.",
            )
            cmd_parser.add_argument(
                "--raw",
                action="store_true",
                help="Wyjście jako czysty tekst (bez Rich formatowania).",
            )
        elif name == "explain":
            cmd_parser.add_argument("case_id", help="Case ID, np. TC-001")
            cmd_parser.add_argument(
                "--proof-id",
                default=None,
                metavar="UUID",
                help="Konkretny proof run ID (domyślnie: najnowszy dla case).",
            )
            cmd_parser.add_argument(
                "--file",
                default=None,
                metavar="PATH",
                help="Plik z tekstem sprawy (fallback gdy tekst nie jest w DB).",
            )
            cmd_parser.add_argument(
                "--dry-run",
                action="store_true",
                help="Pokaż prompt bez wywoływania Gemini.",
            )
            cmd_parser.add_argument(
                "--output",
                default=None,
                metavar="PATH",
                help="Zapisz wyjaśnienie do pliku zamiast na ekran.",
            )
            cmd_parser.add_argument(
                "--raw",
                action="store_true",
                help="Wyjście jako czysty tekst (bez Rich formatowania).",
            )

    args = parser.parse_args()

    try:
        _COMMANDS[args.command][0](args)
    except psycopg2.OperationalError as exc:
        console.print(f"[bold red]DB connection error:[/bold red] {exc}")
        console.print(
            f"[dim]Connection: {DSN['user']}@{DSN['host']}:{DSN['port']}/{DSN['dbname']}[/dim]"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
