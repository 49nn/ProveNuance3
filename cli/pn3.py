#!/usr/bin/env python3
"""pn3 – ProveNuance3 command-line interface."""

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from dataclasses import replace

# Load .env from project root (silently ignored if file doesn't exist)
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass

import psycopg2
from rich.console import Console
from rich.table import Table
from rich.text import Text

# Force UTF-8 output so Rich unicode symbols work on Windows legacy consoles.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

console = Console()

DSN = dict(
    host=os.getenv("PN3_HOST", "localhost"),
    port=int(os.getenv("PN3_PORT", "5432")),
    dbname=os.getenv("PN3_DB", "provenuance"),
    user=os.getenv("PN3_USER", "provenuance"),
    password=os.getenv("PN3_PASSWORD", "provenuance"),
)


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
                f.created_at
            FROM facts f
            LEFT JOIN cases c ON c.id = f.case_id
            ORDER BY f.id
        """)
        rows = cur.fetchall()

    console.print("[bold cyan]Facts[/bold cyan]")
    if not rows:
        console.print("[dim]0 row(s)[/dim]")
        return

    for i, (
        case_id, fact_id, predicate, arity, status, truth_val, conf,
        n_args, args_list, source_id, source_extractor, proof_id, proof_value, n_trace, created_at,
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
                r.created_at
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
        rule_id, module, stratum, enabled, learned, weight, precision, support, created_at
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
            elif relation == "supports":
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

    dag_obj = _coerce_json(proof_dag)
    dag_map = dag_obj if isinstance(dag_obj, dict) else {}

    def _sub_key(value) -> str:
        obj = _coerce_json(value) or {}
        if not isinstance(obj, dict):
            return "-"
        return json.dumps(obj, sort_keys=True, ensure_ascii=False)

    # Best-effort map step -> (atom, status) using (rule_id, substitution).
    step_atom_index: dict[tuple[str, str], list[tuple[str, str]]] = {}
    for atom, node in dag_map.items():
        if not isinstance(node, dict):
            continue
        key = (
            str(node.get("rule_id") or ""),
            _sub_key(node.get("substitution")),
        )
        status = str(node.get("status") or "-")
        step_atom_index.setdefault(key, []).append((str(atom), status))

    console.print("[bold cyan]Proof Run[/bold cyan]")
    console.print(f"proof_id: {run_proof_id}")
    console.print(f"case_id: {case_id}")
    console.print(f"query: {query}")
    console.print(f"result: {result}")
    console.print(f"created_at: {str(created_at)[:19]}")
    console.print(f"steps: {len(step_rows)} linked_facts: {len(fact_rows)}")

    console.print()
    console.print("[bold cyan]Linked Facts[/bold cyan]")
    if not fact_rows:
        console.print("[dim]-[/dim]")
    else:
        for i, (fact_id, predicate, status) in enumerate(fact_rows, start=1):
            status_text = Text(status, style=_STATUS_STYLE.get(status, ""))
            console.print(f"{i}. {fact_id} {predicate} ", status_text)

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
            if candidates:
                atom, atom_status = candidates.pop(0)
            console.print(
                f"{step_order}. atom={atom} status={atom_status} rule={rule_id or '-'} "
                f"sub={sub if sub else '-'} "
                f"used_facts={used if used else '-'}"
            )

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

def _collect_reset_counts(conn, prune_cluster_states: bool) -> dict[str, int]:
    with conn.cursor() as cur:
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


def _apply_reset(conn, prune_cluster_states: bool) -> dict[str, int]:
    result: dict[str, int] = {}
    with conn.cursor() as cur:
        cur.execute("DELETE FROM rules WHERE learned = TRUE")
        result["deleted_learned_rules"] = cur.rowcount

        cur.execute("DELETE FROM proof_runs")
        result["deleted_proof_runs"] = cur.rowcount

        # Czyścimy jawnie, choć zwykle po usunięciu facts i tak byłoby puste.
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

    return result


def cmd_reset_state(args: argparse.Namespace) -> None:
    prune_cluster_states = not args.keep_cluster_states

    with connect() as conn:
        counts = _collect_reset_counts(conn, prune_cluster_states=prune_cluster_states)

        console.print("[bold cyan]Reset Preview[/bold cyan]")
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
            applied = _apply_reset(conn, prune_cluster_states=prune_cluster_states)
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    console.print()
    console.print("[bold green]Reset Applied[/bold green]")
    console.print(f"deleted learned rules: {applied['deleted_learned_rules']}")
    console.print(f"deleted proof runs: {applied['deleted_proof_runs']}")
    console.print(f"deleted neural traces: {applied['deleted_neural_traces']}")
    console.print(f"deleted non-observed facts: {applied['deleted_non_observed_facts']}")
    console.print(f"reset observed facts (proof/truth): {applied['reset_observed_facts']}")
    console.print(f"deleted non-clamped cluster_states: {applied['deleted_non_clamped_cluster_states']}")
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
    from sv.schema import PREDICATE_POSITIONS

    cfg = ProjectConfig.load()

    with DBSession.connect() as session:
        schemas = session.load_cluster_schemas()

    system_prompt = build_system_prompt(schemas, PREDICATE_POSITIONS)

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
              f"[dim](model: {cfg.extractor.gemini_model}, "
              f"backend: {cfg.extractor.backend})[/dim]",
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
        f"Predicates: {len(PREDICATE_POSITIONS)}[/dim]"
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


def cmd_ingest_text(args: argparse.Namespace) -> None:
    """
    Ekstrahuje fakty z tekstu i zapisuje do DB dla podanego case_id.

    Tryby:
      pn3 ingest-text TC-002 --text "Złożyłem zamówienie..."
      pn3 ingest-text TC-002 --file text_cases/TC-002.txt
      pn3 ingest-text TC-002 --file ... --create      # utwórz case jeśli nie istnieje
      pn3 ingest-text TC-002 --file ... --dry-run     # pokaż wynik bez zapisu
    """
    from pathlib import Path

    from config import ProjectConfig
    from db import DBSession
    from nlp import get_extractor

    case_id: str = args.case_id

    if args.file:
        text = Path(args.file).read_text(encoding="utf-8").strip()
    elif args.text:
        text = args.text.strip()
    else:
        console.print("[bold red]Podaj --text lub --file.[/bold red]")
        raise SystemExit(1)

    if not text:
        console.print("[bold red]Tekst jest pusty.[/bold red]")
        raise SystemExit(1)

    source_id = f"{case_id}-TEXT"
    title = args.title or case_id
    cfg = ProjectConfig.load()

    # --backend nadpisuje konfigurację z pliku
    if getattr(args, "backend", None):
        from dataclasses import replace as dc_replace
        cfg = dc_replace(cfg, extractor=dc_replace(cfg.extractor, backend=args.backend))

    with DBSession.connect() as session:
        schemas = session.load_cluster_schemas()

        if args.create:
            with session.conn.transaction():
                _ensure_case_exists(session.conn, case_id, source_id, title)

        console.print(
            f"[dim]Ekstrakcja ({cfg.extractor.backend}) → {case_id}...[/dim]"
        )
        extractor = get_extractor(schemas, cfg)
        result = extractor.extract(text, source_id=source_id)

        if args.dry_run:
            console.print(
                f"[bold yellow]dry-run[/bold yellow] — wynik dla {case_id}:"
            )
            console.print(result.summary())
            return

        session.save_extraction_result(result, case_id=case_id, source_text=text)

    console.print(
        f"[bold green]ingest-text completed[/bold green]: {case_id}"
    )
    console.print(result.summary())


# ---------------------------------------------------------------------------
# run-case
# ---------------------------------------------------------------------------

def cmd_run_case(args: argparse.Namespace) -> None:
    from db import DBSession
    from pipeline.runner import ProposeVerifyRunner

    case_id = args.case_id
    with DBSession.connect() as session:
        schemas = session.load_cluster_schemas()
        entities, facts, rules, cluster_states = session.load_case(case_id)

        runner = ProposeVerifyRunner.from_schemas(schemas)
        result = runner.run(entities, facts, rules, cluster_states)
        session.save_pipeline_result(result, case_id=case_id)

    console.print(f"[bold green]run-case completed[/bold green]: {case_id}")
    console.print(result.summary())


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
    )
    from nn.clamp import apply_clamp
    from nn.config import NNConfig
    from nn.graph_builder import EdgeTypeSpec

    case_ids = _load_case_ids(args.case)
    if not case_ids:
        console.print("[bold red]No cases found.[/bold red]")
        return

    config = replace(NNConfig(), max_epochs=args.epochs)

    with DBSession.connect() as session:
        schemas = session.load_cluster_schemas()

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

        mp_bank = HeteroMessagePassingBank(role_specs + implies_specs)
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
        )

        filtered = [
            r for r in extracted
            if r.body and (r.body[0].predicate, r.head.predicate) in active_cluster_pairs
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

    explainer = LLMExplainer(cfg.explainer)

    if args.dry_run:
        req = explainer.preview_request(
            case_text=case_text,
            facts=facts,
            proof_run=proof_run,
            cluster_states=cluster_states,
            entities=entities,
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
            console.print(
                f"\n[dim]Fakty: {len(facts)} | "
                f"Klastry: {len(cluster_states)} | "
                f"Proof: {'tak' if proof_run else 'brak'}[/dim]"
            )
        return

    console.print(f"[dim]Generuję wyjaśnienie dla {case_id} ({cfg.explainer.gemini_model})...[/dim]")
    explanation = explainer.explain(
        case_text=case_text,
        facts=facts,
        proof_run=proof_run,
        cluster_states=cluster_states,
        entities=entities,
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


_COMMANDS = {
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
    "ingest-text":   (cmd_ingest_text,   "Extract facts from text and save to DB for a given case_id"),
    "run-case":      (cmd_run_case,      "Run full pipeline for one case_id and save results"),
    "learn-rules":   (cmd_learn_rules,   "Train NN proposer and persist extracted learned rules"),
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
        if name == "run-case":
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
                choices=["regex", "llm"],
                default=None,
                help="Backend ekstrakcji: regex (domyślnie z config) lub llm (Gemini).",
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
