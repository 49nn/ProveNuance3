#!/usr/bin/env python3
"""pn3 – ProveNuance3 command-line interface."""

import argparse
import os
import sys

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
                f.fact_id,
                f.predicate,
                f.arity,
                f.status,
                f.truth_value,
                f.truth_confidence,
                COUNT(fa.position)  AS n_args,
                f.source_id,
                f.created_at
            FROM facts f
            LEFT JOIN fact_args fa ON fa.fact_id = f.id
            GROUP BY f.id
            ORDER BY f.id
        """)
        rows = cur.fetchall()

    table = Table(title="Facts", header_style="bold cyan", show_lines=False)
    table.add_column("fact_id",    style="bold yellow", no_wrap=True)
    table.add_column("predicate",  style="cyan")
    table.add_column("arity",      justify="right", style="dim")
    table.add_column("args",       justify="right", style="dim")
    table.add_column("status",     no_wrap=True)
    table.add_column("truth",      justify="center")
    table.add_column("conf",       justify="right", style="dim")
    table.add_column("source",     style="dim", no_wrap=True)
    table.add_column("created_at", style="dim", no_wrap=True)

    for fact_id, predicate, arity, status, truth_val, conf, n_args, source_id, created_at in rows:
        s_style = _STATUS_STYLE.get(status, "")
        status_text = Text(status, style=s_style)
        table.add_row(
            fact_id,
            predicate,
            str(arity) if arity is not None else "–",
            str(n_args),
            status_text,
            truth_val or "–",
            f"{conf:.2f}" if conf is not None else "–",
            source_id or "–",
            str(created_at)[:19],
        )

    console.print(table)
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

    table = Table(title="Rules", header_style="bold cyan", show_lines=False)
    table.add_column("rule_id",    style="bold yellow", no_wrap=True)
    table.add_column("module",     style="cyan")
    table.add_column("stratum",    justify="right")
    table.add_column("enabled",    justify="center")
    table.add_column("learned",    justify="center")
    table.add_column("weight",     justify="right", style="dim")
    table.add_column("precision",  justify="right", style="dim")
    table.add_column("support",    justify="right", style="dim")
    table.add_column("created_at", style="dim", no_wrap=True)

    for rule_id, module, stratum, enabled, learned, weight, precision, support, created_at in rows:
        table.add_row(
            rule_id,
            module,
            str(stratum),
            Text("✓", style="green") if enabled  else Text("✗", style="red"),
            Text("✓", style="yellow") if learned else Text("–", style="dim"),
            f"{weight:.3f}"    if weight    is not None else "–",
            f"{precision:.2f}" if precision is not None else "–",
            str(support)       if support   is not None else "–",
            str(created_at)[:19],
        )

    console.print(table)
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
# main
# ---------------------------------------------------------------------------

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
}


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="pn3",
        description="ProveNuance3 – CLI for inspecting the database",
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True
    for name, (_, help_text) in _COMMANDS.items():
        sub.add_parser(name, help=help_text)

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
