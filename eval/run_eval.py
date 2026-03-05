from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

# Allow running this file directly: `python eval/run_eval.py ...`
# by ensuring repo root is importable (for data_model/db/pipeline/sv imports).
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

if TYPE_CHECKING:
    from db.session import DBSession
    from sv.types import VerifyResult


LABELS: tuple[str, ...] = ("proved", "not_proved", "blocked", "unknown")
_QUERY_RE = re.compile(r"^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\((.*)\)\s*$")


@dataclass
class QueryEvalRow:
    case_id: str
    query_id: int
    query: str
    expected: str
    got: str
    correct: bool
    proof_present: bool
    case_latency_ms: float


def _parse_query_atom(query: str):
    from sv.converter import to_clingo_id
    from sv.types import GroundAtom

    q = query.strip()
    m = _QUERY_RE.match(q)
    if not m:
        return GroundAtom(q.lower(), ())
    pred = m.group(1).strip().lower()
    args_raw = [a.strip() for a in m.group(2).split(",") if a.strip()]
    bindings = tuple((str(i), to_clingo_id(a)) for i, a in enumerate(args_raw))
    return GroundAtom(pred, tuple(sorted(bindings)))


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    arr = sorted(values)
    rank = (len(arr) - 1) * q
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return arr[lo]
    w = rank - lo
    return arr[lo] * (1.0 - w) + arr[hi] * w


def _binary_ece(
    probs: Iterable[float],
    labels: Iterable[int],
    n_bins: int = 10,
) -> float | None:
    p = list(probs)
    y = list(labels)
    if not p:
        return None

    bins: list[list[int]] = [[] for _ in range(n_bins)]
    for i, prob in enumerate(p):
        idx = min(n_bins - 1, max(0, int(prob * n_bins)))
        bins[idx].append(i)

    n = len(p)
    ece = 0.0
    for idxs in bins:
        if not idxs:
            continue
        acc = sum(y[i] for i in idxs) / len(idxs)
        conf = sum(p[i] for i in idxs) / len(idxs)
        ece += (len(idxs) / n) * abs(acc - conf)
    return ece


def _brier_binary(probs: Iterable[float], labels: Iterable[int]) -> float | None:
    p = list(probs)
    y = list(labels)
    if not p:
        return None
    return sum((pi - yi) ** 2 for pi, yi in zip(p, y)) / len(p)


def _precision_recall_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def _proof_signature(result: "VerifyResult") -> tuple[str, ...]:
    entries: list[str] = []
    for atom, node in sorted(
        result.proof_nodes.items(),
        key=lambda kv: (kv[0].predicate, kv[0].bindings),
    ):
        atom_key = f"{atom.predicate}|{tuple(atom.bindings)}"
        pos = tuple(sorted(f"{a.predicate}|{tuple(a.bindings)}" for a in node.pos_used))
        neg = tuple(sorted(f"{a.predicate}|{tuple(a.bindings)}" for a in node.neg_checked))
        subst = tuple(sorted(node.substitution.items()))
        entries.append(f"{atom_key}|{node.rule_id}|{subst}|{pos}|{neg}")
    return tuple(entries)


def _load_case_queries(
    session: "DBSession",
    selected_cases: list[str] | None,
) -> dict[str, list[tuple[int, str, str]]]:
    where = ""
    params: tuple[object, ...] = ()
    if selected_cases:
        where = "WHERE c.case_id = ANY(%s)"
        params = (selected_cases,)

    query = f"""
        SELECT c.case_id, cq.id, cq.query, cq.expected_result
        FROM cases c
        JOIN case_queries cq ON cq.case_id = c.id
        {where}
        ORDER BY c.case_id, cq.id
    """

    out: dict[str, list[tuple[int, str, str]]] = defaultdict(list)
    with session.conn.cursor() as cur:
        cur.execute(query, params)
        for case_id, qid, qtext, expected in cur.fetchall():
            out[str(case_id)].append((int(qid), str(qtext), str(expected)))
    return out


def _storage_metrics(session: "DBSession") -> dict[str, int]:
    tables = (
        "facts",
        "fact_args",
        "fact_neural_trace",
        "cluster_states",
        "proof_runs",
        "proof_steps",
        "rules",
    )
    out: dict[str, int] = {}
    with session.conn.cursor() as cur:
        for t in tables:
            cur.execute(f"SELECT COUNT(*) FROM {t}")
            out[t] = int(cur.fetchone()[0])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute evaluation metrics for ProveNuance3 from DB cases/case_queries.",
    )
    parser.add_argument(
        "--case",
        action="append",
        dest="cases",
        help="Case ID to evaluate (repeatable). Default: all cases with case_queries.",
    )
    parser.add_argument(
        "--replay",
        type=int,
        default=1,
        help="How many times to re-run each case to measure deterministic stability.",
    )
    parser.add_argument(
        "--output-json",
        default="eval_report.json",
        help="Path to output JSON report.",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Optional path to output per-query CSV details.",
    )
    parser.add_argument(
        "--include-details",
        action="store_true",
        help="Include per-query rows in JSON output.",
    )
    args = parser.parse_args()

    replay = max(1, args.replay)

    from data_model.fact import FactStatus
    from db import DBSession
    from pipeline.runner import ProposeVerifyRunner

    with DBSession.connect() as session:
        case_queries = _load_case_queries(session, args.cases)
        if not case_queries:
            raise SystemExit("No case queries found for evaluation.")

        schemas = session.load_cluster_schemas()
        runner = ProposeVerifyRunner.from_schemas(schemas)

        per_query_rows: list[QueryEvalRow] = []
        case_latencies_ms: list[float] = []
        calibration_probs: list[float] = []
        calibration_labels: list[int] = []

        total_cases = len(case_queries)
        successful_cases = 0
        extraction_ready_cases = 0
        deterministic_cases = 0
        case_errors: dict[str, str] = {}

        for case_id, queries in case_queries.items():
            replay_signatures: list[tuple[str, ...]] = []
            replay_classifications: list[tuple[str, ...]] = []
            replay_latencies: list[float] = []
            first_rows: list[QueryEvalRow] = []
            first_calibration: list[tuple[float, int]] = []
            first_extraction_ready = False

            try:
                for _run in range(replay):
                    entities, facts, rules, states = session.load_case(case_id)
                    first_extraction_ready = (len(facts) > 0 and len(states) > 0)

                    t0 = time.perf_counter()
                    nn_facts, nn_states = runner.nn_inference.propose(entities, facts, rules, states)
                    sv_result = runner.verifier.verify(nn_facts, rules, nn_states)
                    elapsed_ms = (time.perf_counter() - t0) * 1000.0
                    replay_latencies.append(elapsed_ms)

                    got_labels: list[str] = []
                    current_rows: list[QueryEvalRow] = []
                    for qid, query_text, expected in queries:
                        query_atom = _parse_query_atom(query_text)
                        got = runner.verifier.classify_query_atom(query_atom, sv_result, rules)
                        got_labels.append(got)
                        proof_present = got == "proved" and query_atom in sv_result.proof_nodes
                        current_rows.append(
                            QueryEvalRow(
                                case_id=case_id,
                                query_id=qid,
                                query=query_text,
                                expected=expected,
                                got=got,
                                correct=(got == expected),
                                proof_present=proof_present,
                                case_latency_ms=elapsed_ms,
                            )
                        )

                    status_by_fact = {f.fact_id: f.status for f in sv_result.updated_facts}
                    current_calibration: list[tuple[float, int]] = []
                    for f in nn_facts:
                        if f.status != FactStatus.inferred_candidate:
                            continue
                        if f.truth.value != "T" or f.truth.confidence is None:
                            continue
                        proved = status_by_fact.get(f.fact_id) == FactStatus.proved
                        current_calibration.append((float(f.truth.confidence), int(proved)))

                    replay_classifications.append(tuple(got_labels))
                    replay_signatures.append(_proof_signature(sv_result))

                    if not first_rows:
                        first_rows = current_rows
                        first_calibration = current_calibration

                successful_cases += 1
                if first_extraction_ready:
                    extraction_ready_cases += 1
                if replay > 1:
                    cls_same = len(set(replay_classifications)) == 1
                    sig_same = len(set(replay_signatures)) == 1
                    if cls_same and sig_same:
                        deterministic_cases += 1
                else:
                    deterministic_cases += 1

                case_latencies_ms.extend(replay_latencies)
                per_query_rows.extend(first_rows)
                for p, y in first_calibration:
                    calibration_probs.append(p)
                    calibration_labels.append(y)
            except Exception as exc:  # noqa: BLE001
                case_errors[case_id] = str(exc)

        total_queries = len(per_query_rows)
        if total_queries == 0:
            raise SystemExit("No evaluated queries produced. Check case data and errors.")

        correct = sum(1 for r in per_query_rows if r.correct)
        accuracy = correct / total_queries

        per_class: dict[str, dict[str, float | int]] = {}
        f1_values: list[float] = []
        for label in LABELS:
            tp = sum(1 for r in per_query_rows if r.expected == label and r.got == label)
            fp = sum(1 for r in per_query_rows if r.expected != label and r.got == label)
            fn = sum(1 for r in per_query_rows if r.expected == label and r.got != label)
            support = sum(1 for r in per_query_rows if r.expected == label)
            p, rr, f1 = _precision_recall_f1(tp, fp, fn)
            f1_values.append(f1)
            per_class[label] = {
                "support": support,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": p,
                "recall": rr,
                "f1": f1,
            }

        macro_f1 = sum(f1_values) / len(f1_values)

        proved_rows = [r for r in per_query_rows if r.got == "proved"]
        proved_with_proof = sum(1 for r in proved_rows if r.proof_present)

        answer_coverage = sum(1 for r in per_query_rows if r.got != "unknown") / total_queries
        case_coverage = successful_cases / total_cases if total_cases > 0 else 0.0
        extraction_coverage = (
            extraction_ready_cases / successful_cases if successful_cases > 0 else 0.0
        )
        proof_coverage = (
            proved_with_proof / len(proved_rows) if proved_rows else None
        )

        ece = _binary_ece(calibration_probs, calibration_labels, n_bins=10)
        brier = _brier_binary(calibration_probs, calibration_labels)

        report = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "config": {
                "selected_cases": args.cases or [],
                "replay": replay,
                "labels": list(LABELS),
            },
            "summary": {
                "cases_total": total_cases,
                "cases_successful": successful_cases,
                "queries_total": total_queries,
                "accuracy": accuracy,
                "macro_f1": macro_f1,
            },
            "coverage": {
                "case_coverage": case_coverage,
                "answer_coverage": answer_coverage,
                "proof_coverage": proof_coverage,
                "extraction_coverage": extraction_coverage,
            },
            "per_class": per_class,
            "calibration": {
                "candidate_count": len(calibration_probs),
                "ece_bin10": ece,
                "brier_binary": brier,
            },
            "cost": {
                "case_latency_ms_p50": _percentile(case_latencies_ms, 0.50),
                "case_latency_ms_p95": _percentile(case_latencies_ms, 0.95),
                "case_latency_ms_mean": (
                    sum(case_latencies_ms) / len(case_latencies_ms) if case_latencies_ms else None
                ),
            },
            "stability": {
                "deterministic_replay_pass_rate": (
                    deterministic_cases / successful_cases if successful_cases > 0 else 0.0
                ),
                "deterministic_cases": deterministic_cases,
            },
            "storage": _storage_metrics(session),
            "errors": case_errors,
        }

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.output_csv:
        out_csv = Path(args.output_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "case_id",
                    "query_id",
                    "query",
                    "expected",
                    "got",
                    "correct",
                    "proof_present",
                    "case_latency_ms",
                ],
            )
            writer.writeheader()
            for row in per_query_rows:
                writer.writerow(asdict(row))

    if args.include_details:
        report_with_details = dict(report)
        report_with_details["per_query"] = [asdict(r) for r in per_query_rows]
        out_json.write_text(
            json.dumps(report_with_details, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    print(f"Saved JSON report: {out_json}")
    if args.output_csv:
        print(f"Saved CSV details: {args.output_csv}")


if __name__ == "__main__":
    main()
