from __future__ import annotations

import re
from typing import Iterator

import pytest

pytest.importorskip("torch")
pytest.importorskip("clingo")
pytest.importorskip("psycopg")

from db import DBSession
from pipeline.runner import ProposeVerifyRunner
from sv.converter import to_clingo_id
from sv.types import GroundAtom


_QUERY_RE = re.compile(r"^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\((.*)\)\s*$")


def _parse_query_atom(query: str) -> GroundAtom:
    q = query.strip()
    m = _QUERY_RE.match(q)
    if not m:
        return GroundAtom(q.lower(), ())
    pred = m.group(1).strip().lower()
    args_raw = [a.strip() for a in m.group(2).split(",") if a.strip()]
    bindings = tuple((str(i), to_clingo_id(a)) for i, a in enumerate(args_raw))
    return GroundAtom(pred, tuple(sorted(bindings)))


def _iter_case_queries(session: DBSession) -> Iterator[tuple[str, str, str]]:
    with session.conn.cursor() as cur:
        cur.execute(
            """
            SELECT c.case_id, cq.query, cq.expected_result
            FROM cases c
            JOIN case_queries cq ON cq.case_id = c.id
            ORDER BY c.case_id, cq.id
            """
        )
        for case_id, query, expected in cur.fetchall():
            yield str(case_id), str(query), str(expected)


@pytest.mark.integration
def test_case_queries_match_expected_results() -> None:
    with DBSession.connect() as session:
        schemas = session.load_cluster_schemas()
        runner = ProposeVerifyRunner.from_schemas(schemas)

        for case_id, query_text, expected in _iter_case_queries(session):
            entities, facts, rules, states = session.load_case(case_id)
            nn_facts, nn_states = runner.nn_inference.propose(entities, facts, rules, states)
            sv_result = runner.verifier.verify(nn_facts, rules, nn_states)

            query_atom = _parse_query_atom(query_text)
            got = runner.verifier.classify_query_atom(query_atom, sv_result, rules)

            # Dopuszczamy "blocked" jako szczególny wariant braku dowodu.
            if expected == "not_proved" and got == "blocked":
                continue
            assert got == expected, f"{case_id} :: {query_text}: expected={expected}, got={got}"
