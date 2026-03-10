from __future__ import annotations

from cli.pn3 import _apply_ontology_reset, _collect_ontology_counts


class _FakeCursor:
    def __init__(self, counts: dict[str, int], rowcounts: dict[str, int]) -> None:
        self._counts = counts
        self._rowcounts = rowcounts
        self._last_count = 0
        self.rowcount = 0
        self.executed: list[str] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, query: str, params=None) -> None:
        del params
        normalized = " ".join(query.split())
        self.executed.append(normalized)
        if normalized.startswith("SELECT COUNT(*) FROM "):
            table = normalized.removeprefix("SELECT COUNT(*) FROM ")
            self._last_count = self._counts[table]
            self.rowcount = 1
            return
        if normalized.startswith("DELETE FROM "):
            table = normalized.removeprefix("DELETE FROM ")
            self.rowcount = self._rowcounts[table]
            return
        raise AssertionError(f"Unexpected query: {normalized}")

    def fetchone(self):
        return (self._last_count,)


class _FakeConn:
    def __init__(self, counts: dict[str, int] | None = None, rowcounts: dict[str, int] | None = None) -> None:
        self.cursor_obj = _FakeCursor(counts or {}, rowcounts or {})

    def cursor(self) -> _FakeCursor:
        return self.cursor_obj


def test_collect_ontology_counts_includes_self_training_tables() -> None:
    conn = _FakeConn(counts={
        "entity_types": 1,
        "predicate_definitions": 2,
        "cluster_definitions": 3,
        "rules WHERE learned = FALSE": 4,
        "rule_modules": 5,
        "entities": 6,
        "facts": 7,
        "self_training_rounds": 8,
        "pseudo_fact_labels": 9,
        "pseudo_cluster_labels": 10,
        "cases": 11,
        "case_queries": 12,
        "sources": 13,
    })

    counts = _collect_ontology_counts(conn)

    assert counts["self_training_rounds"] == 8
    assert counts["pseudo_fact_labels"] == 9
    assert counts["pseudo_cluster_labels"] == 10
    assert counts["case_queries"] == 12


def test_apply_ontology_reset_deletes_self_training_tables_explicitly() -> None:
    conn = _FakeConn(rowcounts={
        "fact_neural_trace": 1,
        "facts": 2,
        "cluster_states": 3,
        "entities": 4,
        "pseudo_fact_labels": 5,
        "pseudo_cluster_labels": 6,
        "self_training_rounds": 7,
        "proof_runs": 8,
        "case_queries": 9,
        "cases": 10,
        "sources": 11,
        "cluster_definitions": 12,
        "predicate_definitions": 13,
        "entity_types": 14,
        "rules": 15,
        "rule_modules": 16,
    })

    result = _apply_ontology_reset(conn)
    executed = conn.cursor_obj.executed

    assert result["pseudo_fact_labels"] == 5
    assert result["pseudo_cluster_labels"] == 6
    assert result["self_training_rounds"] == 7
    assert result["case_queries"] == 9
    assert executed.index("DELETE FROM pseudo_fact_labels") < executed.index("DELETE FROM self_training_rounds")
    assert executed.index("DELETE FROM pseudo_cluster_labels") < executed.index("DELETE FROM self_training_rounds")
    assert executed.index("DELETE FROM self_training_rounds") < executed.index("DELETE FROM cases")
