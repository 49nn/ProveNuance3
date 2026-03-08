from __future__ import annotations

from cli.pn3 import _run_case_runtime_issue


def test_run_case_runtime_issue_requires_enabled_rules() -> None:
    issue = _run_case_runtime_issue(
        cluster_schemas=[object()],
        predicate_positions={"can_withdraw": ["CUSTOMER", "ORDER"]},
        rules=[],
    )

    assert issue is not None
    assert "aktywnych regu" in issue.lower()


def test_run_case_runtime_issue_allows_ready_runtime() -> None:
    issue = _run_case_runtime_issue(
        cluster_schemas=[object()],
        predicate_positions={"can_withdraw": ["CUSTOMER", "ORDER"]},
        rules=[object()],
    )

    assert issue is None
