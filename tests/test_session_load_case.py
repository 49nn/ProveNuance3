from __future__ import annotations

from db.session import DBSession


def test_load_case_can_include_non_observed(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_load_case(conn, case_id: str, *, include_non_observed: bool = False):
        captured["conn"] = conn
        captured["case_id"] = case_id
        captured["include_non_observed"] = include_non_observed
        return [], [], [], []

    monkeypatch.setattr("db.session.load_case_data", _fake_load_case)

    session = DBSession(conn=object())  # type: ignore[arg-type]
    entities, facts, rules, states = session.load_case(
        "TC-159",
        include_non_observed=True,
    )

    assert entities == []
    assert facts == []
    assert rules == []
    assert states == []
    assert captured["case_id"] == "TC-159"
    assert captured["include_non_observed"] is True
