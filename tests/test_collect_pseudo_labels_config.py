from __future__ import annotations

import argparse

import pytest

from cli.pn3train import cmd_collect_pseudo_labels


def test_collect_pseudo_labels_builds_runner_with_temporal_constraints_and_threshold(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    class _FakeSession:
        conn = object()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def load_cluster_schemas(self):
            return []

        def load_predicate_positions(self):
            return {
                "item_delivered_to": ["CUSTOMER", "DATE"],
                "withdrawal_statement_received_on": ["CUSTOMER", "DATE"],
            }

    class _FakeDBSession:
        @staticmethod
        def connect():
            return _FakeSession()

    def _fake_from_schemas(
        schemas,
        *,
        config=None,
        predicate_positions=None,
        max_refinement_rounds=2,
        temporal_constraints=None,
    ):
        captured["schemas"] = schemas
        captured["candidate_fact_threshold"] = config.candidate_fact_threshold
        captured["predicate_positions"] = predicate_positions
        captured["max_refinement_rounds"] = max_refinement_rounds
        captured["temporal_constraints"] = temporal_constraints
        return object()

    monkeypatch.setattr("db.DBSession", _FakeDBSession)
    monkeypatch.setattr("pipeline.runner.ProposeVerifyRunner.from_schemas", _fake_from_schemas)
    monkeypatch.setattr("cli.pn3train.upsert_round", lambda *args, **kwargs: None)
    monkeypatch.setattr("cli.pn3train.list_case_ids_by_split", lambda *args, **kwargs: [])

    args = argparse.Namespace(
        round_id="R2",
        split="train_unlabeled",
        case=None,
        parent_round=None,
        teacher_module="learned_nn",
        candidate_fact_threshold=0.55,
        fact_conf_threshold=0.95,
        cluster_top1_threshold=0.95,
        cluster_margin_threshold=0.80,
        notes=None,
        output=None,
        dry_run=True,
    )

    with pytest.raises(SystemExit):
        cmd_collect_pseudo_labels(args)

    assert captured["candidate_fact_threshold"] == pytest.approx(0.55)
    constraints = captured["temporal_constraints"]
    assert isinstance(constraints, list)
    assert len(constraints) == 1
    assert constraints[0].name == "withdrawal_within_14_days"
