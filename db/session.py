"""
DBSession facade for loading/saving ProveNuance3 domain objects.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import psycopg

from data_model.entity import Entity
from data_model.fact import FactStatus
from nn.graph_builder import ClusterSchema, ClusterStateRow

from .case_repo import load_case as load_case_data
from .case_repo import resolve_case_id_int
from .cluster_repo import upsert_cluster_states
from .connection import connect as connect_db
from .entity_repo import upsert_entity
from .fact_repo import upsert_fact
from .proof_repo import save_proof_run
from .rule_repo import load_rules
from .schema_repo import load_cluster_schemas

if TYPE_CHECKING:
    from data_model.fact import Fact
    from data_model.rule import Rule
    from nlp.result import ExtractionResult
    from pipeline.result import PipelineResult


class DBSession:
    def __init__(self, conn: psycopg.Connection) -> None:
        self.conn = conn
        self._entities_by_case: dict[str, list[Entity]] = {}

    @classmethod
    def connect(cls) -> "DBSession":
        return cls(connect_db())

    # ------------------------------------------------------------------
    # Ontology
    # ------------------------------------------------------------------

    def load_cluster_schemas(self) -> list[ClusterSchema]:
        return load_cluster_schemas(self.conn)

    def load_rules(self, enabled_only: bool = True) -> list["Rule"]:
        return load_rules(self.conn, enabled_only=enabled_only)

    # ------------------------------------------------------------------
    # Case
    # ------------------------------------------------------------------

    def load_case(self, case_id: str) -> tuple[list["Entity"], list["Fact"], list["Rule"], list[ClusterStateRow]]:
        entities, facts, rules, states = load_case_data(self.conn, case_id)
        self._entities_by_case[case_id] = entities
        return entities, facts, rules, states

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save_pipeline_result(self, result: "PipelineResult", case_id: str) -> None:
        case_id_int = resolve_case_id_int(self.conn, case_id)
        known_entities = self._entities_by_case.get(case_id, [])

        with self.conn.transaction():
            for entity in known_entities:
                upsert_entity(self.conn, entity)

            for fact in result.facts:
                upsert_fact(self.conn, fact, case_id_int)

            upsert_cluster_states(self.conn, result.cluster_states, case_id_int)

            if result.proof_nodes:
                proof_result = "proved" if result.proved else "unknown"
                save_proof_run(
                    self.conn,
                    proof_nodes=result.proof_nodes,
                    query="pipeline_result",
                    result=proof_result,
                    case_id_int=case_id_int,
                )

    def save_extraction_result(
        self,
        result: "ExtractionResult",
        case_id: str,
        source_text: str | None = None,
    ) -> None:
        case_id_int = resolve_case_id_int(self.conn, case_id)

        with self.conn.transaction():
            if source_text is not None:
                self._upsert_source(result.source_id, source_text)

            for entity in result.entities:
                upsert_entity(self.conn, entity)

            for fact in result.facts:
                fact_to_save = (
                    fact
                    if fact.status == FactStatus.observed
                    else fact.model_copy(update={"status": FactStatus.observed})
                )
                upsert_fact(self.conn, fact_to_save, case_id_int)

            upsert_cluster_states(self.conn, result.cluster_states, case_id_int)

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        if not self.conn.closed:
            self.conn.close()

    def __enter__(self) -> "DBSession":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is not None and not self.conn.closed:
            self.conn.rollback()
        self.close()
        return False

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _upsert_source(self, source_id: str, source_text: str) -> None:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO sources (source_id, title, content, source_type, source_rank)
                VALUES (%s, %s, %s, 'manual', 0)
                ON CONFLICT (source_id) DO UPDATE SET
                    content = EXCLUDED.content,
                    title = COALESCE(sources.title, EXCLUDED.title)
                """,
                (source_id, source_id, source_text),
            )
