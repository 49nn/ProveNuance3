"""
DBSession facade for loading/saving ProveNuance3 domain objects.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import psycopg

from data_model.cluster import ClusterSchema, ClusterStateRow
from data_model.entity import Entity
from data_model.fact import FactStatus

from .case_repo import load_case as load_case_data
from .case_repo import resolve_case_id_int
from .cluster_repo import resolve_existing_entity_ids as _resolve_existing_entity_ids, upsert_cluster_states
from .connection import connect as connect_db
from .entity_repo import link_or_upsert_entity, upsert_entity
from .fact_repo import attach_proof_run_to_facts, upsert_fact
from .proof_repo import save_proof_run
from .ontology_repo import save_ontology as _save_ontology
from .rule_repo import load_rules, upsert_learned_rules
from .schema_repo import load_cluster_schemas, load_predicate_positions

if TYPE_CHECKING:
    from data_model.fact import Fact
    from data_model.rule import Rule
    from nlp.ontology_builder import OntologyResult
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

    def load_predicate_positions(self) -> dict[str, list[str]]:
        return load_predicate_positions(self.conn)

    def load_rules(
        self,
        enabled_only: bool = True,
        include_learned_modules: list[str] | None = None,
    ) -> list["Rule"]:
        return load_rules(
            self.conn,
            enabled_only=enabled_only,
            include_learned_modules=include_learned_modules,
        )

    def save_ontology(self, result: "OntologyResult") -> None:
        with self.conn.transaction():
            _save_ontology(self.conn, result)
        self.conn.commit()

    def save_learned_rules(
        self,
        rules: list["Rule"],
        module_name: str = "learned_nn",
    ) -> None:
        with self.conn.transaction():
            upsert_learned_rules(self.conn, rules, module_name=module_name)
        self.conn.commit()

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
                proof_run_id = save_proof_run(
                    self.conn,
                    proof_nodes=result.proof_nodes,
                    query="pipeline_result",
                    result=proof_result,
                    case_id_int=case_id_int,
                )
                proved_fact_ids = [
                    f.fact_id
                    for f in result.facts
                    if (
                        f.status == FactStatus.proved
                        and f.provenance is not None
                        and f.provenance.proof_id is not None
                    )
                ]
                attach_proof_run_to_facts(
                    self.conn,
                    fact_ids=proved_fact_ids,
                    proof_id=proof_run_id,
                    case_id_int=case_id_int,
                )
        self.conn.commit()

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

            # Zapisz encje i zbierz mapowanie old_id → actual_db_id
            entity_id_map: dict[str, str] = {}
            for entity in result.entities:
                actual_id = link_or_upsert_entity(self.conn, entity)
                entity_id_map[entity.entity_id] = actual_id

            for fact in result.facts:
                remapped_args = [
                    arg.model_copy(update={
                        "entity_id": (
                            entity_id_map.get(arg.entity_id, arg.entity_id)
                            if arg.entity_id is not None else None
                        ),
                    })
                    for arg in fact.args
                ]
                fact_to_save = fact.model_copy(update={
                    "args": remapped_args,
                    "status": FactStatus.observed,
                })
                upsert_fact(self.conn, fact_to_save, case_id_int)

            # Przepisz entity_id w cluster_states na faktyczny DB-id;
            # pomiń stany których encja nie istnieje w DB (LLM halucynacja / brak w entities).
            remapped: list[ClusterStateRow] = []
            for cs in result.cluster_states:
                actual_eid = entity_id_map.get(cs.entity_id, cs.entity_id)
                remapped.append(ClusterStateRow(
                    entity_id=actual_eid,
                    cluster_name=cs.cluster_name,
                    logits=cs.logits,
                    is_clamped=cs.is_clamped,
                    clamp_hard=cs.clamp_hard,
                    clamp_source=cs.clamp_source,
                ))
            known_eids = _resolve_existing_entity_ids(self.conn, {cs.entity_id for cs in remapped})
            remapped = [cs for cs in remapped if cs.entity_id in known_eids]
            upsert_cluster_states(self.conn, remapped, case_id_int)
        self.conn.commit()

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
