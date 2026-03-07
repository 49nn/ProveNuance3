"""
NeuralInference — entry point pipeline propose-verify.

Workflow:
  1. Oblicz bias pamięci encji (EntityMemoryBiasEncoder).
  2. Zbuduj HeteroData (GraphBuilder).
  3. Zastosuj clamp (apply_clamp).
  4. Uruchom NeuralProposer.forward() z aktywnym NeuralTracer.
  5. Zdekoduj logity faktów → Fact z status=inferred_candidate i neural_trace.
  6. Zdekoduj logity klastrów → zaktualizowane ClusterStateRow.

Deterministyczny: @torch.no_grad(), brak Dropout, stały T.
"""
from __future__ import annotations

import uuid

import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData

from data_model.common import RoleArg, TruthDistribution
from data_model.fact import Fact, FactProvenance, FactSource, FactStatus, NeuralTraceItem
from data_model.rule import Rule
from sv._utils import to_clingo_id
from sv.proof import ground_rule
from sv.types import GroundAtom

from .clamp import apply_clamp
from .config import NNConfig
from .entity_memory import EntityMemoryBiasEncoder
from .graph_builder import (
    ClusterSchema,
    ClusterStateRow,
    GraphBuilder,
    GraphNodeIndex,
)
from .proposer import NeuralProposer
from .trace import NeuralTracer

TRUTH_ORDER = ("T", "F", "U")

# Obserwowane fakty zachowują swój status — nie nadpisujemy
_KEEP_STATUS = {FactStatus.observed, FactStatus.proved, FactStatus.rejected}


class NeuralInference:
    """
    Fasada: przyjmuje dane domenowe, zwraca wzbogacone Fact + zaktualizowane ClusterStateRow.
    """

    def __init__(
        self,
        proposer: NeuralProposer,
        graph_builder: GraphBuilder,
        memory_encoder: EntityMemoryBiasEncoder,
        config: NNConfig,
        predicate_positions: dict[str, list[str]] | None = None,
    ) -> None:
        self.proposer = proposer
        self.graph_builder = graph_builder
        self.memory_encoder = memory_encoder
        self.config = config
        self.predicate_positions = {
            pred.lower(): [role.upper() for role in roles]
            for pred, roles in (predicate_positions or {}).items()
        }
        self.cluster_roles = {
            schema.name.lower(): (
                schema.resolved_entity_role,
                schema.resolved_value_role,
            )
            for schema in self.graph_builder.cluster_schemas
        }

    # ------------------------------------------------------------------

    def propose(
        self,
        entities: list,     # list[Entity]
        facts: list[Fact],
        rules: list[Rule],
        cluster_states: list[ClusterStateRow],
    ) -> tuple[list[Fact], list[ClusterStateRow]]:
        """
        Główny punkt wejściowy.

        Zwraca:
            updated_facts:   Fact z zaktualizowanymi truth.logits, truth.value,
                             truth.confidence, provenance.neural_trace;
                             status = inferred_candidate (jeśli nie był observed/proved).
                             Dodatkowo zawiera nowe candidate fakty wygenerowane
                             z predykcji klastrów NN.
            updated_states:  ClusterStateRow z zaktualizowanymi logitami.
        """
        # 1. Bias pamięci encji
        memory_biases = self.memory_encoder.compute_memory_bias(entities, _dummy_index(self.graph_builder))
        # Pełny indeks budujemy przez build()

        # 2. Buduj graf
        data, node_index, _ = self.graph_builder.build(
            entities=entities,
            facts=facts,
            rules=rules,
            cluster_states=cluster_states,
            memory_biases=None,  # zastąpimy poniżej
        )

        # Oblicz właściwy memory_bias z gotowym node_index
        memory_biases = self.memory_encoder.compute_memory_bias(entities, node_index)

        # Dodaj memory_bias do data
        for schema in self.graph_builder.cluster_schemas:
            node_type = f"c_{schema.name}"
            if schema.name in memory_biases:
                data[node_type].memory_bias = memory_biases[schema.name]

        # 3. Zastosuj clamp (hard → frozen)
        for node_type in data.node_types:
            x = data[node_type].x
            is_clamped = data[node_type].get("is_clamped", torch.zeros(x.size(0), dtype=torch.bool))
            clamp_hard = data[node_type].get("clamp_hard", torch.zeros(x.size(0), dtype=torch.bool))
            logits_out, frozen = apply_clamp(x, is_clamped, clamp_hard, self.config)
            data[node_type].x = logits_out
            data[node_type].is_clamped = is_clamped
            data[node_type].clamp_hard = clamp_hard  # frozen wynika z is_clamped & clamp_hard

        # 4. Forward pass z tracer
        tracer = NeuralTracer(
            top_k=self.config.top_k_trace,
            truth_domain=self.config.truth_domain,
        )
        self.proposer.eval()

        with torch.no_grad():
            logits_cluster, logits_fact = self.proposer(data, node_index, tracer=tracer)

        # 5. Dekoduj fakty
        updated_facts = [
            self._decode_fact(fact, logits_fact, node_index, tracer)
            for fact in facts
        ]

        # 6. Dekoduj klastry
        updated_states = self._decode_cluster_states(
            logits_cluster, node_index, cluster_states
        )

        generated_candidates = self._generate_cluster_candidates(
            updated_states=updated_states,
            existing_facts=updated_facts,
        )
        rule_candidates = self._generate_rule_head_candidates(
            facts=updated_facts,
            updated_states=updated_states,
            rules=rules,
            existing_facts=(updated_facts + generated_candidates),
        )

        return (updated_facts + generated_candidates + rule_candidates), updated_states

    # ------------------------------------------------------------------
    # Dekodowanie faktów
    # ------------------------------------------------------------------

    def _decode_fact(
        self,
        fact: Fact,
        logits_fact: torch.Tensor,  # [N_fact, 3]
        node_index: GraphNodeIndex,
        tracer: NeuralTracer,
    ) -> Fact:
        """Produkuje zaktualizowany Fact (immutable — używa model_copy)."""
        idx = node_index.fact_node_to_idx.get(fact.fact_id)
        if idx is None or logits_fact.numel() == 0:
            return fact  # fakt nie był w grafie — bez zmian

        logits_3 = logits_fact[idx]  # [3]
        p = F.softmax(logits_3, dim=-1)

        tv_idx = int(p.argmax().item())
        tv = TRUTH_ORDER[tv_idx]
        confidence = float(p[tv_idx].item())

        truth_updated = TruthDistribution(
            domain=list(TRUTH_ORDER),
            value=tv,
            confidence=confidence,
            logits={
                "T": float(logits_3[0].item()),
                "F": float(logits_3[1].item()),
                "U": float(logits_3[2].item()),
            },
        )

        neural_trace: list[NeuralTraceItem] = tracer.finalize(fact.fact_id)

        new_provenance = FactProvenance(
            proof_id=fact.provenance.proof_id if fact.provenance else None,
            neural_trace=neural_trace,
        )

        new_status = (
            fact.status
            if fact.status in _KEEP_STATUS
            else FactStatus.inferred_candidate
        )

        return fact.model_copy(
            update={
                "truth": truth_updated,
                "status": new_status,
                "provenance": new_provenance,
            }
        )

    # ------------------------------------------------------------------
    # Dekodowanie stanów klastrów
    # ------------------------------------------------------------------

    def _decode_cluster_states(
        self,
        logits_cluster: dict[str, torch.Tensor],  # "c_{cname}" -> [N, dim]
        node_index: GraphNodeIndex,
        original_states: list[ClusterStateRow],
    ) -> list[ClusterStateRow]:
        """
        Produkuje zaktualizowane ClusterStateRow z finalnymi logitami.
        """
        # Indeks oryginałów do szybkiego wyszukiwania
        orig_lookup: dict[tuple[str, str], ClusterStateRow] = {
            (s.entity_id, s.cluster_name): s for s in original_states
        }

        updated: list[ClusterStateRow] = []

        for node_type, logits in logits_cluster.items():
            cname = node_type[2:] if node_type.startswith("c_") else node_type
            idx_to_entity = node_index.idx_to_cluster_node.get(cname, {})

            for node_idx, entity_id in idx_to_entity.items():
                row_logits = logits[node_idx].tolist()
                orig = orig_lookup.get((entity_id, cname))

                updated.append(
                    ClusterStateRow(
                        entity_id=entity_id,
                        cluster_name=cname,
                        logits=row_logits,
                        is_clamped=orig.is_clamped if orig else False,
                        clamp_hard=orig.clamp_hard if orig else False,
                        clamp_source=orig.clamp_source if orig else "manual",
                    )
                )

        return updated

    # ------------------------------------------------------------------
    # Generowanie nowych candidate facts
    # ------------------------------------------------------------------

    @staticmethod
    def _fact_identity(fact: Fact) -> tuple[str, tuple[tuple[str, str], ...]]:
        values = tuple(
            sorted((a.role, a.entity_id or a.literal_value or "") for a in fact.args)
        )
        return fact.predicate, values

    def _generate_cluster_candidates(
        self,
        updated_states: list[ClusterStateRow],
        existing_facts: list[Fact],
    ) -> list[Fact]:
        """
        Tworzy nowe inferred_candidate na bazie top-1 predykcji klastrów.

        Kandydat:
          - predicate: CLUSTER_NAME.upper()
          - args: (entity_role=entity_id, value_role=literal(top-1))
          - truth: T z confidence top-1
        """
        existing_keys = {self._fact_identity(f) for f in existing_facts}
        out: list[Fact] = []

        for state in updated_states:
            schema = self.graph_builder.schema_by_name(state.cluster_name)
            if schema is None:
                continue
            if not state.logits:
                continue
            if state.is_clamped and state.clamp_hard:
                # To zwykle obserwacje z tekstu/pamięci; nie generujemy z nich nowych kandydatów.
                continue

            logits = torch.tensor(state.logits[: schema.dim], dtype=torch.float32)
            probs = F.softmax(logits, dim=-1)
            top_idx = int(probs.argmax().item())
            confidence = float(probs[top_idx].item())
            if confidence < self.config.candidate_fact_threshold:
                continue

            value = schema.domain[top_idx].lower()
            entity_role = schema.resolved_entity_role
            value_role = schema.resolved_value_role

            args = [
                RoleArg(role=entity_role, entity_id=state.entity_id),
                RoleArg(role=value_role, literal_value=value),
            ]
            predicate = state.cluster_name.upper()
            candidate_seed = f"{predicate}|{state.entity_id}|{value}"
            candidate_fact = Fact(
                fact_id=str(uuid.uuid5(uuid.NAMESPACE_URL, candidate_seed)),
                predicate=predicate,
                arity=len(args),
                args=args,
                truth=TruthDistribution(
                    domain=["T", "F", "U"],
                    value="T",
                    confidence=confidence,
                    logits={
                        "T": confidence,
                        "F": 1.0 - confidence,
                        "U": 0.0,
                    },
                ),
                status=FactStatus.inferred_candidate,
                source=FactSource(
                    source_id="nn",
                    spans=[],
                    extractor="NeuralInference",
                    confidence=confidence,
                ),
                provenance=FactProvenance(neural_trace=[]),
            )

            key = self._fact_identity(candidate_fact)
            if key in existing_keys:
                continue
            existing_keys.add(key)
            out.append(candidate_fact)

        return out

    @staticmethod
    def _derived_positions(rules: list[Rule]) -> dict[str, list[str]]:
        result: dict[str, list[str]] = {}
        for rule in rules:
            pred = rule.head.predicate.lower()
            if pred not in result:
                result[pred] = [arg.role.upper() for arg in rule.head.args]
        return result

    def _fact_to_ground_atom(
        self,
        fact: Fact,
        predicate_positions: dict[str, list[str]],
    ) -> GroundAtom | None:
        pred = fact.predicate.lower()
        roles = predicate_positions.get(pred)
        if roles is None:
            return None

        role_map: dict[str, str] = {
            arg.role.upper(): to_clingo_id(arg.entity_id or arg.literal_value)  # type: ignore[arg-type]
            for arg in fact.args
        }
        bindings = tuple(sorted(
            (role, role_map[role])
            for role in roles
            if role in role_map
        ))
        if len(bindings) != len(roles):
            return None
        return GroundAtom(pred, bindings)

    def _cluster_state_to_ground_atom(
        self,
        state: ClusterStateRow,
    ) -> tuple[GroundAtom, float, str, str, str] | None:
        schema = self.graph_builder.schema_by_name(state.cluster_name)
        if schema is None or not state.logits:
            return None

        logits = torch.tensor(state.logits[: schema.dim], dtype=torch.float32)
        if logits.numel() == 0:
            return None

        sorted_logits = sorted((float(v) for v in logits.tolist()), reverse=True)
        margin = sorted_logits[0] - sorted_logits[1] if len(sorted_logits) > 1 else float("inf")
        if not state.is_clamped and margin < 1.0:
            return None

        probs = F.softmax(logits, dim=-1)
        top_idx = int(probs.argmax().item())
        confidence = float(probs[top_idx].item())
        value = schema.domain[top_idx].lower()
        entity_role, value_role = self.cluster_roles.get(
            state.cluster_name.lower(),
            (schema.resolved_entity_role, schema.resolved_value_role),
        )
        atom = GroundAtom(
            state.cluster_name.lower(),
            tuple(sorted((
                (entity_role, to_clingo_id(state.entity_id)),
                (value_role, to_clingo_id(value)),
            ))),
        )
        return atom, confidence, f"{state.cluster_name}:{state.entity_id}", state.entity_id, value

    def _build_rule_candidate_evidence(
        self,
        facts: list[Fact],
        updated_states: list[ClusterStateRow],
        predicate_positions: dict[str, list[str]],
    ) -> tuple[
        set[GroundAtom],
        dict[GroundAtom, tuple[str, float]],
        dict[GroundAtom, tuple[str, float]],
        dict[str, str],
    ]:
        atoms: set[GroundAtom] = set()
        fact_sources: dict[GroundAtom, tuple[str, float]] = {}
        cluster_sources: dict[GroundAtom, tuple[str, float]] = {}
        id_map: dict[str, str] = {}

        for fact in facts:
            if fact.truth.value != "T":
                continue
            confidence = float(fact.truth.confidence if fact.truth.confidence is not None else 1.0)
            if confidence < 0.5:
                continue
            for arg in fact.args:
                value = arg.entity_id or arg.literal_value
                if value is not None:
                    id_map.setdefault(to_clingo_id(value), value)
            atom = self._fact_to_ground_atom(fact, predicate_positions)
            if atom is None:
                continue
            atoms.add(atom)
            fact_sources[atom] = (fact.fact_id, confidence)

        for state in updated_states:
            item = self._cluster_state_to_ground_atom(state)
            if item is None:
                continue
            atom, confidence, cluster_id, entity_original, value_original = item
            entity_cid = to_clingo_id(entity_original)
            value_cid = to_clingo_id(value_original)
            id_map.setdefault(entity_cid, entity_original)
            id_map.setdefault(value_cid, value_original)
            atoms.add(atom)
            cluster_sources[atom] = (cluster_id, confidence)

        return atoms, fact_sources, cluster_sources, id_map

    @staticmethod
    def _is_candidate_head_supported(predicate: str, cluster_names: set[str]) -> bool:
        pred = predicate.lower()
        if pred.startswith("_sv_") or pred.startswith("ab_"):
            return False
        if pred in cluster_names:
            return False
        return True

    def _make_rule_head_candidate(
        self,
        grounded_rule,
        *,
        fact_sources: dict[GroundAtom, tuple[str, float]],
        cluster_sources: dict[GroundAtom, tuple[str, float]],
        id_map: dict[str, str],
    ) -> Fact | None:
        source_ids: list[str] = []
        support_confidences: list[float] = []
        trace_items: list[NeuralTraceItem] = []

        for atom in grounded_rule.pos_body:
            fact_meta = fact_sources.get(atom)
            if fact_meta is not None:
                fact_id, confidence = fact_meta
                source_ids.append(fact_id)
                support_confidences.append(confidence)
                trace_items.append(NeuralTraceItem(
                    step=0,
                    edge_type=f"rule_head:{grounded_rule.rule_id}",
                    from_fact_id=fact_id,
                    delta_logits={"T": confidence, "F": 0.0, "U": 0.0},
                ))
                continue

            cluster_meta = cluster_sources.get(atom)
            if cluster_meta is not None:
                cluster_id, confidence = cluster_meta
                source_ids.append(cluster_id)
                support_confidences.append(confidence)
                trace_items.append(NeuralTraceItem(
                    step=0,
                    edge_type=f"rule_head:{grounded_rule.rule_id}",
                    from_cluster_id=cluster_id,
                    delta_logits={"T": confidence, "F": 0.0, "U": 0.0},
                ))

        if not support_confidences:
            return None

        confidence = min(support_confidences)
        if confidence < self.config.candidate_fact_threshold:
            return None

        bindings_key = "|".join(f"{role}={value}" for role, value in grounded_rule.head.bindings)
        support_key = "|".join(sorted(source_ids))
        candidate_seed = f"{grounded_rule.rule_id}|{grounded_rule.head.predicate}|{bindings_key}|{support_key}"

        args = [
            RoleArg(role=role, entity_id=id_map.get(value, value))
            for role, value in grounded_rule.head.bindings
        ]
        return Fact(
            fact_id=str(uuid.uuid5(uuid.NAMESPACE_URL, candidate_seed)),
            predicate=grounded_rule.head.predicate.upper(),
            arity=len(args),
            args=args,
            truth=TruthDistribution(
                domain=["T", "F", "U"],
                value="T",
                confidence=confidence,
                logits={
                    "T": confidence,
                    "F": 1.0 - confidence,
                    "U": 0.0,
                },
            ),
            status=FactStatus.inferred_candidate,
            source=FactSource(
                source_id="nn_rule_head",
                spans=[],
                extractor="NeuralInference.rule_head",
                confidence=confidence,
            ),
            provenance=FactProvenance(neural_trace=trace_items),
        )

    def _generate_rule_head_candidates(
        self,
        facts: list[Fact],
        updated_states: list[ClusterStateRow],
        rules: list[Rule],
        existing_facts: list[Fact],
    ) -> list[Fact]:
        if not rules or not self.predicate_positions:
            return []

        predicate_positions = {
            **self.predicate_positions,
            **self._derived_positions(rules),
        }
        cluster_names = {schema.name.lower() for schema in self.graph_builder.cluster_schemas}
        atoms, fact_sources, cluster_sources, id_map = self._build_rule_candidate_evidence(
            facts,
            updated_states,
            predicate_positions,
        )
        if not atoms:
            return []

        existing_keys = {self._fact_identity(fact) for fact in existing_facts}
        best_by_key: dict[tuple[str, tuple[tuple[str, str], ...]], Fact] = {}

        for rule in rules:
            if not rule.body:
                continue
            if not self._is_candidate_head_supported(rule.head.predicate, cluster_names):
                continue
            for grounded_rule in ground_rule(rule, atoms, predicate_positions):
                candidate = self._make_rule_head_candidate(
                    grounded_rule,
                    fact_sources=fact_sources,
                    cluster_sources=cluster_sources,
                    id_map=id_map,
                )
                if candidate is None:
                    continue
                key = self._fact_identity(candidate)
                if key in existing_keys:
                    continue
                current = best_by_key.get(key)
                current_conf = (
                    float(current.truth.confidence)
                    if current is not None and current.truth.confidence is not None
                    else -1.0
                )
                next_conf = float(candidate.truth.confidence or 0.0)
                if current is None or next_conf > current_conf:
                    best_by_key[key] = candidate

        return list(best_by_key.values())


# ---------------------------------------------------------------------------
# Pomocnicze
# ---------------------------------------------------------------------------

def _dummy_index(builder: GraphBuilder) -> GraphNodeIndex:
    """Zwraca pusty GraphNodeIndex — używany gdy potrzebujemy schematu bez danych."""
    from .graph_builder import GraphNodeIndex
    return GraphNodeIndex()
