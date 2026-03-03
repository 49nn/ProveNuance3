"""
GraphBuilder — konwersja Entity/Fact/Rule + ClusterStateRow → torch_geometric HeteroData.

Typy węzłów:
  - "c_{cluster_name}"  (np. "c_customer_type") — jeden typ per definicja klastra
  - "fact"              — węzły reifikowanych faktów n-arnych

Typy krawędzi:
  - ("c_{cluster_name}", "role_of", "fact")    — klaster encji uczestniczy w fakcie
  - ("c_{src_name}", "implies", "c_{dst_name}") — reguła klaster→klaster (learned)
  - ("fact", "supports", "c_{dst_name}")        — reguła fakt→klaster (learned)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
from torch_geometric.data import HeteroData

if TYPE_CHECKING:
    from data_model.entity import Entity
    from data_model.fact import Fact
    from data_model.rule import Rule

TRUTH_ORDER = ("T", "F", "U")


# ---------------------------------------------------------------------------
# Publiczne dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ClusterSchema:
    """Definicja klastra unarnego (z DB: cluster_definitions + cluster_domain_values)."""
    cluster_id: int
    name: str          # np. "customer_type"
    entity_type: str   # np. "CUSTOMER"
    domain: list[str]  # np. ["CONSUMER", "BUSINESS"]

    @property
    def dim(self) -> int:
        return len(self.domain)


@dataclass
class ClusterStateRow:
    """Wiersz z tabeli cluster_states (stan klastra dla jednej encji i przypadku)."""
    entity_id: str
    cluster_name: str
    logits: list[float]   # FLOAT[] z DB; długość == dim klastra
    is_clamped: bool
    clamp_hard: bool
    clamp_source: str     # 'text' | 'memory' | 'manual'


@dataclass
class GraphNodeIndex:
    """Dwukierunkowe odwzorowania ID↔indeks węzłów w HeteroData."""
    # cluster_name -> entity_id -> indeks węzła (w tensoru tego cluster_name)
    cluster_node_to_idx: dict[str, dict[str, int]] = field(default_factory=dict)
    # cluster_name -> indeks węzła -> entity_id
    idx_to_cluster_node: dict[str, dict[int, str]] = field(default_factory=dict)
    # fact_id -> indeks węzła
    fact_node_to_idx: dict[str, int] = field(default_factory=dict)
    # indeks węzła -> fact_id
    idx_to_fact_node: dict[int, str] = field(default_factory=dict)

    def cluster_id_str(self, cluster_name: str, entity_id: str) -> str:
        """Unikalny string dla NeuralTraceItem.from_cluster_id."""
        return f"{cluster_name}:{entity_id}"

    def parse_cluster_id_str(self, s: str) -> tuple[str, str]:
        cluster_name, entity_id = s.split(":", 1)
        return cluster_name, entity_id


@dataclass
class EdgeTypeSpec:
    """Specyfikacja jednego typu krawędzi — jeden LogitMessagePassing per instancja."""
    src_type: str       # np. "c_customer_type"
    relation: str       # np. "role_of"
    dst_type: str       # np. "fact"
    src_dim: int        # rozmiar domeny źródłowego węzła
    dst_dim: int        # rozmiar domeny docelowego węzła
    rule_id: str | None = None


# ---------------------------------------------------------------------------
# GraphBuilder
# ---------------------------------------------------------------------------

class GraphBuilder:
    """
    Buduje HeteroData z instancji Pydantic i wierszy DB.

    Zasada: każdy typ klastra (np. 'customer_type') jest osobnym typem węzłów
    w PyG ('c_customer_type'). Dzięki temu W+ i W- są per-typ, nie dzielone.
    """

    FACT_DIM: int = 3  # T / F / U

    def __init__(self, cluster_schemas: list[ClusterSchema]) -> None:
        self.cluster_schemas = cluster_schemas
        self._schema_by_name: dict[str, ClusterSchema] = {s.name: s for s in cluster_schemas}

    # ------------------------------------------------------------------
    # Publiczne API
    # ------------------------------------------------------------------

    def build(
        self,
        entities: list[Entity],
        facts: list[Fact],
        rules: list[Rule],
        cluster_states: list[ClusterStateRow],
        memory_biases: dict[str, torch.Tensor] | None = None,
    ) -> tuple[HeteroData, GraphNodeIndex, list[EdgeTypeSpec]]:
        """
        Zwraca (data, node_index, edge_type_specs).

        edge_type_specs zawiera tylko te typy krawędzi, dla których istnieje
        przynajmniej jedna krawędź — potrzebne do inicjalizacji HeteroMessagePassingBank.

        memory_biases: opcjonalny dict cluster_name -> Tensor[N, dim] z entity_memory.py.
        """
        data = HeteroData()
        node_index = GraphNodeIndex()

        # Szybkie wyszukiwanie stanów klastrów
        state_lookup: dict[tuple[str, str], ClusterStateRow] = {
            (s.entity_id, s.cluster_name): s for s in cluster_states
        }

        # 1. Węzły klastrów (per typ)
        for schema in self.cluster_schemas:
            self._build_cluster_nodes(
                data, node_index, schema, entities, state_lookup, memory_biases
            )

        # 2. Węzły faktów
        self._build_fact_nodes(data, node_index, facts)

        # 3. Krawędzie role_of
        edge_specs = self._build_role_of_edges(data, node_index, facts)

        # 4. Krawędzie z reguł nauczonych (obecnie zwykle puste — learned=false)
        rule_specs = self._build_rule_edges(data, node_index, rules)
        edge_specs.extend(rule_specs)

        return data, node_index, edge_specs

    # ------------------------------------------------------------------
    # Budowa węzłów klastrów
    # ------------------------------------------------------------------

    def _build_cluster_nodes(
        self,
        data: HeteroData,
        node_index: GraphNodeIndex,
        schema: ClusterSchema,
        entities: list[Entity],
        state_lookup: dict[tuple[str, str], ClusterStateRow],
        memory_biases: dict[str, torch.Tensor] | None,
    ) -> None:
        node_type = f"c_{schema.name}"
        matching = [e for e in entities if e.type == schema.entity_type]

        n = len(matching)
        dim = schema.dim
        logits = torch.zeros(n, dim)
        is_clamped = torch.zeros(n, dtype=torch.bool)
        clamp_hard = torch.zeros(n, dtype=torch.bool)

        node_index.cluster_node_to_idx[schema.name] = {}
        node_index.idx_to_cluster_node[schema.name] = {}

        for i, entity in enumerate(matching):
            node_index.cluster_node_to_idx[schema.name][entity.entity_id] = i
            node_index.idx_to_cluster_node[schema.name][i] = entity.entity_id

            state = state_lookup.get((entity.entity_id, schema.name))
            if state is not None and state.logits:
                loaded = torch.tensor(state.logits[:dim], dtype=torch.float32)
                logits[i, : len(loaded)] = loaded
                is_clamped[i] = state.is_clamped
                clamp_hard[i] = state.clamp_hard

        data[node_type].x = logits
        data[node_type].is_clamped = is_clamped
        data[node_type].clamp_hard = clamp_hard
        data[node_type].domain_size = dim
        data[node_type].domain = schema.domain

        # Bias z pamięci encji (opcjonalny)
        if memory_biases is not None and schema.name in memory_biases:
            data[node_type].memory_bias = memory_biases[schema.name]
        else:
            data[node_type].memory_bias = torch.zeros(n, dim)

    # ------------------------------------------------------------------
    # Budowa węzłów faktów
    # ------------------------------------------------------------------

    def _build_fact_nodes(
        self,
        data: HeteroData,
        node_index: GraphNodeIndex,
        facts: list[Fact],
    ) -> None:
        n = len(facts)
        logits = torch.zeros(n, self.FACT_DIM)
        is_clamped = torch.zeros(n, dtype=torch.bool)
        clamp_hard = torch.zeros(n, dtype=torch.bool)

        for i, fact in enumerate(facts):
            node_index.fact_node_to_idx[fact.fact_id] = i
            node_index.idx_to_fact_node[i] = fact.fact_id

            if fact.truth.logits:
                for j, tv in enumerate(TRUTH_ORDER):
                    if tv in fact.truth.logits:
                        logits[i, j] = fact.truth.logits[tv]
                is_clamped[i] = True
                clamp_hard[i] = True
            elif fact.truth.value is not None:
                # Brak logitów ale jest wartość → hard clamp do tej wartości
                j = TRUTH_ORDER.index(fact.truth.value)
                is_clamped[i] = True
                clamp_hard[i] = True
                # Logity ustawiane przez clamp.py (tu zero → clamp.py je nadpisze)

        data["fact"].x = logits
        data["fact"].is_clamped = is_clamped
        data["fact"].clamp_hard = clamp_hard
        data["fact"].memory_bias = torch.zeros(n, self.FACT_DIM)

    # ------------------------------------------------------------------
    # Budowa krawędzi role_of
    # ------------------------------------------------------------------

    def _build_role_of_edges(
        self,
        data: HeteroData,
        node_index: GraphNodeIndex,
        facts: list[Fact],
    ) -> list[EdgeTypeSpec]:
        """
        Dla każdego faktu i każdego arg.entity_id: znajdź wszystkie klastry tej encji
        i dodaj krawędź (cluster_node → fact_node) w relacji "role_of".
        """
        # cluster_name -> (list src_idx, list dst_idx)
        edges: dict[str, tuple[list[int], list[int]]] = {
            s.name: ([], []) for s in self.cluster_schemas
        }

        for fact in facts:
            fact_idx = node_index.fact_node_to_idx.get(fact.fact_id)
            if fact_idx is None:
                continue
            for arg in fact.args:
                if arg.entity_id is None:
                    continue
                eid = arg.entity_id
                for schema in self.cluster_schemas:
                    cluster_map = node_index.cluster_node_to_idx.get(schema.name, {})
                    if eid in cluster_map:
                        c_idx = cluster_map[eid]
                        edges[schema.name][0].append(c_idx)
                        edges[schema.name][1].append(fact_idx)

        specs: list[EdgeTypeSpec] = []
        for schema in self.cluster_schemas:
            src_list, dst_list = edges[schema.name]
            if not src_list:
                continue
            node_type = f"c_{schema.name}"
            data[node_type, "role_of", "fact"].edge_index = torch.tensor(
                [src_list, dst_list], dtype=torch.long
            )
            specs.append(
                EdgeTypeSpec(
                    src_type=node_type,
                    relation="role_of",
                    dst_type="fact",
                    src_dim=schema.dim,
                    dst_dim=self.FACT_DIM,
                )
            )
        return specs

    # ------------------------------------------------------------------
    # Budowa krawędzi z reguł (learned)
    # ------------------------------------------------------------------

    def _build_rule_edges(
        self,
        data: HeteroData,
        node_index: GraphNodeIndex,
        rules: list[Rule],
    ) -> list[EdgeTypeSpec]:
        """
        Tworzy krawędzie tylko dla reguł z metadata.learned=True.
        Reguły hard (learned=False) obsługuje Symbolic Verifier (Clingo).
        Obecnie wszystkie reguły w ontologii mają learned=False → pusta lista.
        """
        specs: list[EdgeTypeSpec] = []
        # TODO: obsługa learned=True gdy zostaną dodane nauczone reguły
        return specs

    # ------------------------------------------------------------------
    # Pomocnicze
    # ------------------------------------------------------------------

    def schema_by_name(self, name: str) -> ClusterSchema | None:
        return self._schema_by_name.get(name)
