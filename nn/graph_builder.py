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
from data_model.common import ConstTerm, VarTerm

if TYPE_CHECKING:
    from data_model.common import Span
    from data_model.entity import Entity
    from data_model.fact import Fact
    from data_model.rule import Rule

TRUTH_ORDER = ("T", "F", "U")
SUPPORTS_RELATION_PREFIX = "supports:"


def supports_relation(predicate: str, role: str) -> str:
    return f"{SUPPORTS_RELATION_PREFIX}{predicate.lower()}:{role.upper()}"


def parse_supports_relation(relation: str) -> tuple[str, str] | None:
    if not relation.startswith(SUPPORTS_RELATION_PREFIX):
        return None
    payload = relation[len(SUPPORTS_RELATION_PREFIX):]
    predicate, sep, role = payload.rpartition(":")
    if not sep or not predicate or not role:
        return None
    return predicate.lower(), role.upper()


def is_supports_relation(relation: str) -> bool:
    return relation == "supports" or parse_supports_relation(relation) is not None


def same_rule_term(left, right) -> bool:
    if isinstance(left, VarTerm) and isinstance(right, VarTerm):
        return left.var == right.var
    if isinstance(left, ConstTerm) and isinstance(right, ConstTerm):
        return left.const == right.const
    return False


def find_head_entity_term(rule_head_args, entity_role: str):
    entity_role_upper = entity_role.upper()
    for arg in rule_head_args:
        if arg.role.upper() == entity_role_upper:
            return arg.term
    return None


def get_support_binding_roles(rule_args, head_entity_term) -> tuple[str, ...]:
    if head_entity_term is None:
        return ()
    roles = [
        arg.role.upper()
        for arg in rule_args
        if same_rule_term(arg.term, head_entity_term)
    ]
    return tuple(dict.fromkeys(roles))


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
    entity_role: str | None = None
    value_role: str | None = None

    @property
    def dim(self) -> int:
        return len(self.domain)

    @property
    def resolved_entity_role(self) -> str:
        return (self.entity_role or self.entity_type).upper()

    @property
    def resolved_value_role(self) -> str:
        return (self.value_role or "VALUE").upper()


@dataclass
class ClusterStateRow:
    """Wiersz z tabeli cluster_states (stan klastra dla jednej encji i przypadku)."""
    entity_id: str
    cluster_name: str
    logits: list[float]        # FLOAT[] z DB; długość == dim klastra
    is_clamped: bool
    clamp_hard: bool
    clamp_source: str          # 'text' | 'memory' | 'manual'
    source_span: "Span | None" = None  # fragment tekstu, z którego pochodzi clamp


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
        rule_specs = self._build_rule_edges(data, node_index, rules, facts)
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
        # Primary match by entity_type; fallback to entities with pre-existing cluster states
        # (handles ontology entity_type name mismatches, e.g. English vs Polish names).
        by_type = {e.entity_id for e in entities if e.type == schema.entity_type}
        by_state = {eid for (eid, cname) in state_lookup if cname == schema.name}
        relevant = by_type | by_state
        matching = [e for e in entities if e.entity_id in relevant]

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
                # Zachowaj preferowany indeks już na wejściu, aby hard clamp
                # poprawnie odtworzył także F/U, nie tylko domyślne T.
                logits[i, j] = 1.0

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
        facts: list[Fact],
    ) -> list[EdgeTypeSpec]:
        """
        Tworzy krawędzie tylko dla reguł z metadata.learned=True.
        Reguły hard (learned=False) obsługuje Symbolic Verifier (Clingo).
        Wspierane przypadki:
          - cluster -> cluster: body literal jest klastrem, head jest klastrem
          - fact -> cluster:    body literal jest predykatem faktowym, head jest klastrem
        """
        from data_model.rule import LiteralType

        schema_by_name = {s.name: s for s in self.cluster_schemas}

        # (src_type, relation, dst_type) -> (src_idx_list, dst_idx_list)
        edge_bucket: dict[tuple[str, str, str], tuple[list[int], list[int]]] = {}

        def add_edge(src_type: str, relation: str, dst_type: str, src_idx: int, dst_idx: int) -> None:
            key = (src_type, relation, dst_type)
            if key not in edge_bucket:
                edge_bucket[key] = ([], [])
            edge_bucket[key][0].append(src_idx)
            edge_bucket[key][1].append(dst_idx)

        # Fact lookup by fact_id from graph node index.
        fact_by_id = {f.fact_id: f for f in facts}

        for rule in rules:
            if not rule.metadata.learned:
                continue
            if not rule.body:
                continue

            head_pred = rule.head.predicate
            dst_schema = schema_by_name.get(head_pred)
            if dst_schema is None:
                # Obsługujemy tylko learned reguły, których head trafia do klastra.
                continue

            # Wybieramy pozytywne literały, bo tylko one niosą sygnał "supports/implies".
            body_literals = [lit for lit in rule.body if lit.literal_type == LiteralType.pos]
            if not body_literals:
                continue

            # 1) Cluster -> cluster (implies)
            for lit in body_literals:
                src_schema = schema_by_name.get(lit.predicate)
                if src_schema is None:
                    continue

                src_map = node_index.cluster_node_to_idx.get(src_schema.name, {})
                dst_map = node_index.cluster_node_to_idx.get(dst_schema.name, {})
                if not src_map or not dst_map:
                    continue

                for entity_id, src_idx in src_map.items():
                    dst_idx = dst_map.get(entity_id)
                    if dst_idx is not None:
                        add_edge(
                            f"c_{src_schema.name}",
                            "implies",
                            f"c_{dst_schema.name}",
                            src_idx,
                            dst_idx,
                        )

            # 2) Fact -> cluster (supports)
            # Groundujemy literał body względem istniejących faktów i mapujemy zmienne
            # na role head, aby uzyskać docelową encję klastra.
            for lit in body_literals:
                if lit.predicate in schema_by_name:
                    continue  # to był przypadek cluster->cluster

                dst_map = node_index.cluster_node_to_idx.get(dst_schema.name, {})
                if not dst_map:
                    continue
                head_entity_term = find_head_entity_term(
                    rule.head.args,
                    dst_schema.resolved_entity_role,
                )
                support_roles = get_support_binding_roles(lit.args, head_entity_term)
                if not support_roles:
                    continue

                for fact_idx, fact_id in node_index.idx_to_fact_node.items():
                    fact = fact_by_id.get(fact_id)
                    if fact is None:
                        continue
                    if fact.predicate.lower() != lit.predicate.lower():
                        continue

                    role_map: dict[str, str] = {}
                    for arg in fact.args:
                        val = arg.entity_id or arg.literal_value
                        if val is not None:
                            role_map[arg.role.upper()] = val

                    subst: dict[str, str] = {}
                    matched = True
                    for rule_arg in lit.args:
                        role = rule_arg.role.upper()
                        fact_val = role_map.get(role)
                        if fact_val is None:
                            matched = False
                            break
                        term = rule_arg.term
                        if isinstance(term, VarTerm):
                            if term.var == "_":
                                continue
                            existing = subst.get(term.var)
                            if existing is None:
                                subst[term.var] = fact_val
                            elif existing != fact_val:
                                matched = False
                                break
                        elif isinstance(term, ConstTerm):
                            if fact_val != term.const:
                                matched = False
                                break
                    if not matched:
                        continue

                    target_entity_id: str | None = None
                    if isinstance(head_entity_term, VarTerm):
                        target_entity_id = subst.get(head_entity_term.var)
                    elif isinstance(head_entity_term, ConstTerm):
                        target_entity_id = head_entity_term.const

                    if target_entity_id is None:
                        continue
                    dst_idx = dst_map.get(target_entity_id)
                    if dst_idx is None:
                        continue

                    for support_role in support_roles:
                        add_edge(
                            "fact",
                            supports_relation(lit.predicate, support_role),
                            f"c_{dst_schema.name}",
                            fact_idx,
                            dst_idx,
                        )

        specs: list[EdgeTypeSpec] = []
        for (src_type, relation, dst_type), (src_list, dst_list) in edge_bucket.items():
            if not src_list:
                continue

            # Dedup krawędzi dla stabilnego edge_index.
            unique_pairs = sorted(set(zip(src_list, dst_list)))
            src_idx = [p[0] for p in unique_pairs]
            dst_idx = [p[1] for p in unique_pairs]

            data[src_type, relation, dst_type].edge_index = torch.tensor(
                [src_idx, dst_idx],
                dtype=torch.long,
            )

            if src_type == "fact":
                src_dim = self.FACT_DIM
            else:
                src_cluster = src_type[2:] if src_type.startswith("c_") else src_type
                src_schema = schema_by_name[src_cluster]
                src_dim = src_schema.dim

            dst_cluster = dst_type[2:] if dst_type.startswith("c_") else dst_type
            dst_dim = schema_by_name[dst_cluster].dim if dst_cluster in schema_by_name else self.FACT_DIM

            specs.append(
                EdgeTypeSpec(
                    src_type=src_type,
                    relation=relation,
                    dst_type=dst_type,
                    src_dim=src_dim,
                    dst_dim=dst_dim,
                )
            )

        return specs

    # ------------------------------------------------------------------
    # Pomocnicze
    # ------------------------------------------------------------------

    def schema_by_name(self, name: str) -> ClusterSchema | None:
        return self._schema_by_name.get(name)
