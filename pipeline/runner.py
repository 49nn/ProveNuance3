"""
ProposeVerifyRunner — fasada łącząca NeuralInference → SymbolicVerifier.

Workflow (jeden przebieg):
  1. NeuralInference.propose()  → nn_facts (inferred_candidate), nn_states
  2. SymbolicVerifier.verify()  → proved facts, new facts, proof DAG
  3. Złożenie PipelineResult

Typowe użycie:
    runner = ProposeVerifyRunner.from_schemas(cluster_schemas)
    result = runner.run(entities, facts, rules, cluster_states)

Użycie z istniejącymi komponentami:
    runner = ProposeVerifyRunner(nn_inference, verifier)
    result = runner.run(entities, facts, rules, cluster_states)
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from nn.config import NNConfig
from nn.entity_memory import EntityMemoryBiasEncoder
from nn.gating import ExceptionGateBank
from nn.graph_builder import ClusterSchema, ClusterStateRow, EdgeTypeSpec, GraphBuilder
from nn.inference import NeuralInference
from nn.message_passing import HeteroMessagePassingBank
from nn.proposer import NeuralProposer
from sv.verifier import SymbolicVerifier

from .result import PipelineResult

if TYPE_CHECKING:
    from data_model.entity import Entity
    from data_model.fact import Fact
    from data_model.rule import Rule


class ProposeVerifyRunner:
    """
    Fasada propose-verify: jeden przebieg NN → SV.

    Parametry:
        nn_inference:  skonstruowany NeuralInference (z gotowym NeuralProposer)
        verifier:      skonstruowany SymbolicVerifier
    """

    def __init__(
        self,
        nn_inference: NeuralInference,
        verifier: SymbolicVerifier,
    ) -> None:
        self.nn_inference = nn_inference
        self.verifier = verifier

    # ------------------------------------------------------------------
    # Fabryka
    # ------------------------------------------------------------------

    @classmethod
    def from_schemas(
        cls,
        cluster_schemas: list[ClusterSchema],
        config: NNConfig | None = None,
    ) -> ProposeVerifyRunner:
        """
        Buduje runner z definicji klastrów (bez danych).

        Wagi NeuralProposer są zainicjalizowane losowo (Xavier dla W+,
        zeros dla W-, zeros dla biasów). Przed trenowaniem dają słabe
        propozycje — to normalne; SV i tak filtruje przez reguły.

        edge_type_specs: jeden 'role_of' per klaster (klaster→fakt).
        Reguły klaster→klaster (learned=True) są pomijane — gdy będą
        potrzebne, należy przekazać gotowy runner z własnymi specs.
        """
        config = config or NNConfig()

        edge_type_specs: list[EdgeTypeSpec] = [
            EdgeTypeSpec(
                src_type=f"c_{s.name}",
                relation="role_of",
                dst_type="fact",
                src_dim=s.dim,
                dst_dim=GraphBuilder.FACT_DIM,
            )
            for s in cluster_schemas
        ]

        mp_bank = HeteroMessagePassingBank(edge_type_specs)
        gate_bank = ExceptionGateBank(gate_specs=[])
        cluster_type_dims = {s.name: s.dim for s in cluster_schemas}

        proposer = NeuralProposer(config, mp_bank, gate_bank, cluster_type_dims)
        graph_builder = GraphBuilder(cluster_schemas)
        memory_encoder = EntityMemoryBiasEncoder(cluster_schemas, config)

        nn_inference = NeuralInference(proposer, graph_builder, memory_encoder, config)
        verifier = SymbolicVerifier(cluster_schemas)

        return cls(nn_inference, verifier)

    # ------------------------------------------------------------------
    # Główna metoda
    # ------------------------------------------------------------------

    def run(
        self,
        entities: list[Entity],
        facts: list[Fact],
        rules: list[Rule],
        cluster_states: list[ClusterStateRow],
    ) -> PipelineResult:
        """
        Wykonuje jeden przebieg propose-verify.

        1. NN propose: uzupełnia logity faktów i klastrów.
           Fakty nieobserwowane dostają status inferred_candidate.
        2. SV verify: dowodzi inferred_candidate przez reguły Horn+NAF.
           Udowodnione → proved. Reguły produkują new_facts (np. CONTRACT_FORMED).
        3. Składa PipelineResult.

        Zwraca:
            PipelineResult z facts = updated_facts + new_facts,
            cluster_states = zaktualizowane logity z NN,
            new_facts = fakty derywowane przez SV,
            proof_nodes = proof DAG.
        """
        # Faza 1 — Neural
        nn_facts, nn_states = self.nn_inference.propose(
            entities, facts, rules, cluster_states
        )

        # Faza 2 — Symbolic
        sv_result = self.verifier.verify(nn_facts, rules, nn_states)

        # Scalenie: updated_facts (z nowymi statusami) + new_facts (derywowane)
        all_facts = sv_result.updated_facts + sv_result.new_facts

        return PipelineResult(
            facts=all_facts,
            cluster_states=nn_states,
            new_facts=sv_result.new_facts,
            proof_nodes=dict(sv_result.proof_nodes),
        )
