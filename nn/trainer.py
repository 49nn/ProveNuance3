"""
ProposerTrainer — pętla treningu z BPTT przez T kroków.

Strategia maskowania (self-supervised):
  1. Wybierz losowy podzbiór clamped węzłów (frakcja = config.mask_fraction).
  2. Tymczasowo odklampuj je i usuń bezpośredni sygnał clamp z logitów wejściowych.
  3. Uruchom forward pass bez tracer.
  4. Oblicz L na zamaskowanych węzłach (znamy prawdziwą wartość = argmax przed maskowaniem).
  5. Backward + optimizer step.
  6. Przywróć stan clamp.

BPTT przez T kroków jest obsługiwany automatycznie przez PyTorch autograd
(pętla for w NeuralProposer.forward() jest zwykłym grafem obliczeniowym).

SV Feedback (opcjonalne):
  Jeśli sv_provider jest podany i config.sv_feedback_in_training=True,
  po forward passie dekodujemy fakty z logitów i wywołujemy SymbolicVerifier.
  Wynik (CandidateFeedback) trafia do l_sv_feedback w compute_loss.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterator, Protocol, runtime_checkable

import torch
import torch.optim as optim
from torch_geometric.data import HeteroData

from .config import NNConfig
from .graph_builder import ClusterSchema, GraphNodeIndex
from .loss import compute_loss
from .proposer import NeuralProposer

if TYPE_CHECKING:
    from data_model.fact import Fact, FactStatus
    from data_model.rule import Rule
    from .graph_builder import ClusterStateRow


@runtime_checkable
class SVFeedbackProvider(Protocol):
    """
    Protokół dla dostawcy feedbacku z SymbolicVerifier.

    Implementowany w warstwie pipeline/ lub cli/ — bez importu sv/ z nn/.

    Przykład implementacji:
        def sv_provider(facts, rules, cluster_states):
            result = verifier.verify(facts, rules, cluster_states)
            return result.candidate_feedback
    """

    def __call__(
        self,
        facts: list,   # list[Fact]
        rules: list,   # list[Rule]
        cluster_states: list,  # list[ClusterStateRow]
    ) -> list:  # list[CandidateFeedback]
        ...


@dataclass
class TrainingCase:
    """
    Kompletny przypadek treningowy z opcjonalnym kontekstem dla SV feedback.

    Pola facts/rules/cluster_states są opcjonalne — jeśli nie podane,
    l_sv_feedback = 0 (SV nie jest uruchamiany).
    """

    data: HeteroData
    node_index: GraphNodeIndex
    facts: list | None = None           # list[Fact]
    rules: list | None = None           # list[Rule]
    cluster_states: list | None = None  # list[ClusterStateRow]


class ProposerTrainer:
    """
    Zarządza jednym przebiegiem treningu nad zbiorem przypadków.

    Każdy "przypadek" to jeden graf (HeteroData) lub TrainingCase.
    """

    def __init__(
        self,
        proposer: NeuralProposer,
        cluster_schemas: list[ClusterSchema],
        config: NNConfig,
        seed: int = 42,
        sv_provider: SVFeedbackProvider | None = None,
    ) -> None:
        self.proposer = proposer
        self.cluster_schemas = cluster_schemas
        self.config = config
        self.rng = random.Random(seed)
        self._step_counter = 0
        self.sv_provider = sv_provider

        self.optimizer = optim.Adam(proposer.parameters(), lr=config.lr)

    # ------------------------------------------------------------------
    # Główna metoda
    # ------------------------------------------------------------------

    def train_on_case(
        self,
        data: HeteroData,
        node_index: GraphNodeIndex,
        facts: list | None = None,
        rules: list | None = None,
        cluster_states: list | None = None,
    ) -> dict[str, float]:
        """
        Jeden forward+backward+step na pojedynczym grafie przypadku.

        Zwraca dict komponentów straty do logowania.
        """
        self.proposer.train()
        self.optimizer.zero_grad()

        # 1. Zbierz clamped węzły (cluster) ze znajomością prawdziwej wartości
        masked_items, saved_clamp_state = self._sample_and_mask(data)

        # 2. Forward pass (bez tracer — oszczędność pamięci)
        logits_cluster, logits_fact = self.proposer(data, node_index, tracer=None)

        # 3. SV feedback (opcjonalne) — dekoduje fakty z bieżących logitów
        sv_feedback = None
        if (
            self.sv_provider is not None
            and self.config.sv_feedback_in_training
            and facts is not None
        ):
            with torch.no_grad():
                sv_facts = self._decode_facts_for_sv(
                    logits_fact.detach(),
                    node_index,
                    facts,
                    self.config.candidate_fact_threshold,
                )
            sv_feedback = self.sv_provider(
                sv_facts,
                rules or [],
                cluster_states or [],
            )

        # 4. Frozen masks do straty sparsity
        frozen_cluster: dict[str, torch.BoolTensor] = {}
        frozen_fact = torch.zeros(
            logits_fact.size(0) if logits_fact.numel() > 0 else 0,
            dtype=torch.bool,
        )
        for node_type in data.node_types:
            ic = data[node_type].get("is_clamped", torch.zeros(data[node_type].x.size(0), dtype=torch.bool))
            ch = data[node_type].get("clamp_hard", torch.zeros(data[node_type].x.size(0), dtype=torch.bool))
            if node_type == "fact":
                frozen_fact = ic & ch
            else:
                frozen_cluster[node_type] = ic & ch

        # 5. Straty
        total, components = compute_loss(
            logits_cluster=logits_cluster,
            logits_fact=logits_fact,
            data=data,
            node_index=node_index,
            config=self.config,
            cluster_schemas=self.cluster_schemas,
            masked_items=masked_items,
            frozen_cluster=frozen_cluster,
            frozen_fact=frozen_fact,
            sv_feedback=sv_feedback,
        )

        # 6. Backward + step
        total.backward()
        self.optimizer.step()

        # 7. Przywróć stan clamp
        self._restore_clamp(data, saved_clamp_state)

        self._step_counter += 1
        return components

    # ------------------------------------------------------------------
    # SV feedback: dekodowanie faktów z logitów
    # ------------------------------------------------------------------

    @staticmethod
    def _decode_facts_for_sv(
        logits_fact: torch.Tensor,  # [N_fact, 3] — detached
        node_index: GraphNodeIndex,
        original_facts: list,       # list[Fact]
        candidate_threshold: float,
    ) -> list:
        """
        Buduje listę faktów do przekazania do SV:
          - Fakty observed/proved: bez zmian (stanowią bazę LP)
          - Fakty inferred_candidate: aktualizuje status jeśli p(T) >= threshold

        Dla przypadków gold (wszystkie fakty observed), SV nie będzie miał
        inferred_candidate faktów → l_sv_feedback = 0. Feedback jest
        użyteczny gdy dane zawierają inferred_candidate (np. z pipeline).
        """
        from data_model.fact import FactStatus

        fact_by_id: dict[str, object] = {f.fact_id: f for f in original_facts}
        idx_to_fact_id = {v: k for k, v in node_index.fact_node_to_idx.items()}

        base_facts = [
            f for f in original_facts
            if f.status in (FactStatus.observed, FactStatus.proved)
        ]
        base_ids = {f.fact_id for f in base_facts}

        probs = torch.softmax(logits_fact, dim=-1)  # [N, 3]
        candidate_facts = []
        for idx in range(logits_fact.size(0)):
            fact_id = idx_to_fact_id.get(idx)
            if fact_id is None or fact_id in base_ids:
                continue
            orig = fact_by_id.get(fact_id)
            if orig is None:
                continue
            p_true = float(probs[idx, 0].item())
            if p_true >= candidate_threshold:
                candidate_facts.append(
                    orig.model_copy(update={"status": FactStatus.inferred_candidate})
                )

        return base_facts + candidate_facts

    # ------------------------------------------------------------------
    # Maskowanie
    # ------------------------------------------------------------------

    def _sample_and_mask(
        self,
        data: HeteroData,
    ) -> tuple[list[tuple[str, int, int, float]], dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
        """
        Wybiera losowo frakcję clamped węzłów klastrów do maskowania.

        Zwraca:
            masked_items: [(cluster_name, node_idx, true_domain_idx), ...]
            saved_clamp_state: cluster_type -> (is_clamped_copy, clamp_hard_copy, x_copy)
        """
        masked_items: list[tuple[str, int, int, float]] = []
        saved: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

        for node_type in data.node_types:
            if node_type == "fact":
                continue  # maskujemy tylko klastry

            is_clamped = data[node_type].get("is_clamped")
            clamp_hard = data[node_type].get("clamp_hard")
            if is_clamped is None:
                continue
            mask_weight = data[node_type].get(
                "mask_weight",
                torch.ones(data[node_type].x.size(0), dtype=torch.float32),
            )

            # Zapisz oryginał
            saved[node_type] = (
                is_clamped.clone(),
                clamp_hard.clone(),
                data[node_type].x.clone(),
            )

            clamped_indices = is_clamped.nonzero(as_tuple=False).squeeze(1).tolist()
            if not clamped_indices:
                continue

            n_mask = max(1, int(len(clamped_indices) * self.config.mask_fraction))
            to_mask = self.rng.sample(clamped_indices, min(n_mask, len(clamped_indices)))

            # Cluster name: "c_customer_type" → "customer_type"
            cname = node_type[2:] if node_type.startswith("c_") else node_type

            for idx in to_mask:
                # Prawdziwa wartość = argmax bieżących logitów (przed maskowaniem)
                k_true = int(data[node_type].x[idx].argmax().item())
                weight = float(mask_weight[idx].item()) if mask_weight.numel() > idx else 1.0
                masked_items.append((cname, idx, k_true, weight))

                # Odklampuj i usuń wejściowy leak targetu z wcześniej zastosowanego clampu.
                data[node_type].is_clamped[idx] = False
                data[node_type].clamp_hard[idx] = False
                data[node_type].x[idx].zero_()

        return masked_items, saved

    def _restore_clamp(
        self,
        data: HeteroData,
        saved: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    ) -> None:
        """Przywraca oryginalne flagi clamp po backward."""
        for node_type, (orig_is_clamped, orig_clamp_hard, orig_x) in saved.items():
            data[node_type].is_clamped = orig_is_clamped
            data[node_type].clamp_hard = orig_clamp_hard
            data[node_type].x = orig_x

    # ------------------------------------------------------------------
    # Pętla epok
    # ------------------------------------------------------------------

    def train_epochs(
        self,
        cases: list,  # list[TrainingCase] lub list[tuple[HeteroData, GraphNodeIndex]]
    ) -> Iterator[dict[str, float]]:
        """
        Generator: trenuje przez config.max_epochs epok, yield dict straty per przypadek.

        Akceptuje zarówno stary format (tuple) jak i nowy (TrainingCase).
        """
        for epoch in range(self.config.max_epochs):
            for case in cases:
                if isinstance(case, TrainingCase):
                    components = self.train_on_case(
                        case.data,
                        case.node_index,
                        case.facts,
                        case.rules,
                        case.cluster_states,
                    )
                else:
                    data, node_index = case
                    components = self.train_on_case(data, node_index)
                components["epoch"] = float(epoch)
                yield components
