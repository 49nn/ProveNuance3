"""
ProposerTrainer — pętla treningu z BPTT przez T kroków.

Strategia maskowania (self-supervised):
  1. Wybierz losowy podzbiór clamped węzłów (frakcja = config.mask_fraction).
  2. Tymczasowo odklampuj je (usuń z frozen, nie modyfikuj logitów).
  3. Uruchom forward pass bez tracer.
  4. Oblicz L na zamaskowanych węzłach (znamy prawdziwą wartość = argmax przed maskowaniem).
  5. Backward + optimizer step.
  6. Przywróć stan clamp.

BPTT przez T kroków jest obsługiwany automatycznie przez PyTorch autograd
(pętla for w NeuralProposer.forward() jest zwykłym grafem obliczeniowym).
"""
from __future__ import annotations

import random
from typing import Iterator

import torch
import torch.optim as optim
from torch_geometric.data import HeteroData

from .config import NNConfig
from .graph_builder import ClusterSchema, GraphNodeIndex
from .loss import compute_loss
from .proposer import NeuralProposer


class ProposerTrainer:
    """
    Zarządza jednym przebiegiem treningu nad zbiorem przypadków.

    Każdy "przypadek" to jeden graf (HeteroData).
    """

    def __init__(
        self,
        proposer: NeuralProposer,
        cluster_schemas: list[ClusterSchema],
        config: NNConfig,
        seed: int = 42,
    ) -> None:
        self.proposer = proposer
        self.cluster_schemas = cluster_schemas
        self.config = config
        self.rng = random.Random(seed)
        self._step_counter = 0

        self.optimizer = optim.Adam(proposer.parameters(), lr=config.lr)

    # ------------------------------------------------------------------
    # Główna metoda
    # ------------------------------------------------------------------

    def train_on_case(
        self,
        data: HeteroData,
        node_index: GraphNodeIndex,
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

        # 3. Frozen masks do straty sparsity
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

        # 4. Straty
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
        )

        # 5. Backward + step
        total.backward()
        self.optimizer.step()

        # 6. Przywróć stan clamp
        self._restore_clamp(data, saved_clamp_state)

        self._step_counter += 1
        return components

    # ------------------------------------------------------------------
    # Maskowanie
    # ------------------------------------------------------------------

    def _sample_and_mask(
        self,
        data: HeteroData,
    ) -> tuple[list[tuple[str, int, int, float]], dict[str, tuple[torch.Tensor, torch.Tensor]]]:
        """
        Wybiera losowo frakcję clamped węzłów klastrów do maskowania.

        Zwraca:
            masked_items: [(cluster_name, node_idx, true_domain_idx), ...]
            saved_clamp_state: cluster_type -> (is_clamped_copy, clamp_hard_copy)
        """
        masked_items: list[tuple[str, int, int, float]] = []
        saved: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

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
            saved[node_type] = (is_clamped.clone(), clamp_hard.clone())

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

                # Odklampuj: wyzeruj flagi
                data[node_type].is_clamped[idx] = False
                data[node_type].clamp_hard[idx] = False

        return masked_items, saved

    def _restore_clamp(
        self,
        data: HeteroData,
        saved: dict[str, tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        """Przywraca oryginalne flagi clamp po backward."""
        for node_type, (orig_is_clamped, orig_clamp_hard) in saved.items():
            data[node_type].is_clamped = orig_is_clamped
            data[node_type].clamp_hard = orig_clamp_hard

    # ------------------------------------------------------------------
    # Pętla epok
    # ------------------------------------------------------------------

    def train_epochs(
        self,
        cases: list[tuple[HeteroData, GraphNodeIndex]],
    ) -> Iterator[dict[str, float]]:
        """
        Generator: trenuje przez config.max_epochs epok, yield dict straty per przypadek.
        """
        for epoch in range(self.config.max_epochs):
            for data, node_index in cases:
                components = self.train_on_case(data, node_index)
                components["epoch"] = float(epoch)
                yield components
