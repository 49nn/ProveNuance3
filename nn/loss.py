"""
Funkcja straty:

    L = L_mask + λ · L_imp + μ · L_incomp + β · L_sparse

L_mask   — rekonstrukcja maskowanych obserwacji (self-supervised)
L_imp    — soft constraint implikacyjny: p(A=a) ≤ p(B=b)
L_incomp — soft constraint niekompatybilności: p(A=a)·p(B=b) ≈ 0
L_sparse — entropia / sparsity dla nieuchwyconych węzłów

Wszystkie funkcje przyjmują tensory i są w pełni różniczkowalne przez PyTorch.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData

from .config import NNConfig
from .graph_builder import ClusterSchema, GraphNodeIndex


# ---------------------------------------------------------------------------
# Straty cząstkowe
# ---------------------------------------------------------------------------

def l_mask(
    logits_cluster: dict[str, torch.Tensor],  # cluster_name -> [N, dim]
    masked_items: list[tuple[str, int, int]],  # (cluster_name, node_idx, true_domain_idx)
    cluster_schemas: list[ClusterSchema],
) -> torch.Tensor:
    """
    L_mask = -Σ_{v∈M} log p_v^(T)(k_v_true)

    masked_items: lista (cluster_name, node_idx, true_domain_idx)
    """
    schema_by_name = {s.name: s for s in cluster_schemas}
    total = torch.tensor(0.0, requires_grad=True)

    for cname, idx, k_true in masked_items:
        # logits_cluster może mieć klucze z prefiksem "c_" lub bez
        key = cname if cname in logits_cluster else f"c_{cname}"
        if key not in logits_cluster:
            continue
        schema = schema_by_name.get(cname)
        if schema is None:
            continue
        dim = schema.dim
        logit_row = logits_cluster[key][idx, :dim]  # [dim] — bez paddingu
        log_p = F.log_softmax(logit_row, dim=-1)
        total = total - log_p[k_true]

    if masked_items:
        total = total / len(masked_items)

    return total


def l_implication(
    logits_cluster: dict[str, torch.Tensor],
    config: NNConfig,
    node_index: GraphNodeIndex,
    cluster_schemas: list[ClusterSchema],
) -> torch.Tensor:
    """
    L_imp = E[ max(0, p(A=a) - p(B=b)) ]

    Dla każdej pary (cname_A, val_A, cname_B, val_B) w config.implication_constraints
    i dla każdej encji która ma oba klastry:
        penalizuje gdy p(A=a) > p(B=b).
    """
    schema_by_name = {s.name: s for s in cluster_schemas}
    terms: list[torch.Tensor] = []

    for cname_A, val_A, cname_B, val_B in config.implication_constraints:
        s_A = schema_by_name.get(cname_A)
        s_B = schema_by_name.get(cname_B)
        if s_A is None or s_B is None:
            continue
        if val_A not in s_A.domain or val_B not in s_B.domain:
            continue

        idx_A = s_A.domain.index(val_A)
        idx_B = s_B.domain.index(val_B)

        nodes_A = node_index.cluster_node_to_idx.get(cname_A, {})
        nodes_B = node_index.cluster_node_to_idx.get(cname_B, {})

        log_A = logits_cluster.get(cname_A)
        log_B = logits_cluster.get(cname_B)
        if log_A is None or log_B is None:
            continue

        for entity_id, i_A in nodes_A.items():
            i_B = nodes_B.get(entity_id)
            if i_B is None:
                continue
            p_A = F.softmax(log_A[i_A, : s_A.dim], dim=-1)[idx_A]
            p_B = F.softmax(log_B[i_B, : s_B.dim], dim=-1)[idx_B]
            terms.append(F.relu(p_A - p_B))

    if not terms:
        return torch.tensor(0.0)
    return torch.stack(terms).mean()


def l_incompatibility(
    logits_cluster: dict[str, torch.Tensor],
    config: NNConfig,
    node_index: GraphNodeIndex,
    cluster_schemas: list[ClusterSchema],
) -> torch.Tensor:
    """
    L_incomp = E[ p(A=a) · p(B=b) ]
    """
    schema_by_name = {s.name: s for s in cluster_schemas}
    terms: list[torch.Tensor] = []

    for cname_A, val_A, cname_B, val_B in config.incompatibility_constraints:
        s_A = schema_by_name.get(cname_A)
        s_B = schema_by_name.get(cname_B)
        if s_A is None or s_B is None:
            continue
        if val_A not in s_A.domain or val_B not in s_B.domain:
            continue

        idx_A = s_A.domain.index(val_A)
        idx_B = s_B.domain.index(val_B)

        nodes_A = node_index.cluster_node_to_idx.get(cname_A, {})
        nodes_B = node_index.cluster_node_to_idx.get(cname_B, {})

        log_A = logits_cluster.get(cname_A)
        log_B = logits_cluster.get(cname_B)
        if log_A is None or log_B is None:
            continue

        for entity_id, i_A in nodes_A.items():
            i_B = nodes_B.get(entity_id)
            if i_B is None:
                continue
            p_A = F.softmax(log_A[i_A, : s_A.dim], dim=-1)[idx_A]
            p_B = F.softmax(log_B[i_B, : s_B.dim], dim=-1)[idx_B]
            terms.append(p_A * p_B)

    if not terms:
        return torch.tensor(0.0)
    return torch.stack(terms).mean()


def l_sparsity(
    logits_cluster: dict[str, torch.Tensor],  # cluster_name -> [N, dim]
    logits_fact: torch.Tensor,                # [N_fact, 3]
    frozen_cluster: dict[str, torch.BoolTensor],  # cluster_name -> [N]
    frozen_fact: torch.BoolTensor,            # [N_fact]
    cluster_schemas: list[ClusterSchema],
) -> torch.Tensor:
    """
    L_sparse = Σ_{v∉clamp} H(p_v)   (entropia Shannona)

    Obliczana tylko dla nieuchwyconych (niezamrożonych) węzłów.
    Wysoka entropia = niepewność = penalizowana.
    """
    schema_by_name = {s.name: s for s in cluster_schemas}
    terms: list[torch.Tensor] = []

    # Klastry
    for cname, logits in logits_cluster.items():
        schema = schema_by_name.get(cname[2:] if cname.startswith("c_") else cname)
        if schema is None:
            # spróbuj bez prefiksu
            schema = schema_by_name.get(cname)
        if schema is None:
            continue

        frz = frozen_cluster.get(cname, torch.zeros(logits.size(0), dtype=torch.bool))
        free_mask = ~frz  # [N]
        if not free_mask.any():
            continue

        dim = schema.dim
        free_logits = logits[free_mask, :dim]  # [N_free, dim]
        p = F.softmax(free_logits, dim=-1)
        # Entropia: -Σ p log p; clamp dla stabilności numerycznej
        entropy = -(p * (p + 1e-9).log()).sum(dim=-1)  # [N_free]
        terms.append(entropy.mean())

    # Fakty
    if logits_fact.numel() > 0:
        free_fact = ~frozen_fact
        if free_fact.any():
            p_fact = F.softmax(logits_fact[free_fact], dim=-1)  # [N_free, 3]
            entropy_fact = -(p_fact * (p_fact + 1e-9).log()).sum(dim=-1)
            terms.append(entropy_fact.mean())

    if not terms:
        return torch.tensor(0.0)
    return torch.stack(terms).mean()


# ---------------------------------------------------------------------------
# Kompozytowa funkcja straty
# ---------------------------------------------------------------------------

def compute_loss(
    logits_cluster: dict[str, torch.Tensor],
    logits_fact: torch.Tensor,
    data: HeteroData,
    node_index: GraphNodeIndex,
    config: NNConfig,
    cluster_schemas: list[ClusterSchema],
    masked_items: list[tuple[str, int, int]],  # (cluster_name, node_idx, true_domain_idx)
    frozen_cluster: dict[str, torch.BoolTensor],
    frozen_fact: torch.BoolTensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Zwraca (loss_total, dict komponentów straty do logowania).
    """
    lm = l_mask(logits_cluster, masked_items, cluster_schemas)
    li = l_implication(logits_cluster, config, node_index, cluster_schemas)
    lc = l_incompatibility(logits_cluster, config, node_index, cluster_schemas)
    ls = l_sparsity(logits_cluster, logits_fact, frozen_cluster, frozen_fact, cluster_schemas)

    total = lm + config.lambda_imp * li + config.mu_incomp * lc + config.beta_sparse * ls

    components = {
        "L_mask":   float(lm.item()),
        "L_imp":    float(li.item()),
        "L_incomp": float(lc.item()),
        "L_sparse": float(ls.item()),
        "L_total":  float(total.item()),
    }
    return total, components
