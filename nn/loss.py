"""
Funkcja straty:

    L = L_mask + λ · L_imp + μ · L_incomp + β · L_sparse + γ · L_sv_feedback

L_mask        — rekonstrukcja maskowanych obserwacji (self-supervised)
L_imp         — soft constraint implikacyjny: p(A=a) ≤ p(B=b)
L_incomp      — soft constraint niekompatybilności: p(A=a)·p(B=b) ≈ 0
L_sparse      — entropia / sparsity dla nieuchwyconych węzłów
L_sv_feedback — SV outcome jako pseudo-supervision: blocked→F, proved→T

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
    masked_items: list[tuple[str, int, int, float]],  # (cluster_name, node_idx, true_domain_idx, weight)
    cluster_schemas: list[ClusterSchema],
) -> torch.Tensor:
    """
    L_mask = -Σ_{v∈M} log p_v^(T)(k_v_true)

    masked_items: lista (cluster_name, node_idx, true_domain_idx)
    """
    schema_by_name = {s.name: s for s in cluster_schemas}
    total = torch.tensor(0.0, requires_grad=True)

    total_weight = 0.0

    for cname, idx, k_true, weight in masked_items:
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
        total = total - (float(weight) * log_p[k_true])
        total_weight += float(weight)

    if total_weight > 0.0:
        total = total / total_weight

    return total


def l_fact_supervision(
    logits_fact: torch.Tensor,
    data: HeteroData,
) -> torch.Tensor:
    """
    Weighted cross-entropy on fact nodes with explicit supervision targets.

    Expected tensors on data["fact"]:
      - supervision_target: LongTensor[N_fact], -1 means unlabeled
      - supervision_weight: FloatTensor[N_fact], optional
    """
    if logits_fact.numel() == 0 or "fact" not in data.node_types:
        return torch.tensor(0.0)

    targets = data["fact"].get("supervision_target")
    if targets is None or targets.numel() == 0:
        return torch.tensor(0.0, device=logits_fact.device)

    targets = targets.to(device=logits_fact.device, dtype=torch.long)
    mask = targets >= 0
    if not bool(mask.any().item()):
        return torch.tensor(0.0, device=logits_fact.device)

    log_p = F.log_softmax(logits_fact[mask], dim=-1)
    selected_targets = targets[mask]
    losses = -log_p[torch.arange(selected_targets.size(0), device=logits_fact.device), selected_targets]

    weights = data["fact"].get("supervision_weight")
    if weights is not None and weights.numel() == targets.numel():
        weights = weights.to(device=logits_fact.device, dtype=torch.float32)[mask]
        total_weight = float(weights.sum().item())
        if total_weight > 0.0:
            return (losses * weights).sum() / weights.sum()

    return losses.mean()


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


def l_sv_feedback(
    logits_fact: torch.Tensor,  # [N_fact, 3] — w grafie obliczeniowym
    feedback_items: list,        # list[CandidateFeedback] — TYPE_CHECKING import tylko
    node_index: GraphNodeIndex,
    config: NNConfig,
) -> torch.Tensor:
    """
    L_sv_feedback = Σ_i w_i · (-log p_i(target_i)) / Σ w_i

    Dla każdego CandidateFeedback:
      - outcome='blocked' → target=F (idx=1), weight=sv_blocked_weight
      - outcome='proved'  → target=T (idx=0), weight=sv_proved_weight
      - pozostałe         → pomijane

    Gradient płynie przez logits_fact (ciągłe), mimo że targety
    pochodzą z dyskretnych decyzji SV.
    """
    if logits_fact.numel() == 0 or not feedback_items:
        return torch.tensor(0.0)

    terms: list[torch.Tensor] = []
    total_weight = 0.0

    for item in feedback_items:
        if item.outcome == "blocked":
            target_idx, weight = 1, config.sv_blocked_weight   # F=1
        elif item.outcome == "proved":
            target_idx, weight = 0, config.sv_proved_weight    # T=0
        else:
            continue

        idx = node_index.fact_node_to_idx.get(item.fact_id)
        if idx is None or idx >= logits_fact.size(0):
            continue

        log_p = F.log_softmax(logits_fact[idx], dim=-1)
        terms.append(weight * (-log_p[target_idx]))
        total_weight += weight

    if not terms:
        return torch.tensor(0.0, device=logits_fact.device)

    return torch.stack(terms).sum() / max(total_weight, 1e-9)


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
    masked_items: list[tuple[str, int, int, float]],  # (cluster_name, node_idx, true_domain_idx, weight)
    frozen_cluster: dict[str, torch.BoolTensor],
    frozen_fact: torch.BoolTensor,
    sv_feedback: list | None = None,  # list[CandidateFeedback] | None
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Zwraca (loss_total, dict komponentów straty do logowania).
    """
    lm = l_mask(logits_cluster, masked_items, cluster_schemas)
    lf = l_fact_supervision(logits_fact, data)
    li = l_implication(logits_cluster, config, node_index, cluster_schemas)
    lc = l_incompatibility(logits_cluster, config, node_index, cluster_schemas)
    ls = l_sparsity(logits_cluster, logits_fact, frozen_cluster, frozen_fact, cluster_schemas)
    lsv = l_sv_feedback(logits_fact, sv_feedback or [], node_index, config)

    total = (
        lm
        + config.lambda_fact_sup * lf
        + config.lambda_imp * li
        + config.mu_incomp * lc
        + config.beta_sparse * ls
        + config.gamma_sv_feedback * lsv
    )

    components = {
        "L_mask":        float(lm.item()),
        "L_fact_sup":    float(lf.item()),
        "L_imp":         float(li.item()),
        "L_incomp":      float(lc.item()),
        "L_sparse":      float(ls.item()),
        "L_sv_feedback": float(lsv.item()),
        "L_total":       float(total.item()),
    }
    return total, components
