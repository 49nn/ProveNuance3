"""
Mechanizm clampowania logitów (hard / soft).

Hard clamp (clamp_source='text' lub 'manual'):
    s[k_true] = +M,  s[j≠k] = -M
    Węzeł jest zamrożony — nie otrzymuje wkładów z message passing.

Soft clamp (clamp_source='memory'):
    s[k_true] += M
    Węzeł jest elastyczny — może być dalej modyfikowany przez wiadomości.
"""
from __future__ import annotations

import torch

from .config import NNConfig


# Źródła clampowania uznawane za "hard"
HARD_SOURCES: frozenset[str] = frozenset({"text", "manual"})


def apply_clamp(
    logits: torch.Tensor,            # [N, dim] — tensor logitów jednego typu węzłów
    is_clamped: torch.BoolTensor,    # [N]
    clamp_hard: torch.BoolTensor,    # [N] — True = hard, False = soft
    config: NNConfig,
) -> tuple[torch.Tensor, torch.BoolTensor]:
    """
    Nakłada clamping na tensor logitów.

    Zwraca:
        logits_out  — zaktualizowany tensor [N, dim]
        frozen_mask — [N] BoolTensor; True = węzeł nie otrzymuje delta z MP
    """
    logits_out = logits.clone()
    n, dim = logits.shape
    frozen_mask = torch.zeros(n, dtype=torch.bool, device=logits.device)

    if not is_clamped.any():
        return logits_out, frozen_mask

    clamped_indices = is_clamped.nonzero(as_tuple=False).squeeze(1)

    for i in clamped_indices.tolist():
        # Wyznacz indeks wartości prawdziwej (argmax bieżących logitów)
        k_true = int(logits[i].argmax().item())

        if clamp_hard[i]:
            # Hard clamp: s_k = +M, reszta = -M; zamroź węzeł
            logits_out[i] = -config.clamp_hard_M
            logits_out[i, k_true] = config.clamp_hard_M
            frozen_mask[i] = True
        else:
            # Soft clamp: s_k += M; węzeł dalej uczestniczy w MP
            logits_out[i, k_true] += config.clamp_soft_M

    return logits_out, frozen_mask


def apply_clamp_from_value(
    logits: torch.Tensor,            # [N, dim]
    is_clamped: torch.BoolTensor,    # [N]
    clamp_hard: torch.BoolTensor,    # [N]
    clamp_value_idx: torch.LongTensor,  # [N] — indeks domeny (jeśli znany)
    config: NNConfig,
) -> tuple[torch.Tensor, torch.BoolTensor]:
    """
    Wariant gdy indeks wartości jest jawnie podany (np. podczas treningu
    przy maskowaniu i odtwarzaniu oryginalnych clampów).
    """
    logits_out = logits.clone()
    n, dim = logits.shape
    frozen_mask = torch.zeros(n, dtype=torch.bool, device=logits.device)

    clamped_indices = is_clamped.nonzero(as_tuple=False).squeeze(1)

    for i in clamped_indices.tolist():
        k = int(clamp_value_idx[i].item())
        if clamp_hard[i]:
            logits_out[i] = -config.clamp_hard_M
            logits_out[i, k] = config.clamp_hard_M
            frozen_mask[i] = True
        else:
            logits_out[i, k] += config.clamp_soft_M

    return logits_out, frozen_mask


def clamp_source_to_hard(clamp_source: str) -> bool:
    """Mapowanie clamp_source z DB (text/manual/memory) na bool hard."""
    return clamp_source in HARD_SOURCES
