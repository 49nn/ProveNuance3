"""
NNConfig — wszystkie hiperparametry silnika neuronowego.

Wartości domyślne to sensowne punkty startowe do tuningu;
dokumentacja projektu definiuje wzory (T, M, λ, μ, β) ale nie podaje liczb.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class NNConfig:
    # --- Message passing ---
    T: int = 5
    """Liczba kroków message passing (unroll BPTT)."""

    # --- Clamp ---
    clamp_hard_M: float = 10.0
    """M dla hard clamp: s_k = +M, s_{j≠k} = -M."""
    clamp_soft_M: float = 3.0
    """Addytywne M dla soft clamp: s_k += M."""

    # --- Neural trace ---
    top_k_trace: int = 5
    """Maksymalna liczba NeuralTraceItem na fakt (top-k wg |Δs|₂)."""

    # --- Dziedzina prawdziwości ---
    truth_domain: tuple[str, ...] = ("T", "F", "U")
    """Porządek wartości T/F/U — indeksy w tensorze logitów faktów."""

    # --- Staty uczenia ---
    lambda_imp: float = 0.1
    """Waga L_imp w L = L_mask + λ·L_imp + μ·L_incomp + β·L_sparse."""
    mu_incomp: float = 0.1
    """Waga L_incomp."""
    beta_sparse: float = 0.01
    """Waga L_sparse."""
    mask_fraction: float = 0.15
    """Frakcja clampowanych węzłów maskowanych podczas treningu."""

    # --- Optymalizator ---
    lr: float = 1e-3
    """Learning rate (Adam)."""
    max_epochs: int = 50

    # --- Ograniczenia logiczne (dla L_imp / L_incomp) ---
    # Format: (cluster_name_A, domain_value_A, cluster_name_B, domain_value_B)
    implication_constraints: tuple[tuple[str, str, str, str], ...] = field(
        default_factory=tuple
    )
    """p(A=a) ≤ p(B=b) — naruszenie penalizowane przez L_imp."""

    incompatibility_constraints: tuple[tuple[str, str, str, str], ...] = field(
        default_factory=tuple
    )
    """p(A=a) · p(B=b) ≈ 0 — penalizowane przez L_incomp."""
