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

    # --- Candidate generation ---
    candidate_fact_threshold: float = 0.70
    """Minimalna pewność top-1 dla generowania nowych inferred_candidate z klastrów."""

    # --- Dziedzina prawdziwości ---
    truth_domain: tuple[str, ...] = ("T", "F", "U")
    """Porządek wartości T/F/U — indeksy w tensorze logitów faktów."""

    # --- Staty uczenia ---
    lambda_fact_sup: float = 1.0
    """Waga L_fact_sup dla pseudo/gold superwizji na wezłach faktowych."""
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

    # --- SV feedback supervision ---
    gamma_sv_feedback: float = 0.5
    """Waga L_sv_feedback (SV outcome → supervised signal na logitach faktów)."""
    sv_blocked_weight: float = 1.0
    """Per-item waga dla outcome='blocked' (penalizuje logit T)."""
    sv_proved_weight: float = 0.8
    """Per-item waga dla outcome='proved' (nagradza logit T)."""
    sv_feedback_in_training: bool = True
    """Czy uruchamiać SV inline w train_on_case do generowania l_sv_feedback."""
