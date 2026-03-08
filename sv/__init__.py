"""
Symbolic Verifier — publiczne API.

Główny punkt wejścia: SymbolicVerifier.verify()

Przykład użycia:
    from sv import SymbolicVerifier, VerifyResult

    verifier = SymbolicVerifier(cluster_schemas=schemas)
    result   = verifier.verify(facts, rules, cluster_states)
    # result.updated_facts  — fakty ze statusem proved / inferred_candidate
    # result.new_facts       — nowe fakty derywowane przez reguły
    # result.derived_atoms   — cały model stabilny
    # result.proof_nodes     — proof DAG
"""

from sv.proof import ProofRun, ProofStep, build_proof_run
from sv.runner import build_program, rule_to_lp, solve
from sv.stratification import validate_stratification
from sv.temporal import (
    AnyTemporalConstraint,
    TemporalConstraint,
    TemporalCoincidenceConstraint,
    TemporalWindowConstraint,
    temporal_constraints_to_rules,
)
from sv.types import CandidateFeedback, GroundAtom, GroundRule, ProofNode, VerifyResult
from sv.verifier import SymbolicVerifier

__all__ = [
    # Fasada
    "SymbolicVerifier",
    # Constraints temporalne
    "TemporalConstraint",
    "TemporalCoincidenceConstraint",
    "TemporalWindowConstraint",
    "AnyTemporalConstraint",
    "temporal_constraints_to_rules",
    # Wyniki
    "VerifyResult",
    "CandidateFeedback",
    "ProofRun",
    "ProofStep",
    # Typy wewnętrzne (do testów i inspekcji)
    "GroundAtom",
    "GroundRule",
    "ProofNode",
    "validate_stratification",
    # Niskopoziomowe API (do testów jednostkowych)
    "build_program",
    "rule_to_lp",
    "solve",
    "build_proof_run",
]
