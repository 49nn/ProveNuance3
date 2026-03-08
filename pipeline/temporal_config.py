"""
Temporal constraints aktywne w tym projekcie.

To jest jedyne miejsce gdzie definiujesz constrainty temporalne.
CLI (run-case, verify-json), pipeline i llm_extractor załadują je automatycznie.

Przykłady:
    TemporalConstraint       — zdarzenie A musi być PRZED B
    TemporalCoincidenceConstraint — zdarzenia A i B muszą być w tym samym day/week/month/year
    TemporalWindowConstraint — zdarzenie B musi nastąpić w ciągu N dni od A

Każdy constraint musi pasować do predicate_positions z aktywnej ontologii.
Constrainty z nieznanymi predykatami są automatycznie pomijane z warning w logu.
"""
from __future__ import annotations

from sv.temporal import AnyTemporalConstraint


def get_temporal_constraints(
    predicate_positions: dict[str, list[str]],
) -> list[AnyTemporalConstraint]:
    """
    Zwraca listę aktywnych temporal constraints dla aktualnej ontologii.

    Argument predicate_positions pozwala warunkowo dodawać constrainty
    tylko gdy wymagane predykaty istnieją w ontologii.

    Przykład użycia:
        from sv.temporal import TemporalConstraint, TemporalWindowConstraint

        if "order_placed" in predicate_positions and "return_request" in predicate_positions:
            constraints.append(TemporalConstraint(
                name="return_after_purchase",
                earlier_pred="order_placed",   earlier_key_role="ORDER", earlier_date_role="DATE",
                later_pred="return_request",   later_key_role="ORDER",   later_date_role="DATE",
            ))

        if "order_placed" in predicate_positions and "return_request" in predicate_positions:
            constraints.append(TemporalWindowConstraint(
                name="return_within_14_days",
                earlier_pred="order_placed",   earlier_key_role="ORDER", earlier_date_role="DATE",
                later_pred="return_request",   later_key_role="ORDER",   later_date_role="DATE",
                n_days=14,
            ))
    """
    constraints: list[AnyTemporalConstraint] = []

    # ----------------------------------------------------------------
    # Zdefiniuj swoje constrainty tutaj.
    # ----------------------------------------------------------------

    return constraints
