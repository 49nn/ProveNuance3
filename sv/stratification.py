"""
Validation of stratified negation for Horn+NAF rules.
"""
from __future__ import annotations

from data_model.rule import LiteralType, Rule


def validate_stratification(rules: list[Rule]) -> None:
    """
    Raises ValueError when rule strata are inconsistent or violate stratification.

    Constraints:
      - Every predicate must have a single declared stratum.
      - Positive dependency: stratum(body_pred) <= stratum(head_pred)
      - NAF dependency:      stratum(body_pred) <  stratum(head_pred)
    """
    pred_to_strata: dict[str, set[int]] = {}
    for rule in rules:
        pred_to_strata.setdefault(rule.head.predicate, set()).add(rule.metadata.stratum)

    inconsistent = {
        pred: sorted(strata)
        for pred, strata in pred_to_strata.items()
        if len(strata) > 1
    }
    if inconsistent:
        details = "; ".join(
            f"{pred}: {strata}" for pred, strata in sorted(inconsistent.items())
        )
        raise ValueError(f"Inconsistent stratum assignment per predicate: {details}")

    pred_stratum = {pred: next(iter(strata)) for pred, strata in pred_to_strata.items()}

    violations: list[str] = []
    for rule in rules:
        head_pred = rule.head.predicate
        head_stratum = rule.metadata.stratum

        for lit in rule.body:
            body_pred = lit.predicate
            body_stratum = pred_stratum.get(body_pred)
            if body_stratum is None:
                # Extensional/helper predicate without defining rule in this set.
                continue

            if lit.literal_type == LiteralType.pos:
                if body_stratum > head_stratum:
                    violations.append(
                        f"{rule.rule_id}: positive dependency {body_pred}({body_stratum}) > {head_pred}({head_stratum})"
                    )
            else:
                if body_stratum >= head_stratum:
                    violations.append(
                        f"{rule.rule_id}: NAF dependency {body_pred}({body_stratum}) >= {head_pred}({head_stratum})"
                    )

    if violations:
        raise ValueError(
            "Rules are not stratified. Violations: " + " | ".join(violations)
        )
