"""
Extract learned symbolic rules from message-passing weights.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from data_model.common import ConstTerm, RuleArg, VarTerm
from data_model.rule import LiteralType, Rule, RuleBodyLiteral, RuleHead, RuleLanguage, RuleMetadata

from .graph_builder import ClusterSchema, EdgeTypeSpec
from .message_passing import HeteroMessagePassingBank


@dataclass(frozen=True)
class RuleExtractionConfig:
    min_weight: float = 0.5
    top_k_per_source_value: int = 2
    rule_id_prefix: str = "learned.nn"


def _cluster_roles(schema: ClusterSchema) -> tuple[str, str]:
    entity_role = schema.entity_type.upper()
    if schema.name.endswith("_type"):
        return entity_role, "TYPE"
    if schema.name == "payment_method":
        return entity_role, "METHOD"
    return entity_role, "VALUE"


def _score_matrix(spec: EdgeTypeSpec, mp_bank: HeteroMessagePassingBank) -> torch.Tensor:
    module = mp_bank.get_module(spec)
    # Effective matrix used in message passing.
    return (module.W_pos - module.W_neg).detach().cpu()


def extract_rules_from_mp_bank(
    mp_bank: HeteroMessagePassingBank,
    cluster_schemas: list[ClusterSchema],
    config: RuleExtractionConfig | None = None,
) -> list[Rule]:
    """
    Convert strong positive cluster->cluster weights into 1-premise Horn rules.
    """
    cfg = config or RuleExtractionConfig()
    schema_by_name = {s.name: s for s in cluster_schemas}
    extracted: list[Rule] = []

    for spec in mp_bank.specs:
        if not (spec.src_type.startswith("c_") and spec.dst_type.startswith("c_")):
            continue
        src_name = spec.src_type[2:]
        dst_name = spec.dst_type[2:]
        src_schema = schema_by_name.get(src_name)
        dst_schema = schema_by_name.get(dst_name)
        if src_schema is None or dst_schema is None:
            continue

        weights = _score_matrix(spec, mp_bank)  # [src_dim, dst_dim]
        src_entity_role, src_value_role = _cluster_roles(src_schema)
        dst_entity_role, dst_value_role = _cluster_roles(dst_schema)

        for src_i, src_value in enumerate(src_schema.domain):
            row = weights[src_i]
            vals, idxs = torch.topk(row, k=min(cfg.top_k_per_source_value, row.numel()))
            for val, dst_j in zip(vals.tolist(), idxs.tolist()):
                if float(val) < cfg.min_weight:
                    continue
                dst_value = dst_schema.domain[int(dst_j)]

                var_e = VarTerm(var="E")
                body = RuleBodyLiteral(
                    literal_type=LiteralType.pos,
                    predicate=src_schema.name,
                    args=[
                        RuleArg(role=src_entity_role, term=var_e),
                        RuleArg(role=src_value_role, term=ConstTerm(const=src_value.lower())),
                    ],
                )
                head = RuleHead(
                    predicate=dst_schema.name,
                    args=[
                        RuleArg(role=dst_entity_role, term=var_e),
                        RuleArg(role=dst_value_role, term=ConstTerm(const=dst_value.lower())),
                    ],
                )

                rule_id = (
                    f"{cfg.rule_id_prefix}."
                    f"{src_schema.name}.{src_value.lower()}__to__{dst_schema.name}.{dst_value.lower()}"
                )

                extracted.append(
                    Rule(
                        rule_id=rule_id,
                        language=RuleLanguage.horn_naf_stratified,
                        head=head,
                        body=[body],
                        metadata=RuleMetadata(
                            stratum=0,
                            learned=True,
                            weight=float(val),
                            support=None,
                            precision_est=None,
                            last_validated_at=None,
                            constraints=[],
                        ),
                    )
                )

    return extracted
