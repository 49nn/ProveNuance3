"""
Extract learned symbolic rules from message-passing weights.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from data_model.common import ConstTerm, RuleArg, VarTerm
from data_model.rule import LiteralType, Rule, RuleBodyLiteral, RuleHead, RuleLanguage, RuleMetadata

from .graph_builder import (
    ClusterSchema,
    EdgeTypeSpec,
    find_head_entity_term,
    get_support_binding_roles,
    parse_supports_relation,
)
from .message_passing import HeteroMessagePassingBank


@dataclass(frozen=True)
class RuleExtractionConfig:
    min_weight: float = 0.5
    top_k_per_source_value: int = 2
    rule_id_prefix: str = "learned.nn"


def _score_matrix(spec: EdgeTypeSpec, mp_bank: HeteroMessagePassingBank) -> torch.Tensor:
    module = mp_bank.get_module(spec)
    return (module.W_pos - module.W_neg).detach().cpu()


def _cluster_rule_signature(
    src_schema: ClusterSchema,
    src_value: str,
    dst_schema: ClusterSchema,
    dst_value: str,
    weight: float,
    cfg: RuleExtractionConfig,
) -> Rule:
    var_e = VarTerm(var="E")
    body = RuleBodyLiteral(
        literal_type=LiteralType.pos,
        predicate=src_schema.name,
        args=[
            RuleArg(role=src_schema.resolved_entity_role, term=var_e),
            RuleArg(role=src_schema.resolved_value_role, term=ConstTerm(const=src_value.lower())),
        ],
    )
    head = RuleHead(
        predicate=dst_schema.name,
        args=[
            RuleArg(role=dst_schema.resolved_entity_role, term=var_e),
            RuleArg(role=dst_schema.resolved_value_role, term=ConstTerm(const=dst_value.lower())),
        ],
    )
    rule_id = (
        f"{cfg.rule_id_prefix}."
        f"{src_schema.name}.{src_value.lower()}__to__{dst_schema.name}.{dst_value.lower()}"
    )
    return Rule(
        rule_id=rule_id,
        language=RuleLanguage.horn_naf_stratified,
        head=head,
        body=[body],
        metadata=RuleMetadata(
            stratum=0,
            learned=True,
            weight=weight,
            support=None,
            precision_est=None,
            last_validated_at=None,
            constraints=[],
        ),
    )


def _next_var_name(role: str, used: set[str]) -> str:
    base = "".join(ch for ch in role.upper() if ch.isalnum()) or "X"
    candidate = base
    suffix = 2
    while candidate in used:
        candidate = f"{base}{suffix}"
        suffix += 1
    used.add(candidate)
    return candidate


def _fact_rule_body_args(predicate: str, binding_role: str, predicate_positions: dict[str, list[str]]) -> list[RuleArg]:
    roles = predicate_positions.get(predicate.lower(), [])
    if binding_role.upper() not in {role.upper() for role in roles}:
        return []

    used_vars = {"E"}
    args: list[RuleArg] = []
    for role in roles:
        role_upper = role.upper()
        if role_upper == binding_role.upper():
            term = VarTerm(var="E")
        else:
            term = VarTerm(var=_next_var_name(role_upper, used_vars))
        args.append(RuleArg(role=role_upper, term=term))
    return args


def _fact_rule(
    predicate: str,
    binding_role: str,
    dst_schema: ClusterSchema,
    dst_value: str,
    weight: float,
    cfg: RuleExtractionConfig,
    predicate_positions: dict[str, list[str]],
) -> Rule | None:
    body_args = _fact_rule_body_args(predicate, binding_role, predicate_positions)
    if not body_args:
        return None

    head = RuleHead(
        predicate=dst_schema.name,
        args=[
            RuleArg(role=dst_schema.resolved_entity_role, term=VarTerm(var="E")),
            RuleArg(role=dst_schema.resolved_value_role, term=ConstTerm(const=dst_value.lower())),
        ],
    )
    body = RuleBodyLiteral(
        literal_type=LiteralType.pos,
        predicate=predicate.lower(),
        args=body_args,
    )
    rule_id = (
        f"{cfg.rule_id_prefix}."
        f"{predicate.lower()}.{binding_role.lower()}__to__{dst_schema.name}.{dst_value.lower()}"
    )
    return Rule(
        rule_id=rule_id,
        language=RuleLanguage.horn_naf_stratified,
        head=head,
        body=[body],
        metadata=RuleMetadata(
            stratum=0,
            learned=True,
            weight=weight,
            support=None,
            precision_est=None,
            last_validated_at=None,
            constraints=[],
        ),
    )


def fact_cluster_rule_signature(
    rule: Rule,
    cluster_schemas: list[ClusterSchema],
) -> tuple[str, str, str] | None:
    if not rule.body:
        return None
    schema_by_name = {schema.name: schema for schema in cluster_schemas}
    body = next((lit for lit in rule.body if lit.literal_type == LiteralType.pos), None)
    if body is None or body.predicate in schema_by_name:
        return None

    dst_schema = schema_by_name.get(rule.head.predicate)
    if dst_schema is None:
        return None

    head_entity_term = find_head_entity_term(rule.head.args, dst_schema.resolved_entity_role)
    support_roles = get_support_binding_roles(body.args, head_entity_term)
    if not support_roles:
        return None
    return body.predicate.lower(), support_roles[0], dst_schema.name


def extract_rules_from_mp_bank(
    mp_bank: HeteroMessagePassingBank,
    cluster_schemas: list[ClusterSchema],
    config: RuleExtractionConfig | None = None,
    predicate_positions: dict[str, list[str]] | None = None,
) -> list[Rule]:
    """
    Convert strong positive message-passing weights into 1-premise Horn rules.
    """
    cfg = config or RuleExtractionConfig()
    schema_by_name = {schema.name: schema for schema in cluster_schemas}
    predicate_positions = {
        predicate.lower(): [role.upper() for role in roles]
        for predicate, roles in (predicate_positions or {}).items()
    }
    extracted: list[Rule] = []

    for spec in mp_bank.specs:
        if spec.src_type.startswith("c_") and spec.dst_type.startswith("c_"):
            src_name = spec.src_type[2:]
            dst_name = spec.dst_type[2:]
            src_schema = schema_by_name.get(src_name)
            dst_schema = schema_by_name.get(dst_name)
            if src_schema is None or dst_schema is None:
                continue

            weights = _score_matrix(spec, mp_bank)
            for src_i, src_value in enumerate(src_schema.domain):
                row = weights[src_i]
                vals, idxs = torch.topk(row, k=min(cfg.top_k_per_source_value, row.numel()))
                for val, dst_j in zip(vals.tolist(), idxs.tolist()):
                    if float(val) < cfg.min_weight:
                        continue
                    extracted.append(
                        _cluster_rule_signature(
                            src_schema=src_schema,
                            src_value=src_value,
                            dst_schema=dst_schema,
                            dst_value=dst_schema.domain[int(dst_j)],
                            weight=float(val),
                            cfg=cfg,
                        )
                    )
            continue

        if spec.src_type != "fact" or not spec.dst_type.startswith("c_"):
            continue

        parsed = parse_supports_relation(spec.relation)
        if parsed is None:
            continue
        predicate, binding_role = parsed
        dst_schema = schema_by_name.get(spec.dst_type[2:])
        if dst_schema is None:
            continue
        if predicate not in predicate_positions:
            continue

        weights = _score_matrix(spec, mp_bank)
        if weights.size(0) == 0:
            continue
        truth_positive = weights[0]
        vals, idxs = torch.topk(truth_positive, k=min(cfg.top_k_per_source_value, truth_positive.numel()))
        for val, dst_j in zip(vals.tolist(), idxs.tolist()):
            if float(val) < cfg.min_weight:
                continue
            rule = _fact_rule(
                predicate=predicate,
                binding_role=binding_role,
                dst_schema=dst_schema,
                dst_value=dst_schema.domain[int(dst_j)],
                weight=float(val),
                cfg=cfg,
                predicate_positions=predicate_positions,
            )
            if rule is not None:
                extracted.append(rule)

    return extracted
