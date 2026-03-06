"""
Modele danych dla self-trainingu i pseudo-labels.
"""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .cluster import ClusterStateRow
from .common import Confidence, ID
from .fact import Fact

CaseSplit = Literal["train_gold", "train_unlabeled", "holdout"]
RoundStatus = Literal["draft", "collected", "imported", "promoted", "rejected"]


class SelfTrainingRound(BaseModel):
    model_config = ConfigDict(extra="forbid")

    round_id: ID
    parent_round_id: ID | None = None
    status: RoundStatus = "draft"
    teacher_module: str | None = None
    fact_conf_threshold: Confidence = 0.95
    cluster_top1_threshold: Confidence = 0.95
    cluster_margin_threshold: Confidence = 0.80
    notes: str | None = None
    created_at: datetime | None = None
    promoted_at: datetime | None = None


class PseudoFactLabel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    round_id: ID
    case_id: ID
    fact_key: str = Field(min_length=1)
    fact: Fact
    truth_confidence: Confidence
    proof_id: ID | None = None
    accepted: bool = True
    rejection_reason: str | None = None
    promoted: bool = False


class PseudoClusterLabel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    round_id: ID
    case_id: ID
    entity_id: ID
    cluster_name: str = Field(min_length=1)
    value: str = Field(min_length=1)
    state: ClusterStateRow
    top1_confidence: Confidence
    margin: Confidence
    accepted: bool = True
    rejection_reason: str | None = None
    promoted: bool = False
