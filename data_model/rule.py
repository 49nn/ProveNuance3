"""
Model danych dla reguł Horn + NAF ze stratyfikowaną negacją.
Mapuje: schemas/rule.json
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from .common import ID, Confidence, RuleArg


# ---------------------------------------------------------------------------
# Enumeracje
# ---------------------------------------------------------------------------

class RuleLanguage(str, Enum):
    horn_naf_stratified = "horn_naf_stratified"


class LiteralType(str, Enum):
    pos = "pos"   # literał pozytywny
    naf = "naf"   # negacja jako brak dowodu (not p)


# ---------------------------------------------------------------------------
# RuleHead  (rule.json → head)
# ---------------------------------------------------------------------------

class RuleHead(BaseModel):
    model_config = ConfigDict(extra="forbid")

    predicate: Annotated[str, Field(min_length=1)]
    args:      list[RuleArg]


# ---------------------------------------------------------------------------
# RuleBodyLiteral  (rule.json → body items)
# ---------------------------------------------------------------------------

class RuleBodyLiteral(BaseModel):
    model_config = ConfigDict(extra="forbid")

    literal_type: LiteralType
    predicate:    Annotated[str, Field(min_length=1)]
    args:         list[RuleArg]


# ---------------------------------------------------------------------------
# RuleMetadata  (rule.json → metadata)
# ---------------------------------------------------------------------------

class RuleMetadata(BaseModel):
    """
    stratum: poziom stratyfikacji negacji (0 = brak NAF w ciele).
    learned: True gdy wagi wygenerowane gradientowo przez Neural Proposer.
    weight:  siła reguły (ge=0); None gdy reguła twarda / niedouczona.
    """
    model_config = ConfigDict(extra="forbid")

    stratum:           int = Field(ge=0)
    learned:           bool
    weight:            Annotated[float, Field(ge=0.0)] | None   = None
    support:           int | None                               = Field(default=None, ge=0)
    precision_est:     Confidence | None                        = None
    last_validated_at: datetime | None                          = None
    constraints:       list[Annotated[str, Field(min_length=1)]] = Field(default_factory=list)
    source_span_text:  str | None                               = None


# ---------------------------------------------------------------------------
# Rule  (rule.json)
# ---------------------------------------------------------------------------

class Rule(BaseModel):
    """
    Reguła Horn + NAF ze stratyfikowaną negacją.
    body może być puste (unit clause / fakt), np. prepaid(card).
    head.args używa RuleArg z Term (zmienna lub stała), nie entity_id.
    """
    model_config = ConfigDict(extra="forbid")

    rule_id:  ID
    language: RuleLanguage = RuleLanguage.horn_naf_stratified
    head:     RuleHead
    body:     list[RuleBodyLiteral]   # minItems: 0 — puste = unit clause
    metadata: RuleMetadata
