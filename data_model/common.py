"""
Wspólne typy i definicje bazowe.
Mapuje: schemas/common.json → $defs
"""
from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Typy prymitywne (aliasy)
# ---------------------------------------------------------------------------

ID = Annotated[str, Field(min_length=1)]
Confidence = Annotated[float, Field(ge=0.0, le=1.0)]

# common.json → TruthValue
TruthValue = Literal["T", "F", "U"]


# ---------------------------------------------------------------------------
# Span  (common.json → Span)
# ---------------------------------------------------------------------------

class Span(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start: int | None = Field(default=None, ge=0)
    end:   int | None = Field(default=None, ge=0)
    text:  str | None = None  # fragment tekstu źródłowego (text[start:end])


# ---------------------------------------------------------------------------
# ProvenanceItem  (common.json → ProvenanceItem)
# ---------------------------------------------------------------------------

class ProvenanceItem(BaseModel):
    """
    Provenance pojedynczego faktu / encji / slotu.
    span i spans są wzajemnie wykluczające się:
      - span  → jeden zakres tekstu
      - spans → wiele zakresów (np. fakt rozrzucony w tekście)
    """
    model_config = ConfigDict(extra="forbid")

    source_id:  ID
    span:       Span | None              = None
    spans:      list[Span] | None        = None
    extractor:  str | None               = None
    confidence: Confidence | None        = None
    note:       str | None               = None

    @model_validator(mode="after")
    def _span_exclusive(self) -> ProvenanceItem:
        if self.span is not None and self.spans is not None:
            raise ValueError("span i spans są wzajemnie wykluczające się")
        return self


# ---------------------------------------------------------------------------
# TruthDistribution  (common.json → TruthDistribution)
# ---------------------------------------------------------------------------

class TruthDistribution(BaseModel):
    """
    Rozkład prawdziwości faktu nad domeną {T, F, U}.
    Klucze logits są ograniczone do wartości z domain.
    """
    model_config = ConfigDict(extra="forbid")

    domain:     Annotated[list[TruthValue], Field(min_length=2)]
    value:      TruthValue | None                    = None
    confidence: Confidence | None                    = None
    logits:     dict[TruthValue, float] | None       = None

    @field_validator("domain")
    @classmethod
    def _unique_domain(cls, v: list[TruthValue]) -> list[TruthValue]:
        if len(v) != len(set(v)):
            raise ValueError("domain musi zawierać unikalne wartości")
        return v


# ---------------------------------------------------------------------------
# RoleArg  (common.json → RoleArg)
# Argument roli w fakcie reifikowanym.
# Dokładnie jedno z: entity_id (referencja do encji) lub literal_value (wartość inline).
# ---------------------------------------------------------------------------

class RoleArg(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role:          Annotated[str, Field(min_length=1)]
    entity_id:     ID | None                                    = None
    literal_value: Annotated[str, Field(min_length=1)] | None  = None
    type_hint:     str | None                                   = None

    @model_validator(mode="after")
    def _exactly_one_arg(self) -> RoleArg:
        has_entity  = self.entity_id     is not None
        has_literal = self.literal_value is not None
        if has_entity == has_literal:  # oba None lub oba ustawione
            raise ValueError("Dokładnie jedno z entity_id / literal_value musi być ustawione")
        return self


# ---------------------------------------------------------------------------
# Term  (common.json → VarTerm | ConstTerm → Term)
# Używane w regułach (RuleArg).
# ---------------------------------------------------------------------------

class VarTerm(BaseModel):
    """Zmienna w regule, np. {\"var\": \"O\"}. Underscore (_) dozwolony jako zmienna anonimowa."""
    model_config = ConfigDict(extra="forbid")

    var: str = Field(pattern=r"^[a-zA-Z_][a-zA-Z0-9_]*$")


class ConstTerm(BaseModel):
    """Stała w regule, np. {\"const\": \"store\"}."""
    model_config = ConfigDict(extra="forbid")

    const: ID


# Union rozróżniany przez wzajemnie wykluczające się pola (extra='forbid' w obu)
Term = Union[VarTerm, ConstTerm]


# ---------------------------------------------------------------------------
# RuleArg  (common.json → RuleArg)
# Argument roli w regule — używa Term (zmienna lub stała), nie entity_id.
# ---------------------------------------------------------------------------

class RuleArg(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role:      Annotated[str, Field(min_length=1)]
    term:      Term
    type_hint: str | None = None
