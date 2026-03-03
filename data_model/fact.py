"""
Model danych dla faktu reifikowanego (n-arny, z truth distribution i provenance).
Mapuje: schemas/fact.json
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .common import ID, Confidence, RoleArg, Span, TruthDistribution, TruthValue


# ---------------------------------------------------------------------------
# FactStatus  (fact.json → status enum)
# ---------------------------------------------------------------------------

class FactStatus(str, Enum):
    observed            = "observed"
    inferred_candidate  = "inferred_candidate"
    proved              = "proved"
    rejected            = "rejected"
    retracted           = "retracted"


# ---------------------------------------------------------------------------
# NeuralTraceItem  (fact.json → provenance.neural_trace items)
# ---------------------------------------------------------------------------

class NeuralTraceItem(BaseModel):
    """
    Wkład jednej krawędzi message-passing do logitów docelowego faktu.
    Dokładnie jedno źródło: from_fact_id albo from_cluster_id.
    Klucze delta_logits odpowiadają wartościom TruthValue (T/F/U).
    """
    model_config = ConfigDict(extra="forbid")

    from_fact_id:    ID | None                                        = None
    from_cluster_id: ID | None                                        = None
    edge_type:       Annotated[str, Field(min_length=1)]
    delta_logits:    Annotated[dict[TruthValue, float], Field(min_length=1)]
    step:            int = Field(ge=0)

    @model_validator(mode="after")
    def _exactly_one_source(self) -> NeuralTraceItem:
        has_fact    = self.from_fact_id    is not None
        has_cluster = self.from_cluster_id is not None
        if has_fact == has_cluster:
            raise ValueError("Dokładnie jedno z from_fact_id / from_cluster_id musi być ustawione")
        return self


# ---------------------------------------------------------------------------
# FactProvenance  (fact.json → provenance)
# ---------------------------------------------------------------------------

class FactProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    proof_id:      ID | None               = None
    neural_trace:  list[NeuralTraceItem]   = Field(default_factory=list[NeuralTraceItem])


# ---------------------------------------------------------------------------
# FactSource  (fact.json → source)
# ---------------------------------------------------------------------------

class FactSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id:  ID
    spans:      list[Span]     = Field(default_factory=list[Span])
    extractor:  str | None     = None
    confidence: Confidence | None = None


# ---------------------------------------------------------------------------
# FactTime  (fact.json → time)
# ---------------------------------------------------------------------------

class FactTime(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_time: datetime | None = None
    valid_from: datetime | None = None
    valid_to:   datetime | None = None


# ---------------------------------------------------------------------------
# Fact  (fact.json)
# ---------------------------------------------------------------------------

class Fact(BaseModel):
    """
    Fakt reifikowany (n-arny).
    arity jest deklaratywne — równość len(args) == arity jest walidowana aplikacyjnie
    (nie da się wyrazić w JSON Schema 2020-12 bez $data).
    """
    model_config = ConfigDict(extra="forbid")

    fact_id:          ID
    predicate:        Annotated[str, Field(min_length=1)]
    arity:            int | None                                          = Field(default=None, ge=0)
    args:             list[RoleArg]
    truth:            TruthDistribution
    time:             FactTime | None                                     = None
    status:           FactStatus
    source:           FactSource | None                                   = None
    constraints_tags: list[Annotated[str, Field(min_length=1)]]          = Field(default_factory=list)
    provenance:       FactProvenance | None                               = None

    @model_validator(mode="after")
    def _arity_matches_args(self) -> Fact:
        if self.arity is not None and len(self.args) != self.arity:
            raise ValueError(
                f"Liczba args ({len(self.args)}) nie zgadza się z arity ({self.arity})"
            )
        return self
