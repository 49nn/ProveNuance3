"""
Model danych dla encji z pamięcią instancyjną.
Mapuje: schemas/entity.json
"""
from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .common import ID, Confidence, ProvenanceItem


# ---------------------------------------------------------------------------
# MemorySlotEntry  (entity.json → memory_slots additionalProperties items)
# ---------------------------------------------------------------------------

class MemorySlotEntry(BaseModel):
    """
    Jedna wersja wartości slotu encji.
    value i normalized są celowo Any — slot może przechowywać string,
    liczbę, datę lub obiekt złożony.
    """
    model_config = ConfigDict(extra="forbid")

    value:       Any                          # wymagane, dowolny typ JSON
    normalized:  Any                  = None  # opcjonalna znormalizowana forma
    valid_from:  datetime | None      = None
    valid_to:    datetime | None      = None
    confidence:  Confidence | None    = None
    source_rank: Confidence | None    = None
    provenance:  list[ProvenanceItem] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# EntityLinking  (entity.json → linking)
# ---------------------------------------------------------------------------

class EntityLinking(BaseModel):
    model_config = ConfigDict(extra="forbid")

    blocking_keys:     list[Annotated[str, Field(min_length=1)]] = Field(default_factory=list)
    last_linked_from:  list[ID]                                   = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Entity  (entity.json)
# ---------------------------------------------------------------------------

# memory_slots: każdy klucz → lista co najmniej 1 wpisu (entity.json minItems:1)
MemorySlots = dict[str, Annotated[list[MemorySlotEntry], Field(min_length=1)]]


class Entity(BaseModel):
    """
    Encja z pamięcią instancyjną.
    entity_id jest kluczem biznesowym (np. 'O100', 'CUST1', '2026-02-01').
    type odpowiada nazwie z entity_types w DB (np. 'ORDER', 'CUSTOMER', 'DATE').
    """
    model_config = ConfigDict(extra="forbid")

    entity_id:      Annotated[str, Field(min_length=1)]
    type:           Annotated[str, Field(min_length=1)]
    canonical_name: Annotated[str, Field(min_length=1)]
    aliases:        list[Annotated[str, Field(min_length=1)]] = Field(default_factory=list)
    embedding_ref:  str | None                                = None
    created_at:     datetime
    updated_at:     datetime | None                           = None
    provenance:     list[ProvenanceItem]                      = Field(default_factory=list)
    memory_slots:   MemorySlots                               = Field(default_factory=dict)
    linking:        EntityLinking | None                      = None

    @field_validator("aliases")
    @classmethod
    def _unique_aliases(cls, v: list[str]) -> list[str]:
        if len(v) != len(set(v)):
            raise ValueError("aliases muszą być unikalne")
        return v
