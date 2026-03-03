"""
data_model – struktury danych oparte na schemas/*.json (Pydantic v2).

Importuj z konkretnych modułów lub używaj poniższych re-exportów.
"""

from .common import (
    ID,
    Confidence,
    TruthValue,
    Span,
    ProvenanceItem,
    TruthDistribution,
    RoleArg,
    VarTerm,
    ConstTerm,
    Term,
    RuleArg,
)
from .entity import (
    MemorySlotEntry,
    EntityLinking,
    Entity,
)
from .fact import (
    FactStatus,
    NeuralTraceItem,
    FactProvenance,
    FactSource,
    FactTime,
    Fact,
)
from .rule import (
    RuleLanguage,
    LiteralType,
    RuleHead,
    RuleBodyLiteral,
    RuleMetadata,
    Rule,
)

__all__ = [
    # common
    "ID", "Confidence", "TruthValue",
    "Span", "ProvenanceItem", "TruthDistribution",
    "RoleArg", "VarTerm", "ConstTerm", "Term", "RuleArg",
    # entity
    "MemorySlotEntry", "EntityLinking", "Entity",
    # fact
    "FactStatus", "NeuralTraceItem", "FactProvenance",
    "FactSource", "FactTime", "Fact",
    # rule
    "RuleLanguage", "LiteralType", "RuleHead",
    "RuleBodyLiteral", "RuleMetadata", "Rule",
]
