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
from .cluster import (
    ClusterSchema,
    ClusterStateRow,
)
from .self_training import (
    CaseSplit,
    PseudoClusterLabel,
    PseudoFactLabel,
    RoundStatus,
    SelfTrainingRound,
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
    # cluster
    "ClusterSchema", "ClusterStateRow",
    # self-training
    "CaseSplit", "RoundStatus", "SelfTrainingRound",
    "PseudoFactLabel", "PseudoClusterLabel",
    # fact
    "FactStatus", "NeuralTraceItem", "FactProvenance",
    "FactSource", "FactTime", "Fact",
    # rule
    "RuleLanguage", "LiteralType", "RuleHead",
    "RuleBodyLiteral", "RuleMetadata", "Rule",
]
