"""
nn/ — Neural Proposer dla ProveNuance3.

Publiczne API:
  NNConfig                   — hiperparametry
  ClusterSchema              — definicja klastra (z DB)
  ClusterStateRow            — stan klastra (z DB)
  GraphNodeIndex             — indeks węzłów grafu
  EdgeTypeSpec               — specyfikacja typu krawędzi
  GraphBuilder               — buduje HeteroData z Pydantic/DB
  EntityMemoryBiasEncoder    — memory_slots → bias logitów
  HeteroMessagePassingBank   — W+/W- per typ krawędzi
  ExceptionGateBank          — bramki wyjątków ab_*
  NeuralTracer               — rejestruje NeuralTraceItem
  NeuralProposer             — T-krokowy forward pass
  NeuralInference            — entry point pipeline
  ProposerTrainer            — pętla treningu BPTT
"""

from .config import NNConfig
from .entity_memory import EntityMemoryBiasEncoder
from .gating import ExceptionGateBank, GateSpec
from .graph_builder import (
    ClusterSchema,
    ClusterStateRow,
    EdgeTypeSpec,
    GraphBuilder,
    GraphNodeIndex,
)
from .inference import NeuralInference
from .message_passing import HeteroMessagePassingBank, LogitMessagePassing
from .proposer import NeuralProposer
from .trace import NeuralTracer
from .trainer import ProposerTrainer

__all__ = [
    # Konfiguracja
    "NNConfig",
    # Graf
    "ClusterSchema",
    "ClusterStateRow",
    "EdgeTypeSpec",
    "GraphBuilder",
    "GraphNodeIndex",
    # Pamięć encji
    "EntityMemoryBiasEncoder",
    # Message passing
    "HeteroMessagePassingBank",
    "LogitMessagePassing",
    # Bramki wyjątków
    "ExceptionGateBank",
    "GateSpec",
    # Trace
    "NeuralTracer",
    # Proposer
    "NeuralProposer",
    # Pipeline
    "NeuralInference",
    "ProposerTrainer",
]
