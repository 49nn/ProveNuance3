"""
pipeline/ — warstwa integracyjna propose-verify.

Łączy Neural Proposer (nn/) i Symbolic Verifier (sv/) w jeden przebieg.

Przykład użycia:
    from pipeline import ProposeVerifyRunner, PipelineResult
    from nn import ClusterSchema

    schemas = [...]   # z DB: cluster_definitions + cluster_domain_values
    runner  = ProposeVerifyRunner.from_schemas(schemas)

    result  = runner.run(entities, facts, rules, cluster_states)
    print(result.summary())
    for f in result.proved:
        print(f.predicate, f.provenance)
"""

from .result import PipelineResult
from .runner import ProposeVerifyRunner

__all__ = [
    "PipelineResult",
    "ProposeVerifyRunner",
]
