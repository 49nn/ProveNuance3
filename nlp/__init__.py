"""
LLM-based NLP entrypoints.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from .ontology_builder import (
    OntologyResult,
    build_ontology_prompt,
    build_ontology_schema,
    parse_ontology_response,
)
from .result import ExtractionResult

if TYPE_CHECKING:
    from .llm_extractor import LLMExtractor


def get_extractor(
    cluster_schemas: list,
    config=None,
    year: int | None = None,
    predicate_positions: dict[str, list[str]] | None = None,
) -> "LLMExtractor":
    """
    Return the only supported extractor backend: LLMExtractor.
    """
    from config import ExtractorConfig, ProjectConfig

    if config is None:
        ext_cfg = ExtractorConfig()
        resolved_year = year or 2026
    elif isinstance(config, ProjectConfig):
        ext_cfg = config.extractor
        resolved_year = year or config.year
    else:
        ext_cfg = config
        resolved_year = year or 2026

    if ext_cfg.backend != "llm":
        raise ValueError("Only backend='llm' is supported.")

    from .llm_extractor import LLMExtractor
    return LLMExtractor(cluster_schemas, ext_cfg, resolved_year, predicate_positions)


__all__ = [
    "ExtractionResult",
    "get_extractor",
    "OntologyResult",
    "build_ontology_prompt",
    "build_ontology_schema",
    "parse_ontology_response",
]
