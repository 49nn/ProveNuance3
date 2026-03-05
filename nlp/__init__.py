"""
nlp/ — ekstraktor faktów z tekstu polskiego.

Dwa backendy do wyboru przez ProjectConfig:
  - "regex"  TextExtractor  — pure-regex, brak zewnętrznych zależności (domyślnie)
  - "llm"    LLMExtractor   — Google Gemini + pętla zwrotna Symbolic Verifier

Przykład użycia (regex — domyślny):
    from nlp import TextExtractor, ExtractionResult
    from nn import ClusterSchema

    schemas = [...]
    extractor = TextExtractor(schemas)
    result = extractor.extract("...", source_id="case_1")
    print(result.summary())

Przykład użycia (LLM — przez factory):
    from config import ProjectConfig
    from nlp import get_extractor

    cfg = ProjectConfig.load()          # wczytaj project_config.json
    extractor = get_extractor(schemas, cfg)
    result = extractor.extract("...", source_id="case_1")

Przełącznik w project_config.json:
    { "extractor": { "backend": "llm", "gemini_model": "gemini-2.0-flash" } }
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from .extractor import TextExtractor
from .result import ExtractionResult

if TYPE_CHECKING:
    from .llm_extractor import LLMExtractor


def get_extractor(
    cluster_schemas: list,
    config=None,
    year: int | None = None,
) -> "TextExtractor | LLMExtractor":
    """
    Fabryka ekstraktora na podstawie konfiguracji projektu.

    Parametry:
        cluster_schemas:  lista ClusterSchema (z DB / seed_ontology)
        config:           ProjectConfig | ExtractorConfig | None
                          Jeśli None — używa domyślnego TextExtractor (regex)
        year:             rok dla dat; jeśli None — pobiera z config.year lub 2026

    Przykład:
        cfg = ProjectConfig.load()
        extractor = get_extractor(schemas, cfg)
        result = extractor.extract(text, source_id="case_1")
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

    if ext_cfg.backend == "llm":
        from .llm_extractor import LLMExtractor
        return LLMExtractor(cluster_schemas, ext_cfg, resolved_year)

    return TextExtractor(cluster_schemas, resolved_year)


__all__ = [
    "TextExtractor",
    "LLMExtractor",
    "ExtractionResult",
    "get_extractor",
]
