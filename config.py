"""
Project configuration for ProveNuance3.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass
class ExtractorConfig:
    """
    Text extraction configuration.

    Runtime supports only the LLM backend.
    """

    backend: Literal["llm"] = "llm"
    gemini_model: str = "gemini-2.5-flash"
    temperature: float = 0.0
    max_retries: int = 2
    max_output_tokens: int | None = 65536
    sv_verification: bool = True
    api_key_env: str = "GEMINI_API_KEY"
    preflight_timeout_s: int = 120

    def __post_init__(self) -> None:
        if self.backend != "llm":
            raise ValueError("Only backend='llm' is supported.")
        if self.max_output_tokens is not None and self.max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be > 0.")
        if self.preflight_timeout_s <= 0:
            raise ValueError("preflight_timeout_s must be > 0.")


@dataclass
class ExplainerConfig:
    """
    Explanation layer configuration.
    """

    gemini_model: str = "gemini-2.5-flash"
    temperature: float = 0.3
    api_key_env: str = "GEMINI_API_KEY"
    language: str = "pl"
    max_facts: int = 30
    grounded: bool = False


@dataclass
class ProjectConfig:
    """
    Top-level project configuration loaded from JSON.
    """

    extractor: ExtractorConfig = field(default_factory=ExtractorConfig)
    explainer: ExplainerConfig = field(default_factory=ExplainerConfig)
    year: int = 2026

    @classmethod
    def load(cls, path: str | Path = "project_config.json") -> "ProjectConfig":
        """
        Load configuration from JSON. If the file does not exist, use defaults.
        """
        p = Path(path)
        if not p.exists():
            return cls()
        data = json.loads(p.read_text(encoding="utf-8"))
        ext = ExtractorConfig(**data.get("extractor", {}))
        expl = ExplainerConfig(**data.get("explainer", {}))
        return cls(
            extractor=ext,
            explainer=expl,
            year=data.get("year", 2026),
        )
