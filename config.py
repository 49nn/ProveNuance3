"""
Konfiguracja projektu ProveNuance3.

Wczytaj z project_config.json (jeśli istnieje) lub użyj wartości domyślnych:

    from config import ProjectConfig
    cfg = ProjectConfig.load()          # z project_config.json
    cfg = ProjectConfig.load("mój.json")

Przełączanie ekstraktora:

    project_config.json:
    {
      "extractor": { "backend": "llm", "gemini_model": "gemini-2.0-flash" },
      "year": 2026
    }
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass
class ExtractorConfig:
    """
    Konfiguracja warstwy ekstrakcji tekstu.

    backend:
        "regex" — pure-regex TextExtractor (domyślnie, bez zewnętrznych zależności)
        "llm"   — LLMExtractor oparty o Gemini (wymaga google-generativeai + GEMINI_API_KEY)

    gemini_model:   ID modelu Gemini (np. "gemini-2.0-flash", "gemini-1.5-pro")
    temperature:    0.0 = deterministyczne odpowiedzi
    max_retries:    ile razy LLM dostaje szansę na poprawę po odrzuceniu przez SV
    sv_verification: czy uruchamiać Symbolic Verifier jako walidator po ekstrakcji LLM
    api_key_env:    nazwa zmiennej środowiskowej z kluczem API Gemini
    """

    backend: Literal["regex", "llm"] = "regex"
    gemini_model: str = "gemini-2.5-flash"
    temperature: float = 0.0
    max_retries: int = 2
    sv_verification: bool = True
    api_key_env: str = "GEMINI_API_KEY"


@dataclass
class ExplainerConfig:
    """
    Konfiguracja warstwy wyjaśniania wyników pipeline'u.

    gemini_model:  ID modelu Gemini (np. "gemini-2.0-flash")
    temperature:   0.3 = nieco bardziej naturalny język niż deterministyczny 0.0
    api_key_env:   nazwa zmiennej środowiskowej z kluczem API Gemini
    language:      język odpowiedzi — "pl" (polski) lub "en" (angielski)
    max_facts:     maksymalna liczba faktów przekazywanych do promptu
    """

    gemini_model: str = "gemini-2.5-flash"
    temperature: float = 0.3
    api_key_env: str = "GEMINI_API_KEY"
    language: str = "pl"
    max_facts: int = 30
    grounded: bool = False  # True = nie wysyłaj tekstu sprawy, tylko dane strukturalne


@dataclass
class ProjectConfig:
    """
    Główna konfiguracja projektu.

    Przykład (project_config.json):
    {
        "extractor": {
            "backend": "llm",
            "gemini_model": "gemini-2.0-flash",
            "temperature": 0.0,
            "max_retries": 2,
            "sv_verification": true,
            "api_key_env": "GEMINI_API_KEY"
        },
        "explainer": {
            "gemini_model": "gemini-2.0-flash",
            "temperature": 0.3,
            "language": "pl"
        },
        "year": 2026
    }
    """

    extractor: ExtractorConfig = field(default_factory=ExtractorConfig)
    explainer: ExplainerConfig = field(default_factory=ExplainerConfig)
    year: int = 2026

    @classmethod
    def load(cls, path: str | Path = "project_config.json") -> "ProjectConfig":
        """
        Wczytaj konfigurację z pliku JSON.
        Jeśli plik nie istnieje — zwróć domyślną konfigurację (backend=regex).
        """
        p = Path(path)
        if not p.exists():
            return cls()
        data = json.loads(p.read_text(encoding="utf-8"))
        ext_data = data.get("extractor", {})
        ext = ExtractorConfig(**ext_data)
        expl_data = data.get("explainer", {})
        expl = ExplainerConfig(**expl_data)
        return cls(
            extractor=ext,
            explainer=expl,
            year=data.get("year", 2026),
        )
