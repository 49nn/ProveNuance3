"""
Helpers for loading project environment variables from the repo root.
"""
from __future__ import annotations

import os
from pathlib import Path


_ENV_LOADED = False
_DOTENV_AVAILABLE = True


def load_project_env() -> Path:
    global _ENV_LOADED, _DOTENV_AVAILABLE

    env_path = Path(__file__).resolve().parent / ".env"
    if _ENV_LOADED:
        return env_path

    try:
        from dotenv import load_dotenv
    except ImportError:
        _DOTENV_AVAILABLE = False
        _ENV_LOADED = True
        return env_path

    load_dotenv(env_path)
    _ENV_LOADED = True
    _DOTENV_AVAILABLE = True
    return env_path


def get_required_env(var_name: str) -> str:
    env_path = load_project_env()
    value = os.environ.get(var_name, "").strip()
    if value:
        return value

    if not _DOTENV_AVAILABLE:
        raise EnvironmentError(
            f"Brak zmiennej srodowiskowej: {var_name}. "
            "Pakiet python-dotenv nie jest zainstalowany, wiec .env nie zostal zaladowany. "
            f"Plik .env oczekiwany: {env_path}"
        )

    raise EnvironmentError(
        f"Brak zmiennej srodowiskowej: {var_name}. "
        f"Sprawdz plik .env: {env_path}"
    )
