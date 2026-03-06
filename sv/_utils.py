"""
Wewnętrzne narzędzia współdzielone przez moduły sv/.

Wydzielone tutaj, by uniknąć cyklicznych importów między
converter.py, runner.py i proof.py.
"""
from __future__ import annotations

import re

try:
    from unidecode import unidecode as _unidecode
except ImportError:
    def _unidecode(s: str) -> str:  # type: ignore[misc]
        return s

_CLINGO_ID_RE = re.compile(r"[^a-z0-9_]")


def to_clingo_id(s: str) -> str:
    """
    Normalizuje string (entity_id, wartość) do poprawnego atomu Clingo (lowercase).
    Polskie znaki diakrytyczne są transliterowane przed normalizacją:
      "Zamówienie" → "zamowienie"  (nie "zam_wienie")
    UUID "3f9a-..." → "e_3f9a_..." (prefiks gdy zaczyna się od cyfry lub jest pusty).
    """
    safe = _CLINGO_ID_RE.sub("_", _unidecode(s).lower())
    if not safe or safe[0].isdigit():
        safe = "e_" + safe
    return safe
