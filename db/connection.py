"""
Połączenie z PostgreSQL – konfiguracja przez zmienne środowiskowe.
"""
from __future__ import annotations

import os

import psycopg

DSN: dict[str, object] = {
    "host": os.getenv("PN3_HOST", "localhost"),
    "port": int(os.getenv("PN3_PORT", "5432")),
    "dbname": os.getenv("PN3_DB", "provenuance"),
    "user": os.getenv("PN3_USER", "provenuance"),
    "password": os.getenv("PN3_PASSWORD", "provenuance"),
}


def connect() -> psycopg.Connection:
    return psycopg.connect(**DSN)  # type: ignore[arg-type]
