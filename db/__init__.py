"""
Public API for DB integration.
"""
from __future__ import annotations

import psycopg

from .connection import connect as _connect
from .rule_repo import load_rules as _load_rules
from .schema_repo import load_cluster_schemas as _load_cluster_schemas
from .schema_repo import load_predicate_positions as _load_predicate_positions
from .session import DBSession


def connect() -> psycopg.Connection:
    return _connect()


def load_cluster_schemas(conn: psycopg.Connection | None = None):
    if conn is not None:
        return _load_cluster_schemas(conn)
    with _connect() as local_conn:
        return _load_cluster_schemas(local_conn)


def load_predicate_positions(conn: psycopg.Connection | None = None):
    if conn is not None:
        return _load_predicate_positions(conn)
    with _connect() as local_conn:
        return _load_predicate_positions(local_conn)


def load_rules(
    conn: psycopg.Connection | None = None,
    enabled_only: bool = True,
):
    if conn is not None:
        return _load_rules(conn, enabled_only=enabled_only)
    with _connect() as local_conn:
        return _load_rules(local_conn, enabled_only=enabled_only)


__all__ = [
    "DBSession",
    "connect",
    "load_cluster_schemas",
    "load_predicate_positions",
    "load_rules",
]
