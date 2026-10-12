"""Postgres access: a small connection pool and helpers.

Queries name pgvector's type and operator with their schema (extensions.vector,
OPERATOR(extensions.<=>)), because Supabase's transaction pooler doesn't keep
session settings such as search_path between transactions.
"""
import os
from contextlib import contextmanager
from functools import lru_cache

import settings

SCHEMA_FILE = os.path.join(os.path.dirname(__file__), "schema.sql")


@lru_cache(maxsize=1)
def pool():
    import atexit
    from psycopg_pool import ConnectionPool
    # Small pool: the free host has 512 MB, and Supabase's pooler limits connections.
    p = ConnectionPool(settings.DATABASE_URL, min_size=1, max_size=4,
                       kwargs={"prepare_threshold": None},  # the Supabase transaction pooler forbids prepared statements
                       open=True)
    atexit.register(p.close)
    return p


@contextmanager
def connect():
    """A pooled connection; commits on success, rolls back on error."""
    with pool().connection() as conn:
        yield conn


def apply_schema(conn) -> None:
    """Create/upgrade the tables (idempotent)."""
    with open(SCHEMA_FILE, encoding="utf-8") as f:
        conn.execute(f.read())
    conn.commit()


def vector_literal(values) -> str:
    """pgvector text format: '[0.1,0.2,...]' (pass as a parameter, cast with ::vector)."""
    return "[" + ",".join(f"{v:.7g}" for v in values) + "]"
