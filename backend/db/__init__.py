"""Postgres access: a small connection pool and helpers.

Every connection gets `search_path = public, extensions`, where Supabase keeps
pgvector, so the `<=>` operator and the `vector` type resolve.
"""
import os
from contextlib import contextmanager
from functools import lru_cache

import settings

SCHEMA_FILE = os.path.join(os.path.dirname(__file__), "schema.sql")


def _configure(conn):
    conn.execute("set search_path = public, extensions")
    conn.commit()


@lru_cache(maxsize=1)
def pool():
    import atexit
    from psycopg_pool import ConnectionPool
    # Small pool: the free host has 512 MB, and Supabase's pooler limits connections.
    p = ConnectionPool(settings.DATABASE_URL, min_size=1, max_size=4, configure=_configure,
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
    with open(SCHEMA_FILE, encoding="utf-8") as f:
        conn.execute(f.read())
    conn.commit()


def vector_literal(values) -> str:
    """pgvector text format: '[0.1,0.2,...]' (pass as a parameter, cast with ::vector)."""
    return "[" + ",".join(f"{v:.7g}" for v in values) + "]"
