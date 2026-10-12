"""Who is calling? Every /api/me/* and /api/chat request goes through current_user().

AUTH_MODE=supabase (production)
    The frontend logs users in with Supabase Auth and sends the access token as
    `Authorization: Bearer <jwt>`. We verify it:
      - projects with asymmetric signing keys (the default for new projects): via the
        project's public JWKS at {SUPABASE_URL}/auth/v1/.well-known/jwks.json
      - legacy projects: HS256 with SUPABASE_JWT_SECRET
    The user id is the token's `sub` (a UUID), never taken from the request body,
    so a user can only ever reach their own data.

AUTH_MODE=dev (local development only - the default when SUPABASE_URL is unset)
    No login: the caller names a test user in the `X-Dev-User` header; its id is a
    stable UUID derived from that name. Refused if a SUPABASE_URL is configured, so
    it can't be left on by mistake.
"""
import os
import re
import uuid
from functools import lru_cache
from typing import Optional

import jwt
from fastapi import Header, HTTPException

import settings  # noqa: F401  (loads backend/.env before the variables below are read)

SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET", "")
AUTH_MODE = os.getenv("AUTH_MODE", "supabase" if SUPABASE_URL else "dev")
AUDIENCE = "authenticated"


_DEV_NAME = re.compile(r"^[A-Za-z0-9_.\-]{1,64}$")
_DEV_NAMESPACE = uuid.UUID("6f1c7a52-6a0e-4d43-9a1e-2f4f2a6d0c11")


def dev_user_id(name: str) -> str:
    """Stable UUID for a dev-mode test user name."""
    if not _DEV_NAME.match(name or ""):
        raise ValueError(f"Invalid dev user name: {name!r}")
    return str(uuid.uuid5(_DEV_NAMESPACE, name))


@lru_cache(maxsize=1)
def _jwks_client():
    return jwt.PyJWKClient(f"{SUPABASE_URL}/auth/v1/.well-known/jwks.json", cache_keys=True)


def verify_token(token: str) -> dict:
    """Return the token's claims, or raise jwt.PyJWTError."""
    header = jwt.get_unverified_header(token)
    if header.get("alg") == "HS256":
        if not SUPABASE_JWT_SECRET:
            raise jwt.InvalidTokenError("HS256 token but SUPABASE_JWT_SECRET is not set")
        return jwt.decode(token, SUPABASE_JWT_SECRET, algorithms=["HS256"], audience=AUDIENCE)
    key = _jwks_client().get_signing_key_from_jwt(token).key
    return jwt.decode(token, key, algorithms=["ES256", "RS256", "EdDSA"], audience=AUDIENCE)


def current_user(authorization: Optional[str] = Header(default=None),
                 x_dev_user: Optional[str] = Header(default=None)) -> str:
    """FastAPI dependency: the caller's user id (UUID string)."""
    if AUTH_MODE == "dev":
        if SUPABASE_URL:
            raise HTTPException(500, "AUTH_MODE=dev is not allowed when SUPABASE_URL is configured.")
        if not x_dev_user:
            raise HTTPException(401, "Dev mode: send the X-Dev-User header with a test user name.")
        try:
            return dev_user_id(x_dev_user)
        except ValueError as e:
            raise HTTPException(400, str(e))

    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(401, "Not signed in.")
    try:
        claims = verify_token(authorization.split(" ", 1)[1].strip())
    except jwt.ExpiredSignatureError:
        raise HTTPException(401, "Your session has expired. Sign in again.")
    except jwt.PyJWTError:
        raise HTTPException(401, "Invalid sign-in token.")
    try:
        return str(uuid.UUID(claims["sub"]))
    except (KeyError, ValueError):
        raise HTTPException(401, "Invalid sign-in token.")
