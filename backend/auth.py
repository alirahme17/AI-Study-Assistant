"""Who is calling? Every /api/me/* and /api/chat request goes through current_user().

AUTH_MODE=supabase (production)
    The frontend logs users in with Supabase Auth and sends the access token as
    `Authorization: Bearer <jwt>`. We verify it:
      - projects with asymmetric signing keys (the default for new projects): via the
        project's public JWKS at {SUPABASE_URL}/auth/v1/.well-known/jwks.json
      - legacy projects: HS256 with SUPABASE_JWT_SECRET
    The workspace name is derived from the token's `sub` (user id), never from the
    request body, so a user can only ever reach their own data.

AUTH_MODE=dev (local development only - the default when SUPABASE_URL is unset)
    No login: the caller names their workspace in the `X-Dev-User` header.
    Refused if a SUPABASE_URL is configured, so it can't be left on by mistake.
"""
import os
from functools import lru_cache
from typing import Optional

import jwt
from fastapi import Header, HTTPException

from workspace import validate_username

SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET", "")
AUTH_MODE = os.getenv("AUTH_MODE", "supabase" if SUPABASE_URL else "dev")
AUDIENCE = "authenticated"


def workspace_for_user_id(user_id: str) -> str:
    """Supabase user ids are UUIDs: 'u-<uuid>' is a valid, unique workspace name."""
    return validate_username(f"u-{user_id}")


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
    """FastAPI dependency: the caller's workspace name."""
    if AUTH_MODE == "dev":
        if SUPABASE_URL:
            raise HTTPException(500, "AUTH_MODE=dev is not allowed when SUPABASE_URL is configured.")
        if not x_dev_user:
            raise HTTPException(401, "Dev mode: send the X-Dev-User header with a workspace name.")
        try:
            return validate_username(x_dev_user)
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
    return workspace_for_user_id(claims["sub"])
