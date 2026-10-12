"""Encrypt users' Gemini API keys at rest (Fernet: AES-128-CBC + HMAC).

The database only ever holds ciphertext; the ENCRYPTION_KEY lives in the host's
environment variables. Generate one with:
    python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
"""
import base64
import hashlib
from functools import lru_cache

from cryptography.fernet import Fernet, InvalidToken

import settings

_DEV_KEY = base64.urlsafe_b64encode(hashlib.sha256(b"teleco-tutor local development only").digest())


@lru_cache(maxsize=1)
def _fernet() -> Fernet:
    if settings.ENCRYPTION_KEY:
        return Fernet(settings.ENCRYPTION_KEY.encode())
    import auth
    if auth.AUTH_MODE != "dev":
        raise RuntimeError("ENCRYPTION_KEY must be set when real sign-in (Supabase) is enabled.")
    return Fernet(_DEV_KEY)          # local development only


def encrypt(plaintext: str) -> str:
    return _fernet().encrypt(plaintext.encode()).decode()


def decrypt(token: str) -> str:
    try:
        return _fernet().decrypt(token.encode()).decode()
    except InvalidToken:
        raise RuntimeError("A stored API key can't be decrypted (was ENCRYPTION_KEY changed?). Save the key again.")
