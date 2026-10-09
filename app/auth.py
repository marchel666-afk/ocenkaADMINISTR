"""Пользователи и вход: хеширование паролей и подписанная cookie сессии (только стандартная библиотека)."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from pathlib import Path

COOKIE_NAME = "ocenka_session"
SESSION_MAX_AGE = 30 * 24 * 3600  # 30 дней

_SCRYPT = {"n": 2**14, "r": 8, "p": 1, "dklen": 32}


def hash_password(password: str) -> str:
    salt = os.urandom(16)
    digest = hashlib.scrypt(password.encode("utf-8"), salt=salt, **_SCRYPT)
    return "scrypt$" + base64.b64encode(salt).decode() + "$" + base64.b64encode(digest).decode()


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, salt_b64, digest_b64 = stored.split("$")
        if scheme != "scrypt":
            return False
        digest = hashlib.scrypt(password.encode("utf-8"), salt=base64.b64decode(salt_b64), **_SCRYPT)
        return hmac.compare_digest(digest, base64.b64decode(digest_b64))
    except (ValueError, TypeError):
        return False


def load_or_create_secret(path: Path) -> bytes:
    """Секрет для подписи cookie хранится в data/secret.key и создаётся при первом запуске."""
    if path.exists():
        data = path.read_bytes().strip()
        if len(data) >= 32:
            return data
    path.parent.mkdir(parents=True, exist_ok=True)
    data = secrets.token_hex(32).encode()
    path.write_bytes(data)
    return data


class SessionSigner:
    def __init__(self, secret: bytes, max_age: int = SESSION_MAX_AGE):
        self.secret = secret
        self.max_age = max_age

    def _sign(self, payload: bytes) -> str:
        return base64.urlsafe_b64encode(hmac.new(self.secret, payload, hashlib.sha256).digest()).decode().rstrip("=")

    def dumps(self, data: dict) -> str:
        payload = json.dumps({**data, "ts": int(time.time())}, separators=(",", ":")).encode()
        body = base64.urlsafe_b64encode(payload).decode().rstrip("=")
        return f"{body}.{self._sign(payload)}"

    def loads(self, token: str | None) -> dict | None:
        if not token or "." not in token:
            return None
        body, signature = token.rsplit(".", 1)
        try:
            payload = base64.urlsafe_b64decode(body + "=" * (-len(body) % 4))
        except (ValueError, TypeError):
            return None
        if not hmac.compare_digest(signature, self._sign(payload)):
            return None
        try:
            data = json.loads(payload)
        except ValueError:
            return None
        if not isinstance(data, dict) or time.time() - int(data.get("ts", 0)) > self.max_age:
            return None
        return data
