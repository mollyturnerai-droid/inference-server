from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from fastapi import Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db import ApiKey, get_db


@dataclass(frozen=True)
class ApiKeyPrincipal:
    id: str
    name: str
    is_admin: bool


def _extract_api_key(request: Request) -> Optional[str]:
    key = request.headers.get("x-api-key")
    if key:
        return key

    auth = request.headers.get("authorization")
    if auth and auth.lower().startswith("bearer "):
        return auth.split(" ", 1)[1].strip()

    return None


def _hash_api_key(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


async def get_current_api_key(
    request: Request,
    db: Session = Depends(get_db),
) -> ApiKeyPrincipal:
    raw = _extract_api_key(request)
    if not raw:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized",
        )

    # Master key (bootstrap / admin)
    if settings.API_KEY and hmac.compare_digest(raw, settings.API_KEY):
        return ApiKeyPrincipal(id="master", name="master", is_admin=True)

    key_hash = _hash_api_key(raw)
    row = db.query(ApiKey).filter(ApiKey.key_hash == key_hash).first()
    if not row or not row.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized",
        )

    try:
        row.last_used_at = datetime.utcnow()
        db.commit()
    except Exception:
        db.rollback()

    return ApiKeyPrincipal(id=row.id, name=row.name, is_admin=bool(row.is_admin))


async def get_current_api_key_optional(
    request: Request,
    db: Session = Depends(get_db),
) -> Optional[ApiKeyPrincipal]:
    raw = _extract_api_key(request)
    if not raw:
        return None

    if settings.API_KEY and hmac.compare_digest(raw, settings.API_KEY):
        return ApiKeyPrincipal(id="master", name="master", is_admin=True)

    key_hash = _hash_api_key(raw)
    row = db.query(ApiKey).filter(ApiKey.key_hash == key_hash).first()
    if not row or not row.is_active:
        return None

    try:
        row.last_used_at = datetime.utcnow()
        db.commit()
    except Exception:
        db.rollback()

    return ApiKeyPrincipal(id=row.id, name=row.name, is_admin=bool(row.is_admin))


async def require_admin_api_key(
    principal: ApiKeyPrincipal = Depends(get_current_api_key),
) -> ApiKeyPrincipal:
    if not principal.is_admin:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden")
    return principal
