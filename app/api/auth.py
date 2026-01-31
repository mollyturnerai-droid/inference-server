from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
from typing import List, Optional
import secrets
import hashlib

from app.db import ApiKey, get_db
from app.services.auth import require_admin_api_key


router = APIRouter(prefix="/admin", tags=["Admin"])


class ApiKeyCreateRequest(BaseModel):
    name: str
    is_admin: bool = False


class ApiKeyCreateResponse(BaseModel):
    id: str
    name: str
    prefix: str
    api_key: str
    is_admin: bool


class ApiKeyListItem(BaseModel):
    id: str
    name: str
    prefix: str
    is_active: bool
    is_admin: bool


def _generate_raw_api_key() -> str:
    return "isk_" + secrets.token_urlsafe(32)


@router.get("/api-keys", response_model=List[ApiKeyListItem])
async def list_api_keys(
    db: Session = Depends(get_db),
    _admin=Depends(require_admin_api_key),
):
    keys = db.query(ApiKey).order_by(ApiKey.created_at.desc()).all()
    return [
        ApiKeyListItem(
            id=k.id,
            name=k.name,
            prefix=k.prefix,
            is_active=bool(k.is_active),
            is_admin=bool(k.is_admin),
        )
        for k in keys
    ]


@router.post("/api-keys", response_model=ApiKeyCreateResponse)
async def create_api_key(
    body: ApiKeyCreateRequest,
    db: Session = Depends(get_db),
    _admin=Depends(require_admin_api_key),
):
    raw = _generate_raw_api_key()
    prefix = raw[:8]
    key_hash = hashlib.sha256(raw.encode("utf-8")).hexdigest()

    row = ApiKey(
        name=body.name,
        prefix=prefix,
        key_hash=key_hash,
        is_active=True,
        is_admin=bool(body.is_admin),
    )
    db.add(row)
    db.commit()
    db.refresh(row)

    return ApiKeyCreateResponse(
        id=row.id,
        name=row.name,
        prefix=row.prefix,
        api_key=raw,
        is_admin=bool(row.is_admin),
    )


@router.post("/api-keys/{api_key_id}/revoke")
async def revoke_api_key(
    api_key_id: str,
    db: Session = Depends(get_db),
    _admin=Depends(require_admin_api_key),
):
    row = db.query(ApiKey).filter(ApiKey.id == api_key_id).first()
    if not row:
        raise HTTPException(status_code=404, detail="API key not found")
    row.is_active = False
    db.commit()
    return {"success": True}
