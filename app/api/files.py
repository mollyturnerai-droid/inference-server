from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from pathlib import Path
from uuid import uuid4
import os
from app.core.config import settings
from app.services.storage import storage_service

router = APIRouter(prefix="/files", tags=["Files"])


@router.get("/{file_path:path}")
async def serve_file(file_path: str):
    """Serve a file from local storage"""
    if settings.STORAGE_TYPE != "local":
        raise HTTPException(
            status_code=404,
            detail="File serving only available for local storage"
        )

    storage_path = Path(settings.STORAGE_PATH).resolve()
    full_path = (storage_path / file_path).resolve()

    # Security: Ensure path doesn't escape storage directory
    if not str(full_path).startswith(str(storage_path)):
        raise HTTPException(status_code=403, detail="Access denied")

    if not full_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    # Determine content type based on extension
    content_type = "application/octet-stream"
    suffix = full_path.suffix.lower()
    if suffix == ".png":
        content_type = "image/png"
    elif suffix in (".jpg", ".jpeg"):
        content_type = "image/jpeg"
    elif suffix == ".gif":
        content_type = "image/gif"
    elif suffix == ".webp":
        content_type = "image/webp"
    elif suffix == ".wav":
        content_type = "audio/wav"
    elif suffix == ".mp3":
        content_type = "audio/mpeg"
    elif suffix == ".m4a":
        content_type = "audio/mp4"
    elif suffix == ".ogg":
        content_type = "audio/ogg"
    elif suffix == ".mp4":
        content_type = "video/mp4"

    return FileResponse(full_path, media_type=content_type)


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    allowed_types = {
        "audio/wav",
        "audio/x-wav",
        "audio/mpeg",
        "audio/mp4",
        "audio/ogg",
        "video/mp4",
        "image/jpeg",
        "image/png",
    }
    if not file.content_type or file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail="Only audio, video, or image uploads are supported",
        )

    original_name = file.filename or "upload"
    _, ext = os.path.splitext(original_name)
    ext = ext.lower() if ext else ""
    allowed_exts = {".wav", ".mp3", ".m4a", ".ogg", ".mp4", ".jpeg", ".jpg", ".png"}
    if ext not in allowed_exts:
        if file.content_type.startswith("image/"):
            ext = ".png"
        elif file.content_type.startswith("video/"):
            ext = ".mp4"
        else:
            ext = ".wav"

    file_path = f"uploads/{uuid4().hex}{ext}"
    content = await file.read()
    await storage_service.save_file(file_path=file_path, content=content, content_type=file.content_type)

    return {
        "file_path": file_path,
        "url": storage_service.get_public_url(file_path),
        "content_type": file.content_type,
    }
