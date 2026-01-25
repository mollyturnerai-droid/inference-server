from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path
from app.core.config import settings

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

    return FileResponse(full_path, media_type=content_type)
