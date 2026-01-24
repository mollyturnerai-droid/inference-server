import os
import aiofiles
from typing import BinaryIO
from pathlib import Path
from app.core.config import settings


class StorageService:
    def __init__(self):
        self.storage_type = settings.STORAGE_TYPE
        self.storage_path = settings.STORAGE_PATH

        if self.storage_type == "local":
            os.makedirs(self.storage_path, exist_ok=True)

    async def save_file(self, file_path: str, content: bytes) -> str:
        if self.storage_type == "local":
            full_path = os.path.join(self.storage_path, file_path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)

            async with aiofiles.open(full_path, "wb") as f:
                await f.write(content)

            return full_path
        elif self.storage_type == "s3":
            # S3 implementation would go here
            raise NotImplementedError("S3 storage not implemented yet")
        else:
            raise ValueError(f"Unknown storage type: {self.storage_type}")

    async def read_file(self, file_path: str) -> bytes:
        if self.storage_type == "local":
            full_path = os.path.join(self.storage_path, file_path)

            async with aiofiles.open(full_path, "rb") as f:
                return await f.read()
        elif self.storage_type == "s3":
            raise NotImplementedError("S3 storage not implemented yet")
        else:
            raise ValueError(f"Unknown storage type: {self.storage_type}")

    def get_url(self, file_path: str) -> str:
        if self.storage_type == "local":
            return f"file://{os.path.join(self.storage_path, file_path)}"
        elif self.storage_type == "s3":
            return f"s3://{settings.S3_BUCKET}/{file_path}"
        else:
            raise ValueError(f"Unknown storage type: {self.storage_type}")


storage_service = StorageService()
