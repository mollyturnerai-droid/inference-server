import os
import aiofiles
from typing import BinaryIO, Optional
from pathlib import Path
from app.core.config import settings


class StorageService:
    def __init__(self):
        self.storage_type = settings.STORAGE_TYPE
        self.storage_path = settings.STORAGE_PATH
        self._s3_client = None

        if self.storage_type == "local":
            os.makedirs(self.storage_path, exist_ok=True)

    @property
    def s3_client(self):
        """Lazy-load S3 client only when needed"""
        if self._s3_client is None and self.storage_type == "s3":
            import boto3
            self._s3_client = boto3.client(
                's3',
                aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
                region_name=settings.S3_REGION
            )
        return self._s3_client

    async def save_file(self, file_path: str, content: bytes, content_type: str = "application/octet-stream") -> str:
        if self.storage_type == "local":
            full_path = os.path.join(self.storage_path, file_path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)

            async with aiofiles.open(full_path, "wb") as f:
                await f.write(content)

            return file_path
        elif self.storage_type == "s3":
            self.s3_client.put_object(
                Bucket=settings.S3_BUCKET,
                Key=file_path,
                Body=content,
                ContentType=content_type
            )
            return file_path
        else:
            raise ValueError(f"Unknown storage type: {self.storage_type}")

    def save_file_sync(self, file_path: str, content: bytes, content_type: str = "application/octet-stream") -> str:
        """Synchronous version for use in Celery workers"""
        if self.storage_type == "local":
            full_path = os.path.join(self.storage_path, file_path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)

            with open(full_path, "wb") as f:
                f.write(content)

            return file_path
        elif self.storage_type == "s3":
            self.s3_client.put_object(
                Bucket=settings.S3_BUCKET,
                Key=file_path,
                Body=content,
                ContentType=content_type
            )
            return file_path
        else:
            raise ValueError(f"Unknown storage type: {self.storage_type}")

    async def read_file(self, file_path: str) -> bytes:
        if self.storage_type == "local":
            full_path = os.path.join(self.storage_path, file_path)

            async with aiofiles.open(full_path, "rb") as f:
                return await f.read()
        elif self.storage_type == "s3":
            response = self.s3_client.get_object(
                Bucket=settings.S3_BUCKET,
                Key=file_path
            )
            return response['Body'].read()
        else:
            raise ValueError(f"Unknown storage type: {self.storage_type}")

    def get_url(self, file_path: str) -> str:
        """Get internal URL (file:// or s3://) - for backward compatibility"""
        if self.storage_type == "local":
            return f"file://{os.path.join(self.storage_path, file_path)}"
        elif self.storage_type == "s3":
            return f"s3://{settings.S3_BUCKET}/{file_path}"
        else:
            raise ValueError(f"Unknown storage type: {self.storage_type}")

    def get_public_url(self, file_path: str) -> str:
        """Get publicly accessible HTTP URL for a file"""
        if self.storage_type == "local":
            base = (settings.API_BASE_URL or "").rstrip("/")
            if not base or base in {"http://localhost:8000", "http://127.0.0.1:8000"}:
                return f"/v1/files/{file_path}"
            return f"{base}/v1/files/{file_path}"
        elif self.storage_type == "s3":
            # Generate presigned URL for S3
            url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': settings.S3_BUCKET, 'Key': file_path},
                ExpiresIn=settings.S3_PRESIGNED_EXPIRY
            )
            return url
        else:
            raise ValueError(f"Unknown storage type: {self.storage_type}")


storage_service = StorageService()
