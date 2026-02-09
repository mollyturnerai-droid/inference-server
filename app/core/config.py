import secrets

from pydantic_settings import BaseSettings
from typing import Optional


def _generate_secret_key() -> str:
    return secrets.token_urlsafe(32)


class Settings(BaseSettings):
    # Server
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    WORKERS: int = 4

    # Redis
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: Optional[str] = None

    # Database
    DATABASE_URL: str = "sqlite:///./inference.db"
    DATABASE_FORCE_IPV4: bool = True
    DATABASE_HOSTADDR: Optional[str] = None
    DATABASE_POOL_SIZE: int = 5
    DATABASE_MAX_OVERFLOW: int = 10
    DATABASE_POOL_RECYCLE: int = 3600  # Recycle connections after 1 hour
    DATABASE_POOL_TIMEOUT: int = 30

    # Authentication — randomly generated per process when not set via env.
    SECRET_KEY: str = ""
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # Storage
    STORAGE_TYPE: str = "local"  # local or s3
    STORAGE_PATH: str = "/tmp/inference_storage"
    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None
    S3_BUCKET: Optional[str] = None
    S3_REGION: str = "us-east-1"
    S3_PRESIGNED_EXPIRY: int = 3600  # Seconds for presigned URL validity

    # API URL for generating file URLs
    API_BASE_URL: str = "http://localhost:8000"

    CORS_ALLOW_ORIGINS: str = "*"
    CORS_ALLOW_CREDENTIALS: bool = False
    TRUST_PROXY_HEADERS: bool = False
    WEBHOOK_ALLOWED_HOSTS: Optional[str] = None

    API_KEY: Optional[str] = None

    # Model Configuration
    MODEL_CACHE_DIR: str = "/tmp/model_cache"
    MAX_MODEL_CACHE_SIZE_GB: int = 50
    MAX_LOADED_MODELS: int = 0
    MODEL_IDLE_TTL_SECONDS: Optional[int] = None
    TRUST_REMOTE_CODE: bool = False
    DISABLE_SAFETY_CHECKER: bool = True

    BUILD_GIT_SHA: Optional[str] = None
    BUILD_IMAGE_TAG: Optional[str] = None

    # Worker Configuration
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/0"
    WORKER_CONCURRENCY: int = 2
    CELERY_WORKER_POOL: str = "solo"

    # Single Container Mode
    SINGLE_CONTAINER: bool = True  # Run all services in one container

    # Inference Configuration
    DEFAULT_TIMEOUT: int = 300
    MAX_BATCH_SIZE: int = 8
    ENABLE_GPU: bool = True
    ENABLE_MCP: bool = False

    # File upload limit (bytes). Default 50 MB.
    MAX_UPLOAD_SIZE_BYTES: int = 50 * 1024 * 1024

    # SSRF protection: block requests to private/link-local IP ranges by default.
    ALLOW_PRIVATE_URL_FETCH: bool = False

    CATALOG_PATH: str = "/tmp/model_catalog.json"
    CATALOG_ADMIN_TOKEN: Optional[str] = None
    HF_API_TOKEN: Optional[str] = None
    REPLICATE_API_TOKEN: Optional[str] = None
    RECON_ENABLED: bool = True
    RECON_ON_STARTUP: bool = True
    RECON_INTERVAL_MINUTES: int = 1440
    RECON_MAX_MODELS: int = 200
    RECON_SOURCES: str = "huggingface,replicate"
    RECON_TIMEOUT_SECONDS: int = 30

    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"


settings = Settings()

# Generate a random secret key if none was provided (safe default).
if not settings.SECRET_KEY:
    import logging as _logging

    settings.SECRET_KEY = _generate_secret_key()
    _logging.getLogger(__name__).warning(
        "SECRET_KEY was not set — using a random key for this process. "
        "JWTs will not survive restarts. Set SECRET_KEY in env for persistence."
    )
