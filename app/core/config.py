from pydantic_settings import BaseSettings
from typing import Optional


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

    # Authentication
    SECRET_KEY: str = "your-secret-key-change-this"
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

    # Model Configuration
    MODEL_CACHE_DIR: str = "/tmp/model_cache"
    MAX_MODEL_CACHE_SIZE_GB: int = 50

    # Worker Configuration
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/0"
    WORKER_CONCURRENCY: int = 2

    # Single Container Mode
    SINGLE_CONTAINER: bool = True  # Run all services in one container

    # Inference Configuration
    DEFAULT_TIMEOUT: int = 300
    MAX_BATCH_SIZE: int = 8
    ENABLE_GPU: bool = True

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
