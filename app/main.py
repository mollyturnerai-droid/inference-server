from fastapi import FastAPI, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from app.api import api_router
from app.db import engine, Base, SessionLocal, ApiKey
from app.core.config import settings
from app.services.auth import get_current_api_key
from app.services.recon import start_recon_scheduler
from sqlalchemy.engine.url import make_url

# Create database tables (only if database is available)
try:
    Base.metadata.create_all(bind=engine)
except Exception as e:
    print(f"Warning: Could not create database tables: {e}")
    print("Database will be initialized when connection is available")


@app.on_event("startup")
def _start_recon():
    start_recon_scheduler()

def _get_client_ip(request: Request) -> str:
    if settings.TRUST_PROXY_HEADERS:
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip.strip()
    return get_remote_address(request)


# Initialize rate limiter
limiter = Limiter(key_func=_get_client_ip)

# Create FastAPI app
app = FastAPI(
    title="Inference Server",
    description="A full-featured ML inference engine",
    version="1.0.0"
)

# Add rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

_origins_raw = (settings.CORS_ALLOW_ORIGINS or "").strip()
if _origins_raw == "*":
    _allow_origins = ["*"]
else:
    _allow_origins = [o.strip() for o in _origins_raw.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allow_origins,
    allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def require_api_key(request: Request, call_next):
    if request.method == "OPTIONS":
        return await call_next(request)

    public_paths = {
        "/",
        "/health",
        "/health/detailed",
        "/docs",
        "/openapi.json",
        "/redoc",
        "/favicon.ico",
    }

    path = request.url.path
    if path in public_paths or path.startswith("/v1/files/"):
        return await call_next(request)

    raw = request.headers.get("x-api-key")
    if not raw:
        auth = request.headers.get("authorization")
        if auth and auth.lower().startswith("bearer "):
            raw = auth.split(" ", 1)[1].strip()

    if not raw:
        return JSONResponse({"detail": "Unauthorized"}, status_code=401)

    if settings.API_KEY and raw == settings.API_KEY:
        return await call_next(request)

    import hashlib

    key_hash = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    db = SessionLocal()
    try:
        row = db.query(ApiKey).filter(ApiKey.key_hash == key_hash).first()
        if not row or not row.is_active:
            return JSONResponse({"detail": "Unauthorized"}, status_code=401)
        try:
            from datetime import datetime
            row.last_used_at = datetime.utcnow()
            db.commit()
        except Exception:
            db.rollback()
    finally:
        db.close()

    return await call_next(request)


@app.get("/")
async def root():
    return {
        "name": "Inference Server",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health():
    """Basic health check - API is running"""
    return {"status": "healthy"}


@app.get("/health/detailed")
async def health_detailed():
    """Detailed health check with service status"""
    from app.db import engine
    import redis

    status = {
        "api": "healthy",
        "database": "unknown",
        "redis": "unknown",
        "gpu": "unknown"
    }

    # Check database
    try:
        from sqlalchemy import text
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        status["database"] = "healthy"
    except Exception as e:
        status["database"] = f"unavailable: {str(e)[:100]}"

    # Check Redis
    try:
        r = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            password=settings.REDIS_PASSWORD,
            db=settings.REDIS_DB
        )
        r.ping()
        status["redis"] = "healthy"
    except Exception as e:
        status["redis"] = f"unavailable: {str(e)[:100]}"

    # Check GPU
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "N/A"
            status["gpu"] = f"available: {gpu_count} GPU(s) - {gpu_name}"
        else:
            status["gpu"] = "unavailable: No CUDA devices found"
    except Exception as e:
        status["gpu"] = f"unavailable: {str(e)[:100]}"

    overall_status = "healthy" if status["database"] == "healthy" else "degraded"

    return {
        "status": overall_status,
        "services": status,
        "version": "1.0.0"
    }


@app.get("/v1/system/status")
async def system_status(principal=Depends(get_current_api_key)):
    from app.db import engine
    from app.models.model_loader import model_loader
    import redis

    status = {
        "database": "unknown",
        "redis": "unknown",
        "gpu": "unknown",
        "loaded_models": [],
    }

    try:
        from sqlalchemy import text
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        status["database"] = "healthy"
    except Exception as e:
        status["database"] = f"unavailable: {str(e)[:100]}"

    try:
        r = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            password=settings.REDIS_PASSWORD,
            db=settings.REDIS_DB
        )
        r.ping()
        status["redis"] = "healthy"
    except Exception as e:
        status["redis"] = f"unavailable: {str(e)[:100]}"

    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "N/A"
            status["gpu"] = f"available: {gpu_count} GPU(s) - {gpu_name}"
        else:
            status["gpu"] = "unavailable: No CUDA devices found"
    except Exception as e:
        status["gpu"] = f"unavailable: {str(e)[:100]}"

    try:
        status["loaded_models"] = list(model_loader.loaded_models.keys())
    except Exception:
        status["loaded_models"] = []

    overall_status = "healthy" if status["database"] == "healthy" else "degraded"
    return {
        "status": overall_status,
        "services": status,
        "build": {
            "git_sha": settings.BUILD_GIT_SHA,
            "image_tag": settings.BUILD_IMAGE_TAG,
        },
        "model_loader": {
            "max_loaded_models": settings.MAX_LOADED_MODELS,
            "idle_ttl_seconds": settings.MODEL_IDLE_TTL_SECONDS,
        },
    }


@app.get("/v1/system/db-info")
async def db_info(principal=Depends(get_current_api_key)):
    """Return non-sensitive DB connection info to confirm the active database."""
    url = make_url(settings.DATABASE_URL)
    return {
        "driver": url.drivername,
        "host": url.host,
        "port": url.port,
        "database": url.database,
    }


# Include API routes
app.include_router(api_router, prefix="/v1")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        workers=settings.WORKERS,
        reload=True
    )
