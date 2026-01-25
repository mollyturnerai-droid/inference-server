from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from app.api import api_router
from app.db import engine, Base
from app.core.config import settings

# Create database tables (only if database is available)
try:
    Base.metadata.create_all(bind=engine)
except Exception as e:
    print(f"Warning: Could not create database tables: {e}")
    print("Database will be initialized when connection is available")

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)

# Create FastAPI app
app = FastAPI(
    title="Inference Server",
    description="A full-featured ML inference engine",
    version="1.0.0"
)

# Add rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
