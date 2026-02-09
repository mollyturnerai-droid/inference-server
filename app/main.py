from fastapi import FastAPI, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from app.api import api_router
from app.db import engine, Base, SessionLocal, ApiKey
from app.core.config import settings
from app.services.auth import get_current_api_key, extract_api_key, authenticate_api_key
from app.services.recon import start_recon_scheduler
from sqlalchemy.engine.url import make_url
import httpx

# Create database tables (only if database is available)
try:
    Base.metadata.create_all(bind=engine)
    # Ensure new prediction progress columns exist (for existing DBs)
    from sqlalchemy import text
    with engine.connect() as conn:
        conn.execute(text("ALTER TABLE predictions ADD COLUMN IF NOT EXISTS progress DOUBLE PRECISION"))
        conn.execute(text("ALTER TABLE predictions ADD COLUMN IF NOT EXISTS progress_step INTEGER"))
        conn.execute(text("ALTER TABLE predictions ADD COLUMN IF NOT EXISTS progress_total INTEGER"))
        conn.execute(text("ALTER TABLE catalog_models ADD COLUMN IF NOT EXISTS prediction_count INTEGER DEFAULT 0"))
        conn.commit()
except Exception as e:
    print(f"Warning: Could not create database tables: {e}")
    print("Database will be initialized when connection is available")


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


@app.on_event("startup")
def _start_recon():
    start_recon_scheduler()

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

    raw = extract_api_key(request)
    if not raw:
        return JSONResponse({"detail": "Unauthorized"}, status_code=401)

    db = SessionLocal()
    try:
        principal = authenticate_api_key(raw, db)
        if principal is None:
            return JSONResponse({"detail": "Unauthorized"}, status_code=401)
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
    return {
        "status": "healthy",
        "build": {
            "git_sha": settings.BUILD_GIT_SHA,
            "image_tag": settings.BUILD_IMAGE_TAG,
        },
    }


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
        "version": "1.0.0",
        "build": {
            "git_sha": settings.BUILD_GIT_SHA,
            "image_tag": settings.BUILD_IMAGE_TAG,
        },
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


@app.get("/v1/system/hf-status")
async def hf_status(principal=Depends(get_current_api_key)):
    """Return whether HF_API_TOKEN is set and which account it maps to."""
    token = (settings.HF_API_TOKEN or "").strip()
    if not token:
        return {"token_present": False, "authenticated": False}

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                "https://huggingface.co/api/whoami-v2",
                headers={"Authorization": f"Bearer {token}"},
            )
        if resp.status_code != 200:
            return {
                "token_present": True,
                "authenticated": False,
                "status_code": resp.status_code,
            }
        data = resp.json()
        return {
            "token_present": True,
            "authenticated": True,
            "username": data.get("name") or data.get("user"),
        }
    except Exception as exc:
        return {
            "token_present": True,
            "authenticated": False,
            "error": str(exc),
        }


# Include API routes
app.include_router(api_router, prefix="/v1")

# Optional MCP server mounted under /mcp (SSE transport).
# This allows IDEs/agents to interact with the server via MCP without exposing a separate gateway port.
if settings.ENABLE_MCP:
    try:
        import base64
        from typing import Any, Dict, Optional

        from mcp.server.fastmcp import FastMCP
        from mcp_gateway.sse import create_sse_server

        _MCP_BASE_URL = f"http://127.0.0.1:{settings.API_PORT}".rstrip("/")
        _MCP_TIMEOUT_S = float(getattr(settings, "DEFAULT_TIMEOUT", 300) or 300)

        mcp = FastMCP("InferenceServer")
        app.mount("/mcp", create_sse_server(mcp))

        def _coerce_json(value: Any) -> Any:
            if value is None:
                return None
            if isinstance(value, (dict, list, str, int, float, bool)):
                return value
            import json as _json
            return _json.loads(_json.dumps(value, default=str))

        @mcp.tool()
        async def inference_api_request(
            authorization: str,
            method: str,
            path: str,
            query: Optional[Dict[str, Any]] = None,
            json_body: Optional[Any] = None,
            headers: Optional[Dict[str, str]] = None,
        ) -> Dict[str, Any]:
            """Proxy a request to the local Inference Server API.

            Provide `authorization` as a full header value (e.g. "Bearer <api_key>").
            """
            if not path.startswith("/"):
                path = "/" + path
            url = f"{_MCP_BASE_URL}{path}"

            h = {"Authorization": authorization}
            if headers:
                h.update({str(k): str(v) for k, v in headers.items()})

            timeout = httpx.Timeout(_MCP_TIMEOUT_S)
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.request(
                    method.upper(),
                    url,
                    params=query,
                    headers=h,
                    json=json_body,
                )

            content_type = resp.headers.get("content-type", "")
            out: Dict[str, Any] = {
                "status_code": resp.status_code,
                "headers": {k: v for k, v in resp.headers.items() if k.lower() in ("content-type",)},
            }
            if "application/json" in content_type:
                try:
                    out["json"] = resp.json()
                except Exception:
                    out["text"] = resp.text
            else:
                out["text"] = resp.text
            return _coerce_json(out)

        @mcp.tool()
        async def files_upload_base64(
            authorization: str,
            filename: str,
            content_base64: str,
            content_type: str = "application/octet-stream",
        ) -> Dict[str, Any]:
            """Upload a file to /v1/files/upload using base64 content."""
            raw = base64.b64decode(content_base64)
            files = {"file": (filename, raw, content_type)}

            url = f"{_MCP_BASE_URL}/v1/files/upload"
            timeout = httpx.Timeout(_MCP_TIMEOUT_S)
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(url, headers={"Authorization": authorization}, files=files)
            return _coerce_json({"status_code": resp.status_code, "text": resp.text})

    except Exception as _exc:
        # Don't fail the API if MCP isn't available in a given environment.
        print(f"Warning: MCP server disabled due to import/runtime error: {_exc}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        workers=settings.WORKERS,
        reload=True
    )
