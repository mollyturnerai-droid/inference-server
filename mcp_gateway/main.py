from __future__ import annotations

import base64
import json
import os
from typing import Any, Dict, Optional

import httpx
from fastapi import FastAPI

from mcp.server.fastmcp import FastMCP

from mcp_gateway.sse import create_sse_server


INFERENCE_BASE_URL = os.getenv("INFERENCE_BASE_URL", "http://api:8000").rstrip("/")
CATALOG_ADMIN_TOKEN = os.getenv("CATALOG_ADMIN_TOKEN")
DEFAULT_TIMEOUT_SECONDS = float(os.getenv("MCP_GATEWAY_TIMEOUT_SECONDS", "120"))


app = FastAPI()
mcp = FastMCP("InferenceServer")


def _coerce_json(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (dict, list, str, int, float, bool)):
        return value
    return json.loads(json.dumps(value, default=str))


async def _forward_request(
    authorization: str,
    method: str,
    path: str,
    *,
    query: Optional[Dict[str, Any]] = None,
    json_body: Optional[Any] = None,
    headers: Optional[Dict[str, str]] = None,
    files: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if not path.startswith("/"):
        path = "/" + path

    url = f"{INFERENCE_BASE_URL}{path}"

    h = {"Authorization": authorization}
    if headers:
        h.update({str(k): str(v) for k, v in headers.items()})

    # If calling catalog admin endpoints, inject admin token held by gateway.
    if path.startswith("/v1/catalog/admin/") and CATALOG_ADMIN_TOKEN:
        h.setdefault("X-Catalog-Admin-Token", CATALOG_ADMIN_TOKEN)

    timeout = httpx.Timeout(DEFAULT_TIMEOUT_SECONDS)

    async with httpx.AsyncClient(timeout=timeout) as client:
        if files is not None:
            # For multipart, httpx expects {name: (filename, bytes, content_type)} or {name: bytes}
            resp = await client.request(method.upper(), url, params=query, headers=h, files=files)
        else:
            resp = await client.request(method.upper(), url, params=query, headers=h, json=json_body)

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


@app.get("/health")
def health():
    return {"status": "ok"}


# Mount MCP SSE server under /mcp
app.mount("/mcp", create_sse_server(mcp))


@mcp.tool()
async def inference_api_request(
    authorization: str,
    method: str,
    path: str,
    query: Optional[Dict[str, Any]] = None,
    json_body: Optional[Any] = None,
    headers: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Proxy a request to the Inference Server API.

    Parameters:
    - authorization: Full Authorization header value, e.g. "Bearer <token>"
    - method: HTTP method, e.g. GET/POST
    - path: API path, e.g. /v1/models
    - query: Optional query params dict
    - json_body: Optional JSON body
    - headers: Optional extra headers (gateway may inject catalog admin token for /v1/catalog/admin/*)
    """

    return await _forward_request(
        authorization,
        method,
        path,
        query=query,
        json_body=json_body,
        headers=headers,
    )


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

    return await _forward_request(
        authorization,
        "POST",
        "/v1/files/upload",
        files=files,
    )
