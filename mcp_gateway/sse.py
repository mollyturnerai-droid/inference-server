from __future__ import annotations

import os
from mcp.server.fastmcp import FastMCP
from mcp.server.sse import SseServerTransport
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route
from urllib.parse import parse_qs


def _require_api_key(app, expected_key: str):
    async def asgi(scope, receive, send):
        if scope["type"] != "http":
            await app(scope, receive, send)
            return

        headers = {k.decode().lower(): v.decode() for k, v in scope.get("headers", [])}
        key = headers.get("x-api-key")
        if not key:
            qs = parse_qs((scope.get("query_string") or b"").decode())
            key = (qs.get("api_key") or [None])[0]

        if not key or key != expected_key:
            res = JSONResponse({"detail": "Unauthorized"}, status_code=401)
            await res(scope, receive, send)
            return

        await app(scope, receive, send)

    return asgi


def create_sse_server(mcp: FastMCP) -> Starlette:
    transport = SseServerTransport("/messages/")
    expected_key = os.getenv("API_KEY")

    async def handle_sse(request):
        async with transport.connect_sse(
            request.scope, request.receive, request._send
        ) as streams:
            await mcp._mcp_server.run(
                streams[0],
                streams[1],
                mcp._mcp_server.create_initialization_options(),
            )

    authed_messages_app = transport.handle_post_message
    if expected_key:
        authed_messages_app = _require_api_key(authed_messages_app, expected_key)

    if expected_key:
        async def handle_sse_authed(request):
            key = request.headers.get("x-api-key") or request.query_params.get("api_key")
            if not key or key != expected_key:
                return JSONResponse({"detail": "Unauthorized"}, status_code=401)
            return await handle_sse(request)
        sse_endpoint = handle_sse_authed
    else:
        sse_endpoint = handle_sse

    routes = [
        Route("/sse/", endpoint=sse_endpoint),
        Mount("/messages/", app=authed_messages_app),
    ]
    return Starlette(routes=routes)
