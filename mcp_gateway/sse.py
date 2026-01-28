from __future__ import annotations

from mcp.server.fastmcp import FastMCP
from mcp.server.sse import SseServerTransport
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route


def _require_auth_header(app):
    async def asgi(scope, receive, send):
        if scope["type"] != "http":
            await app(scope, receive, send)
            return

        headers = {k.decode().lower(): v.decode() for k, v in scope.get("headers", [])}
        if not headers.get("authorization"):
            res = JSONResponse({"detail": "Missing Authorization header"}, status_code=401)
            await res(scope, receive, send)
            return

        await app(scope, receive, send)

    return asgi


def create_sse_server(mcp: FastMCP) -> Starlette:
    transport = SseServerTransport("/messages/")

    async def handle_sse(request):
        async with transport.connect_sse(
            request.scope, request.receive, request._send
        ) as streams:
            await mcp._mcp_server.run(
                streams[0],
                streams[1],
                mcp._mcp_server.create_initialization_options(),
            )

    authed_messages_app = _require_auth_header(transport.handle_post_message)

    routes = [
        Route("/sse/", endpoint=handle_sse),
        Mount("/messages/", app=authed_messages_app),
    ]
    return Starlette(routes=routes)
