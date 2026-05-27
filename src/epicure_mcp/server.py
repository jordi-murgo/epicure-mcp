"""MCP server bootstrap.

Exposes the Streamable HTTP MCP endpoint at ``/mcp`` plus a ``/healthz``
probe. The token-bucket rate limiter is applied to every non-health
request.
"""

from __future__ import annotations

import logging
from typing import Any

import uvicorn
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route

from .config import Config, load_config
from .ratelimit import RateLimitMiddleware
from .tools import register_all

log = logging.getLogger("epicure_mcp.server")


def _build_mcp(cfg: Config) -> FastMCP:
    # Public server: the host name varies (Azure Container Apps assigns a
    # generated FQDN) and clients may proxy through Claude.ai / Cursor /
    # ChatGPT. Disable DNS-rebinding protection - it defends against
    # browser-initiated attacks against a localhost server, which does not
    # apply when the server is publicly exposed.
    security = TransportSecuritySettings(enable_dns_rebinding_protection=False)
    server = FastMCP(
        name=cfg.server_name,
        instructions=(
            "Read-only access to the Epicure ingredient-embedding model. "
            "Resolve ingredient names with the deterministic matcher, then "
            "call tools to compute axis projections, pairings, factor poles, "
            "GMM mode placement, and atlas coordinates. Call list_targets() "
            "first to see valid axis and mode names for the morph tool."
        ),
        transport_security=security,
        host=cfg.host,
        port=cfg.port,
    )
    register_all(server)
    return server


async def _healthz(request: Request) -> JSONResponse:
    # Light health response; intentionally does NOT touch the bundle so
    # cold starts respond fast.
    return JSONResponse({"status": "ok"})


def build_app(cfg: Config | None = None) -> Starlette:
    cfg = cfg or load_config()
    mcp = _build_mcp(cfg)
    mcp_app = mcp.streamable_http_app()
    routes: list[Any] = [
        Route("/healthz", endpoint=_healthz, methods=["GET"]),
        Mount("/", app=mcp_app),
    ]

    app = Starlette(
        routes=routes,
        lifespan=mcp_app.router.lifespan_context,
    )
    app.add_middleware(
        RateLimitMiddleware,
        per_minute=cfg.rate_limit_per_minute,
        burst=cfg.rate_limit_burst,
    )
    return app


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )
    cfg = load_config()
    log.info("starting epicure-mcp on %s:%d (data_dir=%s)", cfg.host, cfg.port, cfg.data_dir)
    app = build_app(cfg)
    uvicorn.run(app, host=cfg.host, port=cfg.port, log_level="info")


if __name__ == "__main__":
    main()
