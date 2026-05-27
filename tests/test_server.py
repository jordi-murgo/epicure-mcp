"""End-to-end server test: spin up the ASGI app and invoke an MCP tool via
the Streamable HTTP transport."""

from __future__ import annotations

import json

import httpx
import pytest
from asgi_lifespan import LifespanManager

from epicure_mcp.server import build_app

pytestmark = pytest.mark.usefixtures("use_real_bundle")

JSONRPC_HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json, text/event-stream",
}


def _decode(text: str) -> dict:
    """Decode an SSE or plain-JSON MCP response."""
    if text.startswith("event:") or "\ndata:" in text or text.startswith("data:"):
        for line in text.splitlines():
            if line.startswith("data:"):
                return json.loads(line[5:].strip())
    return json.loads(text)


@pytest.mark.anyio
async def test_healthz_and_initialize() -> None:
    app = build_app()
    async with LifespanManager(app) as manager:
        transport = httpx.ASGITransport(app=manager.app)
        await _run_session(transport)


async def _run_session(transport: httpx.ASGITransport) -> None:
    async with httpx.AsyncClient(
        transport=transport, base_url="http://localhost", follow_redirects=True
    ) as client:
        health = await client.get("/healthz")
        assert health.status_code == 200
        assert health.json() == {"status": "ok"}

        # Favicon endpoints should serve the PNG bytes with the right
        # content-type so browser unfurlers / link previews work.
        favicon_ico = await client.get("/favicon.ico")
        assert favicon_ico.status_code == 200
        assert favicon_ico.headers["content-type"] == "image/png"
        assert favicon_ico.content[:8] == b"\x89PNG\r\n\x1a\n"  # PNG magic
        favicon_png = await client.get("/favicon.png")
        assert favicon_png.status_code == 200
        assert favicon_png.content == favicon_ico.content

        init_payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "pytest", "version": "0.1"},
            },
        }
        resp = await client.post("/mcp", json=init_payload, headers=JSONRPC_HEADERS)
        assert resp.status_code == 200
        body = _decode(resp.text)
        assert body["jsonrpc"] == "2.0"
        assert "result" in body
        # MCP server should advertise its icon in serverInfo.icons.
        server_info = body["result"].get("serverInfo") or {}
        icons = server_info.get("icons") or []
        assert icons, "initialize should advertise at least one icon"
        assert icons[0]["mimeType"] == "image/png"
        assert icons[0]["src"].startswith("data:image/png;base64,")
        session_id = resp.headers.get("mcp-session-id")
        assert session_id

        await client.post(
            "/mcp",
            json={"jsonrpc": "2.0", "method": "notifications/initialized"},
            headers={**JSONRPC_HEADERS, "mcp-session-id": session_id},
        )

        list_resp = await client.post(
            "/mcp",
            json={"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
            headers={**JSONRPC_HEADERS, "mcp-session-id": session_id},
        )
        assert list_resp.status_code == 200
        list_body = _decode(list_resp.text)
        tool_names = {t["name"] for t in list_body["result"]["tools"]}
        for required in (
            "compare_on_axis",
            "pairing_score",
            "find_pairings",
            "flavour_correlations",
            "cultural_profile",
            "neighbors",
            "morph",
            "list_targets",
            "list_factors",
            "ingredient_on_factor",
            "pareto_navigate",
            "closest_mode",
            "where_on_atlas",
        ):
            assert required in tool_names, f"missing tool: {required}"

        # The morph.target parameter must surface as a discriminated union
        # so MCP clients can validate their payload before invoking.
        morph_tool = next(
            t for t in list_body["result"]["tools"] if t["name"] == "morph"
        )
        target_schema = morph_tool["inputSchema"]["properties"]["target"]
        branches = target_schema.get("oneOf") or target_schema.get("anyOf")
        assert branches is not None, (
            f"morph.target should be a union; got: {target_schema}"
        )
        # Resolve $ref-style branches against the schema's $defs so we can
        # inspect their `kind` discriminator.
        defs = morph_tool["inputSchema"].get("$defs") or morph_tool[
            "inputSchema"
        ].get("definitions", {})

        def _resolve(branch: dict) -> dict:
            ref = branch.get("$ref")
            if not ref:
                return branch
            return defs.get(ref.rsplit("/", 1)[-1], branch)

        kinds: set[str] = set()
        for branch in branches:
            resolved = _resolve(branch)
            kind_prop = resolved.get("properties", {}).get("kind", {})
            const = kind_prop.get("const")
            enum = kind_prop.get("enum")
            if const:
                kinds.add(const)
            elif enum:
                kinds.update(enum)
        assert kinds == {"direction", "mode", "ingredient"}, (
            f"morph.target must discriminate on kind in {{'direction', 'mode', "
            f"'ingredient'}}; got {kinds}"
        )

        call_resp = await client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {
                    "name": "pairing_score",
                    "arguments": {"ingredient_a": "miso", "ingredient_b": "soy_sauce"},
                },
            },
            headers={**JSONRPC_HEADERS, "mcp-session-id": session_id},
        )
        assert call_resp.status_code == 200
        call_body = _decode(call_resp.text)
        result = call_body["result"]
        text_block = next(
            (b["text"] for b in result.get("content", []) if b.get("type") == "text"), None
        )
        assert text_block is not None
        payload = json.loads(text_block)
        assert payload["resolved_a"] == "miso"
        assert -1.0 <= payload["pairing_score"] <= 1.0


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"
