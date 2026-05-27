"""Tests for src/epicure_mcp/analytics.py."""

from __future__ import annotations

import json
import logging
from typing import Any

import pytest

from epicure_mcp import analytics


@pytest.fixture
def capture_analytics(caplog: pytest.LogCaptureFixture):
    """Capture INFO-level records from the analytics logger."""
    caplog.set_level(logging.INFO, logger="epicure_mcp.analytics")
    yield caplog


def _records(caplog: pytest.LogCaptureFixture) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for rec in caplog.records:
        if rec.name != "epicure_mcp.analytics":
            continue
        try:
            out.append(json.loads(rec.getMessage()))
        except json.JSONDecodeError:
            continue
    return out


# --- log_call decorator ----------------------------------------------------


def test_log_call_sync_happy_path(capture_analytics) -> None:
    @analytics.log_call("widget")
    def doit(a: int, b: int = 2) -> dict[str, int]:
        return {"sum": a + b}

    result = doit(1)
    assert result == {"sum": 3}

    records = _records(capture_analytics)
    assert len(records) == 1
    r = records[0]
    assert r["tool"] == "widget"
    assert r["args"] == {"a": 1, "b": 2}
    assert r["ok"] is True
    assert r["error"] is None
    assert r["result_truncated"] is False
    assert r["result_size_bytes"] == len(b'{"sum":3}')
    assert r["latency_ms"] >= 0
    # Schema invariants:
    for key in (
        "ts",
        "ip_hash",
        "tool",
        "args",
        "result_preview",
        "result_size_bytes",
        "result_truncated",
        "latency_ms",
        "ok",
        "error",
    ):
        assert key in r


def test_log_call_records_error_and_reraises(capture_analytics) -> None:
    @analytics.log_call("boom")
    def explode() -> None:
        raise ValueError("nope")

    with pytest.raises(ValueError, match="nope"):
        explode()

    records = _records(capture_analytics)
    assert len(records) == 1
    r = records[0]
    assert r["tool"] == "boom"
    assert r["ok"] is False
    assert r["error"] == "ValueError: nope"
    assert r["result_size_bytes"] == 0


def test_log_call_truncates_large_results(capture_analytics) -> None:
    big = "x" * (analytics._MAX_PREVIEW_BYTES * 3)

    @analytics.log_call("big")
    def producer() -> str:
        return big

    producer()
    r = _records(capture_analytics)[0]
    assert r["result_truncated"] is True
    assert r["result_size_bytes"] == len(big)
    assert "[truncated]" in r["result_preview"]
    # The preview is bounded by MAX_PREVIEW_BYTES + the truncation marker.
    assert len(r["result_preview"].encode("utf-8")) <= (
        analytics._MAX_PREVIEW_BYTES + len("... [truncated]") + 4
    )


def test_log_call_pydantic_args_serialise(capture_analytics) -> None:
    from epicure_mcp.tools.morph_types import DirectionTarget

    @analytics.log_call("morph_like")
    def f(target: Any) -> str:
        return "ok"

    f(target=DirectionTarget(kind="direction", name="cuisine:Japanese"))
    r = _records(capture_analytics)[0]
    assert r["args"]["target"] == {"kind": "direction", "name": "cuisine:Japanese"}


def test_log_call_async(capture_analytics) -> None:
    import asyncio

    @analytics.log_call("async_widget")
    async def doit(name: str) -> dict[str, str]:
        await asyncio.sleep(0.01)
        return {"hello": name}

    result = asyncio.run(doit("world"))
    assert result == {"hello": "world"}
    r = _records(capture_analytics)[0]
    assert r["tool"] == "async_widget"
    assert r["args"] == {"name": "world"}
    assert r["ok"] is True
    assert r["latency_ms"] >= 10


# --- IP hashing ------------------------------------------------------------


def test_hashed_ip_is_deterministic_within_one_day() -> None:
    analytics._force_salt_for_testing("fixed_salt", "2026-01-01")
    h1 = analytics._hashed_ip("1.2.3.4")
    h2 = analytics._hashed_ip("1.2.3.4")
    assert h1 == h2
    assert h1 != analytics._hashed_ip("1.2.3.5")
    assert h1 and len(h1) == 16  # 16 hex chars (64 bits)


def test_hashed_ip_returns_none_for_missing_ip() -> None:
    assert analytics._hashed_ip(None) is None
    assert analytics._hashed_ip("") is None


def test_salt_rotates_when_date_changes(monkeypatch) -> None:
    analytics._force_salt_for_testing("salt_day_1", "2026-01-01")
    h1 = analytics._hashed_ip("1.2.3.4")

    # Move clock to next day; _current_salt rotates lazily on next call.
    from datetime import UTC, datetime

    class _FakeDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 1, 2, 12, 0, 0, tzinfo=UTC)

    monkeypatch.setattr(analytics.datetime, "datetime", _FakeDatetime)
    h2 = analytics._hashed_ip("1.2.3.4")
    assert h1 != h2  # salt rotated, so hash differs


def test_log_call_uses_current_ip(capture_analytics) -> None:
    analytics._force_salt_for_testing("fixed_salt", "2026-01-01")

    @analytics.log_call("with_ip")
    def f() -> str:
        return "ok"

    token = analytics._set_current_ip_for_testing("9.9.9.9")
    try:
        f()
    finally:
        analytics._reset_current_ip_for_testing(token)

    r = _records(capture_analytics)[0]
    assert r["ip_hash"] is not None
    assert len(r["ip_hash"]) == 16
