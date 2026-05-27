"""Per-tool-call analytics.

Emits one JSON line per ``tools/call`` invocation to stdout, where Azure
Container Apps forwards it to ``ContainerAppConsoleLogs_CL`` in the
Log Analytics workspace. Each record carries:

  - ts             ISO 8601 timestamp (millisecond precision, UTC)
  - ip_hash        SHA-256(salt || ip)[:16], salt rotates daily per replica
  - tool           the registered tool name
  - args           full input argument dict (Pydantic models materialised)
  - result_preview UTF-8 truncated to 4 KB
  - result_size_bytes  size before truncation
  - result_truncated   whether the preview was truncated
  - latency_ms     wall time of the wrapped call
  - ok             False if the call raised
  - error          "<ExceptionType>: <message>" when ok is False

The salt is generated fresh at process startup and rotates at UTC
midnight. This is intentional: IPs are never recoverable, and unique
counts are scoped to one replica's lifetime within one UTC day -- enough
for usage analytics, not enough to follow a single user across time.
"""

from __future__ import annotations

import asyncio
import contextvars
import datetime
import functools
import hashlib
import inspect
import json
import logging
import secrets
import threading
import time
from collections.abc import Callable
from typing import Any

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.types import ASGIApp

log = logging.getLogger("epicure_mcp.analytics")

_MAX_PREVIEW_BYTES = 4096

_current_ip: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "epicure_mcp_current_ip", default=None
)

_salt_lock = threading.Lock()
_salt: str = secrets.token_hex(16)
_salt_date: str = datetime.datetime.now(datetime.UTC).date().isoformat()


# ---------------------------------------------------------------------------
# IP capture
# ---------------------------------------------------------------------------


def _request_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


class ClientContextMiddleware(BaseHTTPMiddleware):
    """Capture the requester IP into a contextvar so tools can hash it.

    The contextvar is set for the duration of each request; tool calls
    invoked inside the same asyncio.Task inherit it transparently.
    """

    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)

    async def dispatch(self, request: Request, call_next):  # type: ignore[override]
        token = _current_ip.set(_request_ip(request))
        try:
            return await call_next(request)
        finally:
            _current_ip.reset(token)


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------


def _today_utc_iso() -> str:
    return datetime.datetime.now(datetime.UTC).date().isoformat()


def _current_salt() -> str:
    """Return today's salt, rotating at UTC midnight."""
    global _salt, _salt_date
    today = _today_utc_iso()
    with _salt_lock:
        if today != _salt_date:
            _salt = secrets.token_hex(16)
            _salt_date = today
        return _salt


def _hashed_ip(ip: str | None) -> str | None:
    if not ip:
        return None
    salt = _current_salt()
    return hashlib.sha256(f"{salt}:{ip}".encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------


def _coerce(value: Any) -> Any:
    """Make Pydantic models / oddly-typed values JSON-friendly."""
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return value


def _coerce_args(args: dict[str, Any]) -> dict[str, Any]:
    return {k: _coerce(v) for k, v in args.items()}


def _serialize_result(result: Any) -> tuple[str, int, bool]:
    if result is None:
        return "", 0, False
    if isinstance(result, str):
        body = result
    else:
        try:
            body = json.dumps(_coerce(result), default=str, separators=(",", ":"))
        except Exception:
            body = repr(result)
    raw_bytes = body.encode("utf-8")
    n = len(raw_bytes)
    if n > _MAX_PREVIEW_BYTES:
        truncated = raw_bytes[:_MAX_PREVIEW_BYTES].decode("utf-8", errors="ignore")
        return truncated + "... [truncated]", n, True
    return body, n, False


def _emit(record: dict[str, Any]) -> None:
    try:
        log.info(json.dumps(record, default=str, separators=(",", ":")))
    except Exception:  # pragma: no cover - logging must never raise
        pass


def _emit_record(
    tool_name: str,
    call_args: dict[str, Any],
    result: Any,
    started: float,
    ok: bool,
    error: str | None,
) -> None:
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    preview, size, truncated = _serialize_result(result) if ok else ("", 0, False)
    record = {
        "ts": datetime.datetime.now(datetime.UTC).isoformat(timespec="milliseconds"),
        "ip_hash": _hashed_ip(_current_ip.get()),
        "tool": tool_name,
        "args": _coerce_args(call_args),
        "result_preview": preview,
        "result_size_bytes": size,
        "result_truncated": truncated,
        "latency_ms": round(elapsed_ms, 2),
        "ok": ok,
        "error": error,
    }
    _emit(record)


# ---------------------------------------------------------------------------
# Decorator
# ---------------------------------------------------------------------------


def log_call(tool_name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorate a tool callable to emit a structured analytics line per call.

    Works for both sync and async callables. Preserves the original function
    signature via ``functools.wraps`` so FastMCP can still introspect it to
    derive the input JSON Schema.
    """

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        sig = inspect.signature(fn)
        is_async = asyncio.iscoroutinefunction(fn)

        if is_async:

            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                call_args = _bind_args(sig, args, kwargs)
                started = time.perf_counter()
                result: Any = None
                ok = True
                error: str | None = None
                try:
                    result = await fn(*args, **kwargs)
                    return result
                except Exception as e:
                    ok = False
                    error = f"{type(e).__name__}: {e}"
                    raise
                finally:
                    _emit_record(tool_name, call_args, result, started, ok, error)

            return async_wrapper

        @functools.wraps(fn)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            call_args = _bind_args(sig, args, kwargs)
            started = time.perf_counter()
            result: Any = None
            ok = True
            error: str | None = None
            try:
                result = fn(*args, **kwargs)
                return result
            except Exception as e:
                ok = False
                error = f"{type(e).__name__}: {e}"
                raise
            finally:
                _emit_record(tool_name, call_args, result, started, ok, error)

        return sync_wrapper

    return decorator


def _bind_args(sig: inspect.Signature, args: tuple, kwargs: dict) -> dict[str, Any]:
    try:
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        return dict(bound.arguments)
    except TypeError:
        # Best-effort fall-back if the call wasn't bindable (shouldn't
        # happen for FastMCP-validated tool invocations).
        return {"args": list(args), "kwargs": kwargs}


# ---------------------------------------------------------------------------
# Test hooks
# ---------------------------------------------------------------------------


def _force_salt_for_testing(salt: str, date_iso: str) -> None:
    """Override the rotating salt. Tests only."""
    global _salt, _salt_date
    with _salt_lock:
        _salt = salt
        _salt_date = date_iso


def _set_current_ip_for_testing(ip: str | None) -> contextvars.Token:
    """Set the IP contextvar without going through middleware. Tests only."""
    return _current_ip.set(ip)


def _reset_current_ip_for_testing(token: contextvars.Token) -> None:
    _current_ip.reset(token)
