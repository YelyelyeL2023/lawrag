"""Access control for the HTTP API and console.

Two settings, both environment variables because they differ per deployment
and a token has no business in a YAML file under version control:

``LAWRAG_API_TOKEN``
    When set, every request except ``/health`` and the console's static files
    must carry ``Authorization: Bearer <token>``. This is what makes it safe to
    bind to anything other than 127.0.0.1.

``LAWRAG_ALLOW_REMOTE_WRITES``
    Without a token, requests that change the corpus (``POST /api/ingest``) are
    accepted only from a loopback client. Before this existed, binding to
    127.0.0.1 was the *only* thing protecting an endpoint that can drop the
    whole schema, and ``lawrag serve --host 0.0.0.0`` removed it silently.
    Docker publishes the port on the host's 127.0.0.1 but the request reaches
    the container from the bridge gateway, so compose sets this to 1 — the
    publish address is what protects that deployment.

Comparison is constant-time. Nothing here needs a new dependency.
"""

from __future__ import annotations

import hmac
import ipaddress
import os

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from ..logging_setup import get_logger

log = get_logger(__name__)

_OPEN_PATHS = ("/health",)
_OPEN_PREFIXES = ("/ui/",)
_WRITE_METHODS = {"POST", "PUT", "PATCH", "DELETE"}
# Writes that change the corpus. /ask and /search are POST but read-only.
_WRITE_PATHS = ("/api/ingest",)


def api_token() -> str:
    return os.environ.get("LAWRAG_API_TOKEN", "").strip()


def remote_writes_allowed() -> bool:
    return os.environ.get("LAWRAG_ALLOW_REMOTE_WRITES", "").strip().lower() in ("1", "true", "yes")


def _is_loopback(host: str | None) -> bool:
    if not host:
        return False
    if host in ("localhost", "testclient"):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _presented_token(request: Request) -> str:
    header = request.headers.get("authorization", "")
    if header.lower().startswith("bearer "):
        return header[7:].strip()
    return request.headers.get("x-lawrag-token", "").strip()


def describe() -> dict:
    """What /api/stats reports, so the console can say how it is protected."""
    return {
        "token_required": bool(api_token()),
        "remote_writes_allowed": remote_writes_allowed(),
    }


class AccessControl(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if path in _OPEN_PATHS or path == "/" or path.startswith(_OPEN_PREFIXES):
            return await call_next(request)

        token = api_token()
        if token:
            if not hmac.compare_digest(_presented_token(request).encode(), token.encode()):
                return JSONResponse(
                    {"detail": "authentication required: send Authorization: Bearer <LAWRAG_API_TOKEN>"},
                    status_code=401, headers={"WWW-Authenticate": "Bearer"})
            return await call_next(request)

        is_write = request.method in _WRITE_METHODS and path.startswith(_WRITE_PATHS)
        if is_write and not remote_writes_allowed():
            client = request.client.host if request.client else None
            if not _is_loopback(client):
                log.warning("refused %s %s from %s: no LAWRAG_API_TOKEN set", request.method,
                            path, client)
                return JSONResponse(
                    {"detail": ("imports are accepted only from this machine unless "
                                "LAWRAG_API_TOKEN is set on the server")},
                    status_code=403)
        return await call_next(request)


def warn_if_exposed(host: str) -> str | None:
    """A startup warning for ``lawrag serve`` on a non-loopback address."""
    if _is_loopback(host) or api_token():
        return None
    return (f"serving on {host} without LAWRAG_API_TOKEN: the console, the query log and "
            "the answering endpoints are readable by anyone who can reach this port. "
            "Imports are still refused from other machines.")
