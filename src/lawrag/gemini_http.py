"""Shared HTTP plumbing for the hosted Gemini endpoints (generation and embeddings)."""

from __future__ import annotations

import time

import httpx

from .logging_setup import get_logger

log = get_logger(__name__)

RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


class MissingApiKeyError(ValueError):
    """A Gemini backend was selected without its API key."""


def require_api_key(api_key: str, setting: str) -> str:
    if not api_key:
        raise MissingApiKeyError(f"a gemini backend is selected but no API key is set; export {setting}")
    return api_key


def post_with_retry(client: httpx.Client, path: str, payload: dict, *,
                    max_retries: int, backoff_s: float) -> dict:
    """Retries overload (429/5xx) and timeouts with exponential backoff;
    any other failure is raised at once. Returns the decoded JSON body."""
    for attempt in range(max_retries + 1):
        last = attempt == max_retries
        try:
            resp = client.post(path, json=payload)
            if resp.status_code not in RETRYABLE_STATUS or last:
                resp.raise_for_status()
                return resp.json()
            reason = f"HTTP {resp.status_code}"
        except httpx.TimeoutException:
            if last:
                raise
            reason = "timeout"
        delay = backoff_s * 2 ** attempt
        log.warning("%s %s, retry %d/%d in %.0fs",
                    path, reason, attempt + 1, max_retries, delay)
        time.sleep(delay)
    raise AssertionError("unreachable")
