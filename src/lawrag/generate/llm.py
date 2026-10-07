"""Generation backends.

``gemini`` calls the hosted Gemini API through its OpenAI-compatible endpoint.
This is the one outbound network call on the query path: the question and the
retrieved statutory text leave the machine.

``echo`` is a stub that returns a well-formed refusal. It exists so the pipeline
is runnable and testable with no weights installed; it never pretends to have
answered.
"""

from __future__ import annotations

import json
import time
from typing import Protocol

from ..config import LLMConfig
from ..logging_setup import get_logger

log = get_logger(__name__)

RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


class LLM(Protocol):
    def complete(self, prompt: str, *, max_tokens: int = 1024,
                 model: str | None = None, temperature: float | None = None) -> str: ...


class EchoLLM:
    """No model installed. Returns a valid abstention rather than a fake answer."""

    name = "echo"

    def __init__(self, cfg: LLMConfig) -> None:
        self.cfg = cfg
        log.warning(
            "LLM backend is 'echo' — no model is loaded and no answer will be "
            "generated. Retrieval is fully functional; set llm.backend=gemini "
            "and LAWRAG_LLM_API_KEY to enable answers."
        )

    def complete(self, prompt: str, *, max_tokens: int = 1024,
                 model: str | None = None, temperature: float | None = None) -> str:
        return json.dumps(
            {
                "answer": "",
                "citations": [],
                "confidence": "low",
                "abstain": True,
                "abstain_reason": "no_model_loaded",
                "escalate": True,
            },
            ensure_ascii=False,
        )


class MissingApiKeyError(ValueError):
    """The gemini backend was selected without LAWRAG_LLM_API_KEY."""


class GeminiLLM:
    """Gemini via its OpenAI-compatible chat completions endpoint."""

    name = "gemini"

    def __init__(self, cfg: LLMConfig) -> None:
        import httpx

        if not cfg.api_key:
            raise MissingApiKeyError(
                "llm.backend is 'gemini' but no API key is set; "
                "export LAWRAG_LLM_API_KEY")
        self.cfg = cfg
        self._client = httpx.Client(
            base_url=cfg.base_url.rstrip("/"),
            timeout=cfg.timeout_s,
            headers={"Authorization": f"Bearer {cfg.api_key}"},
        )

    def complete(self, prompt: str, *, max_tokens: int = 1024,
                 model: str | None = None, temperature: float | None = None) -> str:
        payload = {
            "model": model or self.cfg.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.cfg.temperature if temperature is None else temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        data = self._post_with_retry("/chat/completions", payload)
        return data["choices"][0]["message"]["content"]

    def _post_with_retry(self, path: str, payload: dict) -> dict:
        import httpx

        for attempt in range(self.cfg.max_retries + 1):
            last = attempt == self.cfg.max_retries
            try:
                resp = self._client.post(path, json=payload)
                if resp.status_code not in RETRYABLE_STATUS or last:
                    resp.raise_for_status()
                    return resp.json()
                reason = f"HTTP {resp.status_code}"
            except httpx.TimeoutException:
                if last:
                    raise
                reason = "timeout"
            delay = self.cfg.retry_backoff_s * 2 ** attempt
            log.warning("gemini %s, retry %d/%d in %.0fs",
                        reason, attempt + 1, self.cfg.max_retries, delay)
            time.sleep(delay)
        raise AssertionError("unreachable")


def build_llm(cfg: LLMConfig) -> LLM:
    backend = cfg.backend.lower()
    if backend in ("echo", "none", ""):
        return EchoLLM(cfg)
    if backend == GeminiLLM.name:
        return GeminiLLM(cfg)
    raise ValueError(f"unknown llm backend: {cfg.backend}")
