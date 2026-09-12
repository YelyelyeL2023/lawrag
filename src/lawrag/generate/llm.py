"""Local generation backends.

``llamacpp`` talks to a ``llama-server`` process over its OpenAI-compatible
endpoint. llama.cpp rather than vLLM is deliberate for an 8 GB card and
single-user traffic: vLLM's advantage is batched multi-tenant throughput, which
is not the workload, and llama.cpp exposes the knobs that matter here — MoE
expert offload (``--n-cpu-moe``), KV-cache quantisation, and GGUF quant choice.

``echo`` is a stub that returns a well-formed refusal. It exists so the pipeline
is runnable and testable with no weights installed; it never pretends to have
answered.
"""

from __future__ import annotations

import json
from typing import Protocol

from ..config import LLMConfig
from ..logging_setup import get_logger

log = get_logger(__name__)


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
            "generated. Retrieval is fully functional; set llm.backend=llamacpp "
            "and point base_url at a running llama-server to enable answers."
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


class LlamaCppLLM:
    """OpenAI-compatible client for a local llama-server."""

    name = "llamacpp"

    def __init__(self, cfg: LLMConfig) -> None:
        import httpx

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
        resp = self._client.post("/chat/completions", json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    def health(self) -> bool:
        try:
            r = self._client.get("/models")
            return r.status_code == 200
        except Exception:  # noqa: BLE001
            return False


def build_llm(cfg: LLMConfig) -> LLM:
    backend = cfg.backend.lower()
    if backend in ("echo", "none", ""):
        return EchoLLM(cfg)
    if backend in ("llamacpp", "llama.cpp", "openai", "local"):
        return LlamaCppLLM(cfg)
    raise ValueError(f"unknown llm backend: {cfg.backend}")
