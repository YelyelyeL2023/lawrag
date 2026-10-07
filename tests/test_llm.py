"""Gemini generator client — request shape and key handling, no network."""

from __future__ import annotations

import json

import httpx
import pytest

from lawrag.config import LLMConfig
from lawrag.generate.llm import GeminiLLM, MissingApiKeyError, build_llm

API_KEY = "test-key"
MODEL = "gemini-test"
REPLY = '{"answer": "ok"}'


def test_gemini_backend_refuses_to_start_without_a_key():
    with pytest.raises(MissingApiKeyError):
        build_llm(LLMConfig(backend="gemini", api_key=""))


def test_gemini_sends_bearer_key_and_returns_message_content():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["auth"] = request.headers["Authorization"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": REPLY}}]})

    cfg = LLMConfig(backend="gemini", api_key=API_KEY, model=MODEL)
    llm = build_llm(cfg)
    assert isinstance(llm, GeminiLLM)
    llm._client = httpx.Client(base_url=cfg.base_url, transport=httpx.MockTransport(handler))
    llm._client.headers["Authorization"] = f"Bearer {API_KEY}"

    assert llm.complete("вопрос", max_tokens=50) == REPLY
    assert seen["path"].endswith("/openai/chat/completions")
    assert seen["auth"] == f"Bearer {API_KEY}"
    assert seen["body"]["model"] == MODEL
    assert seen["body"]["max_tokens"] == 50


def _client_replying(statuses: list[int]) -> tuple[GeminiLLM, list[int]]:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        status = statuses[len(calls)]
        calls.append(status)
        if status != 200:
            return httpx.Response(status, json={"error": {"status": "UNAVAILABLE"}})
        return httpx.Response(200, json={"choices": [{"message": {"content": REPLY}}]})

    cfg = LLMConfig(backend="gemini", api_key=API_KEY, max_retries=2, retry_backoff_s=0)
    llm = GeminiLLM(cfg)
    llm._client = httpx.Client(base_url=cfg.base_url, transport=httpx.MockTransport(handler))
    return llm, calls


def test_gemini_retries_an_overloaded_model_then_answers():
    llm, calls = _client_replying([503, 429, 200])
    assert llm.complete("вопрос") == REPLY
    assert calls == [503, 429, 200]


def test_gemini_gives_up_after_max_retries():
    llm, calls = _client_replying([503, 503, 503])
    with pytest.raises(httpx.HTTPStatusError):
        llm.complete("вопрос")
    assert len(calls) == 3


def test_gemini_does_not_retry_a_client_error():
    llm, calls = _client_replying([404])
    with pytest.raises(httpx.HTTPStatusError):
        llm.complete("вопрос")
    assert calls == [404]
