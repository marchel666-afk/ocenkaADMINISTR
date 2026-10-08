import json

import httpx
import pytest

from app import llm as llm_module
from app.llm import LLMError, OpenRouterClient


def make_client(settings, handler):
    return OpenRouterClient(settings, http=httpx.Client(transport=httpx.MockTransport(handler)))


def ok_response(text='{"ok": true}'):
    return httpx.Response(
        200,
        json={
            "model": "anthropic/claude-haiku-5.5",
            "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1200, "completion_tokens": 300, "cost": 0.00027},
        },
    )


def test_request_body_and_parse(settings):
    seen = {}

    def handler(request: httpx.Request):
        seen["url"] = str(request.url)
        seen["auth"] = request.headers["authorization"]
        seen["body"] = json.loads(request.content)
        return ok_response()

    resp = make_client(settings, handler).complete(["правила", "чек-лист"], "расшифровка")
    body = seen["body"]
    assert seen["url"].endswith("/chat/completions")
    assert seen["auth"] == "Bearer test-key"
    assert body["model"] == settings.llm_model
    assert "temperature" not in body  # у новых моделей Claude нестандартная temperature запрещена
    assert body["messages"][0]["content"][-1]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in body["messages"][0]["content"][0]
    assert body["messages"][1] == {"role": "user", "content": "расшифровка"}
    assert body["usage"] == {"include": True}
    assert "reasoning" not in body
    assert resp.text == '{"ok": true}' and resp.cost_usd == pytest.approx(0.00027) and resp.prompt_tokens == 1200


def test_reasoning_option(settings):
    settings.llm_reasoning_effort = "low"
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return ok_response()

    make_client(settings, handler).complete(["a"], "b")
    assert seen["body"]["reasoning"] == {"effort": "low"}


def test_client_error_not_retried(settings):
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(400, json={"error": {"message": "model not found"}})

    with pytest.raises(LLMError, match="model not found"):
        make_client(settings, handler).complete(["a"], "b")
    assert len(calls) == 1


def test_retry_on_rate_limit(settings, monkeypatch):
    monkeypatch.setattr(llm_module.time, "sleep", lambda s: None)
    responses = iter([httpx.Response(429, json={"error": {"message": "slow down"}}), ok_response("готово")])
    resp = make_client(settings, lambda r: next(responses)).complete(["a"], "b")
    assert resp.text == "готово"


def test_truncated_answer(settings):
    def handler(request):
        return httpx.Response(200, json={"choices": [{"message": {"content": "{"}, "finish_reason": "length"}]})

    with pytest.raises(LLMError, match="обрезан"):
        make_client(settings, handler).complete(["a"], "b")


def test_missing_key(settings):
    settings.openrouter_api_key = ""
    with pytest.raises(LLMError, match="OPENROUTER_API_KEY"):
        make_client(settings, lambda r: ok_response()).complete(["a"], "b")


def test_model_override(settings):
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return ok_response()

    make_client(settings, handler).complete(["a"], "b", model="anthropic/claude-sonnet-5.5")
    assert seen["body"]["model"] == "anthropic/claude-sonnet-5.5"
