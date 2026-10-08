"""Клиент языковой модели через OpenRouter (https://openrouter.ai)."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Protocol

import httpx

from .config import Settings

log = logging.getLogger(__name__)

_RETRY_STATUSES = {408, 409, 425, 429, 500, 502, 503, 504, 520, 522, 524, 529}
_RETRY_DELAYS = (3, 10, 30)


class LLMError(RuntimeError):
    pass


@dataclass
class LLMResponse:
    text: str
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0


class LLMClient(Protocol):
    def complete(self, system_parts: list[str], user: str, max_tokens: int | None = None) -> LLMResponse: ...


class OpenRouterClient:
    def __init__(self, settings: Settings, http: httpx.Client | None = None):
        self.settings = settings
        self.http = http or httpx.Client(timeout=settings.llm_timeout_sec)

    def _body(self, system_parts: list[str], user: str, max_tokens: int | None) -> dict:
        system_content = [{"type": "text", "text": part} for part in system_parts]
        if self.settings.llm_prompt_cache and system_content:
            # Неизменная часть запроса (правила + чек-лист) кешируется у провайдера — повторные звонки дешевле.
            system_content[-1]["cache_control"] = {"type": "ephemeral"}
        body: dict = {
            "model": self.settings.llm_model,
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens or self.settings.llm_max_tokens,
            "usage": {"include": True},
        }
        if self.settings.llm_reasoning_effort:
            body["reasoning"] = {"effort": self.settings.llm_reasoning_effort}
        return body

    def complete(self, system_parts: list[str], user: str, max_tokens: int | None = None) -> LLMResponse:
        if not self.settings.openrouter_api_key:
            raise LLMError("Не задан OPENROUTER_API_KEY в файле .env")

        url = f"{self.settings.openrouter_base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.settings.openrouter_api_key}",
            "Content-Type": "application/json",
            "X-Title": "Call quality control",
        }
        body = self._body(system_parts, user, max_tokens)

        last_error = ""
        for attempt in range(len(_RETRY_DELAYS) + 1):
            try:
                resp = self.http.post(url, json=body, headers=headers)
            except httpx.TransportError as e:
                last_error = f"сетевая ошибка: {e}"
            else:
                if resp.status_code == 200:
                    return self._parse(resp)
                last_error = f"HTTP {resp.status_code}: {_error_text(resp)}"
                if resp.status_code not in _RETRY_STATUSES:
                    raise LLMError(f"OpenRouter отклонил запрос — {last_error}")
            if attempt < len(_RETRY_DELAYS):
                log.warning("OpenRouter: %s; повтор через %s с", last_error, _RETRY_DELAYS[attempt])
                time.sleep(_RETRY_DELAYS[attempt])
        raise LLMError(f"OpenRouter недоступен — {last_error}")

    @staticmethod
    def _parse(resp: httpx.Response) -> LLMResponse:
        data = resp.json()
        if data.get("error"):
            raise LLMError(f"OpenRouter вернул ошибку: {data['error'].get('message', data['error'])}")
        choices = data.get("choices") or []
        if not choices:
            raise LLMError("OpenRouter вернул пустой ответ")
        message = choices[0].get("message") or {}
        content = message.get("content") or ""
        if isinstance(content, list):
            content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        if choices[0].get("finish_reason") == "length":
            raise LLMError("Ответ модели обрезан по лимиту длины (увеличьте LLM_MAX_TOKENS)")
        usage = data.get("usage") or {}
        return LLMResponse(
            text=content,
            model=data.get("model", ""),
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            cost_usd=float(usage.get("cost") or 0.0),
        )


def _error_text(resp: httpx.Response) -> str:
    try:
        data = resp.json()
        err = data.get("error", data)
        if isinstance(err, dict):
            return str(err.get("message", err))
        return str(err)
    except ValueError:
        return resp.text[:300]
