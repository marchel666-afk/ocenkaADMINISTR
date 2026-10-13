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
# Модели Claude «размышляют» перед ответом, и размышления расходуют тот же лимит длины ответа.
# Не уложилась — запрос повторяется с вдвое большим лимитом, но не больше этого потолка.
MAX_TOKENS_CAP = 64000


class LLMError(RuntimeError):
    def __init__(self, message: str, spent: "LLMResponse | None" = None):
        super().__init__(message)
        self.spent = spent  # расход попыток, которые не дали ответа: он уже оплачен


@dataclass
class LLMResponse:
    text: str
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    truncated: bool = False  # ответ оборвался на лимите длины


class LLMClient(Protocol):
    def complete(
        self, system_parts: list[str], user: str, max_tokens: int | None = None, model: str | None = None
    ) -> LLMResponse: ...


class OpenRouterClient:
    def __init__(self, settings: Settings, http: httpx.Client | None = None):
        self.settings = settings
        # ждём долго только сам ответ (модель пишет его минутами); подключение не должно висеть столько же
        self.http = http or httpx.Client(timeout=httpx.Timeout(settings.llm_timeout_sec, connect=30))

    def _body(self, system_parts: list[str], user: str, max_tokens: int | None, model: str | None = None) -> dict:
        system_content = [{"type": "text", "text": part} for part in system_parts]
        if self.settings.llm_prompt_cache and system_content:
            # Неизменная часть запроса (правила + чек-лист) кешируется у провайдера — повторные звонки дешевле.
            system_content[-1]["cache_control"] = {"type": "ephemeral"}
        body: dict = {
            "model": model or self.settings.llm_model,
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

    def complete(
        self, system_parts: list[str], user: str, max_tokens: int | None = None, model: str | None = None
    ) -> LLMResponse:
        if not self.settings.openrouter_api_key:
            raise LLMError("Не задан OPENROUTER_API_KEY в файле .env")

        url = f"{self.settings.openrouter_base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.settings.openrouter_api_key}",
            "Content-Type": "application/json",
            "X-Title": "Call quality control",
        }
        limit = max_tokens or self.settings.llm_max_tokens
        spent = LLMResponse(text="")  # расход оборвавшихся попыток тоже оплачен
        while True:
            try:
                result = self._send(url, headers, self._body(system_parts, user, limit, model))
            except LLMError as e:
                if e.spent is None:
                    e.spent = spent
                raise
            spent.prompt_tokens += result.prompt_tokens
            spent.completion_tokens += result.completion_tokens
            spent.cost_usd += result.cost_usd
            if not result.truncated:
                result.prompt_tokens, result.completion_tokens = spent.prompt_tokens, spent.completion_tokens
                result.cost_usd = spent.cost_usd
                return result
            if limit >= MAX_TOKENS_CAP:
                raise LLMError(
                    f"Ответ модели обрезан: не уместился даже в {limit} токенов. Нажмите «Переоценить»; "
                    "если повторяется — сообщите разработчику",
                    spent=spent,
                )
            log.warning("Ответ модели не уместился в %s токенов — повтор с большим лимитом", limit)
            limit = min(limit * 2, MAX_TOKENS_CAP)

    def _send(self, url: str, headers: dict, body: dict) -> LLMResponse:
        last_error = ""
        for attempt in range(len(_RETRY_DELAYS) + 1):
            try:
                resp = self.http.post(url, json=body, headers=headers)
            except httpx.ReadTimeout:
                # модель долго пишет длинный ответ: повтор того же запроса снова занял бы очередь на столько же
                raise LLMError(
                    f"ИИ не ответил за {self.settings.llm_timeout_sec} с. Нажмите «Переоценить»; если повторяется — "
                    "увеличьте LLM_TIMEOUT_SEC в файле .env"
                ) from None
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
        usage = data.get("usage") or {}
        return LLMResponse(
            text=content,
            model=data.get("model", ""),
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            cost_usd=float(usage.get("cost") or 0.0),
            truncated=choices[0].get("finish_reason") == "length",
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
