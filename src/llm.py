"""RAG 답변 생성용 LLM 클라이언트 (JSON 스키마 강제 출력).

    llm = llm_from_env()
    data = llm.complete_json(system, user, schema)   # → dict (스키마를 따르는 JSON)

제공자 선택 (.env의 LLM_PROVIDER, 비우면 자동):
- anthropic  Claude (ANTHROPIC_API_KEY). 모델 ANTHROPIC_MODEL, 기본 claude-opus-5-5
- upstage    Solar (UPSTAGE_API_KEY, 임베딩과 같은 키). 모델 UPSTAGE_CHAT_MODEL, 기본 solar-pro3
자동 선택: ANTHROPIC_API_KEY가 있으면 anthropic, 없으면 upstage.
"""

from __future__ import annotations

import json
import os
import time
from typing import Protocol

import requests
from dotenv import load_dotenv

from src.debug import dblock

DEFAULT_ANTHROPIC_MODEL = "claude-opus-5-5"
DEFAULT_UPSTAGE_MODEL = "solar-pro3"
UPSTAGE_CHAT_URL = "https://api.upstage.ai/v1/chat/completions"


class LLMError(RuntimeError):
    pass


class LLM(Protocol):
    name: str  # 'anthropic:claude-opus-5-5' 등 (응답·로그 표시용)

    def complete_json(self, system: str, user: str, schema: dict) -> dict: ...


def _log_response(name: str, text: str, finish: object) -> None:
    """응답 원문 전문을 파싱 전에 출력한다 (파싱 실패·잘림·거절 때도 모델이 실제로 보낸 내용이 남도록)."""
    dblock(f"LLM 응답 원문 · {name} · 종료 사유 {finish}", text)


def _parse_json(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise LLMError(f"JSON 파싱 실패: {e}: {text[:200]}") from e


class AnthropicLLM:
    def __init__(self, model: str = DEFAULT_ANTHROPIC_MODEL, effort: str = "medium"):
        import anthropic

        self._anthropic = anthropic
        self.client = anthropic.Anthropic()
        self.model = model
        self.effort = effort
        self.name = f"anthropic:{model}"

    def complete_json(self, system: str, user: str, schema: dict) -> dict:
        anthropic = self._anthropic
        try:
            # fallbacks="default": 안전 분류기가 거절하면 서버가 권장 모델로 다시 실행한다
            response = self.client.beta.messages.create(
                model=self.model,
                max_tokens=16000,
                system=system,
                messages=[{"role": "user", "content": user}],
                output_config={"effort": self.effort, "format": {"type": "json_schema", "schema": schema}},
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
        except anthropic.RateLimitError as e:
            raise LLMError(f"요청 한도 초과(429): {e.message}") from e
        except anthropic.APIStatusError as e:
            raise LLMError(f"Claude API 오류({e.status_code}): {e.message}") from e
        except anthropic.APIConnectionError as e:
            raise LLMError(f"Claude API 연결 실패: {e}") from e

        text = next((b.text for b in response.content if b.type == "text"), "")
        _log_response(f"{self.name} (응답 모델 {response.model})", text, response.stop_reason)
        if response.stop_reason == "refusal":
            raise LLMError("모델이 요청을 거절했습니다(refusal).")
        if response.stop_reason == "max_tokens":
            raise LLMError("출력 길이 한도에 걸려 답변이 잘렸습니다.")
        return _parse_json(text)


class UpstageLLM:
    def __init__(self, api_key: str, model: str = DEFAULT_UPSTAGE_MODEL, timeout: float = 120.0):
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.name = f"upstage:{model}"

    def complete_json(self, system: str, user: str, schema: dict) -> dict:
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "response_format": {"type": "json_schema", "json_schema": {"name": "grounded_answer", "strict": True, "schema": schema}},
            "temperature": 0,
        }
        error = ""
        for attempt in range(3):
            try:
                resp = requests.post(UPSTAGE_CHAT_URL, headers={"Authorization": f"Bearer {self.api_key}"},
                                     json=body, timeout=self.timeout)
            except requests.RequestException as e:
                error = str(e)
            else:
                if resp.ok:
                    choice = resp.json()["choices"][0]
                    _log_response(self.name, choice["message"]["content"] or "", choice.get("finish_reason"))
                    if choice.get("finish_reason") == "length":
                        raise LLMError("출력 길이 한도에 걸려 답변이 잘렸습니다.")
                    return _parse_json(choice["message"]["content"])
                error = f"HTTP {resp.status_code}: {resp.text[:300]}"
                if resp.status_code not in (429, 500, 502, 503, 504):
                    break
            time.sleep(2 ** attempt)
        raise LLMError(f"Upstage API 오류: {error}")


def llm_from_env() -> LLM:
    load_dotenv()
    provider = os.getenv("LLM_PROVIDER", "").strip().lower()
    if not provider:
        provider = "anthropic" if os.getenv("ANTHROPIC_API_KEY", "").strip() else "upstage"
    if provider == "anthropic":
        return AnthropicLLM(os.getenv("ANTHROPIC_MODEL", "").strip() or DEFAULT_ANTHROPIC_MODEL)
    if provider == "upstage":
        key = (os.getenv("UPSTAGE_API_KEY") or os.getenv("LLM_API_KEY") or "").strip()
        if not key:
            raise LLMError(".env에 UPSTAGE_API_KEY(또는 ANTHROPIC_API_KEY)가 설정되어 있지 않습니다.")
        return UpstageLLM(key, os.getenv("UPSTAGE_CHAT_MODEL", "").strip() or DEFAULT_UPSTAGE_MODEL)
    raise LLMError(f"알 수 없는 LLM_PROVIDER: {provider} (anthropic | upstage)")
