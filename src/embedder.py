"""Upstage Solar Embedding 2 클라이언트.

- 모델: solar-embedding-2-passage(문서) / solar-embedding-2-query(질의), 1024차원, 정규화된 벡터
- 한도: 요청당 100건·204,800토큰, 분당 100요청·300,000토큰
- 가격: 100만 토큰당 $0.02 (2026-10, VAT 별도)
"""

from __future__ import annotations

import os
import time
from collections import deque
from collections.abc import Iterator

import requests
from dotenv import load_dotenv

API_URL = "https://api.upstage.ai/v1/embeddings"
PASSAGE_MODEL = "solar-embedding-2-passage"
QUERY_MODEL = "solar-embedding-2-query"
DIMENSION = 1024
PRICE_PER_M_TOKENS = 0.02

MAX_BATCH_SIZE = 100
MAX_BATCH_TOKENS = 100_000  # 요청 한도(204,800)의 절반으로 여유 있게
RPM_LIMIT = 90  # 한도(100)보다 낮게
TPM_LIMIT = 250_000  # 한도(300,000)보다 낮게
MAX_RETRIES = 5


class EmbeddingError(RuntimeError):
    pass


class UpstageEmbedder:
    def __init__(self, api_key: str, timeout: float = 60.0):
        self.api_key = api_key
        self.timeout = timeout
        self._window: deque[tuple[float, int]] = deque()  # (요청 시각, 사용 토큰)
        self._chars = 0
        self.total_tokens = 0

    @classmethod
    def from_env(cls) -> "UpstageEmbedder":
        load_dotenv()
        key = os.getenv("UPSTAGE_API_KEY") or os.getenv("LLM_API_KEY")
        if not key:
            raise EmbeddingError(".env에 UPSTAGE_API_KEY가 설정되어 있지 않습니다.")
        return cls(key.strip())

    # ------------------------------------------------------------ 토큰 추정·속도 제한

    def estimate_tokens(self, text: str) -> int:
        """실제 사용량으로 보정한 글자당 토큰 비율로 추정 (첫 요청 전에는 글자 수 = 토큰 수로 보수적 추정)."""
        ratio = self.total_tokens / self._chars if self._chars else 1.0
        return int(len(text) * ratio * 1.1) + 1

    def _throttle(self, est_tokens: int) -> None:
        while True:
            now = time.monotonic()
            while self._window and now - self._window[0][0] > 60:
                self._window.popleft()
            used = sum(t for _, t in self._window)
            if len(self._window) < RPM_LIMIT and used + est_tokens <= TPM_LIMIT:
                return
            time.sleep(max(0.5, 60 - (now - self._window[0][0])))

    def batches(self, texts: list[str]) -> Iterator[tuple[int, int]]:
        """요청 한도에 맞게 (시작, 끝) 인덱스 구간으로 나눈다."""
        start, tokens = 0, 0
        for i, text in enumerate(texts):
            est = self.estimate_tokens(text)
            if i > start and (i - start >= MAX_BATCH_SIZE or tokens + est > MAX_BATCH_TOKENS):
                yield start, i
                start, tokens = i, 0
            tokens += est
        if start < len(texts):
            yield start, len(texts)

    # ------------------------------------------------------------ 호출

    def embed(self, texts: list[str], model: str = PASSAGE_MODEL) -> list[list[float]]:
        if not texts:
            return []
        self._throttle(sum(self.estimate_tokens(t) for t in texts))

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = requests.post(
                    API_URL,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={"model": model, "input": texts},
                    timeout=self.timeout,
                )
            except requests.RequestException as e:
                error = str(e)
            else:
                if resp.ok:
                    return self._handle_response(resp.json(), texts)
                error = f"HTTP {resp.status_code}: {resp.text[:300]}"
                if resp.status_code not in (429, 500, 502, 503, 504):
                    raise EmbeddingError(error)  # 인증 오류·입력 초과 등은 재시도해도 같다
                if retry_after := resp.headers.get("Retry-After"):
                    time.sleep(float(retry_after))
                    continue
            if attempt == MAX_RETRIES:
                raise EmbeddingError(error)
            time.sleep(2**attempt)
        raise EmbeddingError("재시도 횟수 초과")

    def _handle_response(self, body: dict, texts: list[str]) -> list[list[float]]:
        data = sorted(body["data"], key=lambda d: d["index"])
        if len(data) != len(texts):
            raise EmbeddingError(f"응답 개수 불일치: 요청 {len(texts)}건, 응답 {len(data)}건")
        used = body.get("usage", {}).get("total_tokens", 0)
        self._window.append((time.monotonic(), used))
        self._chars += sum(len(t) for t in texts)
        self.total_tokens += used
        return [d["embedding"] for d in data]

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text], model=QUERY_MODEL)[0]

    @property
    def cost_usd(self) -> float:
        return self.total_tokens / 1_000_000 * PRICE_PER_M_TOKENS


def to_pgvector(vec: list[float]) -> str:
    return "[" + ",".join(f"{x:.7g}" for x in vec) + "]"
