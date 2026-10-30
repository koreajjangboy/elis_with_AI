"""FastAPI 백엔드: 하이브리드 검색 + 근거 강제 RAG 질의응답.

실행:
    uvicorn src.api:app --reload --port 8000
    → http://localhost:8000/docs (Swagger UI)

엔드포인트:
    GET  /health   DB·LLM 상태
    POST /search   하이브리드 검색 결과 (LLM 없이)
    POST /ask      근거 조문·별표를 인용한 답변. 근거가 없으면 answer='확인 불가', citations=[]
                   자료요구·언론 보도·뉴스 기사(URL 또는 본문) 질의면 요구할 자료 목록(data_requests)을 함께 준다
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import date

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.collect_suwon import SBORG
from src.db import create_pool
from src.embedder import EmbeddingError, UpstageEmbedder
from src.llm import LLMError, llm_from_env
from src.rag import DEFAULT_TOP_K, answer_question
from src.search import hybrid_search

log = logging.getLogger("elis.api")
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["pool"] = create_pool()
    state["embedder"] = UpstageEmbedder.from_env()
    try:
        state["llm"] = llm_from_env()
    except LLMError as e:  # LLM 키가 없어도 검색 API는 동작하게 한다
        log.warning("LLM 비활성: %s", e)
        state["llm"], state["llm_error"] = None, str(e)
    yield
    state["pool"].close()


app = FastAPI(title="elis_with_AI — 수원시 자치법규 RAG API", version="0.1.0", lifespan=lifespan)


# ---------------------------------------------------------------- 스키마

class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500, examples=["공영주차장 주차요금 감면"])
    limit: int = Field(10, ge=1, le=50)
    org_code: str = SBORG


class SearchHitOut(BaseModel):
    chunk_id: int
    article_id: int | None
    attachment_id: int | None
    ordinance_id: int
    citation: str
    content: str
    ordinance_name: str
    article_title: str
    enforcement_date: date | None
    source_url: str | None
    attachment_kind: str | None
    score: float


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=20000, examples=["공영주차장 주차요금 감면 대상은?"],
                          description="질문, 뉴스 기사 URL(요청 문장과 함께 가능) 또는 붙여넣은 기사 본문")
    top_k: int = Field(DEFAULT_TOP_K, ge=1, le=15, description="검색해 근거 자료로 넘길 조문·별표 수 (관련 별표는 별도로 최대 3건 추가)")
    org_code: str = SBORG


class CitationOut(BaseModel):
    source_id: str
    label: str
    quote: str
    relabeled_from: str | None = Field(None, description="모델이 단 자료 번호가 틀려 바로잡은 경우 원래 번호")


class NewsOut(BaseModel):
    title: str
    url: str | None
    source: str = Field(description="'url'(기사 URL에서 추출) | 'text'(붙여넣은 본문)")
    summary: str = Field(description="현안 요지 (LLM 분석)")
    keywords: list[str]
    body_chars: int
    error: str | None = Field(description="URL을 읽지 못한 사유")


class DataRequestOut(BaseModel):
    item: str = Field(description="행정에 요구할 자료명")
    source_id: str
    label: str = Field(description="근거 조문")
    purpose: str = Field(description="그 조문의 어떤 규정의 이행을 확인하는지")


class SourceOut(BaseModel):
    id: str
    kind: str
    label: str
    ref: str
    ordinance_name: str
    title: str
    content: str
    article_id: int | None
    attachment_id: int | None
    attachment_kind: str | None
    source_url: str | None
    extraction_method: str | None
    has_images: bool
    image_description: str | None
    extracted: bool
    partial: bool
    related_to: str | None
    score: float | None


class AskResponse(BaseModel):
    status: str = Field(description="'answered' | 'unverifiable'")
    answer: str = Field(description="근거가 없으면 정확히 '확인 불가'")
    citations: list[CitationOut] = Field(description="원문 대조를 통과한 인용. 확인 불가면 빈 목록")
    sources: list[SourceOut] = Field(description="LLM에 제공한 근거 자료 전체 (화면의 원문 펼쳐보기용)")
    model: str
    notes: list[str]
    unconfirmed: list[str] = Field(default_factory=list,
                                   description="질문 중 근거 자료로 확인되지 않은 사항 (비어 있지 않으면 부분 답변)")
    mode: str = Field("qa", description="'qa' | 'request'(자료요구) | 'issue'(언론 보도·현안 추론)")
    data_requests: list[DataRequestOut] = Field(default_factory=list,
                                                description="자료요구·현안 모드에서 행정에 요구할 자료 목록 (항목마다 근거 조문)")
    search_queries: list[str] = Field(default_factory=list, description="현안 모드에서 함께 검색한 조례 용어 검색어")
    news: NewsOut | None = Field(None, description="뉴스 기사 입력이면 기사 제목·현안 요지·핵심어")


# ---------------------------------------------------------------- 엔드포인트

@app.get("/health")
def health() -> dict:
    with state["pool"].connection() as conn:
        ordinances, chunks = conn.execute(
            "SELECT (SELECT count(*) FROM ordinances WHERE is_current), (SELECT count(*) FROM chunks)"
        ).fetchone()
    llm = state.get("llm")
    return {
        "status": "ok",
        "ordinances": ordinances,
        "chunks": chunks,
        "llm": llm.name if llm else None,
        "llm_error": state.get("llm_error"),
    }


@app.post("/search", response_model=list[SearchHitOut])
def search(req: SearchRequest) -> list[dict]:
    try:
        with state["pool"].connection() as conn:
            hits = hybrid_search(conn, state["embedder"], req.query, org_code=req.org_code, limit=req.limit)
    except EmbeddingError as e:
        raise HTTPException(502, f"질의 임베딩 실패: {e}") from e
    return [asdict(h) for h in hits]


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> dict:
    if state.get("llm") is None:
        raise HTTPException(503, f"LLM이 설정되지 않았습니다: {state.get('llm_error')}")
    try:
        with state["pool"].connection() as conn:
            result = answer_question(conn, state["embedder"], state["llm"], req.question,
                                     org_code=req.org_code, top_k=req.top_k)
    except EmbeddingError as e:
        raise HTTPException(502, f"질의 임베딩 실패: {e}") from e
    except LLMError as e:
        raise HTTPException(502, f"답변 생성 실패: {e}") from e
    return result.to_dict()
