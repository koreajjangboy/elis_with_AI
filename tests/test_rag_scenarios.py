"""의정 활동 '자료요구' 시나리오 통합 테스트 (실제 DB·임베딩·LLM 호출).

비용·시간이 들므로 기본 pytest에서는 건너뛴다. 실행:
    ELIS_LIVE_TESTS=1 python -m pytest tests/test_rag_scenarios.py -v -s

- 검색 테스트: 정답 조문이 RAG 근거 자료(top_k)에 들어오는지 (임베딩 API만 사용)
- 답변 테스트: answered 시나리오는 정답 조문을 인용했는지, unverifiable 시나리오는 '확인 불가' + 빈 인용인지
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import pytest

from src.evaluate import article_of, norm

pytestmark = pytest.mark.skipif(os.getenv("ELIS_LIVE_TESTS") != "1", reason="ELIS_LIVE_TESTS=1 일 때만 실행 (DB·API 호출)")


@dataclass(frozen=True)
class Scenario:
    id: str
    question: str
    expected_status: str  # 'answered' | 'unverifiable'
    expected: tuple[str, ...]  # 검색·인용되어야 할 조문 (항 생략). unverifiable은 검색만 확인


SCENARIOS = [
    Scenario(
        "보조금_정산",
        "마을만들기 사업 보조금을 받은 추진주체가 사업을 마친 뒤 시에 제출해야 하는 실적보고서와 정산 자료는 무엇인가요?",
        "answered",
        ("수원시 마을만들기 조례 제32조",),
    ),
    Scenario(
        "위원회_심의자료",
        "시 소속 위원회의 회의록에는 어떤 내용이 들어가야 하고, 의원이 요구하면 공개해야 하나요?",
        "answered",
        ("수원시 각종 위원회 설치 및 운영 조례 제15조",),
    ),
    Scenario(
        "공모사업_의회보고",
        "시가 정부 공모사업에 신청하기 전에 시의회에 사전 보고해야 하는 사업은 어떤 경우인가요?",
        "answered",
        ("수원시 공모사업 관리 조례 제7조",),
    ),
    Scenario(
        "재정사업_종결현황",
        "예산안을 심사할 때 종결된 재정사업 현황 자료를 집행부가 의회에 함께 제출해야 하는 근거 규정이 있나요?",
        "answered",
        ("수원시 재정사업 종결 관리 기본 조례 제5조",),
    ),
    Scenario(
        "청년창업_집행실적",  # 조례에는 지원 근거만 있고 연도별 집행액·실적은 없다 → 확인 불가여야 한다
        "2025년도 청년 창업 지원 사업의 예산 집행액과 지원받은 기업 수 현황을 알려 주세요.",
        "unverifiable",
        ("수원시 청년 창업 지원 조례 제5조",),
    ),
]


@pytest.fixture(scope="module")
def ctx():
    from src.db import connect
    from src.embedder import UpstageEmbedder

    os.environ.setdefault("ELIS_DEBUG", "0")
    with connect() as conn:
        yield conn, UpstageEmbedder.from_env()


def _article_set(labels) -> set[str]:
    return {norm(article_of(label)) for label in labels}


@pytest.mark.parametrize("sc", SCENARIOS, ids=[s.id for s in SCENARIOS])
def test_scenario_retrieves_expected_article(ctx, sc: Scenario):
    from src.rag import DEFAULT_TOP_K
    from src.search import hybrid_search

    conn, embedder = ctx
    hits = hybrid_search(conn, embedder, sc.question, limit=DEFAULT_TOP_K)
    ranked = [norm(article_of(h.citation)) for h in hits]
    expected = _article_set(sc.expected)
    rank = next((i for i, c in enumerate(ranked, 1) if c in expected), None)
    print(f"\n[{sc.id}] 정답 순위 {rank or f'{DEFAULT_TOP_K}위 밖'}  상위: {ranked[:3]}")
    assert rank is not None, f"정답 조문 {sc.expected}이 top {DEFAULT_TOP_K}에 없음: {ranked}"


@pytest.mark.parametrize("sc", SCENARIOS, ids=[s.id for s in SCENARIOS])
def test_scenario_answer_status_and_citation(ctx, sc: Scenario):
    from src.llm import llm_from_env
    from src.rag import UNVERIFIABLE, answer_question

    conn, embedder = ctx
    result = answer_question(conn, embedder, llm_from_env(), sc.question)
    cited = _article_set(c.label for c in result.citations)
    print(f"\n[{sc.id}] status={result.status}  인용={sorted(cited)}  notes={result.notes}\n{result.answer}")

    assert result.status == sc.expected_status
    if sc.expected_status == "unverifiable":
        assert result.answer == UNVERIFIABLE and result.citations == []
    else:
        assert cited & _article_set(sc.expected), f"정답 조문 {sc.expected}을 인용하지 않음: {sorted(cited)}"
        assert "근거:" in result.answer
