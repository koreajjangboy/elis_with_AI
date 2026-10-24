"""Kiwi 형태소 분석 기반 키워드 토큰화. 적재(chunks.*_tokens)와 검색 질의에 같은 함수를 써야 한다.

법령 검색에서는 명사가 핵심어다. 동사·형용사 어간(VV, VA)이나 어근(XR)은
'하다', '있다', '정하다'처럼 거의 모든 조문에 등장해 엉뚱한 조문을 끌어올리므로 제외한다.
"""

from __future__ import annotations

from functools import lru_cache

from kiwipiepy import Kiwi

# 일반/고유명사, 수사, 외국어, 한자, 숫자
KEEP_TAGS = {"NNG", "NNP", "NR", "SL", "SH", "SN"}

# 거의 모든 조문에 등장해 변별력이 없는 법령 상투어
STOPWORDS = {
    "조례", "규칙", "규정", "사항", "경우", "관련", "해당", "다음", "이하", "이상", "이내",
    "등", "것", "수", "때", "각", "제", "조", "항", "호", "목", "단", "외", "중", "내",
}


@lru_cache(maxsize=1)
def _kiwi() -> Kiwi:
    return Kiwi()


def _keep(tokens) -> str:
    return " ".join(
        form for t in tokens if t.tag in KEEP_TAGS and (form := t.form.lower()) not in STOPWORDS
    )


def tokenize(text: str) -> str:
    """'주차장 이용요금의 감면대상을 정한다' → '주차장 이용 요금 감면 대상'"""
    return _keep(_kiwi().tokenize(text))


def tokenize_many(texts: list[str]) -> list[str]:
    return [_keep(tokens) for tokens in _kiwi().tokenize(texts)]
