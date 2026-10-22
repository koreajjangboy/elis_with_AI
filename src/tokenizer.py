"""Kiwi 형태소 분석 기반 키워드 토큰화. 적재(chunks.tokens)와 검색 질의에 같은 함수를 써야 한다."""

from __future__ import annotations

from functools import lru_cache

from kiwipiepy import Kiwi

# 검색에 의미 있는 품사: 일반/고유명사, 수사, 외국어, 한자, 숫자, 어근, 동사·형용사 어간
KEEP_TAGS = {"NNG", "NNP", "NR", "SL", "SH", "SN", "XR", "VV", "VA"}


@lru_cache(maxsize=1)
def _kiwi() -> Kiwi:
    return Kiwi()


def tokenize(text: str) -> str:
    """'주차장 이용요금의 감면대상' → '주차장 이용 요금 감면 대상'"""
    return " ".join(t.form.lower() for t in _kiwi().tokenize(text) if t.tag in KEEP_TAGS)


def tokenize_many(texts: list[str]) -> list[str]:
    return [" ".join(t.form.lower() for t in tokens if t.tag in KEEP_TAGS) for tokens in _kiwi().tokenize(texts)]
