"""검색·RAG 내부 처리 과정 디버그 출력.

기본으로 켜져 있고, 환경변수 ELIS_DEBUG=0이면 끈다 (평가 스크립트처럼 질의를 대량으로 돌릴 때).
결과 출력(stdout)과 섞이지 않도록 stderr로 쓴다. 'python -m src.search ... 2>NUL'로 숨길 수 있다.
"""

from __future__ import annotations

import os
import sys


def enabled() -> bool:
    return os.getenv("ELIS_DEBUG", "1").strip().lower() not in ("0", "false", "off", "")


def dprint(*parts: object) -> None:
    if not enabled():
        return
    text = " ".join(str(p) for p in parts)
    try:
        print(text, file=sys.stderr, flush=True)
    except UnicodeEncodeError:  # Windows 파이프 출력(cp949)에서 못 쓰는 문자
        enc = sys.stderr.encoding or "utf-8"
        print(text.encode(enc, "replace").decode(enc), file=sys.stderr, flush=True)


def section(title: str) -> None:
    dprint(f"\n[DEBUG] ── {title} " + "─" * max(0, 60 - len(title)))


def dblock(title: str, text: str) -> None:
    """긴 텍스트(LLM 프롬프트·응답)를 자르지 않고 전문 그대로 출력한다. 시작·끝 표시로 경계를 구분한다."""
    dprint(f"\n[DEBUG] ▼▼▼ {title} (전문 시작) ▼▼▼\n{text}\n[DEBUG] ▲▲▲ {title} (전문 끝) ▲▲▲")


def preview(text: str, width: int = 80) -> str:
    """검색 결과·근거 자료 목록용 한 줄 요약. LLM 요청·응답 로그에는 쓰지 않는다 (dblock 사용)."""
    flat = " ".join(text.split())
    return flat if len(flat) <= width else flat[: width - 1] + "…"
