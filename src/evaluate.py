"""검색 품질 평가 (tests/eval/search_eval.json).

- ordinance(관련 조례 찾기): P@5 = 상위 5개 조례 중 정답 비율(정답 수가 5 미만이면 정답 수로 나눔),
                            R@10 = 상위 10개 조례가 정답 조례를 얼마나 포함하는지
- article(질문 → 조문):     Hit@1, Hit@5, MRR@10 (항 청크는 소속 조로 보고 판정)

실행:
    python -m src.evaluate               # 보너스 적용
    python -m src.evaluate --compare     # 보너스 적용 전후 비교
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

from src.db import connect
from src.embedder import UpstageEmbedder
from src.search import hybrid_search

EVAL_PATH = Path(__file__).resolve().parent.parent / "tests" / "eval" / "search_eval.json"

# 목록 API와 본문 API가 가운뎃점을 다르게 준다 (· U+00B7 / ㆍ U+318D 등) → 비교 전에 통일
_MIDDOTS = str.maketrans({"·": "ㆍ", "‧": "ㆍ", "・": "ㆍ", "•": "ㆍ"})


def norm(name: str) -> str:
    return " ".join(name.translate(_MIDDOTS).split())


def article_of(citation: str) -> str:
    """'…조례 제5조제2항' → '…조례 제5조' (항 표기만 떼고 '제5조의2'는 유지).
    별표 조각 '…조례 별표 1 요금표 (1/3)' → '…조례 별표 1 요금표'."""
    citation = re.sub(r" \(\d+/\d+\)$", "", citation)
    if " 제" not in citation or "별표" in citation or "서식" in citation:
        return citation
    head, sep, tail = citation.rpartition(" 제")
    return f"{head}{sep}{tail.split('제', 1)[0]}" if sep else citation


def evaluate(conn, embedder, data: dict, boost: bool, verbose: bool) -> dict:
    p5s, r10s = [], []
    for case in data["ordinance"]:
        hits = hybrid_search(conn, embedder, case["query"], limit=20, boost=boost, max_per_ordinance=1)
        ranked = list(dict.fromkeys(norm(h.ordinance_name) for h in hits))
        relevant = {norm(n) for n in case["relevant"]}
        p5 = sum(n in relevant for n in ranked[:5]) / min(5, len(relevant))
        r10 = sum(n in relevant for n in ranked[:10]) / len(relevant)
        p5s.append(p5)
        r10s.append(r10)
        if verbose:
            marks = " ".join("O" if n in relevant else "x" for n in ranked[:5])
            print(f"  [조례] {case['query']:<10} P@5 {p5:.2f}  R@10 {r10:.2f}  상위5: {marks}")
            for n in ranked[:5]:
                if n not in relevant:
                    print(f"         오답: {n}")

    hit1s, hit5s, rrs = [], [], []
    for case in data["article"]:
        hits = hybrid_search(conn, embedder, case["query"], limit=10, boost=boost)
        ranked = [norm(article_of(h.citation)) for h in hits]
        expected = {norm(e) for e in case["expected"]}
        rank = next((i for i, c in enumerate(ranked, 1) if c in expected), None)
        hit1s.append(rank == 1)
        hit5s.append(rank is not None and rank <= 5)
        rrs.append(1 / rank if rank else 0)
        if verbose:
            print(f"  [조문] {case['query']:<24} 정답 순위 {rank or '10위 밖'}  (1위: {ranked[0] if ranked else '-'})")

    avg = lambda xs: sum(xs) / len(xs) if xs else 0
    return {"P@5": avg(p5s), "R@10": avg(r10s), "Hit@1": avg(hit1s), "Hit@5": avg(hit5s), "MRR@10": avg(rrs)}


def main() -> None:
    os.environ.setdefault("ELIS_DEBUG", "0")  # 질의를 대량으로 돌리므로 검색 디버그 출력은 기본으로 끈다
    parser = argparse.ArgumentParser(description="검색 품질 평가")
    parser.add_argument("--compare", action="store_true", help="보너스 적용 전후 비교")
    parser.add_argument("-q", "--quiet", action="store_true", help="질의별 결과 생략")
    args = parser.parse_args()

    data = json.loads(EVAL_PATH.read_text(encoding="utf-8"))
    embedder = UpstageEmbedder.from_env()
    variants = [("보너스 없음", False), ("보너스 적용", True)] if args.compare else [("보너스 적용", True)]
    results = {}
    with connect() as conn:
        for label, boost in variants:
            print(f"\n=== {label} ===")
            results[label] = evaluate(conn, embedder, data, boost, verbose=not args.quiet)

    print("\n" + " " * 12 + "  ".join(f"{m:>7}" for m in next(iter(results.values()))))
    for label, metrics in results.items():
        print(f"{label:<10}  " + "  ".join(f"{v:7.2f}" for v in metrics.values()))


if __name__ == "__main__":
    main()
