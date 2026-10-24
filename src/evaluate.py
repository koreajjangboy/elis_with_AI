"""검색 품질 평가 (tests/eval/search_eval.json).

- ordinance(관련 조례 찾기): P@5 = 상위 5개 조례 중 정답 비율(정답 수가 5 미만이면 정답 수로 나눔),
                            R@10 = 상위 10개 조례가 정답 조례를 얼마나 포함하는지
- article(질문 → 조문):     Hit@1, Hit@5, MRR@10 (항 청크는 소속 조로 보고 판정)

질의마다 domain(교통·주차, 환경·생활, 계약·재정 …)을 달아 도메인별로도 집계한다.

실행:
    python -m src.evaluate                       # 보너스 적용
    python -m src.evaluate --compare             # 보너스 적용 전후 비교 (+ 순위가 바뀐 질의)
    python -m src.evaluate --domain 환경·생활 -q  # 특정 도메인만, 질의별 출력 생략
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

from src.db import connect
from src.embedder import UpstageEmbedder
from src.search import hybrid_search

ROOT = Path(__file__).resolve().parent.parent
EVAL_PATH = ROOT / "tests" / "eval" / "search_eval.json"
REPORT_DIR = ROOT / "data" / "eval"  # 평가 결과 JSON (git 제외)

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
    """전체 지표와 도메인별 지표, 질의별 기록을 돌려준다."""
    records = []
    for case in data["ordinance"]:
        hits = hybrid_search(conn, embedder, case["query"], limit=20, boost=boost, max_per_ordinance=1)
        ranked = list(dict.fromkeys(norm(h.ordinance_name) for h in hits))
        relevant = {norm(n) for n in case["relevant"]}
        p5 = sum(n in relevant for n in ranked[:5]) / min(5, len(relevant))
        r10 = sum(n in relevant for n in ranked[:10]) / len(relevant)
        records.append({"type": "ordinance", "query": case["query"], "domain": case.get("domain", "기타"),
                        "P@5": p5, "R@10": r10, "top5": ranked[:5],
                        "wrong_top5": [n for n in ranked[:5] if n not in relevant]})
        if verbose:
            marks = " ".join("O" if n in relevant else "x" for n in ranked[:5])
            print(f"  [조례] {case['query']:<10} P@5 {p5:.2f}  R@10 {r10:.2f}  상위5: {marks}")
            for n in ranked[:5]:
                if n not in relevant:
                    print(f"         오답: {n}")

    for case in data["article"]:
        hits = hybrid_search(conn, embedder, case["query"], limit=10, boost=boost)
        ranked = [norm(article_of(h.citation)) for h in hits]
        expected = {norm(e) for e in case["expected"]}
        rank = next((i for i, c in enumerate(ranked, 1) if c in expected), None)
        records.append({"type": "article", "query": case["query"], "domain": case.get("domain", "기타"),
                        "rank": rank, "Hit@1": rank == 1, "Hit@5": rank is not None and rank <= 5,
                        "MRR@10": 1 / rank if rank else 0, "top1": ranked[0] if ranked else None})
        if verbose:
            print(f"  [조문] {case['query']:<24} 정답 순위 {rank or '10위 밖'}  (1위: {ranked[0] if ranked else '-'})")

    domains = sorted({r["domain"] for r in records})
    return {"overall": summarize(records), "by_domain": {d: summarize([r for r in records if r["domain"] == d]) for d in domains},
            "records": records}


METRICS = {"ordinance": ("P@5", "R@10"), "article": ("Hit@1", "Hit@5", "MRR@10")}


def summarize(records: list[dict]) -> dict:
    """지표별 평균. 해당 유형 질의가 없으면 None (도메인별 표에서 '-')."""
    out = {}
    for kind, metrics in METRICS.items():
        rows = [r for r in records if r["type"] == kind]
        for m in metrics:
            out[m] = sum(float(r[m]) for r in rows) / len(rows) if rows else None
        out[f"n_{kind}"] = len(rows)
    return out


def print_table(title: str, rows: dict[str, dict]) -> None:
    names = [m for ms in METRICS.values() for m in ms]
    width = max([10] + [len(k) * 2 for k in rows])  # 한글 폭 보정
    print(f"\n{title}\n" + " " * width + "  " + "  ".join(f"{m:>7}" for m in names) + "   조례/조문 질의 수")
    fmt = lambda v: f"{v:7.2f}" if v is not None else f"{'-':>7}"
    for label, s in rows.items():
        pad = width - len(label) * 2 + len(label)
        print(f"{label:<{pad}}  " + "  ".join(fmt(s[m]) for m in names) + f"   {s['n_ordinance']}/{s['n_article']}")


def main() -> None:
    os.environ.setdefault("ELIS_DEBUG", "0")  # 질의를 대량으로 돌리므로 검색 디버그 출력은 기본으로 끈다
    parser = argparse.ArgumentParser(description="검색 품질 평가")
    parser.add_argument("--compare", action="store_true", help="보너스 적용 전후 비교")
    parser.add_argument("--domain", help="이 도메인의 질의만 평가 (예: 환경·생활)")
    parser.add_argument("-q", "--quiet", action="store_true", help="질의별 결과 생략")
    parser.add_argument("--out", type=Path, help="결과 JSON 저장 경로 (기본: data/eval/search_<시각>.json)")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    data = json.loads(EVAL_PATH.read_text(encoding="utf-8"))
    if args.domain:
        data = {k: [c for c in data[k] if c.get("domain") == args.domain] for k in METRICS}
        if not any(data.values()):
            raise SystemExit(f"도메인 '{args.domain}'의 질의가 없습니다.")
    embedder = UpstageEmbedder.from_env()
    variants = [("보너스 없음", False), ("보너스 적용", True)] if args.compare else [("보너스 적용", True)]
    results = {}
    with connect() as conn:
        for label, boost in variants:
            print(f"\n=== {label} ===")
            results[label] = evaluate(conn, embedder, data, boost, verbose=not args.quiet)

    last = results[variants[-1][0]]
    print_table(f"[전체] 조례 질의 {last['overall']['n_ordinance']}개 · 조문 질의 {last['overall']['n_article']}개",
                {label: r["overall"] for label, r in results.items()})
    for label, r in results.items():
        print_table(f"[도메인별 · {label}]", r["by_domain"])
    if args.compare:
        before, after = (results[v[0]]["records"] for v in variants)
        changed = [(b, a) for b, a in zip(before, after) if b["type"] == "article" and b["rank"] != a["rank"]]
        if changed:
            print("\n[보너스로 정답 순위가 바뀐 조문 질의]")
            for b, a in changed:
                print(f"  {b['query']:<28} {b['rank'] or '10위 밖'} → {a['rank'] or '10위 밖'}")

    out = args.out or REPORT_DIR / f"search_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n결과 저장: {out}")


if __name__ == "__main__":
    main()
