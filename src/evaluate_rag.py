"""RAG 답변 품질 평가 (tests/eval/rag_eval.json). 실제 DB·임베딩·LLM을 호출한다.

지표
- 상태 정확도     : expected_status(answered/unverifiable)와 최종 status가 같은 비율
- 오답변          : 확인 불가여야 할 질문에 answered로 답한 수 (가장 위험)
- 과잉 거절       : 답할 수 있는 질문을 확인 불가로 처리한 수
- 정답 조문 인용률 : answered 질의 중 정답 조문을 하나 이상 인용한 비율
- 타 조례 혼입률   : answered 질의 중 정답 조례가 아닌 조례를 인용한 비율
- 인용 검증       : LLM 1차 응답 인용구의 원문 일치율, 재생성 발생률, 최종 제외된 인용 수
- 대상 조례 점검   : 질문 대상이 아닌 조례 인용을 감지해 재생성·확인 불가·경고한 질의 수 (src.rag.apply_topic_guard)

실행:
    python -m src.evaluate_rag                     # 전체
    python -m src.evaluate_rag --domain 환경·생활   # 특정 도메인만
    python -m src.evaluate_rag --limit 3            # 앞에서 3개만 (빠른 점검)
    python -m src.evaluate_rag --cases 다른평가셋.json  # 다른 평가셋 ({"cases": [...]} 또는 목록, 질문 필드는 question|query)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

from src.evaluate import REPORT_DIR, ROOT, article_of, norm
from src.rag import TOPIC_NOTE, answer_question, check_citations

RAG_EVAL_PATH = ROOT / "tests" / "eval" / "rag_eval.json"


class RecordingLLM:
    """LLM 응답 원문(dict)을 호출 순서대로 기록한다. 1차 응답의 인용 일치율과 재생성 여부를 보기 위함."""

    def __init__(self, llm):
        self.llm = llm
        self.name = llm.name
        self.calls: list[dict] = []

    def complete_json(self, *args, **kwargs) -> dict:
        raw = self.llm.complete_json(*args, **kwargs)
        self.calls.append(raw)
        return raw


def ordinance_of(article: str) -> str:
    """'수원시 주차장 조례 제12조' / '수원시 주차장 조례 별표 1 공영주차장 주차요금표' → '수원시 주차장 조례'."""
    return norm(re.split(r" (?=제\d+조|[\[〔]?별표|별지|[\[〔]?서식)", article, maxsplit=1)[0])


def load_cases(path: Path) -> list[dict]:
    """{"cases": [...]} 또는 [...] 형식. 질문 필드는 question 또는 query."""
    data = json.loads(path.read_text(encoding="utf-8"))
    cases = data["cases"] if isinstance(data, dict) else data
    out = []
    for c in cases:
        if "question" not in c:
            c = {**c, "question": c["query"]}
        out.append({**c, "expected": c.get("expected") or []})
    return out


def run_case(conn, embedder, llm, case: dict) -> dict:
    recorder = RecordingLLM(llm)
    started = time.perf_counter()
    try:
        result = answer_question(conn, embedder, recorder, case["question"])
    except Exception as e:  # 한 질의의 API 오류로 전체 평가가 멈추지 않게
        return {**case, "error": f"{type(e).__name__}: {e}", "ok": False}
    elapsed = time.perf_counter() - started

    by_id = {s.id: s for s in result.sources}
    cited_articles = sorted({norm(article_of(c.label)) for c in result.citations})
    cited_ordinances = sorted({norm(by_id[c.source_id].ordinance_name) for c in result.citations})
    expected = {norm(e) for e in case["expected"]}
    expected_ords = {ordinance_of(e) for e in case["expected"]}
    retrieved = [norm(article_of(s.label)) for s in result.sources]

    first_total = first_valid = 0
    final_invalid = 0
    if recorder.calls:
        first = recorder.calls[0]
        if first.get("status") == "answered":
            valid, invalid = check_citations(first, result.sources)
            first_total, first_valid = len(valid) + len(invalid), len(valid)
        last = recorder.calls[-1]
        if last.get("status") == "answered":
            final_invalid = len(check_citations(last, result.sources)[1])

    rec = {
        **case,
        "status": result.status,
        "status_ok": result.status == case["expected_status"],
        "retrieved_expected": bool(expected & set(retrieved)) if expected else None,
        "cited_expected": bool(expected & set(cited_articles)) if expected else None,
        "off_topic": sorted(set(cited_ordinances) - expected_ords) if expected and result.status == "answered" else [],
        "cited": cited_articles,
        "llm_calls": len(recorder.calls),
        "regenerated": len(recorder.calls) > 1,
        "first_quotes": first_total,
        "first_valid_quotes": first_valid,
        "final_invalid_quotes": final_invalid,
        "model_said": recorder.calls[0].get("status") if recorder.calls else None,
        "topic_guard": any(TOPIC_NOTE in n for n in result.notes),
        "answer": result.answer,
        "notes": result.notes,
        "seconds": round(elapsed, 1),
    }
    rec["ok"] = rec["status_ok"] and (case["expected_status"] != "answered" or (rec["cited_expected"] and not rec["off_topic"]))
    return rec


def summarize(records: list[dict]) -> dict:
    done = [r for r in records if "error" not in r]
    ans = [r for r in done if r["expected_status"] == "answered"]
    unv = [r for r in done if r["expected_status"] == "unverifiable"]
    answered_ok = [r for r in ans if r["status"] == "answered"]
    ratio = lambda a, b: a / b if b else None
    first_total = sum(r["first_quotes"] for r in done)
    return {
        "n": len(records),
        "errors": len(records) - len(done),
        "status_accuracy": ratio(sum(r["status_ok"] for r in done), len(done)),
        "unverifiable_recall": ratio(sum(r["status"] == "unverifiable" for r in unv), len(unv)),
        "false_answer": sum(r["status"] == "answered" for r in unv),
        "over_refusal": sum(r["status"] == "unverifiable" for r in ans),
        "retrieval_hit": ratio(sum(bool(r["retrieved_expected"]) for r in ans), len(ans)),
        "citation_hit": ratio(sum(bool(r["cited_expected"]) for r in ans), len(ans)),
        "off_topic_rate": ratio(sum(bool(r["off_topic"]) for r in answered_ok), len(answered_ok)),
        "first_quote_valid_rate": ratio(sum(r["first_valid_quotes"] for r in done), first_total),
        "regeneration_rate": ratio(sum(r["regenerated"] for r in done), len(done)),
        "final_invalid_quotes": sum(r["final_invalid_quotes"] for r in done),
        "topic_guard": sum(r.get("topic_guard", False) for r in done),
        "pass": sum(r["ok"] for r in records),
        "avg_seconds": ratio(sum(r["seconds"] for r in done), len(done)),
    }


LABELS = [
    ("status_accuracy", "상태 정확도", "%"),
    ("unverifiable_recall", "확인 불가 재현율", "%"),
    ("false_answer", "오답변(확인 불가여야 하는데 답함)", "건"),
    ("over_refusal", "과잉 거절(답할 수 있는데 확인 불가)", "건"),
    ("retrieval_hit", "정답 조문 검색률(근거 자료 내)", "%"),
    ("citation_hit", "정답 조문 인용률", "%"),
    ("off_topic_rate", "타 조례 혼입률", "%"),
    ("first_quote_valid_rate", "1차 인용구 원문 일치율", "%"),
    ("regeneration_rate", "재생성 발생률", "%"),
    ("final_invalid_quotes", "최종 제외된 인용", "건"),
    ("topic_guard", "질문 대상 조례 점검 작동", "건"),
    ("avg_seconds", "평균 응답 시간", "s"),
]


def fmt(value, unit: str) -> str:
    if value is None:
        return "-"
    return f"{value * 100:.0f}%" if unit == "%" else f"{value:.1f}s" if unit == "s" else f"{value}건"


def print_report(summary: dict, by_domain: dict[str, dict], records: list[dict], model: str) -> None:
    print(f"\n=== RAG 답변 품질 리포트 · {model} · {summary['n']}개 질의 (통과 {summary['pass']}, 오류 {summary['errors']}) ===")
    for key, label, unit in LABELS:
        print(f"  {label:<28} {fmt(summary[key], unit)}")

    cols = [("status_accuracy", "상태정확", "%"), ("citation_hit", "인용적중", "%"), ("off_topic_rate", "혼입", "%"),
            ("first_quote_valid_rate", "1차일치", "%"), ("false_answer", "오답변", "건"), ("over_refusal", "과잉거절", "건")]
    print("\n[도메인별]\n" + " " * 14 + "".join(f"{c[1]:>8}" for c in cols) + "   통과/질의")
    for dom, s in by_domain.items():
        pad = 14 - len(dom)  # 한글 2칸 보정
        print(f"{dom}{' ' * max(pad, 1)}" + "".join(f"{fmt(s[k], u):>10}" for k, _, u in cols) + f"   {s['pass']}/{s['n']}")

    failed = [r for r in records if not r["ok"]]
    if failed:
        print("\n[실패 질의]")
    for r in failed:
        if "error" in r:
            print(f"  ✗ [{r['domain']}] {r['question']}\n      오류: {r['error']}")
            continue
        why = []
        if not r["status_ok"]:
            why.append(f"상태 {r['expected_status']} 기대 → {r['status']} (LLM 1차 판단: {r['model_said']})")
        if r["expected_status"] == "answered" and r["status"] == "answered" and not r["cited_expected"]:
            why.append(f"정답 조문 미인용 (검색{'됨' if r['retrieved_expected'] else ' 안 됨'}) · 인용: {r['cited']}")
        if r["off_topic"]:
            why.append(f"타 조례 혼입: {r['off_topic']}")
        print(f"  ✗ [{r['domain']}] {r['question']}")
        for w in why:
            print(f"      - {w}")
        for n in r["notes"]:
            print(f"      ※ {n}")


def main() -> None:
    os.environ.setdefault("ELIS_DEBUG", "0")
    parser = argparse.ArgumentParser(description="RAG 답변 품질 평가 (DB·LLM 호출)")
    parser.add_argument("--cases", type=Path, default=RAG_EVAL_PATH, help="평가셋 JSON (기본: tests/eval/rag_eval.json)")
    parser.add_argument("--domain", help="이 도메인의 질의만 평가")
    parser.add_argument("--limit", type=int, help="앞에서 N개 질의만 평가")
    parser.add_argument("--out", type=Path, help="결과 JSON 저장 경로 (기본: data/eval/rag_<시각>.json)")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    from src.db import connect
    from src.embedder import UpstageEmbedder
    from src.llm import llm_from_env

    cases = load_cases(args.cases)
    if args.domain:
        cases = [c for c in cases if c["domain"] == args.domain]
    cases = cases[: args.limit] if args.limit else cases
    if not cases:
        raise SystemExit("평가할 질의가 없습니다.")

    llm, embedder = llm_from_env(), UpstageEmbedder.from_env()
    records = []
    with connect() as conn:
        for i, case in enumerate(cases, 1):
            rec = run_case(conn, embedder, llm, case)
            records.append(rec)
            mark = "O" if rec["ok"] else "✗"
            print(f"[{i:>2}/{len(cases)}] {mark} {rec.get('status', 'error'):<12} {case.get('id', '')} {case['question']}")

    summary = summarize(records)
    by_domain = {d: summarize([r for r in records if r["domain"] == d]) for d in sorted({r["domain"] for r in records})}
    print_report(summary, by_domain, records, llm.name)

    out = args.out or REPORT_DIR / f"rag_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"model": llm.name, "summary": summary, "by_domain": by_domain, "records": records},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n결과 저장: {out}")


if __name__ == "__main__":
    main()
