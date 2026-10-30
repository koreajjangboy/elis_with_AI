"""RAG 답변 품질 평가 (tests/eval/rag_eval.json). 실제 DB·임베딩·LLM을 호출한다.

질의별 결과(outcome)
- pass        : answered 질의는 정답 조문을 인용하고 다른 조례가 섞이지 않음 / unverifiable 질의는 '확인 불가'
- partial     : unverifiable 질의에 "조례로는 확인할 수 없음"을 밝히고 관련 절차·책무만 안내한 부분 답변 (고지형).
                실무 보좌에 유용한 정상 답변으로 보고 '실무 통과'에 넣는다
- false_hard  : unverifiable 질의에 그런 고지 없이 답함 (가장 위험)
- over_refusal: answered 질의를 '확인 불가'로 처리
- missing     : answered로 답했지만 정답 조문을 인용하지 않음
- off_topic   : 정답 조례가 아닌 조례를 인용함 (타 조례 혼입)
- no_requests : 자료요구·현안 질의(expect_requests)에 정답 조문을 인용했지만 요구 자료 목록이 없음
통과는 두 가지로 센다: 실무 통과 = pass + partial, 엄격 통과 = pass.

그 밖의 지표
- 정답 조문 검색률·인용률, 타 조례 혼입률
- 인용 검증: LLM 1차 응답 인용구의 원문 일치율, 재생성 발생률, 최종 제외된 인용 수
- 방어선 작동: 조례명 점검(src.rag.apply_topic_guard), 2단계 근거 검증(src.rag.verify_answer)
- 의정활동 모드: 질의 유형 판정 정확도(케이스의 mode), 요구 자료 목록 작성률, 요구 자료 근거 적중률(근거 조문이 정답 조례인 항목 비율)

실행:
    python -m src.evaluate_rag                     # 전체
    python -m src.evaluate_rag --domain 환경·생활   # 특정 도메인만
    python -m src.evaluate_rag --limit 3            # 앞에서 3개만 (빠른 점검)
    python -m src.evaluate_rag --cases 다른평가셋.json  # 다른 평가셋 ({"cases": [...]} 또는 목록, 질문 필드는 question|query)
    python -m src.evaluate_rag --cases tests/eval/rag_eval_request.json  # 자료요구·현안 추론 시나리오
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
from src.rag import ISSUE_SYSTEM_PROMPT, LINKED_ORDINANCE, TOPIC_NOTE, VERIFY_NOTE, VERIFY_SYSTEM_PROMPT, answer_question, check_citations

RAG_EVAL_PATH = ROOT / "tests" / "eval" / "rag_eval.json"
PASSING = {"pass", "partial"}
OUTCOME_LABELS = {"pass": "통과", "partial": "고지형 부분 답변", "false_hard": "고지 없는 오답변", "over_refusal": "과잉 거절",
                  "missing": "정답 조문 미인용", "off_topic": "타 조례 혼입", "no_requests": "요구 자료 목록 없음", "error": "오류"}
CALL_KINDS = {VERIFY_SYSTEM_PROMPT: "verify", ISSUE_SYSTEM_PROMPT: "expand"}


class RecordingLLM:
    """LLM 응답 원문(dict)을 호출 순서대로 기록한다. 답변 생성·근거 검증·현안 검색어 확장 호출을 나눠 센다."""

    def __init__(self, llm):
        self.llm = llm
        self.name = llm.name
        self.calls: list[tuple[str, dict]] = []  # ('answer' | 'verify' | 'expand', 응답)

    def complete_json(self, system: str, user: str, schema: dict) -> dict:
        raw = self.llm.complete_json(system, user, schema)
        self.calls.append((CALL_KINDS.get(system, "answer"), raw))
        return raw

    def answers(self) -> list[dict]:
        return [raw for kind, raw in self.calls if kind == "answer"]


def short(question: str, n: int = 90) -> str:
    """리포트용: 붙여넣은 기사처럼 긴 입력은 첫 줄 앞부분만."""
    first = question.strip().splitlines()[0]
    return first[:n] + ("…" if len(first) > n or "\n" in question.strip() else "")


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


def classify(rec: dict) -> str:
    if rec["expected_status"] == "unverifiable":
        if rec["status"] == "unverifiable":
            return "pass"
        return "partial" if rec["partial"] else "false_hard"
    if rec["status"] == "unverifiable":
        return "over_refusal"
    if not rec["cited_expected"]:
        return "missing"
    if rec["off_topic"]:
        return "off_topic"
    return "no_requests" if rec.get("expect_requests") and not rec.get("data_requests") else "pass"


def run_case(conn, embedder, llm, case: dict) -> dict:
    recorder = RecordingLLM(llm)
    started = time.perf_counter()
    try:
        result = answer_question(conn, embedder, recorder, case["question"])
    except Exception as e:  # 한 질의의 API 오류로 전체 평가가 멈추지 않게
        return {**case, "error": f"{type(e).__name__}: {e}", "outcome": "error", "ok": False, "ok_strict": False}
    elapsed = time.perf_counter() - started

    by_id = {s.id: s for s in result.sources}
    cited_articles = sorted({norm(article_of(c.label)) for c in result.citations})
    cited_ordinances = sorted({norm(by_id[c.source_id].ordinance_name) for c in result.citations})
    expected = {norm(e) for e in case["expected"]}
    expected_ords = {ordinance_of(e) for e in case["expected"]}
    retrieved = [norm(article_of(s.label)) for s in result.sources]

    answers = recorder.answers()
    first_total = first_valid = final_invalid = 0
    if answers and answers[0].get("status") == "answered":
        valid, invalid = check_citations(answers[0], result.sources)
        first_total, first_valid = len(valid) + len(invalid), len(valid)
    if answers and answers[-1].get("status") == "answered":
        final_invalid = len(check_citations(answers[-1], result.sources)[1])

    rec = {
        **case,
        "status": result.status,
        "status_ok": result.status == case["expected_status"],
        "partial": result.partial,
        "unconfirmed": result.unconfirmed,
        "retrieved_expected": bool(expected & set(retrieved)) if expected else None,
        "cited_expected": bool(expected & set(cited_articles)) if expected else None,
        "off_topic": sorted(set(cited_ordinances) - expected_ords - {LINKED_ORDINANCE}) if expected and result.status == "answered" else [],
        "cited": cited_articles,
        "llm_calls": len(answers),
        "verify_calls": sum(kind == "verify" for kind, _ in recorder.calls),
        "mode_detected": result.mode,
        "mode_ok": result.mode == case["mode"] if case.get("mode") else None,
        "search_queries": result.search_queries,
        "news": {"title": result.news.title, "summary": result.news.summary, "keywords": result.news.keywords,
                 "error": result.news.error} if result.news else None,
        "data_requests": [{"item": r.item, "label": r.label, "purpose": r.purpose} for r in result.data_requests],
        "requests_on_expected": sum(ordinance_of(r.label) in expected_ords for r in result.data_requests),
        "regenerated": len(answers) > 1,
        "first_quotes": first_total,
        "first_valid_quotes": first_valid,
        "final_invalid_quotes": final_invalid,
        "model_said": answers[0].get("status") if answers else None,
        "topic_guard": any(TOPIC_NOTE in n for n in result.notes),
        "verify_flagged": any(VERIFY_NOTE in n for n in result.notes),
        "answer": result.answer,
        "notes": result.notes,
        "seconds": round(elapsed, 1),
    }
    rec["outcome"] = classify(rec)
    rec["ok"] = rec["outcome"] in PASSING  # 실무 통과
    rec["ok_strict"] = rec["outcome"] == "pass"
    return rec


def summarize(records: list[dict]) -> dict:
    done = [r for r in records if "error" not in r]
    ans = [r for r in done if r["expected_status"] == "answered"]
    unv = [r for r in done if r["expected_status"] == "unverifiable"]
    answered_ok = [r for r in ans if r["status"] == "answered"]
    count = lambda outcome: sum(r.get("outcome") == outcome for r in done)
    ratio = lambda a, b: a / b if b else None
    first_total = sum(r["first_quotes"] for r in done)
    moded = [r for r in done if r.get("mode_ok") is not None]
    req_answered = [r for r in done if r.get("expect_requests") and r["status"] == "answered"]
    n_items = sum(len(r["data_requests"]) for r in req_answered)
    return {
        "n": len(records),
        "errors": len(records) - len(done),
        "pass": sum(r["ok"] for r in records),
        "pass_strict": sum(r.get("ok_strict", False) for r in records),
        "status_accuracy": ratio(sum(r["status_ok"] for r in done), len(done)),
        "unverifiable_recall": ratio(sum(r["status"] == "unverifiable" for r in unv), len(unv)),
        "unverifiable_safe": ratio(sum(r.get("outcome") in PASSING for r in unv), len(unv)),
        "false_answer_hard": count("false_hard"),
        "partial_answers": count("partial"),
        "partial_in_answered": sum(bool(r.get("partial")) for r in answered_ok),
        "over_refusal": count("over_refusal"),
        "retrieval_hit": ratio(sum(bool(r["retrieved_expected"]) for r in ans), len(ans)),
        "citation_hit": ratio(sum(bool(r["cited_expected"]) for r in ans), len(ans)),
        "off_topic_rate": ratio(sum(bool(r["off_topic"]) for r in answered_ok), len(answered_ok)),
        "first_quote_valid_rate": ratio(sum(r["first_valid_quotes"] for r in done), first_total),
        "regeneration_rate": ratio(sum(r["regenerated"] for r in done), len(done)),
        "final_invalid_quotes": sum(r["final_invalid_quotes"] for r in done),
        "topic_guard": sum(r.get("topic_guard", False) for r in done),
        "verify_flagged": sum(r.get("verify_flagged", False) for r in done),
        "avg_seconds": ratio(sum(r["seconds"] for r in done), len(done)),
        "mode_accuracy": ratio(sum(bool(r["mode_ok"]) for r in moded), len(moded)),
        "request_list_rate": ratio(sum(bool(r["data_requests"]) for r in req_answered), len(req_answered)),
        "request_grounded_rate": ratio(sum(r["requests_on_expected"] for r in req_answered), n_items),
        "avg_requests": ratio(n_items, sum(bool(r["data_requests"]) for r in req_answered)),
    }


LABELS = [
    ("status_accuracy", "상태 정확도", "%"),
    ("unverifiable_safe", "확인 불가 질의 안전 처리율 (거절+고지형)", "%"),
    ("false_answer_hard", "고지 없는 오답변 (위험)", "건"),
    ("partial_answers", "고지형 부분 답변 (확인 불가 질의, 실무 통과)", "건"),
    ("partial_in_answered", "부분 답변 표시 (정답 있는 질의)", "건"),
    ("over_refusal", "과잉 거절 (답할 수 있는데 확인 불가)", "건"),
    ("retrieval_hit", "정답 조문 검색률 (근거 자료 내)", "%"),
    ("citation_hit", "정답 조문 인용률", "%"),
    ("off_topic_rate", "타 조례 혼입률", "%"),
    ("first_quote_valid_rate", "1차 인용구 원문 일치율", "%"),
    ("regeneration_rate", "재생성 발생률", "%"),
    ("final_invalid_quotes", "최종 제외된 인용", "건"),
    ("topic_guard", "조례명 점검 작동", "건"),
    ("verify_flagged", "2단계 근거 검증 작동", "건"),
    ("avg_seconds", "평균 응답 시간", "s"),
]
REQUEST_LABELS = [
    ("mode_accuracy", "질의 유형 판정 정확도 (자료요구·현안)", "%"),
    ("request_list_rate", "요구 자료 목록 작성률 (답변한 자료요구·현안 질의)", "%"),
    ("request_grounded_rate", "요구 자료 근거 적중률 (근거가 정답 조례인 항목)", "%"),
    ("avg_requests", "질의당 요구 자료 수", "n"),
]


def fmt(value, unit: str) -> str:
    if value is None:
        return "-"
    if unit == "n":
        return f"{value:.1f}건"
    return f"{value * 100:.0f}%" if unit == "%" else f"{value:.1f}s" if unit == "s" else f"{value}건"


def print_report(summary: dict, by_domain: dict[str, dict], records: list[dict], model: str) -> None:
    print(f"\n=== RAG 답변 품질 리포트 · {model} · {summary['n']}개 질의 ===")
    print(f"  실무 통과 {summary['pass']} (고지형 부분 답변 {summary['partial_answers']} 포함) · "
          f"엄격 통과 {summary['pass_strict']} · 오류 {summary['errors']}")
    for key, label, unit in LABELS + [x for x in REQUEST_LABELS if summary[x[0]] is not None]:
        print(f"  {label:<34} {fmt(summary[key], unit)}")

    cols = [("status_accuracy", "상태정확", "%"), ("citation_hit", "인용적중", "%"), ("off_topic_rate", "혼입", "%"),
            ("false_answer_hard", "위험오답", "건"), ("partial_answers", "고지형", "건"), ("over_refusal", "과잉거절", "건")]
    print("\n[도메인별]\n" + " " * 14 + "".join(f"{c[1]:>8}" for c in cols) + "   실무통과/질의")
    for dom, s in by_domain.items():
        pad = 14 - len(dom)  # 한글 2칸 보정
        print(f"{dom}{' ' * max(pad, 1)}" + "".join(f"{fmt(s[k], u):>10}" for k, _, u in cols) + f"   {s['pass']}/{s['n']}")

    partial = [r for r in records if r.get("outcome") == "partial"]
    if partial:
        print("\n[고지형 부분 답변 · 실무 통과]")
    for r in partial:
        print(f"  △ [{r['domain']}] {short(r['question'])}\n      확인되지 않은 사항: {', '.join(r['unconfirmed']) or '(답변 본문에 고지)'}"
              f" · 인용: {r['cited']}")

    listed = [r for r in records if r.get("data_requests")]
    if listed:
        print("\n[요구 자료 목록 · 자료요구·현안 질의]")
    for r in listed:
        mode = {"request": "자료요구", "issue": "현안 추론"}.get(r["mode_detected"], r["mode_detected"])
        print(f"  ▣ [{mode}] {short(r['question'])}" + (f"\n      함께 검색: {', '.join(r['search_queries'])}" if r["search_queries"] else ""))
        if r.get("news"):
            print(f"      기사: {r['news']['title']} · 현안 요지: {r['news']['summary']}")
        print(f"      인용: {r['cited']} · 근거 적중 {r['requests_on_expected']}/{len(r['data_requests'])}")
        for i, d in enumerate(r["data_requests"], 1):
            print(f"      {i}. {d['item']}  ← {d['label']}")

    failed = [r for r in records if not r["ok"]]
    if failed:
        print("\n[실패 질의]")
    for r in failed:
        if "error" in r:
            print(f"  ✗ [{r['domain']}] {short(r['question'])}\n      오류: {r['error']}")
            continue
        why = [OUTCOME_LABELS[r["outcome"]]]
        if r["outcome"] in ("over_refusal", "false_hard"):
            why.append(f"LLM 1차 판단: {r['model_said']}")
        if r["outcome"] == "missing":
            why.append(f"검색{'됨' if r['retrieved_expected'] else ' 안 됨'} · 인용: {r['cited']}")
        if r["off_topic"]:
            why.append(f"섞인 조례: {r['off_topic']}")
        if r.get("mode_ok") is False:
            why.append(f"질의 유형 {r['mode_detected']} (기대 {r['mode']})")
        print(f"  ✗ [{r['domain']}] {short(r['question'])}\n      - " + " · ".join(why))
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
    marks = {"pass": "O", "partial": "△"}
    records = []
    with connect() as conn:
        for i, case in enumerate(cases, 1):
            rec = run_case(conn, embedder, llm, case)
            records.append(rec)
            print(f"[{i:>2}/{len(cases)}] {marks.get(rec['outcome'], '✗')} {rec.get('status', 'error'):<12} "
                  f"{case.get('id', '')} {short(case['question'])}")

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
