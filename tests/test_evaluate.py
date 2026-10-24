from src.evaluate import summarize as summarize_search
from src.evaluate_rag import ordinance_of, summarize


def test_ordinance_of_strips_article():
    assert ordinance_of("수원시 주차장 조례 제12조") == "수원시 주차장 조례"
    assert ordinance_of("수원시 폐기물 관리에 관한 조례 제22조의3") == "수원시 폐기물 관리에 관한 조례"
    assert ordinance_of("수원시 재정사업 종결 관리 기본 조례 제5조") == "수원시 재정사업 종결 관리 기본 조례"


def _rec(expected_status, status, cited_expected=None, off_topic=(), first=(0, 0), regenerated=False, ok=True):
    return {"expected_status": expected_status, "status": status, "status_ok": expected_status == status,
            "retrieved_expected": cited_expected, "cited_expected": cited_expected, "off_topic": list(off_topic),
            "first_quotes": first[0], "first_valid_quotes": first[1], "regenerated": regenerated, "topic_guard": False,
            "final_invalid_quotes": 0, "ok": ok, "seconds": 1.0}


def test_rag_summary_counts_false_answers_and_mixing():
    records = [
        _rec("answered", "answered", True, first=(2, 2)),
        _rec("answered", "answered", True, off_topic=["다른 조례"], first=(2, 1), regenerated=True, ok=False),
        _rec("answered", "unverifiable", False, ok=False),
        _rec("unverifiable", "unverifiable"),
        _rec("unverifiable", "answered", first=(1, 1), ok=False),
        {"expected_status": "answered", "error": "Timeout", "ok": False},
    ]
    s = summarize(records)
    assert s["n"] == 6 and s["errors"] == 1 and s["pass"] == 2
    assert s["false_answer"] == 1 and s["over_refusal"] == 1
    assert s["status_accuracy"] == 3 / 5
    assert s["unverifiable_recall"] == 1 / 2
    assert s["citation_hit"] == 2 / 3
    assert s["off_topic_rate"] == 1 / 2  # answered로 답한 정답 질의 2건 중 1건
    assert s["first_quote_valid_rate"] == 4 / 5
    assert s["regeneration_rate"] == 1 / 5


def test_search_summary_handles_missing_query_type():
    s = summarize_search([{"type": "article", "Hit@1": True, "Hit@5": True, "MRR@10": 1.0}])
    assert s["P@5"] is None and s["Hit@1"] == 1.0 and s["n_article"] == 1 and s["n_ordinance"] == 0


def test_ordinance_of_handles_annex_labels():
    assert ordinance_of("수원시 주차장 조례 별표 1 공영주차장 주차요금표") == "수원시 주차장 조례"
    assert ordinance_of("수원시 주차장 조례 [별표 2] 감면 기준(제5조 관련)") == "수원시 주차장 조례"
