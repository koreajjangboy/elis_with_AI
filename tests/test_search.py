from src.evaluate import article_of
from src.ordin_parser import article_label
from src.search import match_ratio, score_candidates, term_weights
from src.tokenizer import tokenize


def test_tokenizer_keeps_nouns_drops_verbs_and_boilerplate():
    assert tokenize("주차장 이용요금의 감면대상을 정한다") == "주차장 이용 요금 감면 대상"
    assert "정하" not in tokenize("시장이 정하는 바에 따른다").split()
    assert "조례" not in tokenize("수원시 청년 기본 조례").split()


def test_article_label_and_citation_parsing():
    assert article_label("3") == "제3조"
    assert article_label("3의2") == "제3조의2"
    assert article_of("수원시 주차장 조례 제4조제1항") == "수원시 주차장 조례 제4조"
    assert article_of("수원시 주차장 조례 제5조의2") == "수원시 주차장 조례 제5조의2"


def test_generic_terms_get_zero_weight():
    # 100개 조례 중 90개 이름에 '수원시'가 있으면 일반어 → 가중치 0
    w = term_weights(["수원시", "청년"], n_docs=100, df={"수원시": 90, "청년": 8})
    assert w["수원시"] == 0 and w["청년"] > 0
    assert match_ratio("수원시 주차장", w) == 0  # 일반어만 일치하면 보너스 없음
    assert match_ratio("수원시 청년 기본", w) == 1


def _row(cid, aid, oid, name_tok, title_tok, vec_r=None, kw_r=None, att_id=None, kind=None):
    return (cid, aid, att_id, oid, f"c{cid}", "본문", f"조례{oid}", "제목", None, name_tok, title_tok, vec_r, kw_r,
            None, kind)


def test_name_match_outranks_body_only_match():
    weights = {"청년": 3.0, "창업": 1.0}  # 질의 '청년 창업'
    rows = [
        _row(1, 1, 1, "도시 계획", "목적", vec_r=1, kw_r=1),  # 본문만 최상위
        _row(2, 2, 2, "청년 기본", "목적", vec_r=30),  # 조례명 일치
        _row(3, 3, 2, "청년 기본", "창업 지원", kw_r=40),  # 조례명 + 조 제목(나머지 핵심어 '창업') 일치
    ]
    hits = score_candidates(rows, weights, limit=3)
    assert [h.chunk_id for h in hits] == [3, 2, 1]
    assert hits[1].score > hits[2].score

    no_boost = score_candidates(rows, weights, limit=3, boost=False)
    assert no_boost[0].chunk_id == 1


def test_title_boost_ignores_terms_already_in_ordinance_name():
    # '재난관리기금 용도': 조례명에 있는 '기금'이 조 제목에 있어도 보너스 없음, '용도'만 평가
    weights = {"기금": 1.0, "용도": 1.0}
    rows = [
        _row(1, 1, 1, "재난 관리 기금", "기금 설치", vec_r=1),
        _row(2, 2, 1, "재난 관리 기금", "기금 용도", vec_r=2),
    ]
    hits = score_candidates(rows, weights, limit=2)
    assert hits[0].chunk_id == 2
    assert hits[1].title_match == 0


def test_dedup_article_and_cap_per_ordinance():
    weights = {"청년": 1.0}
    rows = [
        _row(1, 10, 1, "청년", "", vec_r=1),
        _row(2, 10, 1, "청년", "", vec_r=2),  # 같은 조문의 항 청크 → 제외
        _row(3, 11, 1, "청년", "", vec_r=3),
        _row(4, 12, 1, "청년", "", vec_r=4),  # 같은 조례 3번째 → 제외
        _row(5, 20, 2, "도시", "", vec_r=5),
    ]
    hits = score_candidates(rows, weights, limit=5, max_per_ordinance=2)
    assert [h.chunk_id for h in hits] == [1, 3, 5]


def test_attachment_pieces_deduplicated_per_attachment():
    weights = {"관할": 1.0}
    rows = [
        _row(1, None, 1, "통반", "관할", vec_r=1, att_id=7),
        _row(2, None, 1, "통반", "관할", vec_r=2, att_id=7),  # 같은 별표의 다른 조각 → 제외
        _row(3, None, 2, "통반", "관할", vec_r=3, att_id=8),
    ]
    hits = score_candidates(rows, weights, limit=5)
    assert [h.chunk_id for h in hits] == [1, 3]


def test_blank_forms_penalized_unless_form_intent():
    weights = {"출산": 1.0}
    rows = [
        _row(1, None, 1, "출산", "출산 신청서", vec_r=1, att_id=7, kind="서식"),  # 빈 신청서 양식
        _row(2, 20, 1, "출산", "지원 대상", vec_r=2),  # 조문
    ]
    assert [h.chunk_id for h in score_candidates(rows, weights, limit=2)] == [2, 1]
    assert [h.chunk_id for h in score_candidates(rows, weights, limit=2, form_intent=True)] == [1, 2]


def test_article_of_handles_attachment_citations():
    assert article_of("수원시 주차장 조례 별표 1 공영주차장 주차요금표 (1/3)") == "수원시 주차장 조례 별표 1 공영주차장 주차요금표"
    assert article_of("수원시 주차장 조례 [별표 2] 설치기준(제14조 관련)") == "수원시 주차장 조례 [별표 2] 설치기준(제14조 관련)"
