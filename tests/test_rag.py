from src.chunker import UNEXTRACTED_NOTE
from src.rag import (
    TRUNCATED_MARK,
    UNVERIFIABLE,
    Source,
    annex_ref,
    build_user_prompt,
    mentioned_annexes,
    quote_in_source,
    validate_answer,
)

ARTICLE = "제5조(요금의 감면) ① 시장은 다음 각 호의 자동차에 대하여 주차요금을 감면한다.\n1. 장애인 자동차: 100분의 50\n② 감면 기준은 별표 2와 같다."
ANNEX = "[별표 2] 감면 기준(제5조 관련)\n구분 | 감면율\n경형자동차 | 50퍼센트"


def _sources() -> list[Source]:
    return [
        Source(id="S1", kind="article", label="수원시 주차장 조례 제5조", ref="제5조", ordinance_name="수원시 주차장 조례",
               title="요금의 감면", content=ARTICLE, article_id=10),
        Source(id="S2", kind="attachment", label="수원시 주차장 조례 [별표 2] 감면 기준(제5조 관련)", ref="[별표 2]",
               ordinance_name="수원시 주차장 조례", title="[별표 2] 감면 기준(제5조 관련)", content=ANNEX,
               attachment_id=7, attachment_kind="별표", related_to="S1", extraction_method="document-parse"),
        Source(id="S3", kind="attachment", label="수원시 주차장 조례 [별표 3] 배치도", ref="[별표 3]",
               ordinance_name="수원시 주차장 조례", title="[별표 3] 배치도", content=UNEXTRACTED_NOTE,
               attachment_id=8, extracted=False),
    ]


def test_unverifiable_forces_exact_text_and_empty_citations():
    raw = {"status": "unverifiable", "answer": "아마 50% 정도일 것입니다.", "citations": [{"source_id": "S1", "quote": "주차요금을 감면한다"}]}
    result = validate_answer(raw, _sources(), "test")
    assert result.status == "unverifiable"
    assert result.answer == UNVERIFIABLE
    assert result.citations == []


def test_valid_citation_kept_and_reference_present():
    raw = {"status": "answered",
           "answer": "장애인 자동차는 주차요금의 100분의 50을 감면합니다(수원시 주차장 조례 제5조제1항).\n근거: 수원시 주차장 조례 제5조",
           "citations": [{"source_id": "S1", "quote": "1. 장애인 자동차: 100분의 50"}]}
    result = validate_answer(raw, _sources(), "test")
    assert result.status == "answered"
    assert [c.label for c in result.citations] == ["수원시 주차장 조례 제5조"]
    assert result.notes == []


def test_fabricated_quote_turns_into_unverifiable():
    raw = {"status": "answered", "answer": "전기차는 전액 면제됩니다(제5조).",
           "citations": [{"source_id": "S1", "quote": "전기자동차는 주차요금을 면제한다"}]}
    result = validate_answer(raw, _sources(), "test")
    assert result.status == "unverifiable" and result.answer == UNVERIFIABLE and result.citations == []
    assert any("일치하지 않는 인용 1건" in n for n in result.notes)


def test_quote_from_wrong_source_or_unknown_id_is_dropped():
    raw = {"status": "answered", "answer": "경형자동차는 50퍼센트 감면합니다. [별표 2]",
           "citations": [{"source_id": "S1", "quote": "경형자동차 | 50퍼센트"},  # S2의 문장을 S1으로 인용
                         {"source_id": "S9", "quote": "경형자동차"},
                         {"source_id": "S2", "quote": "경형자동차 | 50퍼센트"}]}
    result = validate_answer(raw, _sources(), "test")
    assert [c.source_id for c in result.citations] == ["S2"]


def test_placeholder_attachment_cannot_be_evidence():
    raw = {"status": "answered", "answer": "배치도는 [별표 3]과 같습니다.",
           "citations": [{"source_id": "S3", "quote": "원문 파일을 확인하세요"}]}
    assert validate_answer(raw, _sources(), "test").status == "unverifiable"


def test_missing_reference_is_appended():
    raw = {"status": "answered", "answer": "장애인 자동차는 절반을 감면합니다.",
           "citations": [{"source_id": "[S1]", "quote": "장애인 자동차: 100분의 50"},
                         {"source_id": "S2", "quote": "경형자동차 | 50퍼센트"}]}
    result = validate_answer(raw, _sources(), "test")
    assert result.answer.endswith("근거: 수원시 주차장 조례 제5조, 수원시 주차장 조례 [별표 2] 감면 기준(제5조 관련)")


def test_quote_matching_ignores_whitespace_and_handles_ellipsis():
    src = _sources()[0]
    assert quote_in_source("다음 각 호의   자동차에\n대하여", src)
    assert quote_in_source("시장은 다음 각 호의 … 주차요금을 감면한다", src)
    assert not quote_in_source("시장은 모든 자동차의 주차요금을 감면한다", src)
    truncated = Source(id="S1", kind="article", label="x", ref="x", ordinance_name="x", title="", content="본문" + TRUNCATED_MARK)
    assert not quote_in_source("이하 생략", truncated)


def test_annex_refs_and_mentions():
    assert annex_ref("[별표 1] 공영주차장 요금표(제4조제1항 관련)") == "[별표 1]"
    assert annex_ref("〔별표 2의3〕 기준") == "〔별표 2의3〕"
    assert annex_ref("별표 1 공영주차장 주차요금표") == "별표 1"
    assert mentioned_annexes("요금은 별표 1과 같고, 기준은 별표2의3 및 별표 4-1에 따른다.") == {(1, 0), (2, 3), (4, 1)}


def test_user_prompt_marks_ocr_related_and_unextracted():
    prompt = build_user_prompt("감면 기준은?", _sources())
    assert "[S1] 수원시 주차장 조례 제5조(요금의 감면)  <조문>" in prompt
    assert "<별표, S1의 관련 별표, OCR 추출본>" in prompt
    assert "<별표, 본문 미추출>" in prompt
    assert prompt.endswith("질문: 감면 기준은?")


def test_quote_tolerates_llm_ellipsis_and_wrapping_quotes():
    src = _sources()[0]
    assert quote_in_source("시장은 다음 각 호의 자동차에 대하여…", src)
    assert quote_in_source("…주차요금을 감면한다.", src)
    assert quote_in_source("시장은 다음 각 호의 자동차에 대하여....", src)  # 마침표 하나가 더 붙은 '...'
    assert quote_in_source("시장은 다음 각 호의 자동차에 대하여 . . .", src)
    assert quote_in_source("① 시장은 다음 각 호의⋯주차요금을 감면한다", src)
    assert quote_in_source("시장은 (중략) 주차요금을 감면한다", src)
    assert quote_in_source("시장은 […] 주차요금을 감면한다", src)
    assert quote_in_source("“1. 장애인 자동차: 100분의 50”", src)
    assert quote_in_source("'② 감면 기준은 별표 2와 같다.'", src)


def test_quote_matching_stays_strict():
    src = _sources()[0]
    assert not quote_in_source("…", src)  # 생략 표기뿐인 인용
    assert not quote_in_source("“ ... ”", src)
    assert not quote_in_source("시장은 다음 각 호의 자동차에 대하여 주차요금을 면제한다…", src)  # 감면 → 면제
    assert not quote_in_source("주차요금을 감면한다 … 시장은 다음 각 호의", src)  # 조각 순서가 원문과 다름
    assert not quote_in_source("시장은 … 전기자동차", src)  # 조각 하나라도 원문에 없으면 탈락
    assert not quote_in_source("시장은 … 다", src)  # 1글자 조각으로 끼워 맞추기 방지
    assert not quote_in_source("시장은 생략 주차요금을 감면한다", src)  # 괄호 없는 '생략'은 원문 단어로 취급


def test_split_quote():
    from src.rag import split_quote

    assert split_quote("“가나다…라마바.”") == ["가나다", "라마바."]
    assert split_quote("(이하 생략) 가나다 [중략] 라마바 ....") == ["가나다", "라마바"]
    assert split_quote("「장애인복지법」 제32조") == ["「장애인복지법」 제32조"]


# 실제 수원시 주차장 조례 제4조처럼 본문 중간에 괄호형 개정 이력이 남아 있는 원문
REVISED = ("① 공영주차장의 주차요금은 별표1과 같이 하고, 민영주차장의 주차요금은 해당 주차장 관리자가 따로 정한다. "
           "(개정 2009.01.08) (개정 2011.06.17)\n"
           "② 「도로교통법」제2조제22호에 따른 긴급자동차에 대해서는 주차요금을 면제한다. <개정 2012. 11. 14.>\n"
           "⑥ 시장은 제1항에도 불구하고 개별 공영주차장에 대하여 주차요금을 감면하여 운영할 수 있다. "
           "(개정 2008.07.14)(종전 제5항에서 이동 2019.12.31) (개정 2024.11.14.) [본조신설 2020. 1. 1.]\n"
           "⑦ 감면 대상(제5조 관련)은 시장이 정한다.")


def _revised_source() -> Source:
    return Source(id="S1", kind="article", label="수원시 주차장 조례 제4조", ref="제4조",
                  ordinance_name="수원시 주차장 조례", title="주차요금 및 가산금", content=REVISED)


def test_quote_spanning_revision_notes_passes():
    src = _revised_source()
    # 개정 표기를 빼고 다음 항까지 자연스럽게 이어 붙인 인용
    assert quote_in_source("해당 주차장 관리자가 따로 정한다. ② 「도로교통법」제2조제22호에 따른 긴급자동차에 대해서는", src)
    assert quote_in_source("주차요금을 면제한다. ⑥ 시장은 제1항에도 불구하고", src)
    assert quote_in_source("감면하여 운영할 수 있다. ⑦ 감면 대상", src)  # 괄호형·꺾쇠형·대괄호형이 한꺼번에 낀 구간
    # 개정 표기를 그대로 포함한 인용, 표기를 다르게 옮겨 적은 인용도 통과
    assert quote_in_source("따로 정한다. (개정 2009.01.08) (개정 2011.06.17)", src)
    assert quote_in_source("주차요금을 면제한다. <개정 2012.11.14.>", src)


def test_revision_note_stripping_keeps_strictness():
    src = _revised_source()
    assert not quote_in_source("(개정 2009.01.08)", src)  # 이력 표기뿐인 인용은 근거가 아니다
    assert not quote_in_source("긴급자동차에 대해서는 주차요금을 감면한다.", src)  # 본문 불일치(면제 → 감면)
    assert not quote_in_source("감면 대상은 시장이 정한다", src)  # '(제5조 관련)'은 이력이 아니므로 지우지 않는다
    assert quote_in_source("감면 대상(제5조 관련)은 시장이 정한다", src)


def test_match_text_only_affects_matching():
    from src.rag import _match_text

    src = _revised_source()
    assert "(개정 2009.01.08)" in src.content  # 원문(content)은 그대로
    assert _match_text("정한다. (개정 2009.01.08) <개정 2012. 11. 14.> [본조신설 2020. 1. 1.] (종전 제5항에서 이동 2019.12.31)") == "정한다."
    assert _match_text("(이하 \"시장\"이라 한다)") == "(이하\"시장\"이라한다)"


# 실제 수원시 각종 위원회 설치 및 운영 조례 제15조 (자료요구 시나리오에서 LLM이 ③항 앞에 조 제목을 붙여 인용함)
COMMITTEE = ("제15조(회의록의 작성 및 공개)\n① 시장은 다른 법령ㆍ조례 등에 특별한 규정이 있는 경우를 제외하고, 회의의 개최일시, "
             "장소, 출석위원, 심의안건, 발언내용, 회의결과 등을 포함한 위원회 회의록을 기록ㆍ보관하여야 한다.\n"
             "② 회의록은 위원장이 지정한 위원이 서명하여야 한다.\n"
             "③ 제1항에 따라 작성된 회의록은 정당한 청구인의 청구가 있는 경우에는 공개하여야 한다.")


def _committee_source() -> Source:
    return Source(id="S1", kind="article", label="수원시 각종 위원회 설치 및 운영 조례 제15조", ref="제15조",
                  ordinance_name="수원시 각종 위원회 설치 및 운영 조례", title="회의록의 작성 및 공개", content=COMMITTEE)


def test_quote_with_article_heading_before_later_paragraph_passes():
    src = _committee_source()
    assert quote_in_source("제15조(회의록의 작성 및 공개) ③ 제1항에 따라 작성된 회의록은 정당한 청구인의 청구가 있는 경우에는 공개하여야 한다.", src)
    assert quote_in_source("제15조(회의록의 작성 및 공개)", src)
    # 제목만 떼어 줄 뿐 본문은 여전히 글자 그대로여야 한다
    assert not quote_in_source("제15조(회의록의 작성 및 공개) ③ 회의록은 의원이 요구하면 공개하여야 한다.", src)
    assert not quote_in_source("제15조(회의록의 공개) ③ 제1항에 따라 작성된 회의록은", src)  # 제목이 다르면 분리하지 않음
    # 다른 자료의 제목을 붙인 인용은 그대로 불일치
    other = Source(id="S2", kind="article", label="x 제9조", ref="제9조", ordinance_name="x", title="회의록", content="제9조(회의록) 회의록을 작성한다.")
    assert not quote_in_source("제15조(회의록의 작성 및 공개) 회의록을 작성한다.", other)


def test_inline_article_mention_does_not_replace_basis_line():
    raw = {"status": "answered",
           "answer": "수원시 주차장 조례 제5조제1항에 따라 장애인 자동차는 100분의 50을 감면합니다.",
           "citations": [{"source_id": "S1", "quote": "1. 장애인 자동차: 100분의 50"}]}
    result = validate_answer(raw, _sources(), "test")
    assert result.answer.splitlines()[-1] == "근거: 수원시 주차장 조례 제5조"


def test_basis_line_ref_must_not_match_longer_article_number():
    from src.rag import mentions_ref

    assert mentions_ref("근거: 수원시 주차장 조례 제5조", "제5조")
    assert mentions_ref("근거: 수원시 주차장 조례 제5조제1항", "제5조")
    assert not mentions_ref("근거: 수원시 주차장 조례 제15조", "제5조")
    assert not mentions_ref("근거: 수원시 주차장 조례 제5조의2", "제5조")
    raw = {"status": "answered", "answer": "감면합니다.\n근거: 수원시 주차장 조례 제15조",
           "citations": [{"source_id": "S1", "quote": "1. 장애인 자동차: 100분의 50"},
                         {"source_id": "S2", "quote": "경형자동차 | 50퍼센트"}]}
    last = validate_answer(raw, _sources(), "test").answer.splitlines()[-1]
    assert last == "근거: 수원시 주차장 조례 제15조, 수원시 주차장 조례 제5조, 수원시 주차장 조례 [별표 2] 감면 기준(제5조 관련)"
