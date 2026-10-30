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


def test_quote_from_wrong_source_is_relabeled_only_when_unambiguous():
    raw = {"status": "answered", "answer": "경형자동차는 50퍼센트 감면합니다. [별표 2]",
           "citations": [{"source_id": "S1", "quote": "경형자동차 | 50퍼센트"},  # S2의 문장을 S1으로 인용 → S2로 바로잡음
                         {"source_id": "S9", "quote": "자동차"},  # 없는 번호 + S1·S2 둘 다에 있음 → 어느 자료인지 몰라 제외
                         {"source_id": "S3", "quote": "원문에 없는 문장"},
                         {"source_id": "S2", "quote": "경형자동차 | 50퍼센트"}]}
    result = validate_answer(raw, _sources(), "test")
    assert [(c.source_id, c.relabeled_from) for c in result.citations] == [("S2", "S1"), ("S2", None)]


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


def test_user_prompt_puts_question_first_and_separates_sections():
    prompt = build_user_prompt("  감면 기준은?\n", _sources())
    assert prompt.startswith("[사용자 질문]\n감면 기준은?\n\n[근거 자료] 아래 3건(S1, S2, S3)만 근거로 쓸 수 있다.")
    q, s, r = (prompt.index(h) for h in ("[사용자 질문]", "[근거 자료]", "[답변 요청]"))
    assert q < s < prompt.index("[S1] 수원시 주차장 조례 제5조") < r
    assert prompt.count("감면 기준은?") == 2  # 맨 앞 + 답변 요청


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


# ---- 질문 대상 조례 점검 (타 조례 혼입 방지)
YOUTH_W = {"청년": 5.4, "창업": 6.5, "지원": 2.2, "사업": 4.5}  # '청년 창업 지원 사업 … 환수 사유' 질문의 조례명 핵심어


def _topic_sources() -> list[Source]:
    mk = lambda sid, name, ref, text: Source(id=sid, kind="article", label=f"{name} {ref}", ref=ref, ordinance_name=name,
                                             title="", content=text)
    return [mk("S1", "수원시 청년 창업 지원 조례", "제5조", "제5조(사업지원) 시장은 청년 창업 지원을 위하여 사업을 지원할 수 있다."),
            mk("S2", "수원시 양성평등 기본 조례", "제50조", "제50조(환수) 거짓이나 부정한 방법으로 지원받은 경우 환수한다.")]


def test_off_topic_only_when_question_points_to_an_ordinance():
    from src.rag import off_topic_ordinances

    retrieved = ["수원시 청년 창업 지원 조례", "수원시 양성평등 기본 조례"]
    assert off_topic_ordinances(YOUTH_W, ["수원시 양성평등 기본 조례"], retrieved) == ["수원시 양성평등 기본 조례"]
    assert off_topic_ordinances(YOUTH_W, ["수원시 청년 창업 지원 조례"], retrieved) == []
    # 질문이 특정 조례를 가리키지 않으면(일치도 0.5 이상인 조례 없음) 판단하지 않는다: 정답 조례도 일치도 0일 수 있음
    fund_w = {"기후": 6.9, "대응": 6.5, "기금": 5.4}
    assert off_topic_ordinances(fund_w, ["수원시 탄소중립 기본 조례"], ["수원시 탄소중립 기본 조례", "수원시 재난관리기금 운용ㆍ관리 조례"]) == []


def test_topic_guard_unverifiable_when_only_off_topic_cited():
    from src.rag import TOPIC_NOTE, Answer, Citation, apply_topic_guard

    src = _topic_sources()
    wrong = Answer("answered", "허위로 지원받으면 환수합니다.\n근거: 수원시 양성평등 기본 조례 제50조",
                   [Citation("S2", src[1].label, "거짓이나 부정한 방법으로 지원받은 경우 환수한다")], src, "test")
    result = apply_topic_guard(wrong, YOUTH_W)
    assert result.status == "unverifiable" and result.answer == UNVERIFIABLE and result.citations == []
    assert any(TOPIC_NOTE in n and "양성평등" in n for n in result.notes)


def test_topic_guard_warns_on_partial_mixing_and_ignores_on_topic():
    from src.rag import TOPIC_NOTE, Answer, Citation, apply_topic_guard

    src = _topic_sources()
    on = Citation("S1", src[0].label, "청년 창업 지원을 위하여 사업을 지원할 수 있다")
    off = Citation("S2", src[1].label, "부정한 방법으로 지원받은 경우 환수한다")
    mixed = apply_topic_guard(Answer("answered", "…", [on, off], src, "test"), YOUTH_W)
    assert mixed.status == "answered" and len(mixed.citations) == 2
    assert any(TOPIC_NOTE in n for n in mixed.notes)
    clean = apply_topic_guard(Answer("answered", "…", [on], src, "test"), YOUTH_W)
    assert clean.status == "answered" and clean.notes == []


def test_repair_feedback_lists_off_topic_ordinances():
    from src.rag import repair_feedback

    text = repair_feedback({"answer": "이전"}, [], ["수원시 양성평등 기본 조례"])
    assert "질문이 묻는 조례·사업이 아닙니다" in text and "- 수원시 양성평등 기본 조례" in text
    assert "글자 그대로 존재하지 않습니다" not in text


def test_quote_with_own_ordinance_name_prefix_passes():
    src = _committee_source()
    assert quote_in_source("수원시 각종 위원회 설치 및 운영 조례 제15조(회의록의 작성 및 공개)", src)
    assert quote_in_source("수원시 각종 위원회 설치 및 운영 조례 제15조(회의록의 작성 및 공개) ③ 제1항에 따라 작성된 회의록은", src)
    assert not quote_in_source("수원시 각종 위원회 설치 및 운영 조례", src)  # 조례명뿐인 인용은 근거가 아니다
    assert not quote_in_source("수원시 인권 기본 조례 제15조(회의록의 작성 및 공개)", src)  # 다른 조례명은 떼지 않는다
    assert not quote_in_source("수원시 각종 위원회 설치 및 운영 조례 ③ 회의록은 의원이 요구하면 공개하여야 한다.", src)


# ---- 2단계 근거 검증 (질문 대상·서술 부합)
SUBCOMMITTEE = Source(id="S1", kind="article", label="수원시 도시계획 조례 제80조", ref="제80조", ordinance_name="수원시 도시계획 조례",
                      title="분과위원회", content="제80조(분과위원회) ④ 분과위원회의 회의는 재적위원 과반수의 출석으로 개의하고 출석위원 과반수의 찬성으로 의결한다.")
RECUSAL_Q = "도시계획위원회의 심의 사항과 위원 제척·회피에 관한 조문은 어디에 있나요?"





def test_prompt_allows_partial_answers_and_limits_unverifiable():
    from src.rag import SYSTEM_PROMPT

    prompt = build_user_prompt("청년 창업 지원금 1인당 한도는?", _sources())
    assert "조례에 구체적 기준은 규정되어 있지 않음" in prompt and "관련된 조항이 전혀 없을 때만" in prompt
    assert "조례에 구체적 기준은 규정되어 있지 않음" in SYSTEM_PROMPT and "근거 자료에서 확인되지 않음" in SYSTEM_PROMPT



def test_prompt_has_request_and_issue_modes():
    from src.rag import MODE_ISSUE, MODE_REQUEST, SYSTEM_PROMPT

    assert "[자료요구 모드]" in SYSTEM_PROMPT and "[현안 추론 모드]" in SYSTEM_PROMPT and "data_requests" in SYSTEM_PROMPT
    assert "자료를 행정에 요구할 수 있습니다" in SYSTEM_PROMPT and "금지" in SYSTEM_PROMPT  # 과잉 거절 금지
    assert "사실로 단정하지 않는다" in SYSTEM_PROMPT  # 보도 사실관계는 단정하지 않음
    request = build_user_prompt("공영주차장 운영 자료요구 해줘", _sources(), MODE_REQUEST)
    assert "[자료요구 모드]" in request and request.index("[자료요구 모드]") > request.index("[답변 요청]")
    assert "본 현안과 관련하여" in build_user_prompt("공영주차장 관리 부실 지적", _sources(), MODE_ISSUE)
    assert "data_requests는 빈 배열" in build_user_prompt("감면 대상은?", _sources())


class _FakeLLM:
    name = "fake"

    def __init__(self, reply=None, error=None):
        self.reply, self.error, self.calls = reply, error, []

    def complete_json(self, system, user, schema):
        self.calls.append((system, user))
        if self.error:
            raise self.error
        return self.reply


def _recusal_answer():
    from src.rag import Answer, Citation

    cit = Citation("S1", SUBCOMMITTEE.label, "출석위원 과반수의 찬성으로 의결한다")
    return Answer("answered", "위원의 제척·회피는 제80조제4항에 규정되어 있습니다.\n근거: 수원시 도시계획 조례 제80조",
                  [cit], [SUBCOMMITTEE], "test")


def test_verify_prompt_carries_question_answer_and_cited_text():
    from src.rag import build_verify_prompt

    prompt = build_verify_prompt(RECUSAL_Q, _recusal_answer())
    assert prompt.startswith(f"[사용자 질문]\n{RECUSAL_Q}")
    assert "[S1] 수원시 도시계획 조례 제80조(분과위원회)" in prompt and "재적위원 과반수" in prompt
    assert "“출석위원 과반수의 찬성으로 의결한다”" in prompt
    assert "근거:" not in prompt  # 답변 본문만 보낸다


def test_verify_answer_returns_flagged_citations_and_tolerates_failure():
    from src.rag import VERIFY_SYSTEM_PROMPT, verify_answer

    llm = _FakeLLM({"subject": "도시계획위원회 위원 제척", "checks": [
        {"source_id": "S1", "about_subject": True, "supports_answer": False, "reason": "의결 정족수 조항이다"},
        {"source_id": "S9", "about_subject": False, "supports_answer": False, "reason": "인용되지 않은 자료"}]})
    bad = verify_answer(llm, RECUSAL_Q, _recusal_answer())
    assert list(bad) == ["S1"] and "답변 서술이 원문과 다름" in bad["S1"] and "정족수" in bad["S1"]
    assert llm.calls[0][0] == VERIFY_SYSTEM_PROMPT
    assert verify_answer(_FakeLLM(error=RuntimeError("timeout")), RECUSAL_Q, _recusal_answer()) == {}
    ok = _FakeLLM({"subject": "x", "checks": [{"source_id": "S1", "about_subject": True, "supports_answer": True, "reason": ""}]})
    assert verify_answer(ok, RECUSAL_Q, _recusal_answer()) == {}


def test_apply_verification_drops_flagged_or_turns_unverifiable():
    from src.rag import VERIFY_NOTE, apply_verification

    only = apply_verification(_recusal_answer(), {"S1": "답변 서술이 원문과 다름"})
    assert only.status == "unverifiable" and only.citations == [] and any(VERIFY_NOTE in n for n in only.notes)

    src = _sources()
    from src.rag import Answer, Citation
    mixed = Answer("answered", "…", [Citation("S1", src[0].label, "장애인 자동차: 100분의 50"),
                                      Citation("S2", src[1].label, "경형자동차 | 50퍼센트")], src, "test")
    kept = apply_verification(mixed, {"S2": "질문 대상과 다른 조문"})
    assert kept.status == "answered" and [c.source_id for c in kept.citations] == ["S1"]
    assert any(VERIFY_NOTE in n for n in kept.notes)


def test_unconfirmed_items_mark_partial_answers():
    raw = {"status": "answered", "answer": "보험금 지급 절차는 제5조와 같습니다.\n근거: 수원시 주차장 조례 제5조",
           "citations": [{"source_id": "S1", "quote": "장애인 자동차: 100분의 50"}], "unconfirmed": ["작년 지급 건수", " "]}
    result = validate_answer(raw, _sources(), "test")
    assert result.unconfirmed == ["작년 지급 건수"] and result.partial
    full = validate_answer({**raw, "unconfirmed": []}, _sources(), "test")
    assert full.unconfirmed == [] and not full.partial
    disclosed = validate_answer({**raw, "answer": "지급 건수는 조례로는 확인할 수 없습니다. 지급 절차는 제5조와 같습니다.",
                                 "unconfirmed": []}, _sources(), "test")
    assert disclosed.partial  # unconfirmed가 비어도 본문에서 스스로 밝혔으면 부분 답변


# ---- 의정활동 모드 (자료요구 · 현안 추론)

def test_detect_mode_separates_requests_issues_and_plain_questions():
    from src.rag import MODE_ISSUE, MODE_QA, MODE_REQUEST, detect_mode

    for q in ["마을만들기 사업 관련 자료요구 해줘", "청년 창업 지원 사업 현황 자료를 뽑아줘",
              "공공주차장 운영과 관련한 자료요구 해주세요", "소규모 공동주택 지원 사업 자료 요청 목록",
              "대행업체 평가 위원회의 운영 실태를 점검하기 위해 어떤 조문 근거로 어떤 자료를 요구할 수 있나요?"]:
        assert detect_mode(q) == MODE_REQUEST, q
    for q in ["수원시 공영주차장 관리 부실 지적 잇따라", "청년 창업 지원금 집행 논란",
              "[단독] 지하차도 침수 대응 늑장", "소규모 공동주택 지원금 특혜 의혹 보도"]:
        assert detect_mode(q) == MODE_ISSUE, q
    for q in ["공영주차장 주차요금 감면 대상은?", "생활폐기물 수집·운반 대행업체 평가 시 어떤 자료를 요구할 수 있나요?",
              "2026년도 공영주차장 실제 연간 총수입 금액은 얼마인가요?"]:
        assert detect_mode(q) == MODE_QA, q


def test_data_requests_keep_only_items_grounded_on_valid_citations():
    raw = {"status": "answered", "answer": "제5조에 따른 감면 실적 자료를 요구할 수 있습니다.\n근거: 수원시 주차장 조례 제5조",
           "citations": [{"source_id": "S1", "quote": "장애인 자동차: 100분의 50"}], "unconfirmed": [],
           "data_requests": [
               {"item": "최근 3년간 장애인 자동차 주차요금 감면 실적", "source_id": "S1", "purpose": "제5조 감면 이행 확인"},
               {"item": "최근 3년간 장애인 자동차 주차요금 감면 실적", "source_id": "[S1]", "purpose": "중복"},
               {"item": "경형자동차 감면 내역", "source_id": "S2", "purpose": "인용되지 않은 자료"},
               {"item": " ", "source_id": "S1", "purpose": ""}]}
    result = validate_answer(raw, _sources(), "test")
    assert [(r.item, r.source_id, r.label) for r in result.data_requests] == [
        ("최근 3년간 장애인 자동차 주차요금 감면 실적", "S1", "수원시 주차장 조례 제5조")]
    assert any("요구 자료 1건" in n for n in result.notes)
    assert validate_answer({**raw, "status": "unverifiable"}, _sources(), "test").data_requests == []


def test_apply_verification_drops_requests_of_rejected_citations():
    from src.rag import Answer, Citation, DataRequest, apply_verification

    src = _sources()
    answer = Answer("answered", "…", [Citation("S1", src[0].label, "장애인 자동차: 100분의 50"),
                                       Citation("S2", src[1].label, "경형자동차 | 50퍼센트")], src, "test",
                    data_requests=[DataRequest("감면 실적", "S1", src[0].label, ""), DataRequest("경형 감면 내역", "S2", src[1].label, "")])
    kept = apply_verification(answer, {"S2": "질문 대상과 다른 조문"})
    assert [r.item for r in kept.data_requests] == ["감면 실적"]


def test_merge_hits_interleaves_queries_without_duplicates():
    from src.rag import merge_hits
    from src.search import SearchHit

    def hit(cid):
        return SearchHit(chunk_id=cid, article_id=cid, attachment_id=None, ordinance_id=1, citation="", content="",
                         ordinance_name="", article_title="", enforcement_date=None, vec_rank=None, kw_rank=None)

    merged = merge_hits([[hit(1), hit(2), hit(3)], [hit(9), hit(1)], [hit(7)]], limit=5)
    assert [h.chunk_id for h in merged] == [1, 2, 9, 7, 3]  # 원 질문 상위 절반(2건)을 먼저, 나머지는 번갈아


def test_analyze_issue_uses_llm_and_tolerates_failure():
    from src.rag import ISSUE_SYSTEM_PROMPT, analyze_issue

    q = "수원시 공영주차장 관리 부실 지적"
    llm = _FakeLLM({"issue": "공영주차장 위탁 관리 부실", "keywords": ["공영주차장", "위탁", "공영주차장"],
                    "queries": ["공영주차장 위탁관리", q, "공영주차장 위탁관리", "주차장 관리상황 보고", "a", "b"]})
    assert analyze_issue(llm, q) == ("공영주차장 위탁 관리 부실", ["공영주차장", "위탁"], ["공영주차장 위탁관리", "주차장 관리상황 보고", "a"])
    assert llm.calls[0][0] == ISSUE_SYSTEM_PROMPT and q in llm.calls[0][1]
    assert analyze_issue(_FakeLLM(error=RuntimeError("timeout")), q) == ("", [], [])


def test_news_question_carries_article_and_user_request():
    from src.news_parser import NewsArticle
    from src.rag import NEWS_DEFAULT_REQUEST, NEWS_PROMPT_CHARS, SYSTEM_PROMPT, news_question

    url = "https://www.segye.com/newsView/20261022507676"
    news = NewsArticle(title="장애인 자립주택부터 청년 이자 지원까지", body="가" * (NEWS_PROMPT_CHARS + 10), url=url, source="url")
    asked = news_question(news, f"{url} 장애인 주거 관련 자료요구 해줘")
    assert asked.startswith(f"[뉴스 기사] {url}\n제목: 장애인 자립주택부터")
    assert "(이하 생략)" in asked and asked.endswith("사용자 요청: 장애인 주거 관련 자료요구 해줘")
    pasted = NewsArticle(title="제목", body="본문", source="text")
    assert news_question(pasted, "제목\n본문").endswith(NEWS_DEFAULT_REQUEST)
    assert "[뉴스/현안 기반 유추 분석 지침]" in SYSTEM_PROMPT and "협약서" in SYSTEM_PROMPT


def test_verify_prompt_treats_request_guidance_as_not_a_claim():
    from src.rag import VERIFY_SYSTEM_PROMPT

    assert "자료를 행정에 요구할 수 있다" in VERIFY_SYSTEM_PROMPT and "현안" in VERIFY_SYSTEM_PROMPT


def test_verification_strips_sentences_about_dropped_ordinances():
    from src.rag import Answer, Citation, apply_verification

    youth = Source(id="S1", kind="article", label="수원시 청년 창업 지원 조례 제5조", ref="제5조", ordinance_name="수원시 청년 창업 지원 조례",
                   title="사업지원", content="시장은 청년 창업을 위하여 다음 각 호의 사업을 지원할 수 있다.")
    other = Source(id="S2", kind="article", label="수원시 효사랑 지원금 지급 조례 제8조", ref="제8조", ordinance_name="수원시 효사랑 지원금 지급 조례",
                   title="환수조치", content="시장은 지원금의 전부 또는 일부를 환수할 수 있다.")
    body = ("법적 근거는 수원시 청년 창업 지원 조례 제5조입니다. 지원금의 환수는 수원시 효사랑 지원금 지급 조례 제8조에 규정되어 있습니다. "
            "보도된 사실관계는 조례로 확인할 수 없습니다.\n근거: 수원시 청년 창업 지원 조례 제5조, 수원시 효사랑 지원금 지급 조례 제8조")
    answer = Answer("answered", body, [Citation("S1", youth.label, "청년 창업을 위하여"), Citation("S2", other.label, "환수할 수 있다")],
                    [youth, other], "test")
    result = apply_verification(answer, {"S2": "질문 대상과 다른 조문: 다른 지원금의 환수 조항"})
    assert "효사랑" not in result.answer
    assert "청년 창업 지원 조례 제5조입니다." in result.answer and "확인할 수 없습니다." in result.answer
    assert result.answer.endswith("근거: 수원시 청년 창업 지원 조례 제5조")
    assert any("문장도 지웠습니다" in n for n in result.notes)


def test_issue_expansion_prompt_asks_for_article_title_terms():
    from src.rag import ISSUE_SYSTEM_PROMPT

    assert "조 제목" in ISSUE_SYSTEM_PROMPT and "도시공원 금지행위" in ISSUE_SYSTEM_PROMPT
    assert "보도 표현" in ISSUE_SYSTEM_PROMPT  # 불법·단속·논란 같은 보도 어휘는 검색어에서 뺀다


def test_wrong_source_id_is_relabeled_when_quote_exists_in_exactly_one_other_source():
    raw = {"status": "answered", "answer": "감면율은 50퍼센트입니다.\n근거: 수원시 주차장 조례 [별표 2]",
           "citations": [{"source_id": "S1", "quote": "경형자동차 | 50퍼센트"},  # 실제로는 S2([별표 2])의 문장
                         {"source_id": "S1", "quote": "원문에 없는 문장"}],
           "unconfirmed": [], "data_requests": [{"item": "경형자동차 감면 실적", "source_id": "S1", "purpose": ""}]}
    result = validate_answer(raw, _sources(), "test")
    assert [(c.source_id, c.relabeled_from) for c in result.citations] == [("S2", "S1")]
    assert [r.source_id for r in result.data_requests] == ["S2"]  # 같은 틀린 번호를 단 요구 자료도 따라간다
    assert any("자료 번호가 틀린 인용 1건" in n for n in result.notes)
    assert any("일치하지 않는 인용 1건" in n for n in result.notes)


def test_quote_skipping_enumerated_items_matches_in_order_only():
    plan = Source(id="S1", kind="article", label="수원시 주거복지 조례 제5조", ref="제5조", ordinance_name="수원시 주거복지 조례",
                  title="주거복지종합계획 수립·시행",
                  content=("제5조(주거복지종합계획 수립·시행)\n① 시장은 5년 단위로 주거복지종합계획을 수립·시행하여야 한다.\n"
                           "③ 주거복지종합계획에는 다음 각 호의 사항이 포함되어야 한다.\n1. 주거복지 정책의 기본 방향\n"
                           "2. 주거실태조사에 관한 사항\n5. 주거지원이 필요한 계층에 대한 임대주택 우선공급 및 주거비 지원 등에 관한 사항"))
    skip = "③ 주거복지종합계획에는 다음 각 호의 사항이 포함되어야 한다. 5. 주거지원이 필요한 계층에 대한 임대주택 우선공급 및 주거비 지원 등에 관한 사항"
    assert quote_in_source(skip, plan)
    assert quote_in_source("① 시장은 5년 단위로 주거복지종합계획을 수립·시행하여야 한다. ③ 주거복지종합계획에는 다음 각 호의 사항이 포함되어야 한다.", plan)
    assert not quote_in_source("5. 주거지원이 필요한 계층에 대한 임대주택 우선공급 1. 주거복지 정책의 기본 방향", plan)  # 순서가 뒤바뀜
    assert not quote_in_source("③ 주거복지종합계획에는 다음 각 호의 사항이 포함되어야 한다. 5. 청년 보증금 이자 지원", plan)  # 없는 호


def test_request_mode_adds_operating_articles_of_core_ordinance():
    from src.rag import pick_sibling_articles

    # 주차장 조례 일부 (조문 순서)
    articles = [(1, "목적"), (2, "주차장수급실태조사"), (3, "주차요금 및 가산금"), (4, "공영주차장의 위탁관리"),
                (5, "관리ㆍ감독ㆍ관리상황의 보고 등"), (6, "계약의 해지 등"), (7, "공무수행사인의 행위제한 등"),
                (8, "공영주차장의 주차요금 징수방법"), (9, "부설주차장의 설치기준")]
    picked = [aid for aid, _ in pick_sibling_articles(articles, core_id=4, exclude=set(), limit=3)]
    assert picked == [5, 6, 2]  # 가까운 운영 조문 먼저, 같은 거리면 뒤쪽 — 목적·행위제한·설치기준은 제외
    assert [aid for aid, _ in pick_sibling_articles(articles, 4, exclude={5, 6}, limit=2)] == [2, 8]
    assert pick_sibling_articles(articles, core_id=99, exclude=set()) == []


def test_topic_guard_rejects_ordinance_of_another_district():
    from src.rag import off_topic_ordinances

    q = "탑동지구 공공개발 사업 부지의 평당 감정평가액은 얼마인가요?"
    w = {"탑동": 3.0, "공공": 1.0, "개발": 1.0, "부지": 1.0}
    other = "수원시 신동지구 도시개발사업 시행 조례"
    assert off_topic_ordinances(w, [other], [other], q) == [other]
    assert off_topic_ordinances(w, [other], [other], "신동지구 토지 평가 방법은?") == []
    assert off_topic_ordinances(w, ["수원시 공유재산 관리 조례"], [other], q) == []  # 지구명이 없는 조례는 이 규칙과 무관


def test_request_checklist_goes_into_answer_body_before_basis_line():
    from src.rag import CHECKLIST_TITLE, DataRequest, with_request_checklist

    reqs = [DataRequest("최근 3년간 예술인 창작 공간 지원 실적", "S1", "수원시 예술인 복지증진에 관한 조례 제7조", ""),
            DataRequest("최근 3년간 예술인 지원 예산 편성 및 집행 내역(불용액 포함)", "S2", "수원시 예술인 복지증진에 관한 조례 제8조", "")]
    text = "제5조에 따른 사업의 추진 실적 등의 자료를 행정에 요구할 수 있습니다.\n\n근거: 수원시 예술인 복지증진에 관한 조례 제5조"
    out = with_request_checklist(text, reqs)
    assert out.endswith("근거: 수원시 예술인 복지증진에 관한 조례 제5조")
    assert f"{CHECKLIST_TITLE}\n1. 최근 3년간 예술인 창작 공간 지원 실적 (근거: 수원시 예술인 복지증진에 관한 조례 제7조)\n2. " in out
    assert with_request_checklist(out, reqs) == out and with_request_checklist(text, []) == text


def test_request_mode_for_first_person_requests_and_program_law_questions():
    from src.rag import MODE_QA, MODE_REQUEST, detect_mode

    assert detect_mode("지역예술인 지원 사업과 관련한 자료를 요구하고 싶습니다") == MODE_REQUEST
    assert detect_mode("지역예술인 지원 사업 관련 조례를 알려줘") == MODE_REQUEST
    assert detect_mode("2026년도 청년 창업 지원 사업의 실제 예산 집행 총액은 얼마인가요?") == MODE_QA


def test_outsourcing_ordinance_is_not_off_topic():
    from src.rag import LINKED_ORDINANCE, off_topic_ordinances

    w = {"예술인": 3.0, "지원": 0.5}
    retrieved = ["수원시 예술인 복지증진에 관한 조례", LINKED_ORDINANCE]
    assert off_topic_ordinances(w, [LINKED_ORDINANCE, "수원시 예술인 복지증진에 관한 조례"], retrieved) == []
