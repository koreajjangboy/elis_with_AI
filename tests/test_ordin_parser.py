import xml.etree.ElementTree as ET

from src.ordin_parser import (
    extract_law_references,
    parse_article_text,
    parse_ordinance_detail,
    parse_search_result,
)

LIST_XML = """<?xml version="1.0" encoding="UTF-8"?>
<OrdinSearch>
  <totalCnt>1234</totalCnt>
  <law id="1">
    <자치법규일련번호>1500001</자치법규일련번호>
    <자치법규명>서울특별시 주차장 설치 및 관리 조례</자치법규명>
    <자치법규ID>2000001</자치법규ID>
    <공포일자>20240105</공포일자>
    <공포번호>9000</공포번호>
    <지자체기관명>서울특별시</지자체기관명>
    <자치법규종류>조례</자치법규종류>
    <시행일자>20240105</시행일자>
  </law>
</OrdinSearch>"""

DETAIL_XML = """<?xml version="1.0" encoding="UTF-8"?>
<LawService>
  <자치법규기본정보>
    <자치법규ID>2000001</자치법규ID>
    <자치법규명>서울특별시 주차장 설치 및 관리 조례</자치법규명>
    <지자체기관명>서울특별시</지자체기관명>
    <자치법규종류>C0001</자치법규종류>
    <공포일자>20240105</공포일자>
    <시행일자>20240105</시행일자>
  </자치법규기본정보>
  <조문>
    <조><조문번호>000000</조문번호><조문여부>N</조문여부><조내용>제1장 총칙</조내용></조>
    <조><조문번호>000100</조문번호><조문여부>Y</조문여부><조제목>목적</조제목>
      <조내용>제1조(목적) 이 조례는 「주차장법」 제12조에 따라 필요한 사항을 규정함을 목적으로 한다.</조내용></조>
    <조><조문번호>000200</조문번호><조문여부>Y</조문여부><조제목>감면</조제목>
      <조내용>제2조(감면) ① 다음 각 호의 어느 하나에 해당하는 경우 요금을 감면한다.
1. 「장애인복지법」에 따른 장애인
2. 국가유공자
가. 본인
나. 유족
② 감면 절차는 규칙으로 정한다.</조내용></조>
  </조문>
  <부칙><부칙공포일자>20240105</부칙공포일자><부칙공포번호>9000</부칙공포번호><부칙내용>이 조례는 공포한 날부터 시행한다.</부칙내용></부칙>
</LawService>"""


def test_parse_search_result():
    result = parse_search_result(ET.fromstring(LIST_XML))
    assert result["total_count"] == "1234"
    item = result["items"][0]
    assert item["serial_no"] == "1500001"
    assert item["promulgation_date"] == "2024-01-05"


def test_parse_article_paragraphs_items_subitems():
    art = parse_article_text("제2조(감면) ① 다음 경우 감면한다.\n1. 장애인\n2. 국가유공자\n가. 본인\n나. 유족\n② 절차는 규칙으로 정한다.")
    assert art["article_no"] == "2" and art["title"] == "감면"
    p1, p2 = art["paragraphs"]
    assert p1["number"] == 1 and [i["number"] for i in p1["items"]] == ["1", "2"]
    assert [s["number"] for s in p1["items"][1]["subitems"]] == ["가", "나"]
    assert p2["number"] == 2 and p2["items"] == []


def test_single_paragraph_article_and_branch_number():
    art = parse_article_text("제3조의2(정의) 이 조례에서 사용하는 용어는 2024. 1. 1. 기준이다.")
    assert art["article_no"] == "3의2"
    assert len(art["paragraphs"]) == 1
    assert art["paragraphs"][0]["number"] is None
    assert art["paragraphs"][0]["items"] == []  # 문장 속 날짜를 호로 오인하지 않음


def test_extract_law_references_dedup():
    assert extract_law_references("「주차장법」 및 「주차장법 시행령」, 「주차장법」") == ["주차장법", "주차장법 시행령"]


def test_parse_ordinance_detail():
    o = parse_ordinance_detail(ET.fromstring(DETAIL_XML), org_code="6110000")
    assert o["org_code"] == "6110000"
    assert o["kind"] == "조례"
    assert o["chapters"] == ["제1장 총칙"]
    assert len(o["articles"]) == 2
    assert o["articles"][0]["chapter"] == "제1장 총칙"
    assert o["articles"][1]["paragraphs"][0]["items"][0]["text"] == "「장애인복지법」에 따른 장애인"
    assert o["referenced_laws"] == ["주차장법", "장애인복지법"]
    assert o["addenda"][0]["text"].startswith("이 조례는")
