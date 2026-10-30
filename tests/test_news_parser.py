"""뉴스 입력 처리 단위 테스트 (네트워크 없이)."""

import pytest

from src.news_parser import (
    NewsArticle,
    NewsError,
    check_public_url,
    extract_url,
    looks_like_article,
    parse_article_html,
    read_news,
)

URL = "https://www.segye.com/newsView/20261022507676"
PARAGRAPH = "경기 수원시가 장애인과 무주택 청년, 신혼부부의 주거안정을 위한 맞춤형 복지 정책을 강화한다고 밝혔다."
HTML = f"""<html><head><title>사이트 제목 | 세계일보</title>
<meta property="og:title" content="장애인 자립주택부터 청년 이자 지원까지"></head>
<body><header><nav>정치 경제 사회</nav></header>
<article itemprop="articleBody">
<p>{PARAGRAPH}</p><p>시는 '청년·신혼희망터치 보증금 이자지원사업'을 통해 298가구를 선정했다.<br>연간 청년은 최대 50만원을 지원받는다.</p>
<script>var ad = 1;</script>
<p>홍길동 기자 hong@segye.com</p><p>ⓒ 세계일보 무단전재 및 재배포 금지</p>
</article><footer>회사소개</footer></body></html>"""
ARTICLE_TEXT = "수원시, 장애인 자립주택 공급\n" + "\n".join([PARAGRAPH] * 6)


def test_parse_article_html_extracts_title_and_body_without_noise():
    article = parse_article_html(HTML, URL)
    assert article.title == "장애인 자립주택부터 청년 이자 지원까지" and article.url == URL and article.source == "url"
    assert PARAGRAPH in article.body and "298가구" in article.body and "최대 50만원" in article.body
    for noise in ("var ad", "정치 경제", "회사소개", "기자 hong", "무단전재"):
        assert noise not in article.body


def test_parse_article_html_falls_back_to_paragraphs():
    html = f"<html><head><title>제목만</title></head><body><div><p>{PARAGRAPH}</p><p>짧은 줄</p></div></body></html>"
    article = parse_article_html(html)
    assert article.title == "제목만" and article.body == PARAGRAPH


def test_detects_urls_and_pasted_articles():
    assert extract_url(f"{URL}. 이 기사 관련 자료요구 해줘") == URL
    assert extract_url("공영주차장 감면 대상은?") is None
    assert looks_like_article(ARTICLE_TEXT)
    assert not looks_like_article("마을만들기 사업 관련 자료요구 해줘")


def test_read_news_uses_pasted_text_and_url_fallbacks():
    pasted = read_news(ARTICLE_TEXT)
    assert pasted.source == "text" and pasted.title == "수원시, 장애인 자립주택 공급" and PARAGRAPH in pasted.body
    assert read_news("공영주차장 감면 대상은?") is None

    ok = NewsArticle(title="t", body="b", url=URL, source="url")
    assert read_news(f"{URL} 자료요구", fetch=lambda url: ok) is ok

    def blocked(url):
        raise NewsError("기사를 불러오지 못했습니다: HTTP 403")

    fallback = read_news(f"{URL}\n{ARTICLE_TEXT}", fetch=blocked)  # 크롤링이 막혀도 함께 붙여넣은 본문으로
    assert fallback.ok and fallback.url == URL and "403" in fallback.error and PARAGRAPH in fallback.body
    failed = read_news(f"{URL} 이 기사 분석해줘", fetch=blocked)
    assert not failed.ok and failed.url == URL and "403" in failed.error


@pytest.mark.parametrize("url", ["http://127.0.0.1/admin", "http://10.0.0.5/", "http://192.168.0.1/", "http://[::1]/",
                                 "http://169.254.169.254/latest/meta-data", "file:///etc/passwd", "ftp://example.com/a"])
def test_check_public_url_blocks_internal_and_non_http(url):
    with pytest.raises(NewsError):
        check_public_url(url)


def test_short_brief_counts_as_article_but_long_question_does_not():
    brief = ("[가상 기사] 수원시 공영주차장 위탁 운영 곳곳 허점\n"
             "공영주차장에서 요금 정산 오류와 무인정산기 고장이 잇따르고 있다는 민원이 늘고 있다.\n"
             "야간 관리 인력이 배치되지 않아 불법 주차가 방치됐다는 지적도 나왔다.\n"
             "위탁 기관에 대한 관리·감독과 수입·지출 보고를 점검해야 한다는 목소리가 나온다.\n"
             "시 관계자는 위탁 계약 조건을 다시 살펴보고 현장 점검을 강화하겠다고 밝혔다.")
    assert 200 <= len(brief) < 300
    assert looks_like_article(brief)
    question = ("수원시 공영주차장 위탁 운영과 관련해서 관리수탁자의 선정 방법과 시에 납부해야 하는 금액, 그리고 주차요금 징수 방법과 "
                "감면 대상, 위탁 계약 해지 사유까지 조례에 어떻게 규정되어 있는지 한꺼번에 정리해 주실 수 있을까요? 자료요구에 쓰려고 합니다.")
    question += " 특히 위탁대행료 산출기준과 매년 조정하는 방식이 궁금합니다. 관련 별표와 서식이 있다면 해당 조문과 함께 빠짐없이 알려 주세요."
    assert len(question) >= 200 and "\n" not in question and not looks_like_article(question)  # 한 줄짜리 긴 질문
