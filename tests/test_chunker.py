from src.chunker import ARTICLE_SPLIT_CHARS, build_articles, is_local_ordinance, strip_revision_notes


def _ordinance(*raw_texts: str) -> dict:
    return {
        "name": "수원시 주차장 조례",
        "articles": [
            {"article_no": "", "title": "", "key": f"{i:04d}00", "chapter": "제1장 총칙", "raw_text": t}
            for i, t in enumerate(raw_texts, 1)
        ],
        "addenda": [{"promulgation_date": "2024-01-05", "promulgation_no": "9000", "text": "이 조례는 공포한 날부터 시행한다."}],
    }


def test_strip_revision_notes():
    text, notes = strip_revision_notes("① 시장은 감면한다.<개정 2024. 12. 30.>\n② 다만, 예외로 한다. <단서 신설 2026.09.30.>\n[본조신설 2020. 1. 1.]")
    assert text == "① 시장은 감면한다.\n② 다만, 예외로 한다."
    assert notes == ["<개정 2024. 12. 30.>", "<단서 신설 2026.09.30.>", "[본조신설 2020. 1. 1.]"]


def test_strip_moved_notes_and_html_but_keep_annex_refs():
    text, notes = strip_revision_notes('<p style="font-family:굴림체">③ [별표 1]에 따른다.<종전 제2항에서 이동 2020.1.1.><2016.08.19. 호 신설></p>')
    assert text == "③ [별표 1]에 따른다."
    assert notes == ["<종전 제2항에서 이동 2020.1.1.>", "<2016.08.19. 호 신설>"]


def test_short_article_single_chunk_with_header():
    (article, addendum) = build_articles(_ordinance("제1조(목적) 이 조례는 「주차장법」에 따른 사항을 규정한다.<개정 2024. 1. 5.>"))
    assert [c.granularity for c in article.chunks] == ["article"]
    chunk = article.chunks[0]
    assert chunk.citation == "수원시 주차장 조례 제1조"
    assert chunk.embed_text.startswith("[수원시 주차장 조례]\n제1조(목적)")
    assert "<개정" not in chunk.content and article.revision_notes == ["<개정 2024. 1. 5.>"]
    assert article.law_references == ["주차장법"]
    assert addendum.article_no == "부칙" and addendum.chunks == []


def test_long_article_adds_paragraph_chunks():
    filler = "가" * (ARTICLE_SPLIT_CHARS // 2)
    (article, _) = build_articles(_ordinance(f"제5조(감면) ① {filler}\n1. 장애인\n가. 본인\n② {filler}"))
    assert [c.granularity for c in article.chunks] == ["article", "paragraph", "paragraph"]
    p1 = article.chunks[1]
    assert p1.citation == "수원시 주차장 조례 제5조제1항"
    assert p1.embed_text.startswith("[수원시 주차장 조례] 제5조(감면)\n①")
    assert "1. 장애인\n가. 본인" in p1.content  # 호·목은 항 청크에 포함


def test_deleted_paragraph_skipped_in_paragraph_chunks():
    filler = "가" * ARTICLE_SPLIT_CHARS
    (article, _) = build_articles(_ordinance(f"제5조(감면) ① {filler}\n② <삭제 2020. 1. 1.>\n③ 다른 내용"))
    assert [c.paragraph_no for c in article.chunks if c.granularity == "paragraph"] == [1, 3]


def test_deleted_article_has_no_chunk():
    (article, _) = build_articles(_ordinance("제3조 삭제 <2020. 1. 1.>"))
    assert article.article_no == "3" and article.chunks == []


def test_is_local_ordinance():
    assert is_local_ordinance("수원시 사무위탁 조례", "경기도 수원시")
    assert is_local_ordinance("경기도 주차장 조례", "경기도 수원시")
    assert not is_local_ordinance("주차장법 시행규칙", "경기도 수원시")


def test_deleted_paragraph_range_skipped():
    filler = "가" * ARTICLE_SPLIT_CHARS
    (article, _) = build_articles(_ordinance(f"제23조(휴가) ① {filler}\n⑥ ~ ⑨ <삭제 2025.07.10.>\n⑩ 다른 내용"))
    assert [c.paragraph_no for c in article.chunks if c.granularity == "paragraph"] == [1, 10]
