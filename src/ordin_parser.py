"""자치법규 API XML 응답 파서.

응답 태그의 정확한 중첩 구조가 가이드에 명시되어 있지 않으므로,
태그 이름으로 트리 어디서든 찾는 방식으로 방어적으로 파싱한다.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET


def _text(el: ET.Element | None) -> str:
    return (el.text or "").strip() if el is not None else ""


def _find_text(root: ET.Element, *tags: str) -> str:
    """여러 후보 태그 중 처음 값이 있는 것을 반환."""
    for tag in tags:
        for el in root.iter(tag):
            if _text(el):
                return _text(el)
    return ""


def _fmt_date(value: str) -> str:
    return f"{value[:4]}-{value[4:6]}-{value[6:8]}" if re.fullmatch(r"\d{8}", value) else value


# ---------------------------------------------------------------- 목록

LIST_FIELDS = {
    "serial_no": "자치법규일련번호",
    "ordin_id": "자치법규ID",
    "name": "자치법규명",
    "org_name": "지자체기관명",
    "kind": "자치법규종류",
    "promulgation_date": "공포일자",
    "promulgation_no": "공포번호",
    "enforcement_date": "시행일자",
    "revision_type": "제개정구분명",
    "field": "자치법규분야명",
    "detail_link": "자치법규상세링크",
}


def parse_search_result(root: ET.Element) -> dict:
    items = []
    # 항목 태그는 보통 <law>지만, 이름과 무관하게 자치법규명을 가진 요소를 항목으로 본다
    rows = [el for el in root.iter() if el.find("자치법규명") is not None]
    for el in rows:
        item = {key: _text(el.find(tag)) for key, tag in LIST_FIELDS.items()}
        for key in ("promulgation_date", "enforcement_date"):
            item[key] = _fmt_date(item[key])
        items.append(item)
    return {"total_count": _find_text(root, "totalCnt"), "items": items}


# ---------------------------------------------------------------- 본문 텍스트 구조화

PARAGRAPH_MARKS = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳㉑㉒㉓㉔㉕㉖㉗㉘㉙㉚"
_PARA_SPLIT = re.compile(f"(?=[{PARAGRAPH_MARKS}])")
_ITEM_RE = re.compile(r"^(\d+(?:의\d+)?)\.\s*(.*)")  # 호: 1. / 2의2.
_SUBITEM_RE = re.compile(r"^([가-하])\.\s*(.*)")  # 목: 가. 나.
_ARTICLE_HEAD_RE = re.compile(r"^제(\d+)조(?:의(\d+))?\s*(?:\(([^)]*)\))?\s*")
_LAW_REF_RE = re.compile(r"「([^」]+)」")


def _split_items(text: str) -> tuple[str, list[dict]]:
    """항 텍스트를 (본문, 호 목록)으로 나눈다. 호/목은 줄 시작 기준으로만 인식한다
    (문장 중간의 '2024. 1. 1.' 같은 날짜를 호로 오인하지 않기 위함)."""
    body_lines: list[str] = []
    items: list[dict] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if m := _ITEM_RE.match(line):
            items.append({"number": m.group(1), "text": m.group(2), "subitems": []})
        elif (m := _SUBITEM_RE.match(line)) and items:
            items[-1]["subitems"].append({"number": m.group(1), "text": m.group(2)})
        elif items:
            target = items[-1]["subitems"][-1] if items[-1]["subitems"] else items[-1]
            target["text"] += " " + line
        else:
            body_lines.append(line)
    return " ".join(body_lines), items


def parse_article_text(text: str) -> dict:
    """조내용 텍스트를 조 머리(번호·제목)와 항/호/목 구조로 분해."""
    text = text.strip()
    head = _ARTICLE_HEAD_RE.match(text)
    article_no, title = "", ""
    if head:
        article_no = head.group(1) + (f"의{head.group(2)}" if head.group(2) else "")
        title = head.group(3) or ""
        text = text[head.end():]

    segments = [s for s in _PARA_SPLIT.split(text) if s.strip()]
    paragraphs = []
    lead = ""
    for seg in segments:
        seg = seg.strip()
        if seg[0] in PARAGRAPH_MARKS:
            body, items = _split_items(seg[1:])
            paragraphs.append({"number": PARAGRAPH_MARKS.index(seg[0]) + 1, "text": body, "items": items})
        else:
            lead = seg  # 항 기호 없이 시작하는 단일 항 조문

    if lead:
        body, items = _split_items(lead)
        paragraphs.insert(0, {"number": None, "text": body, "items": items})
    return {"article_no": article_no, "title": title, "paragraphs": paragraphs}


def extract_law_references(text: str) -> list[str]:
    """본문에서 「」로 인용된 법령명을 등장 순서대로 중복 없이 추출."""
    return list(dict.fromkeys(_LAW_REF_RE.findall(text)))


# ---------------------------------------------------------------- 본문

def _collect_article_text(el: ET.Element) -> str:
    """조내용 + (있다면) 하위 항/호 요소 텍스트를 합친다."""
    parts = [_text(el.find("조내용"))]
    for para in el.iter("항"):
        parts.append(_text(para.find("항내용")) or _text(para))
        for item in para.iter("호"):
            parts.append(_text(item.find("호내용")) or _text(item))
    return "\n".join(p for p in parts if p)


# 본문 API는 자치법규종류를 코드로 반환한다 (목록 API는 이름으로 반환)
KIND_CODES = {
    "C0001": "조례",
    "C0002": "규칙",
    "C0003": "훈령",
    "C0004": "예규",
    "C0006": "기타",
    "C0010": "고시",
    "C0011": "의회규칙",
}


def parse_ordinance_detail(root: ET.Element, org_code: str = "") -> dict:
    info = {
        "ordin_id": _find_text(root, "자치법규ID"),
        "serial_no": _find_text(root, "자치법규일련번호"),
        "name": _find_text(root, "자치법규명"),
        "org_name": _find_text(root, "지자체기관명"),
        # 응답에는 자치단체 코드가 없으므로 요청에 사용한 org 코드를 기록한다
        "org_code": org_code,
        "kind": KIND_CODES.get(kind := _find_text(root, "자치법규종류"), kind),
        "promulgation_date": _fmt_date(_find_text(root, "공포일자")),
        "promulgation_no": _find_text(root, "공포번호"),
        "enforcement_date": _fmt_date(_find_text(root, "시행일자")),
        "department": _find_text(root, "담당부서명"),
        "revision_info": _find_text(root, "제개정정보"),
    }

    articles, structure = [], []
    for el in root.iter("조"):
        if el.find("조내용") is None:
            continue
        is_article = _text(el.find("조문여부")) != "N"
        raw = _collect_article_text(el)
        if not is_article:  # 편·장·절·관 제목
            structure.append(raw)
            continue
        parsed = parse_article_text(raw)
        parsed["title"] = parsed["title"] or _text(el.find("조제목"))
        parsed["key"] = _text(el.find("조문번호"))
        parsed["chapter"] = structure[-1] if structure else ""
        parsed["raw_text"] = raw
        articles.append(parsed)

    addenda = [
        {
            "promulgation_date": _fmt_date(_find_text(el, "부칙공포일자")),
            "promulgation_no": _find_text(el, "부칙공포번호"),
            "text": _find_text(el, "부칙내용"),
        }
        for el in root.iter("부칙")
        if _find_text(el, "부칙내용")
    ]

    full_text = "\n".join(a["raw_text"] for a in articles)
    return {
        **info,
        "chapters": structure,
        "articles": articles,
        "addenda": addenda,
        "referenced_laws": extract_law_references(full_text),
    }
