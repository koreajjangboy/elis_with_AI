"""파싱된 조례를 DB 적재 단위(조문·청크·인용 법령)로 변환.

청킹 규칙 (docs/db_design.md 2절):
- 조 단위 청크를 기본으로 만들고, 긴 조(ARTICLE_SPLIT_CHARS 초과 + 항 2개 이상)는 항 단위 청크를 추가한다.
- 호·목은 쪼개지 않고 소속 항 청크에 포함한다.
- 임베딩 입력에는 조례명·조 제목 헤더를 붙인다.
- '<개정 2024. 12. 30.>' 같은 개정 표기는 revision_notes로 분리하고 검색 텍스트에서 제거한다.
- 삭제된 조와 부칙은 articles에만 저장하고 검색 청크는 만들지 않는다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.ordin_parser import PARAGRAPH_MARKS, article_label, extract_law_references, parse_article_text

ARTICLE_SPLIT_CHARS = 800
MAX_EMBED_CHARS = 6000  # Solar Embedding 2 입력 한도(8k 토큰) 대비 여유

_REVISION_KEYWORDS = "개정|신설|삭제|시행|종전|이동"
# '<개정 2024. 12. 30.>', '<단서 신설 2026.09.30.>', '<2020. 1. 1.>'(삭제 조문), '[본조신설 2020. 1. 1.]' 등
_REVISION_RE = re.compile(
    rf"<(?=[^<>]*(?:{_REVISION_KEYWORDS}))(?=[^<>]*\d{{4}})[^<>]*>"  # 키워드·연도 순서 무관
    r"|<\s*\d{4}\s*\.\s*\d{1,2}\s*\.\s*\d{1,2}\s*\.?\s*(?:,[^<>]*)?>"
    rf"|\[[^\[\]]*(?:{_REVISION_KEYWORDS})[^\[\]]*\d{{4}}[^\[\]]*\]"
)
# 인용 대조용 (src/rag.py): 위 패턴 + 괄호형 '(개정 2009.01.08)', '(종전 제5항에서 이동 2019.12.31)', '(본조개정 2016.09.28)'.
# 키워드와 4자리 연도가 모두 있는 괄호만 해당하므로 '(제5조 관련)', '(이하 "시장"이라 한다)'는 건드리지 않는다.
# 괄호형은 청킹(strip_revision_notes)에는 적용하지 않는다: 적용하면 저장된 청크·임베딩과 달라져 전체 재적재가 필요하다.
REVISION_NOTE_RE = re.compile(
    rf"{_REVISION_RE.pattern}|\((?=[^()]*(?:{_REVISION_KEYWORDS}))(?=[^()]*\d{{4}})[^()]*\)"
)
_HTML_TAG_RE = re.compile(r"</?[a-zA-Z][^<>]*>")  # 일부 조문에 섞인 '<p style=...>' 등
_SPACES_RE = re.compile(r"[ \t]+")


@dataclass
class ChunkRecord:
    granularity: str  # 'article' | 'paragraph'
    paragraph_no: int | None
    citation: str
    content: str
    embed_text: str


@dataclass
class ArticleRecord:
    seq: int
    article_key: str
    article_no: str
    title: str
    chapter: str
    body: str
    structure: dict
    revision_notes: list[str]
    chunks: list[ChunkRecord] = field(default_factory=list)
    law_references: list[str] = field(default_factory=list)


def strip_revision_notes(text: str) -> tuple[str, list[str]]:
    notes = [m.group(0) for m in _REVISION_RE.finditer(text)]
    cleaned = _HTML_TAG_RE.sub("", _REVISION_RE.sub("", text))
    cleaned = "\n".join(_SPACES_RE.sub(" ", line).strip() for line in cleaned.splitlines())
    return cleaned.strip(), notes


def render_paragraph(p: dict) -> str:
    mark = PARAGRAPH_MARKS[p["number"] - 1] + " " if p["number"] else ""
    lines = [f"{mark}{p['text']}".strip()]
    for item in p["items"]:
        lines.append(f"{item['number']}. {item['text']}")
        lines.extend(f"{sub['number']}. {sub['text']}" for sub in item["subitems"])
    return "\n".join(line for line in lines if line)


def _is_deleted(structure: dict) -> bool:
    text = " ".join(p["text"] for p in structure["paragraphs"]).strip()
    has_items = any(p["items"] for p in structure["paragraphs"])
    return not has_items and text in ("", "삭제", "삭제.")


def build_articles(ordinance: dict) -> list[ArticleRecord]:
    name = ordinance["name"]
    records: list[ArticleRecord] = []

    for seq, art in enumerate(ordinance["articles"], 1):
        cleaned, notes = strip_revision_notes(art["raw_text"])
        structure = parse_article_text(cleaned)
        no, title = art["article_no"] or structure["article_no"], art["title"] or structure["title"]
        record = ArticleRecord(
            seq=seq,
            article_key=art.get("key", ""),
            article_no=no,
            title=title,
            chapter=art.get("chapter", ""),
            body=art["raw_text"],
            structure=structure,
            revision_notes=notes,
            law_references=extract_law_references(cleaned),
        )
        records.append(record)
        if _is_deleted(structure):
            continue

        label = article_label(no)
        head = label + (f"({title})" if title else "")
        record.chunks.append(
            ChunkRecord(
                granularity="article",
                paragraph_no=None,
                citation=f"{name} {label}",
                content=cleaned,
                embed_text=f"[{name}]\n{cleaned}"[:MAX_EMBED_CHARS],
            )
        )

        numbered = [p for p in structure["paragraphs"] if p["number"]]
        if len(cleaned) > ARTICLE_SPLIT_CHARS and len(numbered) >= 2:
            for p in numbered:
                # 삭제된 항('② <삭제 …>', '⑥ ~ ⑨ <삭제 …>') → 개정 표기를 떼면 빈 항
                if not p["text"].strip(" 삭제.~∼-") and not p["items"]:
                    continue
                content = render_paragraph(p)
                record.chunks.append(
                    ChunkRecord(
                        granularity="paragraph",
                        paragraph_no=p["number"],
                        citation=f"{name} {label}제{p['number']}항",
                        content=content,
                        embed_text=f"[{name}] {head}\n{content}"[:MAX_EMBED_CHARS],
                    )
                )

    # 부칙: 표시·시행일 질의용으로 저장만 한다
    for i, addendum in enumerate(ordinance["addenda"], 1):
        cleaned, notes = strip_revision_notes(addendum["text"])
        records.append(
            ArticleRecord(
                seq=len(ordinance["articles"]) + i,
                article_key="",
                article_no="부칙",
                title=f"부칙 <{addendum['promulgation_date']}, 제{addendum['promulgation_no']}호>",
                chapter="",
                body=addendum["text"],
                structure={"paragraphs": [{"number": None, "text": cleaned, "items": []}]},
                revision_notes=notes,
            )
        )
    return records


# ---------------------------------------------------------------- 별표·서식

ATTACHMENT_CHUNK_CHARS = 1500


@dataclass
class AttachmentChunk:
    chunk_seq: int
    citation: str
    content: str
    embed_text: str


def split_lines(text: str, max_chars: int = ATTACHMENT_CHUNK_CHARS) -> list[str]:
    """줄 경계에서 max_chars 이하로 나눈다. 한 줄이 너무 길면 그 줄만 글자 수로 자른다."""
    pieces, current = [], ""
    for line in text.splitlines():
        while len(line) > max_chars:
            if current:
                pieces.append(current)
                current = ""
            pieces.append(line[:max_chars])
            line = line[max_chars:]
        if current and len(current) + 1 + len(line) > max_chars:
            pieces.append(current)
            current = line
        else:
            current = f"{current}\n{line}" if current else line
    if current.strip():
        pieces.append(current)
    return [p.strip() for p in pieces if p.strip()]


# 본문을 추출하지 못한 별표(이미지형·배포용 문서 등)도 제목으로는 검색되게 하는 자리표시 청크.
# 답변 생성 시 이 문구만으로는 근거가 될 수 없으므로 LLM은 '확인 불가'와 원문 링크로 안내하게 된다.
UNEXTRACTED_NOTE = "(본문 텍스트 미추출: 이미지형 문서 등으로 내용을 읽지 못했습니다. 원문 파일을 확인하세요.)"


def build_attachment_chunks(ordinance_name: str, title: str, text: str | None) -> list[AttachmentChunk]:
    """별표 텍스트를 검색 청크로. 조각마다 조례명·별표 제목 헤더를 붙여 단독으로도 맥락이 통하게 한다.
    텍스트가 없으면(OCR 대기 등) 제목만 담은 자리표시 청크 1개를 만들어 별표가 검색에서 누락되지 않게 한다."""
    title = " ".join(title.split())
    pieces = split_lines(text or "") or [UNEXTRACTED_NOTE]
    return [
        AttachmentChunk(
            chunk_seq=i,
            citation=f"{ordinance_name} {title}" + (f" ({i}/{len(pieces)})" if len(pieces) > 1 else ""),
            content=piece,
            embed_text=f"[{ordinance_name}] {title}\n{piece}"[:MAX_EMBED_CHARS],
        )
        for i, piece in enumerate(pieces, 1)
    ]


def is_local_ordinance(law_name: str, org_name: str) -> bool:
    """「」 인용이 국가 법령이 아니라 자치법규(자기 지자체 조례 등)인지 판별."""
    first = law_name.split()[0] if law_name.split() else ""
    return law_name.endswith("조례") or first in org_name.split()
