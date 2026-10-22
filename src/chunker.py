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

from src.ordin_parser import PARAGRAPH_MARKS, extract_law_references, parse_article_text

ARTICLE_SPLIT_CHARS = 800
MAX_EMBED_CHARS = 6000  # Solar Embedding 2 입력 한도(8k 토큰) 대비 여유

# '<개정 2024. 12. 30.>', '<단서 신설 2026.09.30.>', '<2020. 1. 1.>'(삭제 조문), '[본조신설 2020. 1. 1.]' 등
_REVISION_RE = re.compile(
    r"<[^<>]*(?:개정|신설|삭제|시행)[^<>]*\d{4}[^<>]*>"
    r"|<\s*\d{4}\s*\.\s*\d{1,2}\s*\.\s*\d{1,2}\s*\.?\s*(?:,[^<>]*)?>"
    r"|\[[^\[\]]*(?:개정|신설|삭제|시행|종전|이동)[^\[\]]*\d{4}[^\[\]]*\]"
)
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
    cleaned = _REVISION_RE.sub("", text)
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

        head = f"제{no}조" + (f"({title})" if title else "")
        record.chunks.append(
            ChunkRecord(
                granularity="article",
                paragraph_no=None,
                citation=f"{name} 제{no}조",
                content=cleaned,
                embed_text=f"[{name}]\n{cleaned}"[:MAX_EMBED_CHARS],
            )
        )

        numbered = [p for p in structure["paragraphs"] if p["number"]]
        if len(cleaned) > ARTICLE_SPLIT_CHARS and len(numbered) >= 2:
            for p in numbered:
                content = render_paragraph(p)
                record.chunks.append(
                    ChunkRecord(
                        granularity="paragraph",
                        paragraph_no=p["number"],
                        citation=f"{name} 제{no}조제{p['number']}항",
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


def is_local_ordinance(law_name: str, org_name: str) -> bool:
    """「」 인용이 국가 법령이 아니라 자치법규(자기 지자체 조례 등)인지 판별."""
    first = law_name.split()[0] if law_name.split() else ""
    return law_name.endswith("조례") or first in org_name.split()
