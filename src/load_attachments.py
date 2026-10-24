"""별표·서식 첨부파일 → 텍스트 추출 → attachments 적재 → 검색 청크 생성.

임베딩은 기존 파이프라인이 embedding IS NULL인 청크를 처리하므로 이어서
`python -m src.load_to_db --embed-only`를 실행하면 된다.

OCR 정책 (--ocr):
    none    로컬 추출만 (hwp/hwpx/txt, 무료)
    needs   로컬 추출이 실패했거나 텍스트가 거의 없는 이미지형 문서만 Upstage Document Parse로 (기본)
    images  그림이 포함된 문서(도면·그림 속 글자)까지 Document Parse로 다시 추출 (페이지당 과금)

재실행 시 파일 해시와 추출 방식이 같으면 건너뛰므로 중단 후 이어서 처리된다.

실행:
    python -m src.load_attachments
    python -m src.load_attachments --ocr none
"""

from __future__ import annotations

import argparse
import hashlib
import xml.etree.ElementTree as ET
from collections import Counter

import psycopg
from psycopg.types.json import Jsonb

from src.attachment_extract import ExtractResult, FallbackExtractor, UpstageDocumentParse, extract_text
from src.chunker import build_attachment_chunks
from src.collect_attachments import attachment_path
from src.collect_suwon import RAW_DIR, SBORG
from src.db import connect
from src.load_to_db import _bulk_insert
from src.ordin_parser import parse_annexes
from src.tokenizer import tokenize_many

ATTACHMENT_CHUNK_COLS = (
    ("attachment_id", "bigint"), ("ordinance_id", "bigint"), ("org_code", "text"), ("granularity", "text"),
    ("chunk_seq", "int"), ("citation", "text"), ("content", "text"), ("embed_text", "text"),
    ("tokens", "text"), ("name_tokens", "text"), ("title_tokens", "text"),
)

UPSERT_SQL = """
INSERT INTO attachments (ordinance_id, annex_key, annex_no, annex_branch_no, kind, title, related_article,
                         file_type, source_url, file_path, file_hash, text, extraction_method,
                         extraction_status, extraction_error, has_images, extracted_at)
SELECT ordinance_id, annex_key, annex_no, annex_branch_no, kind, title, related_article,
       file_type, source_url, file_path, file_hash, text, extraction_method,
       extraction_status, extraction_error, has_images, now()
FROM jsonb_to_recordset(%s::jsonb) AS x(
    ordinance_id bigint, annex_key text, annex_no int, annex_branch_no int, kind text, title text,
    related_article text, file_type text, source_url text, file_path text, file_hash text, text text,
    extraction_method text, extraction_status text, extraction_error text, has_images boolean)
ON CONFLICT (ordinance_id, annex_key) DO UPDATE SET
    annex_no = EXCLUDED.annex_no, annex_branch_no = EXCLUDED.annex_branch_no, kind = EXCLUDED.kind,
    title = EXCLUDED.title, related_article = EXCLUDED.related_article, file_type = EXCLUDED.file_type,
    source_url = EXCLUDED.source_url, file_path = EXCLUDED.file_path, file_hash = EXCLUDED.file_hash,
    text = EXCLUDED.text, extraction_method = EXCLUDED.extraction_method,
    extraction_status = EXCLUDED.extraction_status, extraction_error = EXCLUDED.extraction_error,
    has_images = EXCLUDED.has_images, extracted_at = now()
RETURNING id, annex_key
"""


def wants_fallback(result: ExtractResult, ocr: str) -> bool:
    if ocr == "none":
        return False
    if result.status in ("needs_ocr", "error"):
        return True
    return ocr == "images" and result.has_images


def extract(annex: dict, data: bytes, ocr: str, fallback: FallbackExtractor | None) -> ExtractResult:
    result = extract_text(data, annex["file_type"])
    if fallback and wants_fallback(result, ocr):
        better = fallback.extract(data, attachment_path(annex).name)
        if better.status == "ok":
            better.has_images = better.has_images or result.has_images
            return better
        result.error = (result.error + f" / {fallback.name}: {better.error}").strip(" /")
    return result


def process_ordinance(
    conn: psycopg.Connection, ordinance: tuple, existing: dict, ocr: str, fallback, stats: Counter
) -> None:
    ordinance_id, mst, name = ordinance
    xml_path = RAW_DIR / f"{mst}.xml"
    if not xml_path.exists():
        return
    annexes = parse_annexes(ET.parse(xml_path).getroot())
    if not annexes:
        return

    rows = []
    for annex in annexes:
        path = attachment_path(annex)
        base = {
            "ordinance_id": ordinance_id, "annex_key": annex["key"], "annex_no": int(annex["no"]),
            "annex_branch_no": int(annex["branch_no"]), "kind": annex["kind"], "title": annex["title"],
            "related_article": annex["related_article"] or None, "file_type": annex["file_type"],
            "source_url": annex["url"], "file_path": path.relative_to(RAW_DIR.parent.parent.parent).as_posix(),
            "file_hash": None, "text": None, "extraction_method": None, "extraction_error": None,
            "has_images": False,
        }
        if annex["kind"] == "삭제" or not annex["url"]:
            rows.append({**base, "file_path": None, "extraction_status": "skipped"})
            stats["skipped"] += 1
            continue
        if annex["content"]:  # API가 본문을 직접 준 경우 (서울시교육청 등)
            rows.append({**base, "text": annex["content"], "extraction_method": "api", "extraction_status": "ok"})
            stats["api_text"] += 1
            continue
        if not path.exists():
            rows.append({**base, "extraction_status": "pending", "extraction_error": "파일 미수집"})
            stats["pending"] += 1
            continue

        data = path.read_bytes()
        file_hash = hashlib.sha256(data).hexdigest()
        prev = existing.get((ordinance_id, annex["key"]))
        # 같은 파일이고 이미 성공했으며 OCR 정책상 다시 할 일이 없으면 건너뜀
        if prev and prev[0] == file_hash and prev[1] == "ok" and not (ocr == "images" and prev[2] and prev[3] != "document-parse"):
            stats["unchanged"] += 1
            continue
        result = extract(annex, data, ocr, fallback)
        rows.append({**base, "file_hash": file_hash, "text": result.text, "extraction_method": result.method,
                     "extraction_status": result.status, "extraction_error": result.error or None,
                     "has_images": result.has_images})
        stats[f"{result.method}:{result.status}"] += 1

    if not rows:
        return
    with conn.transaction(), conn.cursor() as cur:
        ids = dict((key, aid) for aid, key in cur.execute(UPSERT_SQL, (Jsonb(rows),)).fetchall())
        # 다시 추출한 별표의 이전 청크를 지우고 새로 만든다 (임베딩은 이후 --embed-only로)
        cur.execute("DELETE FROM chunks WHERE attachment_id = ANY(%s)", ([ids[r["annex_key"]] for r in rows],))
        # 추출에 실패한 별표(needs_ocr 등)도 제목 자리표시 청크로 검색되게 한다. 삭제된 별표만 제외
        pieces = [(ids[r["annex_key"]], r, c) for r in rows if r["extraction_status"] != "skipped"
                  for c in build_attachment_chunks(name, r["title"], r["text"] if r["extraction_status"] == "ok" else None)]
        if pieces:
            body = tokenize_many([c.content for _, _, c in pieces])
            titles = tokenize_many([r["title"] for _, r, _ in pieces])
            name_tok = tokenize_many([name])[0]
            _bulk_insert(cur, "chunks", ATTACHMENT_CHUNK_COLS, [
                (aid, ordinance_id, SBORG, "attachment", c.chunk_seq, c.citation, c.content, c.embed_text, b, name_tok, t)
                for (aid, _, c), b, t in zip(pieces, body, titles)
            ])
            stats["chunks"] += len(pieces)


def main() -> None:
    parser = argparse.ArgumentParser(description="별표·서식 텍스트 추출 및 적재")
    parser.add_argument("--ocr", choices=("none", "needs", "images"), default="needs")
    args = parser.parse_args()

    fallback = UpstageDocumentParse() if args.ocr != "none" else None
    with connect() as conn:
        ordinances = conn.execute(
            "SELECT id, mst, name FROM ordinances WHERE org_code = %s AND is_current ORDER BY id", (SBORG,)
        ).fetchall()
        existing = {
            (oid, key): (h, status, has_img, method)
            for oid, key, h, status, has_img, method in conn.execute(
                "SELECT ordinance_id, annex_key, file_hash, extraction_status, has_images, extraction_method FROM attachments"
            ).fetchall()
        }
        stats: Counter = Counter()
        for i, ordinance in enumerate(ordinances, 1):
            try:
                process_ordinance(conn, ordinance, existing, args.ocr, fallback, stats)
            except Exception as e:  # 해당 조례의 별표만 롤백되고 계속 진행
                stats["failed"] += 1
                print(f"  [실패] {ordinance[2]}: {type(e).__name__}: {str(e)[:200]}")
            if i % 100 == 0 or i == len(ordinances):
                print(f"  {i}/{len(ordinances)} {dict(stats)}")

        summary = conn.execute(
            "SELECT extraction_status, count(*), sum(has_images::int) FROM attachments GROUP BY 1 ORDER BY 1"
        ).fetchall()
    print("완료. 상태별 (건수, 그림 포함):", {s: (n, img) for s, n, img in summary})
    print("다음: python -m src.load_to_db --embed-only  (별표 청크 임베딩)")


if __name__ == "__main__":
    main()
