"""수집된 수원시 조례 XML → 청킹 → PostgreSQL 적재 → Upstage 임베딩.

단계마다 다시 실행해도 안전하다(체크포인트):
- 적재: 조례 1건 = 트랜잭션 1개. 본문 XML 해시가 같으면 건너뛰고, 실패한 조례만 롤백된다.
- 임베딩: embedding IS NULL인 청크만 처리하고 배치마다 커밋하므로 중단된 지점부터 이어진다.

실행:
    python -m src.load_to_db --dry-run      # DB·API 없이 청킹 결과·예상 토큰만 확인
    python -m src.load_to_db                # 적재 + 임베딩
    python -m src.load_to_db --skip-embed   # 적재만
    python -m src.load_to_db --embed-only   # 임베딩만 (이어하기)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import date

import psycopg
from psycopg.types.json import Jsonb

from src.chunker import ArticleRecord, build_articles, is_local_ordinance
from src.collect_suwon import LIST_PATH, RAW_DIR, SBORG
from src.db import connect
from src.embedder import PASSAGE_MODEL, EmbeddingError, UpstageEmbedder, to_pgvector
from src.ordin_parser import parse_ordinance_detail
from src.tokenizer import tokenize_many

EMBED_FETCH_SIZE = 500  # DB에서 한 번에 가져올 미임베딩 청크 수


def _to_date(value: str) -> date | None:
    return date.fromisoformat(value) if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value or "") else None


def load_manifest() -> list[dict]:
    if not LIST_PATH.exists():
        raise SystemExit(f"목록 파일이 없습니다: {LIST_PATH}\n먼저 python -m src.collect_suwon 을 실행하세요.")
    return json.loads(LIST_PATH.read_text(encoding="utf-8"))


def prepare(item: dict) -> tuple[bytes, str, dict, list[ArticleRecord]] | None:
    """XML 읽기 → 파싱 → 청킹. 파일이 없으면 None."""
    path = RAW_DIR / f"{item['serial_no']}.xml"
    if not path.exists():
        return None
    xml = path.read_bytes()
    ordinance = parse_ordinance_detail(ET.fromstring(xml), org_code=SBORG)
    ordinance["org_name"] = ordinance["org_name"] or item["org_name"]
    return xml, hashlib.sha256(xml).hexdigest(), ordinance, build_articles(ordinance)


# ---------------------------------------------------------------- 적재

def load_one(conn: psycopg.Connection, item: dict, prepared) -> str:
    xml, content_hash, o, articles = prepared
    mst = int(item["serial_no"])

    with conn.transaction():
        row = conn.execute("SELECT id, content_hash FROM ordinances WHERE mst = %s", (mst,)).fetchone()
        if row and row[1] == content_hash:
            return "unchanged"

        meta = (
            int(o["ordin_id"] or item["ordin_id"]), o["name"], o["org_name"], SBORG, o["kind"] or item["kind"],
            _to_date(o["promulgation_date"]), o["promulgation_no"], _to_date(o["enforcement_date"]),
            item.get("revision_type"), o["department"], item.get("field"), content_hash,
        )
        if row:  # 같은 버전인데 내용이 다름 (파서 변경 등) → 하위 데이터 재생성
            ordinance_id = row[0]
            conn.execute(
                """UPDATE ordinances SET ordin_id=%s, name=%s, org_name=%s, org_code=%s, kind=%s,
                       promulgation_date=%s, promulgation_no=%s, enforcement_date=%s, revision_type=%s,
                       department=%s, field=%s, content_hash=%s, fetched_at=now()
                   WHERE id=%s""",
                (*meta, ordinance_id),
            )
            conn.execute("DELETE FROM articles WHERE ordinance_id = %s", (ordinance_id,))  # chunks·인용 CASCADE
            status = "updated"
        else:  # 새 버전 → 같은 법규의 이전 현행 버전을 내린다
            conn.execute("UPDATE ordinances SET is_current = false WHERE ordin_id = %s AND is_current", (meta[0],))
            ordinance_id = conn.execute(
                """INSERT INTO ordinances (ordin_id, name, org_name, org_code, kind, promulgation_date,
                       promulgation_no, enforcement_date, revision_type, department, field, content_hash, mst)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id""",
                (*meta, mst),
            ).fetchone()[0]
            status = "inserted"

        all_chunks = [c for a in articles for c in a.chunks]
        tokens = iter(tokenize_many([c.content for c in all_chunks]))
        with conn.cursor() as cur:
            for a in articles:
                article_id = cur.execute(
                    """INSERT INTO articles (ordinance_id, seq, article_key, article_no, title, chapter,
                           body, structure, revision_notes)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id""",
                    (ordinance_id, a.seq, a.article_key, a.article_no, a.title, a.chapter,
                     a.body, Jsonb(a.structure), a.revision_notes),
                ).fetchone()[0]
                if a.chunks:
                    cur.executemany(
                        """INSERT INTO chunks (article_id, ordinance_id, org_code, granularity, paragraph_no,
                               citation, content, embed_text, tokens)
                           VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                        [(article_id, ordinance_id, SBORG, c.granularity, c.paragraph_no,
                          c.citation, c.content, c.embed_text, next(tokens)) for c in a.chunks],
                    )
                if a.law_references:
                    cur.executemany(
                        """INSERT INTO law_references (ordinance_id, article_id, law_name, is_ordinance)
                           VALUES (%s, %s, %s, %s) ON CONFLICT DO NOTHING""",
                        [(ordinance_id, article_id, name, is_local_ordinance(name, o["org_name"]))
                         for name in a.law_references],
                    )
    return status


def load_all(conn: psycopg.Connection, items: list[dict]) -> None:
    run_id = conn.execute(
        "INSERT INTO ingest_runs (org_code, listed) VALUES (%s, %s) RETURNING id", (SBORG, len(items))
    ).fetchone()[0]
    stats, errors = Counter(), []

    for i, item in enumerate(items, 1):
        try:
            prepared = prepare(item)
            if prepared is None:
                stats["missing"] += 1
                errors.append({"mst": item["serial_no"], "error": "XML 파일 없음 (collect_suwon 재실행 필요)"})
                continue
            stats[load_one(conn, item, prepared)] += 1
        except Exception as e:  # 해당 조례만 롤백되고 계속 진행
            stats["failed"] += 1
            errors.append({"mst": item["serial_no"], "name": item["name"], "error": f"{type(e).__name__}: {e}"[:500]})
            print(f"  [실패] {item['serial_no']} {item['name']}: {e}")
        if i % 50 == 0 or i == len(items):
            print(f"  적재 {i}/{len(items)} {dict(stats)}")

    # 목록 전체를 문제없이 반영했을 때만, 목록에서 빠진(폐지 등) 조례를 현행에서 내린다
    if not errors:
        retired = conn.execute(
            "UPDATE ordinances SET is_current = false WHERE org_code = %s AND is_current AND NOT (mst = ANY(%s))",
            (SBORG, [int(item["serial_no"]) for item in items]),
        ).rowcount
        if retired:
            print(f"  목록에 없는 조례 {retired}건을 현행에서 제외")

    conn.execute(
        """UPDATE ingest_runs SET finished_at = now(), inserted = %s, updated = %s, unchanged = %s,
               failed = %s, errors = %s WHERE id = %s""",
        (stats["inserted"], stats["updated"], stats["unchanged"], stats["failed"] + stats["missing"],
         Jsonb(errors), run_id),
    )
    print(f"적재 완료 (ingest_runs.id={run_id}): {dict(stats)}")


# ---------------------------------------------------------------- 임베딩

def embed_pending(conn: psycopg.Connection, embedder: UpstageEmbedder) -> None:
    pending = conn.execute("SELECT count(*) FROM chunks WHERE embedding IS NULL").fetchone()[0]
    print(f"임베딩 대상: {pending}개 청크")
    done, failed_ids = 0, []

    while True:
        rows = conn.execute(
            """SELECT id, embed_text FROM chunks
               WHERE embedding IS NULL AND NOT (id = ANY(%s)) ORDER BY id LIMIT %s""",
            (failed_ids, EMBED_FETCH_SIZE),
        ).fetchall()
        if not rows:
            break
        texts = [r[1] for r in rows]
        for start, end in embedder.batches(texts):
            batch = rows[start:end]
            try:
                vectors = embedder.embed(texts[start:end], model=PASSAGE_MODEL)
            except EmbeddingError as e:
                failed_ids.extend(r[0] for r in batch)  # 이번 실행에서는 건너뛰고 다음 실행 때 재시도
                print(f"  [실패] 청크 {batch[0][0]}~{batch[-1][0]}: {str(e)[:200]}")
                continue
            with conn.transaction(), conn.cursor() as cur:  # 배치 단위 커밋 = 체크포인트
                cur.executemany(
                    "UPDATE chunks SET embedding = %s::vector, embed_model = %s WHERE id = %s",
                    [(to_pgvector(v), PASSAGE_MODEL, r[0]) for v, r in zip(vectors, batch)],
                )
            done += len(batch)
            print(f"  임베딩 {done}/{pending} (누적 {embedder.total_tokens:,}토큰, ${embedder.cost_usd:.4f})")

    print(f"임베딩 완료: 성공 {done}, 실패 {len(failed_ids)}")
    if failed_ids:
        print("실패한 청크는 embedding이 NULL로 남아 있어 --embed-only로 다시 실행하면 재시도합니다.")


# ---------------------------------------------------------------- dry-run

def dry_run(items: list[dict]) -> None:
    stats, chars, sample = Counter(), 0, None
    longest = (0, "")
    for item in items:
        prepared = prepare(item)
        if prepared is None:
            stats["missing"] += 1
            continue
        _, _, o, articles = prepared
        stats["ordinances"] += 1
        stats["articles"] += sum(a.article_no != "부칙" for a in articles)
        stats["addenda"] += sum(a.article_no == "부칙" for a in articles)
        stats["law_refs"] += sum(len(a.law_references) for a in articles)
        for a in articles:
            for c in a.chunks:
                stats[f"chunks_{c.granularity}"] += 1
                chars += len(c.embed_text)
                if len(c.embed_text) > longest[0]:
                    longest = (len(c.embed_text), c.citation)
                if sample is None and c.granularity == "paragraph":
                    sample = c

    print(f"[dry-run] {dict(stats)}")
    print(f"  임베딩 입력 총 {chars:,}자 (토큰 ≈ 글자 수로 보수 추정 시 ${chars / 1e6 * 0.02:.4f})")
    print(f"  가장 긴 청크: {longest[1]} ({longest[0]:,}자)")
    if sample:
        print(f"\n  [항 청크 예시] {sample.citation}\n  ---\n{sample.embed_text[:600]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="수원시 조례 DB 적재 및 임베딩")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="DB·API 없이 청킹 결과만 확인")
    mode.add_argument("--skip-embed", action="store_true", help="적재만 하고 임베딩은 생략")
    mode.add_argument("--embed-only", action="store_true", help="미임베딩 청크만 임베딩")
    args = parser.parse_args()

    if args.dry_run:
        dry_run(load_manifest())
        return

    embedder = None if args.skip_embed else UpstageEmbedder.from_env()  # 키 누락을 적재 전에 확인
    with connect() as conn:
        if not args.embed_only:
            load_all(conn, load_manifest())
        if embedder:
            embed_pending(conn, embedder)


if __name__ == "__main__":
    main()
