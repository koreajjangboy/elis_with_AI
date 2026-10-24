"""수집된 수원시 조례 XML → 청킹 → PostgreSQL 적재 → Upstage 임베딩.

단계마다 다시 실행해도 안전하다(체크포인트):
- 적재: 조례 25건 = 트랜잭션 1개(테이블별 COPY). 본문 XML 해시가 같으면 건너뛰고,
  묶음이 실패하면 1건씩 다시 시도해 실패한 조례만 롤백된다.
- 임베딩: embedding IS NULL인 청크만 처리하고 배치마다 커밋하므로 중단된 지점부터 이어진다.

실행:
    python -m src.load_to_db --dry-run      # DB·API 없이 청킹 결과·예상 토큰만 확인
    python -m src.load_to_db                # 적재 + 임베딩
    python -m src.load_to_db --skip-embed   # 적재만
    python -m src.load_to_db --embed-only   # 임베딩만 (이어하기)
    python -m src.load_to_db --retokenize   # 토크나이저 변경 후 키워드 토큰만 재계산
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import date

import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from psycopg.types.json import Jsonb

from src.chunker import ArticleRecord, build_articles, is_local_ordinance
from src.collect_suwon import LIST_PATH, RAW_DIR, SBORG
from src.db import connect
from src.embedder import PASSAGE_MODEL, EmbeddingError, UpstageEmbedder
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

LOAD_BATCH = 25  # 조례 N건을 트랜잭션 1개 + 테이블별 COPY 1회로 적재 (원격 DB 왕복 최소화)

# (컬럼명, 타입). 행 묶음을 JSON 하나로 보내 jsonb_to_recordset으로 펼친다
ORDINANCE_COLS = (("id", "bigint"), ("mst", "bigint"), ("ordin_id", "bigint"), ("name", "text"),
                  ("org_name", "text"), ("org_code", "text"), ("kind", "text"), ("promulgation_date", "date"),
                  ("promulgation_no", "text"), ("enforcement_date", "date"), ("revision_type", "text"),
                  ("department", "text"), ("field", "text"), ("content_hash", "text"))
ARTICLE_COLS = (("id", "bigint"), ("ordinance_id", "bigint"), ("seq", "int"), ("article_key", "text"),
                ("article_no", "text"), ("title", "text"), ("chapter", "text"), ("body", "text"),
                ("structure", "jsonb"), ("revision_notes", "text[]"))
CHUNK_COLS = (("article_id", "bigint"), ("ordinance_id", "bigint"), ("org_code", "text"), ("granularity", "text"),
              ("paragraph_no", "int"), ("citation", "text"), ("content", "text"), ("embed_text", "text"),
              ("tokens", "text"), ("name_tokens", "text"), ("title_tokens", "text"))
REF_COLS = (("ordinance_id", "bigint"), ("article_id", "bigint"), ("law_name", "text"), ("is_ordinance", "boolean"))

Prepared = tuple[bytes, str, dict, list[ArticleRecord]]


def _bulk_insert(cur: psycopg.Cursor, table: str, cols: tuple[tuple[str, str], ...], rows: list[tuple]) -> None:
    """여러 행을 INSERT 문 하나로 넣는다. 원격 DB에서는 행 단위 전송(executemany·COPY)이
    행마다 네트워크 지연을 겪어, 전체를 JSON 파라미터 하나로 보내는 편이 훨씬 빠르다."""
    if not rows:
        return
    names = ", ".join(n for n, _ in cols)
    record_def = ", ".join(f"{n} {t}" for n, t in cols)
    payload = [{n: (v.isoformat() if isinstance(v, date) else v) for (n, _), v in zip(cols, row)} for row in rows]
    cur.execute(
        f"INSERT INTO {table} ({names}) SELECT {names} FROM jsonb_to_recordset(%s::jsonb) AS x({record_def})",
        (Jsonb(payload),),
    )


def load_batch(
    conn: psycopg.Connection, batch: list[tuple[dict, Prepared]], existing: dict[int, tuple[int, str]]
) -> list[str]:
    """조례 여러 건을 트랜잭션 1개로 적재하고 건별 상태를 반환한다. 실패하면 묶음 전체가 롤백된다.

    원격 DB 왕복 지연이 커서 ID를 시퀀스에서 미리 받고 행은 테이블별 COPY 한 번으로 보낸다.
    """
    statuses, todo = [], []
    for item, prepared in batch:
        row = existing.get(int(item["serial_no"]))
        if row and row[1] == prepared[1]:
            statuses.append("unchanged")
        else:
            statuses.append("updated" if row else "inserted")
            todo.append((item, prepared, row))
    if not todo:
        return statuses

    new_ids, article_ids = conn.execute(
        """SELECT ARRAY(SELECT nextval('ordinances_id_seq') FROM generate_series(1, %s)),
                  ARRAY(SELECT nextval('articles_id_seq') FROM generate_series(1, %s))""",
        (len(todo), sum(len(p[3]) for _, p, _ in todo)),
    ).fetchone()
    article_id_iter = iter(article_ids)
    body_tokens = iter(tokenize_many([c.content for _, p, _ in todo for a in p[3] for c in a.chunks]))
    name_tokens = iter(tokenize_many([p[2]["name"] for _, p, _ in todo]))
    title_tokens = iter(tokenize_many([a.title or "" for _, p, _ in todo for a in p[3]]))

    ord_rows, updates, art_rows, chunk_rows, ref_rows = [], [], [], [], []
    for (item, (_, content_hash, o, articles), row), new_id in zip(todo, new_ids):
        meta = (
            int(item["serial_no"]), int(o["ordin_id"] or item["ordin_id"]), o["name"], o["org_name"], SBORG,
            o["kind"] or item["kind"], _to_date(o["promulgation_date"]), o["promulgation_no"],
            _to_date(o["enforcement_date"]), item.get("revision_type"), o["department"], item.get("field"),
            content_hash,
        )
        ordinance_id = row[0] if row else new_id
        if row:
            updates.append((*meta, ordinance_id))
        else:
            ord_rows.append((ordinance_id, *meta))
        n_tok = next(name_tokens)
        for a in articles:
            aid, t_tok = next(article_id_iter), next(title_tokens)
            art_rows.append((aid, ordinance_id, a.seq, a.article_key, a.article_no, a.title, a.chapter,
                             a.body, a.structure, a.revision_notes))
            chunk_rows += [(aid, ordinance_id, SBORG, c.granularity, c.paragraph_no, c.citation, c.content,
                            c.embed_text, next(body_tokens), n_tok, t_tok) for c in a.chunks]
            ref_rows += [(ordinance_id, aid, name, is_local_ordinance(name, o["org_name"]))
                         for name in a.law_references]

    with conn.transaction(), conn.cursor() as cur:
        for values in updates:  # 같은 버전인데 내용이 바뀐 경우(파서 변경 등) → 하위 데이터 재생성. 드물다
            cur.execute(
                """UPDATE ordinances SET mst=%s, ordin_id=%s, name=%s, org_name=%s, org_code=%s, kind=%s,
                       promulgation_date=%s, promulgation_no=%s, enforcement_date=%s, revision_type=%s,
                       department=%s, field=%s, content_hash=%s, fetched_at=now()
                   WHERE id=%s""",
                values,
            )
            cur.execute("DELETE FROM articles WHERE ordinance_id = %s", (values[-1],))  # chunks·인용 CASCADE
        if ord_rows:  # 새 버전이 들어오는 법규의 이전 현행 버전을 내린다
            cur.execute(
                "UPDATE ordinances SET is_current = false WHERE is_current AND ordin_id = ANY(%s)",
                ([r[2] for r in ord_rows],),
            )
        _bulk_insert(cur, "ordinances", ORDINANCE_COLS, ord_rows)
        _bulk_insert(cur, "articles", ARTICLE_COLS, art_rows)
        _bulk_insert(cur, "chunks", CHUNK_COLS, chunk_rows)
        _bulk_insert(cur, "law_references", REF_COLS, ref_rows)

    for (item, prepared, row), new_id in zip(todo, new_ids):  # 커밋된 뒤에만 반영
        existing[int(item["serial_no"])] = (row[0] if row else new_id, prepared[1])
    return statuses


def load_all(conn: psycopg.Connection, items: list[dict]) -> None:
    run_id = conn.execute(
        "INSERT INTO ingest_runs (org_code, listed) VALUES (%s, %s) RETURNING id", (SBORG, len(items))
    ).fetchone()[0]
    stats, errors = Counter(), []
    existing = {
        mst: (oid, h)
        for mst, oid, h in conn.execute(
            "SELECT mst, id, content_hash FROM ordinances WHERE org_code = %s", (SBORG,)
        ).fetchall()
    }

    def fail(item: dict, e: Exception) -> None:
        stats["failed"] += 1
        errors.append({"mst": item["serial_no"], "name": item["name"], "error": f"{type(e).__name__}: {e}"[:500]})
        print(f"  [실패] {item['serial_no']} {item['name']}: {e}")

    for start in range(0, len(items), LOAD_BATCH):
        batch = []
        for item in items[start:start + LOAD_BATCH]:
            try:
                prepared = prepare(item)
            except Exception as e:
                fail(item, e)
                continue
            if prepared is None:
                stats["missing"] += 1
                errors.append({"mst": item["serial_no"], "error": "XML 파일 없음 (collect_suwon 재실행 필요)"})
                continue
            batch.append((item, prepared))

        try:
            stats.update(load_batch(conn, batch, existing))
        except Exception:  # 묶음이 롤백되면 1건씩 다시 넣어 실패한 조례만 골라낸다
            for one in batch:
                try:
                    stats.update(load_batch(conn, [one], existing))
                except Exception as e:
                    fail(one[0], e)
        print(f"  적재 {min(start + LOAD_BATCH, len(items))}/{len(items)} {dict(stats)}")

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
    register_vector(conn)  # numpy 배열 ↔ vector 바이너리 변환
    pending = conn.execute("SELECT count(*) FROM chunks WHERE embedding IS NULL").fetchone()[0]
    print(f"임베딩 대상: {pending}개 청크")
    done, failed_ids = 0, []

    while True:
        rows = conn.execute(
            """SELECT id, embed_text FROM chunks
               WHERE embedding IS NULL AND NOT (id = ANY(%s::bigint[])) ORDER BY id LIMIT %s""",
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
                # 벡터는 바이너리(float32)로 보낸다: 텍스트 대비 전송량 약 1/3 (원격 DB 업로드가 병목)
                cur.execute(
                    """UPDATE chunks c SET embedding = x.e, embed_model = %b
                       FROM unnest(%b::bigint[], %b::vector[]) AS x(id, e)
                       WHERE c.id = x.id""",
                    (PASSAGE_MODEL, [r[0] for r in batch], [np.asarray(v, dtype=np.float32) for v in vectors]),
                )
            done += len(batch)
            print(f"  임베딩 {done}/{pending} (누적 {embedder.total_tokens:,}토큰, ${embedder.cost_usd:.4f})")

    print(f"임베딩 완료: 성공 {done}, 실패 {len(failed_ids)}")
    if failed_ids:
        print("실패한 청크는 embedding이 NULL로 남아 있어 --embed-only로 다시 실행하면 재시도합니다.")


# ---------------------------------------------------------------- 토큰 재계산

RETOKENIZE_BATCH = 1000


def retokenize(conn: psycopg.Connection) -> None:
    """토크나이저를 바꾼 뒤 저장된 본문·조례명·조 제목 토큰을 다시 계산한다 (임베딩은 그대로)."""
    rows = conn.execute(
        """SELECT c.id, c.content, o.name, COALESCE(a.title, '')
           FROM chunks c JOIN articles a ON a.id = c.article_id JOIN ordinances o ON o.id = c.ordinance_id
           ORDER BY c.id"""
    ).fetchall()
    print(f"토큰 재계산 대상: {len(rows)}개 청크")
    for start in range(0, len(rows), RETOKENIZE_BATCH):
        batch = rows[start:start + RETOKENIZE_BATCH]
        body = tokenize_many([r[1] for r in batch])
        names = tokenize_many([r[2] for r in batch])
        titles = tokenize_many([r[3] for r in batch])
        with conn.transaction(), conn.cursor() as cur:
            cur.execute(
                """UPDATE chunks c SET tokens = x.tokens, name_tokens = x.name_tokens, title_tokens = x.title_tokens
                   FROM jsonb_to_recordset(%s::jsonb) AS x(id bigint, tokens text, name_tokens text, title_tokens text)
                   WHERE c.id = x.id""",
                (Jsonb([{"id": r[0], "tokens": b, "name_tokens": n, "title_tokens": t}
                        for b, n, t, r in zip(body, names, titles, batch)]),),
            )
        print(f"  {min(start + RETOKENIZE_BATCH, len(rows))}/{len(rows)}")
    print("토큰 재계산 완료")


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
    mode.add_argument("--retokenize", action="store_true", help="저장된 청크의 키워드 토큰만 다시 계산")
    args = parser.parse_args()

    if args.dry_run:
        dry_run(load_manifest())
        return
    if args.retokenize:
        with connect() as conn:
            retokenize(conn)
        return

    embedder = None if args.skip_embed else UpstageEmbedder.from_env()  # 키 누락을 적재 전에 확인
    with connect() as conn:
        if not args.embed_only:
            load_all(conn, load_manifest())
        if embedder:
            embed_pending(conn, embedder)


if __name__ == "__main__":
    main()
