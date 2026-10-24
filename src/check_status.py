"""DB 적재·임베딩 상태 점검.

조례·조문·청크·별표 건수, 임베딩 누락, 데이터 정합성(고아 행, 청크 없는 조례 등)을 한 번에 확인한다.
수집 목록(data/raw/{org}/_list.json)이 있으면 목록 건수와도 대조한다.

문제가 있으면 종료 코드 1을 반환하므로 배치·CI에서 적재 후 검증 단계로 쓸 수 있다.
같은 점검을 Supabase SQL Editor에서 하려면 db/verify.sql을 사용한다.

실행:
    python -m src.check_status
"""

from __future__ import annotations

import json
import sys

from src.collect_suwon import LIST_PATH, SBORG
from src.db import connect

CHECKS = [
    # (설명, SQL, 정상 조건: 결과가 0이어야 하면 True)
    ("현행 아닌 조례 수(과거 버전)", "SELECT count(*) FROM ordinances WHERE org_code = %(org)s AND NOT is_current", False),
    ("임베딩 누락 청크", "SELECT count(*) FROM chunks WHERE org_code = %(org)s AND embedding IS NULL", True),
    ("임베딩 차원 불일치(1024 아님)",
     "SELECT count(*) FROM chunks WHERE org_code = %(org)s AND embedding IS NOT NULL AND vector_dims(embedding) <> 1024", True),
    ("검색 청크가 하나도 없는 현행 조례",
     """SELECT count(*) FROM ordinances o WHERE o.org_code = %(org)s AND o.is_current
        AND NOT EXISTS (SELECT 1 FROM chunks c WHERE c.ordinance_id = o.id)""", True),
    ("조문 없이 별표로만 구성된 조례(예: 시민헌장)",
     """SELECT count(*) FROM ordinances o WHERE o.org_code = %(org)s AND o.is_current
        AND NOT EXISTS (SELECT 1 FROM chunks c WHERE c.ordinance_id = o.id AND c.article_id IS NOT NULL)
        AND EXISTS (SELECT 1 FROM chunks c WHERE c.ordinance_id = o.id AND c.attachment_id IS NOT NULL)""", False),
    ("소속(조문/별표)이 없거나 둘 다인 청크",
     "SELECT count(*) FROM chunks WHERE (article_id IS NULL) = (attachment_id IS NULL)", True),
    ("키워드 토큰이 빈 청크", "SELECT count(*) FROM chunks WHERE org_code = %(org)s AND tokens = ''", True),
    ("조례명 토큰이 빈 청크", "SELECT count(*) FROM chunks WHERE org_code = %(org)s AND name_tokens = ''", True),
    ("잘못된 가지조문 인용 표기(제N의M조)", r"SELECT count(*) FROM chunks WHERE citation ~ '제\d+의\d+조'", True),
    ("검색 청크가 없는 별표·서식(삭제 제외)",
     """SELECT count(*) FROM attachments a WHERE a.extraction_status <> 'skipped'
        AND NOT EXISTS (SELECT 1 FROM chunks c WHERE c.attachment_id = a.id)""", True),
    ("삭제 표기인데 '삭제'로 분류되지 않은 별표(파서 수정 전 적재분)",
     r"""SELECT count(*) FROM attachments WHERE kind <> '삭제' AND extraction_status = 'ok'
        AND coalesce(text, '') = '' AND title ~ '삭제'""", True),
    ("본문 미추출(OCR 필요·미지원·오류) 별표 → 자리표시 청크로만 검색됨",
     "SELECT count(*) FROM attachments WHERE extraction_status IN ('needs_ocr', 'unsupported', 'error', 'pending')", False),
]


def section(title: str) -> None:
    print(f"\n■ {title}")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # 파이프·리디렉션 시 Windows 기본(cp949)으로 못 쓰는 문자 대비
    with connect() as conn:
        q = lambda sql, **kw: conn.execute(sql, {"org": SBORG, **kw}).fetchall()

        section(f"조례 (org_code={SBORG})")
        current, total = q("""SELECT count(*) FILTER (WHERE is_current), count(*)
                              FROM ordinances WHERE org_code = %(org)s""")[0]
        print(f"  현행 조례           {current:>7,}건   (전체 버전 {total:,}건)")
        if LIST_PATH.exists():
            listed = len(json.loads(LIST_PATH.read_text(encoding="utf-8")))
            mark = "일치" if listed == current else f"불일치 (차이 {listed - current:+d})"
            print(f"  수집 목록           {listed:>7,}건   → {mark}")
        articles, addenda = q("""SELECT count(*) FILTER (WHERE a.article_no <> '부칙'), count(*) FILTER (WHERE a.article_no = '부칙')
                                 FROM articles a JOIN ordinances o ON o.id = a.ordinance_id
                                 WHERE o.org_code = %(org)s AND o.is_current""")[0]
        print(f"  조문 / 부칙         {articles:>7,}개 / {addenda:,}개")
        print(f"  인용 법령(「」)       {q('SELECT count(*) FROM law_references r JOIN ordinances o ON o.id = r.ordinance_id WHERE o.org_code = %(org)s')[0][0]:>7,}건")

        section("검색 청크 · 임베딩")
        print(f"  {'구분':<14}{'청크':>8}{'임베딩':>8}{'누락':>6}  모델")
        rows = q("""SELECT granularity, count(*), count(embedding), count(*) - count(embedding),
                           string_agg(DISTINCT embed_model, ', ')
                    FROM chunks WHERE org_code = %(org)s GROUP BY 1 ORDER BY 1""")
        label = {"article": "조 단위", "paragraph": "항 단위", "attachment": "별표·서식"}
        for g, n, e, miss, model in rows:
            print(f"  {label.get(g, g):<12}{n:>8,}{e:>8,}{miss:>6,}  {model or '-'}")
        n, e = sum(r[1] for r in rows), sum(r[2] for r in rows)
        print(f"  {'합계':<12}{n:>8,}{e:>8,}{n - e:>6,}  → 임베딩 완료율 {e / n:.1%}" if n else "  청크 없음")

        section("별표·서식 (attachments)")
        rows = q("""SELECT a.kind, a.extraction_status, count(*), sum(a.has_images::int),
                           string_agg(DISTINCT a.extraction_method, ', ')
                    FROM attachments a JOIN ordinances o ON o.id = a.ordinance_id
                    WHERE o.org_code = %(org)s GROUP BY 1, 2 ORDER BY 1, 2""")
        print(f"  {'종류':<6}{'상태':<12}{'건수':>6}{'그림포함':>8}  추출 방식")
        for kind, status, cnt, img, methods in rows:
            print(f"  {kind:<5}{status:<12}{cnt:>6,}{img:>8,}  {methods or '-'}")
        with_annex = q("""SELECT count(DISTINCT a.ordinance_id) FROM attachments a
                          JOIN ordinances o ON o.id = a.ordinance_id WHERE o.org_code = %(org)s""")[0][0]
        print(f"  별표·서식이 있는 조례: {with_annex:,}건")

        section("정합성 점검")
        problems = 0
        for desc, sql, must_be_zero in CHECKS:
            value = q(sql)[0][0]
            if must_be_zero:
                ok = value == 0
                problems += not ok
                print(f"  [{'OK' if ok else '문제'}] {desc}: {value:,}")
            else:
                print(f"  [정보] {desc}: {value:,}")

        section("최근 적재 이력 (ingest_runs)")
        for rid, started, finished, listed, ins, upd, unch, failed in q(
            """SELECT id, started_at, finished_at, listed, inserted, updated, unchanged, failed
               FROM ingest_runs WHERE org_code = %(org)s ORDER BY id DESC LIMIT 3"""
        ):
            state = "완료" if finished else "중단됨"
            print(f"  #{rid} {started:%Y-%m-%d %H:%M} {state:<4} 목록 {listed} / 신규 {ins} / 갱신 {upd} / 변경없음 {unch} / 실패 {failed}")

    print(f"\n결과: {'이상 없음' if problems == 0 else f'문제 {problems}건 — 위 [문제] 항목 확인'}")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
