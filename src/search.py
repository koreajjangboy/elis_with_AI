"""하이브리드 검색: 벡터 + 본문 키워드(RRF) + 조례명·조 제목 일치 보너스.

점수 = RRF(벡터 순위, 본문 키워드 순위) + NAME_BOOST × 조례명 일치도 + TITLE_BOOST × 조 제목 일치도

- 후보: 벡터 상위 50 ∪ 본문 키워드 상위 50 ∪ 조례명·조 제목(heading_tsv) 일치 상위 200
- 검색 단위: 조문 청크와 별표·서식 청크(attachment_id)를 함께 검색한다
- 일치도: 검색어 핵심어 중 조례명(또는 조 제목)에 들어 있는 비율. 핵심어마다 IDF 가중치를 주어
  '수원시'처럼 대부분의 조례명에 들어 있는 단어는 일치해도 점수에 거의 기여하지 않는다.
- RRF 최대값은 약 0.033(=2/61)이므로 NAME_BOOST 0.05면 조례명이 완전히 일치하는 조문이
  본문만 일치하는 조문보다 항상 위에 온다.

실행:
    python -m src.search "청년 창업 지원"
    python -m src.search "공영주차장 주차요금" --no-boost   # 보너스 없이 비교

내부 처리 과정(토큰화·후보 수·상위 결과 점수)은 stderr로 출력된다. 끄려면 ELIS_DEBUG=0 (src/debug.py).
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass

import psycopg

from src.collect_suwon import SBORG
from src.debug import dprint, section
from src.embedder import UpstageEmbedder, to_pgvector
from src.tokenizer import tokenize, tokenize_many

RRF_K = 60
VEC_CANDIDATES = 50
KW_CANDIDATES = 50
HEAD_CANDIDATES = 200
NAME_BOOST = 0.05
TITLE_BOOST = 0.03
GENERIC_TERM_RATIO = 0.3  # 조례명의 30% 이상에 나오는 단어는 변별력 없는 일반어로 본다
# 조례당 최대 결과 수. 질문에 답할 조문을 찾을 때는 제한하지 않는다(정답 조문이 잘릴 수 있음).
# '관련 조례 목록' 화면처럼 다양성이 필요할 때만 1~2로 지정한다.
MAX_PER_ORDINANCE: int | None = None
# 서식(신청서·대장 등 빈 양식)은 '지원 대상', '신청인' 같은 단어가 많아 내용 질문에도 걸린다.
# 검색어에 양식을 찾는 의도가 없으면 점수를 낮춘다 (기준을 담은 서식도 있어 제외하지는 않음).
FORM_PENALTY = 0.6
FORM_INTENT_WORDS = ("서식", "양식", "신청서", "신고서", "대장", "증명서", "확인서", "동의서",
                     "계획서", "보고서", "명부", "서약서", "표지", "별지")

CANDIDATE_SQL = """
WITH vec AS (
    SELECT id, row_number() OVER (ORDER BY embedding <=> %(emb)s::vector) AS r
    FROM chunks
    WHERE org_code = %(org)s AND embedding IS NOT NULL
    ORDER BY embedding <=> %(emb)s::vector
    LIMIT %(vec_cand)s
), kw AS (
    -- 핵심어 중 하나라도(OR) 들어간 청크를 후보로 하되, 모두(AND) 들어간 청크를 먼저 정렬한다
    SELECT id, row_number() OVER (ORDER BY tsv @@ all_q DESC, ts_rank_cd(tsv, any_q) DESC) AS r
    FROM chunks, to_tsquery('simple', %(all_q)s) all_q, to_tsquery('simple', %(any_q)s) any_q
    WHERE org_code = %(org)s AND tsv @@ any_q
    ORDER BY tsv @@ all_q DESC, ts_rank_cd(tsv, any_q) DESC
    LIMIT %(kw_cand)s
), head AS (
    SELECT id
    FROM chunks, to_tsquery('simple', %(head_q)s) q
    WHERE org_code = %(org)s AND heading_tsv @@ q
    ORDER BY ts_rank(heading_tsv, q) DESC
    LIMIT %(head_cand)s
), cand AS (
    SELECT id FROM vec UNION SELECT id FROM kw UNION SELECT id FROM head
)
SELECT c.id, c.article_id, c.attachment_id, c.ordinance_id, c.citation, c.content, o.name,
       COALESCE(a.title, att.title), o.enforcement_date, c.name_tokens, c.title_tokens, vec.r, kw.r,
       att.source_url, att.kind
FROM cand
JOIN chunks c ON c.id = cand.id
JOIN ordinances o ON o.id = c.ordinance_id AND o.is_current
LEFT JOIN articles a ON a.id = c.article_id
LEFT JOIN attachments att ON att.id = c.attachment_id
LEFT JOIN vec ON vec.id = c.id
LEFT JOIN kw ON kw.id = c.id
"""


@dataclass
class SearchHit:
    chunk_id: int
    article_id: int | None  # 조문 청크
    attachment_id: int | None  # 별표·서식 청크
    ordinance_id: int
    citation: str
    content: str
    ordinance_name: str
    article_title: str
    enforcement_date: object
    vec_rank: int | None
    kw_rank: int | None
    source_url: str | None = None  # 별표·서식 원본 파일 (화면에서 원문 링크)
    attachment_kind: str | None = None  # '별표' | '서식'
    rrf: float = 0.0
    name_match: float = 0.0
    title_match: float = 0.0
    score: float = 0.0


# ---------------------------------------------------------------- 핵심어 가중치

_name_df_cache: dict[str, tuple[int, dict[str, int]]] = {}


def name_document_frequency(conn: psycopg.Connection, org_code: str) -> tuple[int, dict[str, int]]:
    """(현행 조례 수, 단어별 그 단어가 조례명에 들어 있는 조례 수). 프로세스당 1회 계산."""
    if org_code not in _name_df_cache:
        names = [r[0] for r in conn.execute(
            "SELECT name FROM ordinances WHERE org_code = %s AND is_current", (org_code,)
        ).fetchall()]
        df: dict[str, int] = {}
        for tokens in tokenize_many(names):
            for term in set(tokens.split()):
                df[term] = df.get(term, 0) + 1
        _name_df_cache[org_code] = (len(names), df)
    return _name_df_cache[org_code]


def term_weights(terms: list[str], n_docs: int, df: dict[str, int]) -> dict[str, float]:
    """IDF 가중치. 조례명의 GENERIC_TERM_RATIO 이상에 나오는 일반어는 0."""
    weights = {}
    for t in terms:
        d = df.get(t, 0)
        weights[t] = 0.0 if n_docs and d / n_docs >= GENERIC_TERM_RATIO else math.log((n_docs + 1) / (d + 1)) + 1
    return weights


def to_tsquery_text(terms: list[str], operator: str) -> str:
    # tsquery 문법 오류를 막기 위해 영숫자·한글만 남긴다
    clean = ["".join(ch for ch in t if ch.isalnum()) for t in terms]
    return f" {operator} ".join(t for t in dict.fromkeys(clean) if t)


# ---------------------------------------------------------------- 점수 계산 (DB 무관)

def match_ratio(field_tokens: str, weights: dict[str, float]) -> float:
    total = sum(weights.values())
    if not total:
        return 0.0
    present = set(field_tokens.split())
    return sum(w for t, w in weights.items() if t in present) / total


def score_candidates(
    rows: list[tuple],
    weights: dict[str, float],
    limit: int,
    boost: bool = True,
    max_per_ordinance: int | None = MAX_PER_ORDINANCE,
    form_intent: bool = False,
) -> list[SearchHit]:
    hits = []
    for (cid, aid, att_id, oid, citation, content, name, title, eff, name_tok, title_tok, vec_r, kw_r, url, kind) in rows:
        hit = SearchHit(cid, aid, att_id, oid, citation, content, name, title or "", eff, vec_r, kw_r, url, kind)
        hit.rrf = (1 / (RRF_K + vec_r) if vec_r else 0) + (1 / (RRF_K + kw_r) if kw_r else 0)
        hit.name_match = match_ratio(name_tok, weights)
        # 조 제목은 조례명으로 이미 설명되지 않는 나머지 핵심어로만 평가한다.
        # ('재난관리기금 용도' → '용도'만) 그래야 같은 조례 안의 조문끼리 변별력이 생긴다.
        in_name = set(name_tok.split())
        hit.title_match = match_ratio(title_tok, {t: w for t, w in weights.items() if t not in in_name})
        hit.score = hit.rrf + (NAME_BOOST * hit.name_match + TITLE_BOOST * hit.title_match if boost else 0)
        if boost and kind == "서식" and not form_intent:
            hit.score *= FORM_PENALTY
        if hit.score > 0:
            hits.append(hit)
    hits.sort(key=lambda h: (-h.score, h.chunk_id))

    # 같은 조문(조·항 청크)·같은 별표(나뉜 조각)는 하나만, 한 조례는 최대 max_per_ordinance개까지
    results, seen_units, per_ordinance = [], set(), {}
    for h in hits:
        unit = ("article", h.article_id) if h.article_id else ("attachment", h.attachment_id)
        if unit in seen_units or (max_per_ordinance and per_ordinance.get(h.ordinance_id, 0) >= max_per_ordinance):
            continue
        seen_units.add(unit)
        per_ordinance[h.ordinance_id] = per_ordinance.get(h.ordinance_id, 0) + 1
        results.append(h)
        if len(results) >= limit:
            break
    return results


# ---------------------------------------------------------------- 검색

def hybrid_search(
    conn: psycopg.Connection,
    embedder: UpstageEmbedder,
    query: str,
    org_code: str = SBORG,
    limit: int = 10,
    boost: bool = True,
    max_per_ordinance: int | None = MAX_PER_ORDINANCE,
) -> list[SearchHit]:
    all_terms = list(dict.fromkeys(tokenize(query).split()))
    n_docs, df = name_document_frequency(conn, org_code)
    weights = term_weights(all_terms, n_docs, df)
    # 일반어를 빼고 검색하되, 일반어뿐인 질의(예: '수원시 조례')는 그대로 쓴다
    key_terms = [t for t in all_terms if weights[t] > 0] or all_terms

    section("하이브리드 검색: 질의 분석")
    dprint(f"  원본 검색어   : {query!r}")
    dprint(f"  Kiwi 토큰     : {all_terms or '(없음)'}")
    dprint("  IDF 가중치    : " + (", ".join(f"{t}={w:.2f}" + ("(일반어)" if w == 0 else "") for t, w in weights.items()) or "(없음)"))
    dprint(f"  검색 핵심어   : {key_terms or '(없음)'}  → AND: {to_tsquery_text(key_terms, '&') or '-'} / OR: {to_tsquery_text(key_terms, '|') or '-'}")

    started = time.perf_counter()
    embedding = embedder.embed_query(query)
    embed_ms = (time.perf_counter() - started) * 1000
    params = {
        "emb": to_pgvector(embedding),
        "org": org_code,
        "vec_cand": VEC_CANDIDATES,
        "kw_cand": KW_CANDIDATES,
        "head_cand": HEAD_CANDIDATES if boost else 0,
        "head_q": to_tsquery_text(key_terms, "|") or "''",
        "all_q": to_tsquery_text(key_terms, "&") or "''",
        "any_q": to_tsquery_text(key_terms, "|") or "''",
    }
    started = time.perf_counter()
    rows = conn.execute(CANDIDATE_SQL, params).fetchall()
    sql_ms = (time.perf_counter() - started) * 1000
    form_intent = any(w in query for w in FORM_INTENT_WORDS)
    hits = score_candidates(rows, weights, limit, boost, max_per_ordinance, form_intent)

    # 후보 행 구조: (..., vec_rank[11], kw_rank[12], ...) — vec·kw 순위가 둘 다 없으면 제목 일치(head)로만 들어온 후보
    n_vec = sum(1 for r in rows if r[11])
    n_kw = sum(1 for r in rows if r[12])
    n_head_only = sum(1 for r in rows if not r[11] and not r[12])
    dprint(f"  후보 청크     : {len(rows)}개 (벡터 {n_vec} · 본문 키워드 {n_kw} · 제목 일치만 {n_head_only})"
           f"  | 임베딩 {embed_ms:.0f}ms · SQL {sql_ms:.0f}ms")
    dprint(f"  옵션          : 보너스 {'켜짐' if boost else '꺼짐'} · 서식 의도 {'있음' if form_intent else '없음(서식 감점)'}"
           f" · 조례당 최대 {max_per_ordinance or '제한 없음'}")
    print_hits(hits)
    return hits


def print_hits(hits: list[SearchHit]) -> None:
    section(f"하이브리드 검색: 상위 {len(hits)}건")
    if not hits:
        dprint("  (결과 없음)")
        return
    # 한글은 화면에서 2칸을 차지하므로 머리글은 폭을 직접 맞춘다 (아래 행 서식과 같은 열 위치)
    dprint("  순위     점수      RRF조례명  제목 벡터 본문  구분    조례명 / 인용")
    for i, h in enumerate(hits, 1):
        kind = "조문" if h.article_id else (h.attachment_kind or "별표")
        dprint(f"  {i:<5}{h.score:>8.4f}  {h.rrf:>7.4f}{h.name_match:>6.0%}{h.title_match:>6.0%}"
               f"{h.vec_rank or '-':>5}{h.kw_rank or '-':>5}  {kind:<6}{h.ordinance_name}")
        dprint(" " * 56 + f"└ {h.citation}" + (f"({h.article_title})" if h.article_id and h.article_title else ""))


def main() -> None:
    from src.db import connect

    parser = argparse.ArgumentParser(description="수원시 조례 하이브리드 검색")
    parser.add_argument("query")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--no-boost", action="store_true", help="조례명·조 제목 보너스 없이 검색 (비교용)")
    args = parser.parse_args()

    with connect() as conn:
        hits = hybrid_search(conn, UpstageEmbedder.from_env(), args.query, limit=args.limit, boost=not args.no_boost)
    print(f"검색어: {args.query}  (핵심어: {tokenize(args.query)})\n")
    for i, h in enumerate(hits, 1):
        detail = (f"벡터 {h.vec_rank or '-'}위, 본문 {h.kw_rank or '-'}위, "
                  f"조례명 일치 {h.name_match:.0%}, 조 제목 일치 {h.title_match:.0%}")
        print(f"[{i}] {h.citation}({h.article_title})  score {h.score:.4f}")
        print(f"    {detail}")
        if h.source_url:
            print(f"    원문: {h.source_url}")
        print(f"    {h.content[:120].replace(chr(10), ' ')}\n")


if __name__ == "__main__":
    main()
