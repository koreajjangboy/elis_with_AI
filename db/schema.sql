-- elis_with_AI 스키마 (PostgreSQL 15+ / pgvector 0.7+)
-- 관계: ordinances 1 ─ N articles 1 ─ N chunks
--       ordinances 1 ─ N law_references
-- 설계 설명: docs/db_design.md

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- ─────────────────────────────────────────────── 조례 메타데이터 (버전 단위)
CREATE TABLE ordinances (
    id                 BIGSERIAL PRIMARY KEY,
    mst                BIGINT      NOT NULL UNIQUE,  -- 자치법규일련번호: 개정될 때마다 바뀜 (버전 키)
    ordin_id           BIGINT      NOT NULL,         -- 자치법규ID: 개정과 무관하게 유지 (법규 키)
    name               TEXT        NOT NULL,
    org_name           TEXT        NOT NULL,         -- '경기도 수원시'
    org_code           TEXT        NOT NULL,         -- 요청에 사용한 기관코드 '3740000'
    kind               TEXT        NOT NULL,         -- 조례/규칙/...
    promulgation_date  DATE,
    promulgation_no    TEXT,
    enforcement_date   DATE,
    revision_type      TEXT,                         -- 제정/일부개정/전부개정...
    department         TEXT,
    field              TEXT,                         -- 자치법규분야명
    is_current         BOOLEAN     NOT NULL DEFAULT TRUE,
    content_hash       TEXT        NOT NULL,         -- 본문 XML 해시: 재수집 시 변경 감지
    fetched_at         TIMESTAMPTZ NOT NULL DEFAULT now()
);
-- 한 법규에 현행 버전은 하나만
CREATE UNIQUE INDEX ordinances_current_uq ON ordinances (ordin_id) WHERE is_current;
CREATE INDEX ordinances_org_idx ON ordinances (org_code, kind) WHERE is_current;
CREATE INDEX ordinances_name_trgm_idx ON ordinances USING gin (name gin_trgm_ops);  -- 조례명 부분일치

-- ─────────────────────────────────────────────── 조문 (화면 표시·인용용 원본 구조)
CREATE TABLE articles (
    id             BIGSERIAL PRIMARY KEY,
    ordinance_id   BIGINT  NOT NULL REFERENCES ordinances(id) ON DELETE CASCADE,
    seq            INT     NOT NULL,              -- 조례 내 순서
    article_key    TEXT    NOT NULL,              -- API 조문번호 '000200'
    article_no     TEXT    NOT NULL,              -- '2', '3의2' (부칙은 '부칙')
    title          TEXT,
    chapter        TEXT,                          -- 소속 장·절 '제2장 주차장의 관리'
    body           TEXT    NOT NULL,              -- 조 전체 원문
    structure      JSONB   NOT NULL,              -- {paragraphs:[{number,text,items:[{number,text,subitems}]}]}
    revision_notes TEXT[],                        -- 본문에서 분리한 '<개정 2024. 12. 30.>' 표기
    UNIQUE (ordinance_id, seq)
);

-- ─────────────────────────────────────────────── 검색 청크 (임베딩 + 키워드)
CREATE TABLE chunks (
    id             BIGSERIAL PRIMARY KEY,
    article_id     BIGINT  NOT NULL REFERENCES articles(id) ON DELETE CASCADE,
    ordinance_id   BIGINT  NOT NULL REFERENCES ordinances(id) ON DELETE CASCADE,  -- 비정규화: 조인 없이 필터
    org_code       TEXT    NOT NULL,                                              -- 비정규화: 지자체 필터
    granularity    TEXT    NOT NULL CHECK (granularity IN ('article', 'paragraph')),
    paragraph_no   INT,                     -- granularity='paragraph'일 때 항 번호
    citation       TEXT    NOT NULL,        -- '수원시 주차장 조례 제5조제2항' (답변 인용용)
    content        TEXT    NOT NULL,        -- 원문 텍스트 (항 청크는 소속 호·목 포함)
    embed_text     TEXT    NOT NULL,        -- 임베딩 입력: '[조례명] 제5조(제목)\n' + content
    tokens         TEXT    NOT NULL,        -- Kiwi 형태소 분석 결과(공백 구분), 키워드 검색 원천
    tsv            TSVECTOR GENERATED ALWAYS AS (to_tsvector('simple', tokens)) STORED,
    embedding      VECTOR(1024),            -- Upstage solar-embedding-2-passage (1024차원). NULL=미임베딩
    embed_model    TEXT
);
CREATE INDEX chunks_tsv_idx ON chunks USING gin (tsv);
CREATE INDEX chunks_embedding_idx ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX chunks_filter_idx ON chunks (org_code, ordinance_id);

-- ─────────────────────────────────────────────── 인용 법령 (상위법령 근사치)
CREATE TABLE law_references (
    ordinance_id   BIGINT NOT NULL REFERENCES ordinances(id) ON DELETE CASCADE,
    article_id     BIGINT NOT NULL REFERENCES articles(id) ON DELETE CASCADE,
    law_name       TEXT   NOT NULL,         -- 「」 안의 법령명 '주차장법'
    is_ordinance   BOOLEAN NOT NULL,        -- 자기 지자체 조례 인용 여부 ('수원시 사무위탁 조례')
    PRIMARY KEY (ordinance_id, article_id, law_name)
);
CREATE INDEX law_references_law_idx ON law_references (law_name);

-- ─────────────────────────────────────────────── 수집 이력 (재시작·감사용)
CREATE TABLE ingest_runs (
    id            BIGSERIAL PRIMARY KEY,
    org_code      TEXT        NOT NULL,
    started_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at   TIMESTAMPTZ,
    listed        INT,
    inserted      INT,
    updated       INT,
    unchanged     INT,
    failed        INT,
    errors        JSONB
);
