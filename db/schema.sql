-- elis_with_AI 스키마 (PostgreSQL 15+ / pgvector 0.7+)
-- 관계: ordinances 1 ─ N articles    1 ─ N chunks
--       ordinances 1 ─ N attachments 1 ─ N chunks   (chunk는 조문 또는 별표 중 하나에 속함)
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

-- ─────────────────────────────────────────────── 별표·서식 (첨부 문서)
-- API는 별표 본문을 비워 두고 첨부파일(hwp/hwpx/txt) URL만 준다. 추출한 텍스트를 여기 두고,
-- 검색 단위(조각 + 임베딩)는 chunks에 넣어 조문과 함께 검색한다.
CREATE TABLE attachments (
    id                 BIGSERIAL PRIMARY KEY,
    ordinance_id       BIGINT  NOT NULL REFERENCES ordinances(id) ON DELETE CASCADE,
    annex_key          TEXT    NOT NULL,          -- API 별표키
    annex_no           INT,
    annex_branch_no    INT,
    kind               TEXT    NOT NULL CHECK (kind IN ('별표', '서식', '삭제', '기타')),  -- 제목 접두어로 판별
    title              TEXT    NOT NULL,          -- '[별표 2] 통ㆍ반 명칭 및 관할구역'
    related_article    TEXT,                      -- 제목의 '(제3조 관련)' → '제3조'
    file_type          TEXT,                      -- hwp / hwpx / txt …
    source_url         TEXT,                      -- 원본 다운로드 URL (화면에서 원문 링크)
    file_path          TEXT,                      -- 로컬 저장 경로
    file_hash          TEXT,                      -- 파일 sha256: 재처리 시 변경 감지
    text               TEXT,                      -- 추출된 전체 텍스트
    extraction_method  TEXT,                      -- hwp5 / hwpx / txt / document-parse …
    extraction_status  TEXT    NOT NULL DEFAULT 'pending'
                       CHECK (extraction_status IN ('pending', 'ok', 'needs_ocr', 'unsupported', 'error', 'skipped')),
    extraction_error   TEXT,
    has_images         BOOLEAN NOT NULL DEFAULT false,  -- 그림 포함 (도면 등 일부 정보가 그림에만 있을 수 있음)
    extracted_at       TIMESTAMPTZ,
    -- 이미지형 별표 후처리 (docs/attachment_design.md 5절). text(원문 추출본)와 분리해 'AI 해석'임을 구분한다
    image_description  TEXT,                      -- 멀티모달 모델의 그림·도면 설명 (원문 아님)
    image_described_by TEXT,                      -- 설명을 만든 모델명
    reviewed_at        TIMESTAMPTZ,               -- 사람이 OCR·설명을 원문과 대조 검수한 시각 (NULL=미검수)
    UNIQUE (ordinance_id, annex_key)
);
CREATE INDEX attachments_status_idx ON attachments (extraction_status);
CREATE INDEX attachments_visual_queue_idx ON attachments (id)   -- OCR·멀티모달 처리 대기열
    WHERE extraction_status = 'needs_ocr' OR (has_images AND image_description IS NULL);

-- ─────────────────────────────────────────────── 검색 청크 (임베딩 + 키워드)
CREATE TABLE chunks (
    id             BIGSERIAL PRIMARY KEY,
    article_id     BIGINT  REFERENCES articles(id) ON DELETE CASCADE,       -- 조문 청크
    attachment_id  BIGINT  REFERENCES attachments(id) ON DELETE CASCADE,    -- 별표·서식 청크
    ordinance_id   BIGINT  NOT NULL REFERENCES ordinances(id) ON DELETE CASCADE,  -- 비정규화: 조인 없이 필터
    org_code       TEXT    NOT NULL,                                              -- 비정규화: 지자체 필터
    granularity    TEXT    NOT NULL CONSTRAINT chunks_granularity_check
                           CHECK (granularity IN ('article', 'paragraph', 'attachment')),
    paragraph_no   INT,                     -- granularity='paragraph'일 때 항 번호
    chunk_seq      INT,                     -- 긴 별표를 나눈 순서
    citation       TEXT    NOT NULL,        -- '수원시 주차장 조례 제5조제2항' (답변 인용용)
    content        TEXT    NOT NULL,        -- 원문 텍스트 (항 청크는 소속 호·목 포함)
    embed_text     TEXT    NOT NULL,        -- 임베딩 입력: '[조례명] 제5조(제목)\n' + content
    tokens         TEXT    NOT NULL,        -- 본문 Kiwi 명사 토큰(공백 구분), 키워드 검색 원천
    name_tokens    TEXT    NOT NULL DEFAULT '',  -- 조례명 토큰 (제목 일치 보너스용)
    title_tokens   TEXT    NOT NULL DEFAULT '',  -- 조 제목 토큰 (제목 일치 보너스용)
    tsv            TSVECTOR GENERATED ALWAYS AS (to_tsvector('simple', tokens)) STORED,
    heading_tsv    TSVECTOR GENERATED ALWAYS AS (
                       setweight(to_tsvector('simple', name_tokens), 'A') ||
                       setweight(to_tsvector('simple', title_tokens), 'B')) STORED,
    embedding      VECTOR(1024),            -- Upstage solar-embedding-2-passage (1024차원). NULL=미임베딩
    embed_model    TEXT,
    CONSTRAINT chunks_owner_check CHECK ((article_id IS NULL) <> (attachment_id IS NULL))
);
CREATE INDEX chunks_attachment_idx ON chunks (attachment_id) WHERE attachment_id IS NOT NULL;
CREATE INDEX chunks_tsv_idx ON chunks USING gin (tsv);
CREATE INDEX chunks_heading_tsv_idx ON chunks USING gin (heading_tsv);
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

-- ─────────────────────────────────────────────── Supabase 보안
-- Supabase는 public 스키마를 REST API(anon 키)로 노출하므로 RLS를 켜서 API 접근을 막는다.
-- 정책이 없으면 API로는 읽기·쓰기 모두 불가하며, 테이블 소유자(직접 DB 연결)는 영향받지 않는다.
ALTER TABLE ordinances     ENABLE ROW LEVEL SECURITY;
ALTER TABLE articles       ENABLE ROW LEVEL SECURITY;
ALTER TABLE chunks         ENABLE ROW LEVEL SECURITY;
ALTER TABLE law_references ENABLE ROW LEVEL SECURITY;
ALTER TABLE ingest_runs    ENABLE ROW LEVEL SECURITY;
ALTER TABLE attachments    ENABLE ROW LEVEL SECURITY;
