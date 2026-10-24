-- 002: 별표·서식(첨부파일) 저장과 검색
-- 첨부 문서 자체는 attachments에, 검색 단위(텍스트 조각 + 임베딩)는 기존 chunks에 둔다.
-- 조문 청크와 같은 테이블·인덱스로 한 번에 하이브리드 검색하기 위함이다.

CREATE TABLE IF NOT EXISTS attachments (
    id                 BIGSERIAL PRIMARY KEY,
    ordinance_id       BIGINT  NOT NULL REFERENCES ordinances(id) ON DELETE CASCADE,
    annex_key          TEXT    NOT NULL,          -- API 별표키
    annex_no           INT,                       -- 별표번호 (1, 2, …)
    annex_branch_no    INT,                       -- 별표가지번호
    kind               TEXT    NOT NULL CHECK (kind IN ('별표', '서식', '삭제', '기타')),
    title              TEXT    NOT NULL,          -- '[별표 2] 통ㆍ반 명칭 및 관할구역'
    related_article    TEXT,                      -- 제목의 '(제3조 관련)' → '제3조'
    file_type          TEXT,                      -- hwp / hwpx / txt / pdf …
    source_url         TEXT,                      -- 원본 다운로드 URL (화면에서 원문 링크로 사용)
    file_path          TEXT,                      -- 로컬 저장 경로 (data/raw/{org}/attachments/…)
    file_hash          TEXT,                      -- 파일 sha256: 재처리 시 변경 감지
    text               TEXT,                      -- 추출된 전체 텍스트
    extraction_method  TEXT,                      -- hwp5 / hwpx / txt / document-parse …
    extraction_status  TEXT    NOT NULL DEFAULT 'pending'
                       CHECK (extraction_status IN ('pending', 'ok', 'needs_ocr', 'unsupported', 'error', 'skipped')),
    extraction_error   TEXT,
    has_images         BOOLEAN NOT NULL DEFAULT false,  -- 그림 포함 (도면 등 일부 정보가 그림에만 있을 수 있음)
    extracted_at       TIMESTAMPTZ,
    UNIQUE (ordinance_id, annex_key)
);
CREATE INDEX IF NOT EXISTS attachments_status_idx ON attachments (extraction_status);
ALTER TABLE attachments ENABLE ROW LEVEL SECURITY;

-- chunks: 조문 청크(article_id) 또는 별표 청크(attachment_id) 중 하나에 속한다
ALTER TABLE chunks ALTER COLUMN article_id DROP NOT NULL;
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS attachment_id BIGINT REFERENCES attachments(id) ON DELETE CASCADE;
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS chunk_seq INT;  -- 긴 별표를 나눈 순서
ALTER TABLE chunks DROP CONSTRAINT IF EXISTS chunks_granularity_check;
ALTER TABLE chunks ADD CONSTRAINT chunks_granularity_check
    CHECK (granularity IN ('article', 'paragraph', 'attachment'));
ALTER TABLE chunks DROP CONSTRAINT IF EXISTS chunks_owner_check;
ALTER TABLE chunks ADD CONSTRAINT chunks_owner_check
    CHECK ((article_id IS NULL) <> (attachment_id IS NULL));
CREATE INDEX IF NOT EXISTS chunks_attachment_idx ON chunks (attachment_id) WHERE attachment_id IS NOT NULL;
