-- 001: 조례명·조 제목 일치 보너스용 컬럼과 인용 표기 수정
-- 적용 후 `python -m src.load_to_db --retokenize`로 토큰을 다시 계산하고,
-- `--embed-only`로 표기 수정 때문에 NULL이 된 임베딩을 다시 만든다.

ALTER TABLE chunks
    ADD COLUMN IF NOT EXISTS name_tokens  TEXT NOT NULL DEFAULT '',
    ADD COLUMN IF NOT EXISTS title_tokens TEXT NOT NULL DEFAULT '';

ALTER TABLE chunks
    ADD COLUMN IF NOT EXISTS heading_tsv TSVECTOR GENERATED ALWAYS AS (
        setweight(to_tsvector('simple', name_tokens), 'A') ||
        setweight(to_tsvector('simple', title_tokens), 'B')) STORED;

CREATE INDEX IF NOT EXISTS chunks_heading_tsv_idx ON chunks USING gin (heading_tsv);

-- 가지번호 조문 인용 표기: '제3의2조' → '제3조의2'
-- 임베딩 입력이 바뀐 청크(항 청크 머리말)는 임베딩을 비워 다시 만든다.
UPDATE chunks
SET citation   = regexp_replace(citation, '제(\d+)의(\d+)조', '제\1조의\2', 'g'),
    embedding  = CASE WHEN embed_text ~ '제\d+의\d+조' THEN NULL ELSE embedding END,
    embed_text = regexp_replace(embed_text, '제(\d+)의(\d+)조', '제\1조의\2', 'g')
WHERE citation ~ '제\d+의\d+조' OR embed_text ~ '제\d+의\d+조';
