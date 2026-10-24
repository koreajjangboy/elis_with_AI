-- 003: 이미지형 별표의 OCR·멀티모달 후처리를 위한 필드 (설계: docs/attachment_design.md 5절)
-- text(원문 추출본)와 image_description(AI가 그림을 해석한 설명)을 분리해,
-- 답변에서 '원문'과 'AI 해석'을 구분해 인용할 수 있게 한다.

ALTER TABLE attachments ADD COLUMN IF NOT EXISTS image_description TEXT;   -- 멀티모달 모델의 그림·도면 설명 (원문 아님)
ALTER TABLE attachments ADD COLUMN IF NOT EXISTS image_described_by TEXT;  -- 설명을 만든 모델명
ALTER TABLE attachments ADD COLUMN IF NOT EXISTS reviewed_at TIMESTAMPTZ;  -- 사람이 OCR·설명을 원문과 대조 검수한 시각 (NULL=미검수)

-- OCR·멀티모달 처리 대기열: 본문을 못 읽었거나, 그림이 있는데 아직 그림 설명이 없는 별표
CREATE INDEX IF NOT EXISTS attachments_visual_queue_idx ON attachments (id)
    WHERE extraction_status = 'needs_ocr' OR (has_images AND image_description IS NULL);
