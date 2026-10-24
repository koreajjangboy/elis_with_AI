-- 적재·임베딩 상태 검증 쿼리 (Supabase SQL Editor 또는 psql에서 실행)
-- 파이썬으로 한 번에 보려면: python -m src.check_status  (문제 있으면 종료 코드 1)
-- 수원시 org_code = '3740000'

-- 1. 조례 건수 (현행 / 전체 버전)
SELECT count(*) FILTER (WHERE is_current) AS current_ordinances, count(*) AS all_versions
FROM ordinances WHERE org_code = '3740000';

-- 2. 조례 종류별 건수
SELECT kind, count(*) FROM ordinances
WHERE org_code = '3740000' AND is_current GROUP BY kind ORDER BY count(*) DESC;

-- 3. 청크·임베딩 완료율 (구분별)
SELECT granularity,
       count(*)                                   AS chunks,
       count(embedding)                           AS embedded,
       count(*) - count(embedding)                AS missing,
       round(100.0 * count(embedding) / count(*), 1) AS pct,
       string_agg(DISTINCT embed_model, ', ')     AS models
FROM chunks WHERE org_code = '3740000'
GROUP BY ROLLUP (granularity) ORDER BY granularity NULLS LAST;

-- 4. 별표·서식 추출 상태
SELECT kind, extraction_status, count(*), sum(has_images::int) AS with_images,
       string_agg(DISTINCT extraction_method, ', ') AS methods
FROM attachments GROUP BY 1, 2 ORDER BY 1, 2;

-- 5. 정합성 점검: 모두 0이어야 정상
SELECT '임베딩 누락 청크' AS check, count(*) FROM chunks WHERE embedding IS NULL
UNION ALL
SELECT '임베딩 차원 ≠ 1024', count(*) FROM chunks WHERE embedding IS NOT NULL AND vector_dims(embedding) <> 1024
UNION ALL
SELECT '청크 없는 현행 조례', count(*) FROM ordinances o
 WHERE o.is_current AND NOT EXISTS (SELECT 1 FROM chunks c WHERE c.ordinance_id = o.id)
UNION ALL
SELECT '키워드 토큰이 빈 청크', count(*) FROM chunks WHERE tokens = ''
UNION ALL
SELECT '청크 없는 별표·서식(삭제 제외)', count(*) FROM attachments a
 WHERE a.extraction_status <> 'skipped' AND NOT EXISTS (SELECT 1 FROM chunks c WHERE c.attachment_id = a.id);

-- 6. 문제 행 확인 (5번에서 0이 아닌 항목이 있을 때)
SELECT id, granularity, citation, left(content, 60) FROM chunks WHERE tokens = '' LIMIT 50;

-- 7. 표본 확인: 특정 조례의 조문·별표 청크가 모두 들어갔는지
SELECT c.granularity, c.citation, c.embedding IS NOT NULL AS embedded, left(c.content, 80) AS preview
FROM chunks c JOIN ordinances o ON o.id = c.ordinance_id
WHERE o.is_current AND o.name LIKE '%주차장%'
ORDER BY o.name, c.attachment_id NULLS FIRST, c.id
LIMIT 100;

-- 8. 벡터 검색 동작 확인: 임의 청크와 가장 가까운 청크 5개 (자기 자신이 1위, 거리 0이어야 정상)
WITH probe AS (SELECT id, embedding FROM chunks WHERE embedding IS NOT NULL ORDER BY id LIMIT 1)
SELECT c.id, c.citation, round((c.embedding <=> p.embedding)::numeric, 4) AS distance
FROM chunks c, probe p ORDER BY c.embedding <=> p.embedding LIMIT 5;
