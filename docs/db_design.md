# DB 설계 및 청킹·적재 전략 (수원시 MVP)

대상: 경기도 수원시 조례 약 707건 (`org=6410000`, `sborg=3740000`, 2026-10 기준)
스키마: [`db/schema.sql`](../db/schema.sql)

## 1. 테이블 구조

```
ordinances (조례 버전)  1 ── N  articles (조문)  1 ── N  chunks (검색 단위)
     │                                                    ├ embedding  vector(1024)  ← HNSW
     └── 1 ── N  law_references (「」 인용 법령)            └ tsv        tsvector      ← GIN
```

| 테이블 | 역할 | 핵심 설계 |
|---|---|---|
| `ordinances` | 조례 메타데이터 | `mst`(일련번호)는 개정마다 바뀌는 **버전 키**, `ordin_id`는 유지되는 **법규 키**. `is_current` 부분 유니크 인덱스로 현행 1건 보장 |
| `articles` | 조문 원본 구조 | 화면 표시·인용용. 항·호·목 구조는 `structure JSONB`에 보관 (검색에는 쓰지 않음) |
| `chunks` | 검색 단위 | 벡터 + 키워드 인덱스를 한 행에. `ordinance_id`, `org_code`를 비정규화해 조인 없이 필터 |
| `law_references` | 상위법령 근사 | API에 조례별 연계 법령이 없어 본문 「」 인용으로 대체. "주차장법 관련 조례" 같은 질의에 사용 |
| `ingest_runs` | 수집 이력 | 건수·실패 목록 기록 |

## 2. 청킹 전략

**원칙: 조(條) 단위가 기본, 긴 조만 항(項) 단위로 분할. 호·목은 쪼개지 않는다.**

- 호·목은 단독으로는 의미가 없다 (예: `2. 국가유공자` — 무엇의 2호인지 알 수 없음). 항 청크 안에 포함한다.
- 분할 기준: 조 본문이 약 800자를 넘고 항이 2개 이상이면 항마다 청크를 만들고, 조 전체 청크도 함께 둔다 (`granularity` 컬럼으로 구분).
  - 짧은 질의는 항 청크가, 맥락이 필요한 질의는 조 청크가 걸린다. 결과는 같은 `article_id`로 묶어 중복 제거한다.
- **맥락 헤더**: 임베딩 입력(`embed_text`)에 조례명과 조 제목을 붙인다.
  `[수원시 주차장 설치 및 관리 조례] 제5조(요금의 감면) ② ...`
  "수원시"나 주제어가 본문에 없어도 검색되게 하는 가장 효과가 큰 장치다.
- **개정 표기 분리**: `<개정 2024. 12. 30.>` 같은 표기는 `revision_notes`로 빼고 검색 텍스트에서 제거한다 (임베딩 노이즈).
- **부칙**: `article_no='부칙'`인 article로 저장하되 기본 검색에서는 제외한다 (시행일·경과조치 질의에서만 사용).
- 예상 규모: 707건 × 평균 20~30조 ≈ 조 청크 1.5~2만 개 + 항 청크 일부. 이 규모에서는 HNSW도 충분히 빠르다.

## 3. 키워드 검색: 한국어 처리

PostgreSQL 기본 전문검색에는 한국어 형태소 분석기가 없다. 선택지:

| 방식 | 장점 | 단점 |
|---|---|---|
| **Kiwi(파이썬) 토큰 → `to_tsvector('simple')`** ✅ | 어떤 관리형 Postgres에서도 동작, 명사 추출 품질 좋음 | 질의도 같은 Kiwi로 토큰화해야 함 |
| pgroonga / pg_bigm 확장 | DB 안에서 처리 | 관리형 DB 지원이 제한적 (Supabase는 pgroonga 지원) |
| ParadeDB `pg_search` (진짜 BM25) | 점수 품질 최고 | 배포 선택지가 좁음 |

MVP는 **Kiwi + `simple` tsvector**를 쓴다. 조례명 부분일치(예: "주차장 조례")는 `pg_trgm` 인덱스로 보완한다.
나중에 BM25 품질이 부족하면 `tokens` 컬럼을 그대로 둔 채 검색 엔진만 바꿀 수 있다.

## 4. 하이브리드 검색 쿼리 (RRF, 1회 왕복)

```sql
-- $1: 질의 임베딩, $2: Kiwi로 토큰화한 질의 ('주차 요금 감면'), $3: org_code
WITH vec AS (
    SELECT id, row_number() OVER (ORDER BY embedding <=> $1) AS r
    FROM chunks WHERE org_code = $3
    ORDER BY embedding <=> $1 LIMIT 50
), kw AS (
    SELECT id, row_number() OVER (ORDER BY ts_rank_cd(tsv, q) DESC) AS r
    FROM chunks, plainto_tsquery('simple', $2) q
    WHERE org_code = $3 AND tsv @@ q
    ORDER BY ts_rank_cd(tsv, q) DESC LIMIT 50
)
SELECT c.id, c.citation, c.content, o.name, o.enforcement_date,
       COALESCE(1.0/(60+vec.r), 0) + COALESCE(1.0/(60+kw.r), 0) AS score
FROM vec FULL JOIN kw USING (id)
JOIN chunks c ON c.id = COALESCE(vec.id, kw.id)
JOIN ordinances o ON o.id = c.ordinance_id AND o.is_current
ORDER BY score DESC LIMIT 20;
```

- `plainto_tsquery`는 AND 조건이라 결과가 적을 수 있다. 0건이면 OR(`to_tsquery('a | b')`)로 재시도한다.
- 리랭커는 이 상위 20건에만 적용한다 (2단계 과제).

## 5. 수집·적재 파이프라인

**단계를 분리해 각각 다시 실행해도 안전하게(멱등) 만든다.** 네트워크·임베딩 실패가 서로 영향을 주지 않는다.

```
src/
  law_api.py, ordin_parser.py      # 기존 재사용
  ingest/
    collect.py   # ① 목록 전체 페이지 조회 → 본문 XML을 data/raw/3740000/{mst}.xml로 저장
    load.py      # ② XML 파싱 → ordinances/articles/chunks/law_references 적재 (임베딩 제외)
    embed.py     # ③ embedding IS NULL인 청크를 배치로 임베딩
    chunker.py   # 조/항 분할, 맥락 헤더, 개정 표기 분리
    tokenizer.py # Kiwi 토큰화 (적재·질의 공용)
  db.py          # 커넥션 풀 (psycopg 3)
```

**① collect.py**
- `display=100`으로 8페이지 순회 → 목록에서 `mst` 확보 (총건수와 실제 수집 건수를 비교·기록).
- 이미 저장된 `{mst}.xml`은 건너뜀 → 중단돼도 이어서 실행 가능.
- 요청 간 0.3~0.5초 대기, 실패 시 지수 백오프 3회 재시도, 최종 실패는 `ingest_runs.errors`에 기록.
- 원본 XML을 남겨 두면 파서를 고쳐도 API를 다시 부르지 않고 ②부터 재실행할 수 있다.

**② load.py** — 조례 1건 = 트랜잭션 1개
- `content_hash`가 같으면 건너뜀(unchanged).
- 같은 `ordin_id`에 새 `mst`가 오면 기존 행을 `is_current=false`로 바꾸고 새 버전 insert.
- `mst`가 같은데 해시가 다르면(파서 변경 등) 해당 조례의 articles를 지우고 다시 넣음 (CASCADE로 chunks 정리).
- 실패한 조례만 롤백되고 나머지는 계속 진행.

**③ embed.py**
- `WHERE embedding IS NULL` 대상만 64~128개씩 배치 임베딩 → 중단돼도 이어서 실행 가능.
- `embed_model` 컬럼에 모델명을 기록해, 모델을 바꾸면 `UPDATE ... SET embedding=NULL`로 재임베딩.

## 6. FastAPI 연동·배포 팁

- **DB 왕복 1회**: 위 RRF 쿼리처럼 벡터·키워드·메타데이터 조인을 SQL 한 번에 처리한다. 파이썬에서 결과를 합치지 않는다.
- **비동기 커넥션 풀**: `psycopg_pool.AsyncConnectionPool`(또는 asyncpg)을 앱 시작 시 생성하고, `pgvector.psycopg.register_vector_async`로 벡터 타입을 등록한다.
- **질의 임베딩이 가장 느린 단계**:
  - 로컬 bge-m3(약 2GB)를 웹 서버에 올리면 콜드스타트와 메모리 비용이 크다. MVP는 **임베딩 API** 또는 별도 임베딩 서비스를 권장한다.
  - 같은 질의는 LRU 캐시로 재사용한다.
- **필터는 SQL로 내린다**: 지자체·시행일·조례종류 필터는 `WHERE`에 넣는다. `org_code`가 chunks에 비정규화되어 있어 조인 없이 걸린다. 여러 지자체로 확장해 필터가 매우 선택적이 되면 `hnsw.iterative_scan = relaxed_order`(pgvector 0.8+)를 켠다.
- **Kiwi는 프로세스당 1회 로드**: 앱 시작 시 싱글턴으로 띄운다 (적재 때와 같은 버전·사용자 사전).
- **호스팅**: pgvector를 지원하는 관리형 Postgres(Supabase, Neon, AWS RDS 등). 로컬 개발은 `pgvector/pgvector:pg17` Docker 이미지.
- **응답 구성**: 검색은 chunk 단위, 표시는 `articles.structure`로 조 전체를 렌더링하고 `citation`으로 근거를 표기한다.

## 7. 결정이 필요한 사항

1. **임베딩 모델** (스키마의 벡터 차원이 이것에 따라 정해짐)
   - bge-m3 (1024차원, 무료·로컬, 한국어 양호)
   - OpenAI `text-embedding-3-small` (1536차원, API, 저렴)
   - Upstage `solar-embedding` (4096차원 → `halfvec` 필요, 한국어 특화)
2. **배포 대상 DB**: Supabase / Neon / 자체 Docker 중 선택 (한국어 확장 사용 가능 여부에 영향)
