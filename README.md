# elis_with_AI

전국 자치단체 자치법규(조례·규칙)를 기반으로 한 검색 강화형 RAG 시스템.

## 목표
- 자치법규를 조문 단위(조·항·호)로 구조화하여 수집·색인
- 하이브리드 검색(BM25 + 임베딩), 메타데이터 필터(자치단체·시행일), 리랭킹
- 답변 시 근거 조문 인용 (예: OO시 OO조례 제5조 제2항)

## 진행 단계
- [x] 0. 저장소 및 기본 골격
- [x] 1. 범위·사용 시나리오 정의
- [x] 2. 데이터 확보 (국가법령정보센터 Open API 검증)
- [x] 3. 데이터 모델·청킹 설계
- [x] 4. 검색 파이프라인 구현 (하이브리드 검색 + 근거 강제 RAG API + Streamlit)
- [x] 5. 평가셋 구축 및 성능 측정 (검색·RAG 평가 파이프라인)

## 디렉터리 구조
```
data/       # 원본/가공 데이터 (git 제외)
docs/       # 설계 문서
src/        # 소스 코드
tests/      # 테스트
```

## 설정
```bash
cp .env.example .env   # 인증키 입력
py -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
```

> open.law.go.kr에서 OPEN API 활용 신청 시 **호출하는 서버의 IP/도메인을 등록**해야 합니다.
> 미등록 시 `사용자 정보 검증에 실패하였습니다` 또는 `필수입력요소 검증에 실패하였습니다` 오류가 납니다.

## 실행
```bash
.venv/Scripts/python -m src.fetch_sample                  # 서울특별시 조례 상위 3건
.venv/Scripts/python -m src.fetch_sample --org-name 수원시 --limit 5
.venv/Scripts/python -m src.fetch_sample --org-name "서울특별시 중구" --query 주차장
.venv/Scripts/python -m src.fetch_sample --org 6410000 --sborg 3740000   # 기관코드 직접 지정
.venv/Scripts/python -m pytest -q                         # 단위 테스트 (DB·API 불필요)
ELIS_LIVE_TESTS=1 .venv/Scripts/python -m pytest tests/test_rag_scenarios.py -v -s   # 자료요구 시나리오 통합 테스트 (DB·LLM 호출)
```
결과: `data/raw/sample_ordinance.json` (파싱 결과), `data/raw/*.xml` (원본 응답)

## 수원시 MVP 파이프라인
설계: [docs/db_design.md](docs/db_design.md), 스키마: [db/schema.sql](db/schema.sql)

```bash
# 0. 클라우드 PostgreSQL에 스키마 적용 (최초 1회)
psql "$DATABASE_URL" -f db/schema.sql

# 1. 원본 XML 수집 → data/raw/3740000/{mst}.xml  (중단 후 재실행하면 이어받음)
.venv/Scripts/python -m src.collect_suwon

# 2. 청킹 결과·예상 비용 확인 (DB·API 불필요)
.venv/Scripts/python -m src.load_to_db --dry-run

# 3. DB 적재 + Upstage 임베딩  (중단 후 재실행하면 이어서 처리)
.venv/Scripts/python -m src.load_to_db
.venv/Scripts/python -m src.load_to_db --embed-only   # 임베딩만 재시도

# 4. 별표·서식: 첨부파일 수집 → 텍스트 추출·적재 → 임베딩
.venv/Scripts/python -m src.collect_attachments
.venv/Scripts/python -m src.load_attachments            # --ocr none|needs|images
.venv/Scripts/python -m src.load_to_db --embed-only

# 5. 적재·임베딩 상태 검증 (문제 있으면 종료 코드 1)
.venv/Scripts/python -m src.check_status

# 6. 검색·평가
.venv/Scripts/python -m src.search "청년 창업 지원"
.venv/Scripts/python -m src.evaluate --compare
.venv/Scripts/python -m src.evaluate_rag
```

스키마 변경은 `db/migrations/`의 SQL을 순서대로 적용한다: `python -m src.migrate db/migrations/00N_….sql`
별표·서식 설계: [docs/attachment_design.md](docs/attachment_design.md)

### 적재 상태 검증
- `python -m src.check_status`: 조례·조문·청크·별표 건수, 임베딩 완료율, 정합성 점검(임베딩 누락·차원 불일치·청크 없는 조례·빈 토큰 등), 최근 적재 이력.
- Supabase SQL Editor에서는 [`db/verify.sql`](db/verify.sql)의 쿼리를 하나씩 실행한다 (5번 정합성 점검은 모두 0이어야 정상).

## 질의응답 서비스 (FastAPI + Streamlit)

```
Streamlit(app.py) ──HTTP──▶ FastAPI(src/api.py) ──▶ 하이브리드 검색(src/search.py) ──▶ Supabase(pgvector)
                                                └─▶ 근거 강제 RAG(src/rag.py) ──▶ LLM(src/llm.py: Claude | Solar)
```

```bash
# 0. (최초 1회) 이미지형 별표 필드 마이그레이션
.venv/Scripts/python -m src.migrate db/migrations/003_attachment_visual.sql

# 1. API 서버  →  http://localhost:8000/docs
.venv/Scripts/python -m uvicorn src.api:app --port 8000

# 2. 웹 앱 (다른 터미널)  →  http://localhost:8501
.venv/Scripts/streamlit run app.py

# CLI로 바로 질문
.venv/Scripts/python -m src.rag "공영주차장 주차요금 감면 대상은?"
```

LLM은 `.env`의 `ANTHROPIC_API_KEY`가 있으면 Claude(`claude-opus-5-5`), 없으면 임베딩과 같은 Upstage 키로 Solar(`solar-pro3`)를 쓴다.
`LLM_PROVIDER=anthropic|upstage`로 고정할 수 있다.

| API | 설명 |
|---|---|
| `GET /health` | DB 건수, 사용 중인 LLM |
| `POST /search` `{"query", "limit"}` | 하이브리드 검색 결과 (LLM 미사용) |
| `POST /ask` `{"question", "top_k"}` | 근거 인용 답변: `status`(answered/unverifiable), `answer`, `citations[{source_id, label, quote}]`, `sources[]`, `notes[]` |

**근거 강제(확인 불가) 원칙** — 프롬프트 + 코드 검증 2중:
1. 시스템 프롬프트: 제공된 조문·별표만 근거로 답하고, 근거가 없으면 `확인 불가` + 빈 인용. 답변 끝에 `근거: 조례명 제N조, [별표 N]`.
2. JSON 스키마 강제 출력 → 서버가 모든 인용구를 근거 원문과 글자 단위로 대조. 불일치 인용이 있으면 1회 재생성,
   그래도 유효한 인용이 없으면 `확인 불가`로 바꾼다. 답변에 근거 표기가 빠지면 `근거:` 줄을 자동으로 붙인다.
3. 조문이 검색되면 관련 별표(`(제N조 관련)`, 본문의 `별표 N` 언급)를 근거 자료에 함께 넣는다.
4. 질문 대상 조례 점검: 질문이 특정 조례를 분명히 가리킬 때(검색된 조례 중 조례명 일치도 0.5 이상) 조례명 핵심어가 하나도 겹치지 않는
   조례를 인용하면 그 조례를 빼라는 피드백으로 1회 재생성한다. 그래도 그런 조례만 인용하면 `확인 불가`, 일부면 `notes`에 경고를 남긴다.

알려진 한계: 질문마다 새로 검색한다(대화 맥락을 검색에 쓰지 않음). 인용 검증은 '인용구가 원문에 있는가'까지만 보장하며,
답변 문장 하나하나가 인용으로 뒷받침되는지는 검사하지 않는다 → `notes`에 제외된 인용이 표시되면 원문을 확인할 것.

## 평가 (5단계)

| 평가 | 데이터셋 | 실행 | 지표 |
|---|---|---|---|
| 검색 | `tests/eval/search_eval.json` (조례 질의 19, 조문 질의 36, 9개 도메인) | `python -m src.evaluate --compare` | P@5, R@10 (관련 조례 찾기) / Hit@1, Hit@5, MRR@10 (질문→조문) |
| RAG 답변 | `tests/eval/rag_eval.json` (answered 20, 확인 불가 10) | `python -m src.evaluate_rag` | 상태 정확도, 오답변·과잉 거절, 정답 조문 인용률, 타 조례 혼입률, 1차 인용구 원문 일치율, 재생성률 |
| RAG 답변 | `tests/eval/rag_eval_ext.json` (현안 30건: answered 22, 확인 불가 8) | `python -m src.evaluate_rag --cases tests/eval/rag_eval_ext.json` | 상태 정확도, 오답변·과잉 거절, 정답 조문 인용률, 타 조례 혼입률, 1차 인용구 원문 일치율, 재생성률 |

- 정답은 검색 결과가 아니라 조례명·조 제목을 DB에서 직접 조회해 정했다 (검색 시스템에 유리한 편향 방지).
- 모든 질의에 `domain`(교통·주차, 환경·생활, 계약·재정, 복지 …)이 있어 도메인별 표가 함께 나온다. `--domain 환경·생활`로 한 도메인만 돌릴 수 있다.
- `--compare`는 보너스(조례명·조 제목 일치) 적용 전후 지표와, 정답 순위가 바뀐 질의를 보여준다.
- 확인 불가 질의는 조례로 답할 수 없는 유형(연도별 실적·통계, 타 지자체, 개별 사실관계, 상위법 조항, 연락처)으로 구성했다.
  여기에 answered로 답하면 **오답변**으로 집계한다(가장 위험한 실패).
- 결과 JSON(질의별 순위·답변·인용)은 `data/eval/`에 저장된다(git 제외).
