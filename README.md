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
| `POST /ask` `{"question", "top_k"}` | 근거 인용 답변: `status`(answered/unverifiable), `answer`, `citations[{source_id, label, quote}]`, `sources[]`, `notes[]`, `unconfirmed[]`(조례로 확인되지 않은 사항 → 부분 답변) |

**근거 강제 원칙** — 목표는 의원·정책지원관이 그대로 인용해도 되는 답이다. 가장 피해야 할 실패는 **잘못된 법적 근거**
(다른 조례·다른 조문을 질문 사항의 규정처럼 제시)이고, 그다음이 근거가 있는데 거절하는 과잉 거절이다. 방어선은 4단:

1. **프롬프트**: 제공된 조문·별표만 근거로 답한다. 관련 조항은 있는데 세부 기준(한도·금액 등)이 없으면 거절하지 않고
   "조례에 구체적 기준은 규정되어 있지 않음"을 밝힌 뒤 확인되는 책무·절차를 답하며, 확인되지 않은 사항은 `unconfirmed`에 적는다.
   `확인 불가`는 관련 조항이 전혀 없을 때(실적·통계, 연락처, 타 지자체, 개별 사안)만. 자료요구·운영 현황 요청은 위탁·징수·운영 기준 조문으로 안내.
2. **인용구 원문 대조** (코드): 모든 인용구를 근거 원문과 글자 단위로 대조한다(공백·생략 표기·개정 이력, 앞에 붙인 조례명·조 제목만 허용).
   불일치가 있으면 1회 재생성, 그래도 유효한 인용이 없으면 `확인 불가`. 근거 표기가 빠지면 `근거:` 줄을 붙인다.
3. **조례명 점검** (코드): 질문이 특정 조례를 분명히 가리킬 때(조례명 일치도 0.5 이상) 조례명 핵심어가 하나도 겹치지 않는 조례를 인용하면
   1회 재생성, 그래도 그런 조례만 인용하면 `확인 불가`.
4. **2단계 근거 검증** (별도 LLM 호출): 인용마다 "그 조문이 질문 대상(사업·시설·지역)을 규정하는가", "답변 서술이 조문 원문과 맞는가"를 판정한다.
   단어 겹침으로는 못 잡는 실패 — 이름에 '과태료'가 같이 들어간 다른 법률의 과태료 조례, 의결 정족수 조항을 '제척 규정'이라 부르는 설명 — 를 겨냥한다.
   문제가 있으면 1회 재생성, 그래도 남으면 그 인용을 빼고(남는 인용이 없으면 `확인 불가`) `notes`에 경고. 검증 호출이 실패하면 1~3단 결과로 답한다.

조문이 검색되면 관련 별표(`(제N조 관련)`, 본문의 `별표 N` 언급)를 근거 자료에 함께 넣는다.

알려진 한계: 질문마다 새로 검색한다(대화 맥락 미사용). 정답 조문이 검색되지 않으면(예: '기금'의 정의가 다른 조에 있는 경우) 부분 답변이나 거절로 끝난다.
2단계 검증도 LLM 판정이라 놓치거나 과하게 걸 수 있다 → `notes`에 검증 경고가 있으면 원문을 확인할 것.

## 평가 (5단계)

| 평가 | 데이터셋 | 실행 | 지표 |
|---|---|---|---|
| 검색 | `tests/eval/search_eval.json` (조례 질의 19, 조문 질의 36, 9개 도메인) | `python -m src.evaluate --compare` | P@5, R@10 (관련 조례 찾기) / Hit@1, Hit@5, MRR@10 (질문→조문) |
| RAG 답변 | `tests/eval/rag_eval.json` (answered 20, 확인 불가 10) | `python -m src.evaluate_rag` | 실무·엄격 통과, 고지 없는 오답변, 고지형 부분 답변, 과잉 거절, 정답 조문 인용률, 타 조례 혼입률, 인용 일치율, 방어선 작동 수 |
| RAG 답변 | `tests/eval/rag_eval_ext.json` (현안 30건: answered 22, 확인 불가 8) | `python -m src.evaluate_rag --cases tests/eval/rag_eval_ext.json` | (위와 같음) |

- 정답은 검색 결과가 아니라 조례명·조 제목을 DB에서 직접 조회해 정했다 (검색 시스템에 유리한 편향 방지).
- 모든 질의에 `domain`(교통·주차, 환경·생활, 계약·재정, 복지 …)이 있어 도메인별 표가 함께 나온다. `--domain 환경·생활`로 한 도메인만 돌릴 수 있다.
- `--compare`는 보너스(조례명·조 제목 일치) 적용 전후 지표와, 정답 순위가 바뀐 질의를 보여준다.
- 확인 불가 질의는 조례로 답할 수 없는 유형(연도별 실적·통계, 타 지자체, 개별 사실관계, 상위법 조항, 연락처)으로 구성했다.
  여기에 답하더라도 "조례로는 확인할 수 없음"을 밝히고(`unconfirmed` 또는 본문) 관련 절차·책무만 안내하면 **고지형 부분 답변**으로 보고
  실무 통과에 넣는다. 그런 고지 없이 답하면 **고지 없는 오답변**(가장 위험한 실패). 엄격 통과(고지형 제외)도 함께 출력한다.
- 결과 JSON(질의별 순위·답변·인용)은 `data/eval/`에 저장된다(git 제외).
