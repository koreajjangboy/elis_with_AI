# elis_with_AI

전국 자치단체 자치법규(조례·규칙)를 기반으로 한 검색 강화형 RAG 시스템.

## 목표
- 자치법규를 조문 단위(조·항·호)로 구조화하여 수집·색인
- 하이브리드 검색(BM25 + 임베딩), 메타데이터 필터(자치단체·시행일), 리랭킹
- 답변 시 근거 조문 인용 (예: OO시 OO조례 제5조 제2항)

## 진행 단계
- [x] 0. 저장소 및 기본 골격
- [ ] 1. 범위·사용 시나리오 정의
- [ ] 2. 데이터 확보 (국가법령정보센터 Open API 검증)
- [ ] 3. 데이터 모델·청킹 설계
- [ ] 4. 검색 파이프라인 구현
- [ ] 5. 평가셋 구축 및 성능 측정

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
.venv/Scripts/python -m pytest -q                         # 파서 테스트
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
```
