# 별표·서식(Attachment) DB화 설계

자치법규의 **[별표]**(요금표·기준표·관할구역표 등)와 **[서식]**(신청서·대장 등 양식)을 조문과 같은 수준으로 검색하고
답변 근거로 인용하기 위한 설계다. 스키마: [`db/schema.sql`](../db/schema.sql), 마이그레이션: `db/migrations/002_attachments.sql`, `003_attachment_visual.sql`

## 1. 왜 별도로 다뤄야 하나

- 금액·비율·기준 같은 **실무 질문의 답은 조문이 아니라 별표에 있는 경우가 많다.**
  예: `수원시 주차장 조례 제4조`는 "공영주차장의 주차요금은 별표1과 같이 하고"라고만 하고, 실제 감면 대상·비율은 `[별표 1]`에 있다.
- 국가법령정보센터 자치법규 API는 **별표 본문(`별표내용`)을 비워 두고 첨부파일 URL만 준다.** 파일을 받아 텍스트를 추출해야 한다.
- 일부 별표는 도면·스캔 이미지라서 텍스트 추출만으로는 내용을 얻을 수 없다 (OCR·멀티모달 처리 필요).

**현황 (수원시 707건, 2026-10-23 DB 기준)**

| 항목 | 값 |
|---|---|
| 별표·서식이 있는 조례 | 218건 |
| 별표 / 서식 / 삭제 | 310 / 453 / 25 (총 788건) |
| 파일 형식 | hwp 757 · hwpx 16 · txt 15 |
| 추출 방식 | hwp5 731 · hwpx 16 · txt 15 · document-parse(OCR) 1 |
| 그림 포함(`has_images`) | 31건 |
| 관련 조문 표기 `(제N조 관련)` 있음 | 183건 |
| 검색 청크 | 1,100개 (텍스트 중앙값 약 350자, 최대 28만 자 → 최대 201조각) |

## 2. 데이터 흐름

```
API 본문 XML ──parse_annexes()──▶ 별표 메타 (키·번호·제목·종류·관련 조문·파일 URL)
                                     │
            collect_attachments ─────┤  첨부파일 다운로드 → data/raw/{org}/attachments/
                                     ▼
            load_attachments   ── extract_text() ── 성공 ──▶ attachments.text (status=ok)
                                     │              실패/이미지형 ──▶ FallbackExtractor(OCR) ──▶ ok | needs_ocr
                                     ▼
                     build_attachment_chunks() ──▶ chunks (attachment_id, granularity='attachment')
                                     ▼
            load_to_db --embed-only ──▶ chunks.embedding (조문 청크와 같은 인덱스)
                                     ▼
            search.hybrid_search / rag.answer_question ──▶ 조문 + 별표 통합 검색·인용
```

## 3. 테이블 설계

```
ordinances 1 ─ N attachments (문서 메타 · 추출 텍스트 · 추출 상태 · 이미지 후처리)
                    1 ─ N chunks (attachment_id: 검색 조각 + 임베딩)
ordinances 1 ─ N articles    1 ─ N chunks (article_id)
```

요구사항의 예시 이름 `ordinance_attachments`에 해당하는 테이블이 **`attachments`** 다 (002에서 이미 이 이름으로 배포됨).

| 컬럼 | 설명 |
|---|---|
| `ordinance_id`, `annex_key` | 조례 버전 + API 별표키. `UNIQUE`로 재적재 시 upsert |
| `annex_no`, `annex_branch_no` | 별표 번호·가지번호 (`[별표 2의3]` → 2, 3). 조문의 "별표 2" 언급과 연결할 때 사용 |
| `kind` | `별표` / `서식` / `삭제` / `기타`. API `별표구분`은 별표도 '서식'으로 주므로 **제목 접두어로 판별** |
| `title` | `[별표 1] 공영주차장 요금표(제4조제1항 관련)` |
| `related_article` | 제목의 `(제4조제1항 관련)` → `제4조`. 조문 검색 결과에 관련 별표를 붙일 때 사용 |
| `file_type`, `source_url`, `file_path`, `file_hash` | 원본 파일 정보. 해시가 같으면 재추출 생략, `source_url`은 화면의 원문 링크 |
| `text` | 추출한 전체 텍스트 (원문) |
| `extraction_method` | `hwp5` / `hwpx` / `txt` / `api` / `document-parse` |
| `extraction_status` | `ok` / `needs_ocr` / `unsupported` / `error` / `pending`(파일 미수집) / `skipped`(삭제된 별표) |
| `has_images` | 그림 포함. 그림 속 정보(도면·흐름도)는 `text`에 없을 수 있음을 화면·답변에 경고 |
| `image_description` | **(003 신설)** 멀티모달 모델이 만든 그림·도면 설명. **원문이 아니므로 `text`와 분리** |
| `image_described_by` | **(003 신설)** 설명을 만든 모델명 (재생성·감사용) |
| `reviewed_at` | **(003 신설)** 사람이 OCR 결과·그림 설명을 원문과 대조 검수한 시각. NULL=미검수 |

`chunks`는 `article_id`와 `attachment_id` 중 **정확히 하나**를 가진다 (`chunks_owner_check`).
임베딩을 `attachments`가 아닌 `chunks`에 두는 이유: ① 조문과 별표를 **한 인덱스·한 쿼리로** 하이브리드 검색, ② 수십만 자짜리 별표는 어차피 여러 조각으로 나눠야 한다.

## 4. 파싱 (`src/ordin_parser.py`)

- `parse_annexes(root)`: `<별표단위>`마다 키·번호·가지번호·제목·종류·관련 조문·파일 형식·URL·본문을 뽑는다. 본문이 있으면(일부 기관) 파일 없이 바로 사용한다.
- `annex_kind(title)`: `[별표 …]`, `〔별표〕`, `별표2` → 별표 / `[별지 제1호서식]`, `…서식` → 서식.
- `is_deleted_annex(title)`: 라벨을 떼고 `<삭제 2017.04.03 조례 제3655호>`, `(삭제 2011.04.01)` 같은 부속어를 지웠을 때 남는 것이 없으면 삭제된 별표다.
  `[별표 2] 삭제할 사항의 처리기준`처럼 실제 제목이 남으면 삭제가 아니다.
- `related_article`: 제목의 `(제3조의2 관련)` → `제3조의2`.

## 5. 텍스트 추출과 이미지형 별표 (`src/attachment_extract.py`, `src/load_attachments.py`)

| 형식 | 로컬 추출기 | 비고 |
|---|---|---|
| hwp (HWP 5.x OLE) | `olefile` + BodyText 레코드(PARA_TEXT) 파싱 | 표는 셀 단위 줄로 나옴. 배포용·암호화 문서는 `unsupported` |
| hwpx (OWPML zip) | section XML 파싱 | 표를 행마다 `셀 \| 셀`로 복원 |
| txt | cp949/utf-8 디코딩 | 괘선 표 그대로 |

**이미지형 별표 판정**: 로컬 추출 결과가 30자(`MIN_TEXT_CHARS`) 미만이고 문서에 그림(BinData)이 있으면 `needs_ocr`.

**OCR·멀티모달 확장 구조**
1. `FallbackExtractor` 프로토콜(`extract(data, filename) -> ExtractResult`): OCR 엔진을 교체할 수 있는 접점.
   기본 구현은 **Upstage Document Parse**(HWP 직접 입력, 그림 속 글자 OCR). Clova OCR, Tesseract, Claude/GPT 비전도 같은 프로토콜만 구현하면 된다.
   - 주의: Document Parse의 `chart_recognition`은 끈다. 켜면 도면을 차트로 오인해 **원문에 없는 수치표를 만들어 냈다.**
2. 정책 `--ocr none | needs | images`: 로컬만 / 실패·이미지형만 OCR(기본) / 그림 포함 문서 전부 OCR(페이지당 과금).
3. **(향후) 그림 설명 단계**: OCR로도 의미가 안 나오는 도면·흐름도·위치도는 멀티모달 모델로 설명을 만들어 `image_description`에 넣는다.
   - 대상 큐: `attachments_visual_queue_idx` (`needs_ocr` 이거나 `has_images AND image_description IS NULL`).
   - 원문(`text`)을 덮어쓰지 않는다. RAG 프롬프트에는 `[AI 그림 설명 — 원문 아님]`으로 구분해 넣고, 시스템 프롬프트가 **단독 근거로 쓰지 못하게** 한다.
   - `reviewed_at`이 채워진(사람이 검수한) 설명만 근거로 허용하는 식으로 정책을 강화할 수 있다.
4. **누락 방지(자리표시 청크)**: 본문을 못 읽은 별표(`needs_ocr`·`unsupported`·`error`·`pending`)도
   `[조례명] [별표 N] 제목` + "본문 텍스트 미추출 … 원문 파일을 확인하세요" 청크를 만들어 **제목으로는 검색되게** 한다.
   이 청크는 근거 검증에서 인용 대상이 될 수 없으므로, 답변은 '확인 불가'가 되고 화면은 원문 파일 링크를 보여 준다.

## 6. 청킹 (`src/chunker.py`)

- `build_attachment_chunks(ordinance_name, title, text)`: 줄 경계에서 1,500자(`ATTACHMENT_CHUNK_CHARS`) 이하로 나눈다. 한 줄이 너무 길면 그 줄만 글자 수로 자른다.
- 조각마다 임베딩 입력에 `[조례명] [별표 N] 제목` 헤더를 붙여 단독으로도 맥락이 통하게 한다.
- 인용 표기(`citation`): `수원시 주차장 조례 [별표 1] 공영주차장 요금표 (2/3)` (여러 조각일 때만 순번).
- 키워드 토큰: 본문은 `tokens`, 별표 제목은 `title_tokens`(제목 일치 보너스), 조례명은 `name_tokens`.
- 텍스트가 없으면 자리표시 청크 1개 (5절 4번).

## 7. 검색·RAG 연계

**검색 (`src/search.py`)**: 조문·별표 청크를 같은 후보 풀에서 RRF + 조례명/제목 보너스로 순위화한다.
- 같은 별표의 여러 조각은 1건으로 묶는다 (`attachment_id` 기준 중복 제거).
- **서식 감점**(`FORM_PENALTY=0.6`): 빈 양식은 '신청인', '지원 대상' 같은 단어가 많아 내용 질문에도 걸리므로, 질문에 양식 의도(서식·신청서·대장 등)가 없으면 점수를 낮춘다.

**RAG (`src/rag.py`)**
- **관련 별표 자동 첨부**: 조문이 검색되면 같은 조례의 별표 중 ① 제목이 `(제N조 관련)`이거나 ② 조문 본문이 `별표 N`을 언급하는 것을 근거 자료에 덧붙인다 (조문당 2건, 전체 3건).
  "제4조 → 별표1과 같이 한다"처럼 **조문만으로는 답이 없는 질문**을 해결한다.
- 긴 별표(4,000자 초과)는 검색된 조각만 싣고 `일부 발췌`로 표시한다.
- 근거 자료 머리에 `<별표, S1의 관련 별표, OCR 추출본, 본문 미추출, 일부 발췌>` 태그를 붙인다.
  OCR 추출본의 수치는 "(원문 확인 필요)"를 덧붙이도록 프롬프트로 지시한다.
- 인용 검증: 인용구는 근거 자료 원문에 글자 그대로 있어야 한다 (공백·가운뎃점 차이만 허용). 자리표시 청크는 인용 불가.

**화면 (`app.py`)**: 별표 근거는 아코디언에 원문을 펼치고 인용 구절을 형광펜으로 표시한다. OCR·그림 포함·미추출 배지와 원문 파일 내려받기 버튼을 함께 보여 준다.

## 8. 검증

```bash
python -m src.check_status     # 별표 상태표 + 정합성 점검 (문제 있으면 종료 코드 1)
```
- `검색 청크가 없는 별표·서식(삭제 제외)` = 0 이어야 한다 (자리표시 청크 도입 후).
- `삭제 표기인데 '삭제'로 분류되지 않은 별표` = 0 이어야 한다.
- `본문 미추출 별표` 건수는 OCR 처리 대상(정보 항목).

SQL Editor에서 직접 보려면 [`db/verify.sql`](../db/verify.sql) 4·5번 쿼리.

## 9. 남은 과제

- 그림 설명 파이프라인 구현 (`image_description` 채우기, 검수 화면, `reviewed_at` 기반 근거 허용 정책).
- HWP 5.x 표의 행 구조 복원 (현재는 셀 단위 줄 → 표 질의 정확도 한계). hwpx처럼 `셀 | 셀` 행으로 만들거나, 표가 많은 별표만 Document Parse로 재추출.
- 별표 개정 이력: 지금은 조례 버전(`mst`)마다 별표를 다시 적재한다. 별표만 바뀐 개정의 차이 비교는 미지원.
