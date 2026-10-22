"""수원시 조례 원본 XML 수집.

목록을 전부 조회해 data/raw/3740000/_list.json(목록 스냅샷)으로 저장하고,
본문 XML을 data/raw/3740000/{mst}.xml로 하나씩 저장한다.
이미 받은 파일은 건너뛰므로 중단 후 다시 실행하면 이어서 받는다.

실행:
    python -m src.collect_suwon            # 이어받기
    python -m src.collect_suwon --refresh  # 기존 파일도 다시 받기 (개정 반영 확인용)
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import requests

from src.law_api import LawApiClient, LawApiError
from src.ordin_parser import parse_search_result

ORG = "6410000"  # 경기도
SBORG = "3740000"  # 수원시
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / SBORG
LIST_PATH = RAW_DIR / "_list.json"
FAILED_PATH = RAW_DIR / "_failed.json"

PAGE_SIZE = 100
REQUEST_INTERVAL = 0.4  # 초. 공공 API 부하를 줄이기 위한 요청 간격
MAX_RETRIES = 3


def atomic_write(path: Path, data: bytes) -> None:
    """임시 파일에 쓴 뒤 교체해, 중단돼도 반쯤 쓰인 파일이 남지 않게 한다."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def with_retry(fn, *args, **kwargs):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fn(*args, **kwargs)
        except (requests.RequestException, LawApiError) as e:
            if attempt == MAX_RETRIES:
                raise
            wait = 2**attempt
            print(f"    재시도 {attempt}/{MAX_RETRIES - 1} ({wait}초 후): {str(e)[:100]}")
            time.sleep(wait)


def fetch_list(client: LawApiClient) -> list[dict]:
    items, page, total = [], 1, None
    while True:
        root, _ = with_retry(
            client.search_ordinances, org=ORG, sborg=SBORG, kind="조례", display=PAGE_SIZE, page=page
        )
        result = parse_search_result(root)
        total = total or int(result["total_count"] or 0)
        items.extend(result["items"])
        print(f"  목록 {page}페이지: 누적 {len(items)}/{total}건")
        if len(result["items"]) < PAGE_SIZE or len(items) >= total:
            break
        page += 1
        time.sleep(REQUEST_INTERVAL)

    # 페이지 경계에서 중복이 생길 수 있어 mst 기준으로 정리
    unique = list({item["serial_no"]: item for item in items}.values())
    if len(unique) != total:
        print(f"  [주의] 목록 총건수 {total}건, 실제 수집 {len(unique)}건")
    return unique


def collect(refresh: bool = False) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    client = LawApiClient.from_env()

    print("[1/2] 목록 조회")
    items = fetch_list(client)
    atomic_write(LIST_PATH, json.dumps(items, ensure_ascii=False, indent=2).encode("utf-8"))

    print(f"[2/2] 본문 수집 ({len(items)}건)")
    failed, skipped, downloaded = [], 0, 0
    for i, item in enumerate(items, 1):
        mst = item["serial_no"]
        path = RAW_DIR / f"{mst}.xml"
        if path.exists() and not refresh:
            skipped += 1
            continue
        try:
            _, xml = with_retry(client.get_ordinance, mst=mst)
            atomic_write(path, xml)
            downloaded += 1
        except Exception as e:
            failed.append({"mst": mst, "name": item["name"], "error": str(e)[:300]})
            print(f"  [실패] {mst} {item['name']}: {str(e)[:100]}")
        if i % 50 == 0:
            print(f"  진행 {i}/{len(items)} (신규 {downloaded}, 건너뜀 {skipped}, 실패 {len(failed)})")
        time.sleep(REQUEST_INTERVAL)

    atomic_write(FAILED_PATH, json.dumps(failed, ensure_ascii=False, indent=2).encode("utf-8"))
    print(f"\n완료: 신규 {downloaded}, 건너뜀 {skipped}, 실패 {len(failed)} → {RAW_DIR}")
    if failed:
        print(f"실패 목록: {FAILED_PATH} (다시 실행하면 실패분만 재시도)")


def main() -> None:
    parser = argparse.ArgumentParser(description="수원시 조례 원본 XML 수집")
    parser.add_argument("--refresh", action="store_true", help="이미 받은 본문도 다시 받기")
    args = parser.parse_args()
    try:
        collect(refresh=args.refresh)
    except LawApiError as e:
        raise SystemExit(f"[API 오류] {e}")


if __name__ == "__main__":
    main()
