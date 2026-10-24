"""수원시 조례 별표·서식 첨부파일 수집.

수집된 조례 본문 XML(data/raw/3740000/{mst}.xml)의 <별표단위>에서 첨부파일 URL을 읽어
data/raw/3740000/attachments/{별표키}.{확장자}로 저장한다. 이미 받은 파일은 건너뛴다.
삭제된 별표('삭제 2018.11.16' 등)는 받지 않는다.

실행:
    python -m src.collect_attachments            # 이어받기
    python -m src.collect_attachments --refresh  # 기존 파일도 다시 받기
"""

from __future__ import annotations

import argparse
import json
import time
import xml.etree.ElementTree as ET

import requests

from src.collect_suwon import LIST_PATH, RAW_DIR, REQUEST_INTERVAL, atomic_write
from src.ordin_parser import parse_annexes

ATTACHMENT_DIR = RAW_DIR / "attachments"
FAILED_PATH = ATTACHMENT_DIR / "_failed.json"
MAX_RETRIES = 3


def attachment_path(annex: dict):
    return ATTACHMENT_DIR / f"{annex['key']}.{annex['file_type'] or 'bin'}"


def download(url: str) -> bytes:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            if resp.content[:15].lower().startswith((b"<!doctype", b"<html")):  # 오류 안내 페이지
                raise requests.RequestException("파일 대신 HTML 응답")
            return resp.content
        except requests.RequestException:
            if attempt == MAX_RETRIES:
                raise
            time.sleep(2**attempt)
    raise RuntimeError("unreachable")


def collect(refresh: bool = False) -> None:
    ATTACHMENT_DIR.mkdir(parents=True, exist_ok=True)
    items = json.loads(LIST_PATH.read_text(encoding="utf-8"))
    annexes = []
    for item in items:
        xml_path = RAW_DIR / f"{item['serial_no']}.xml"
        if xml_path.exists():
            annexes += [(item, a) for a in parse_annexes(ET.parse(xml_path).getroot())]
    targets = [(item, a) for item, a in annexes if a["kind"] != "삭제" and a["url"]]
    print(f"별표·서식 {len(annexes)}건 중 다운로드 대상 {len(targets)}건 (삭제·URL 없음 제외)")

    failed, skipped, downloaded = [], 0, 0
    for i, (item, annex) in enumerate(targets, 1):
        path = attachment_path(annex)
        if path.exists() and not refresh:
            skipped += 1
            continue
        try:
            atomic_write(path, download(annex["url"]))
            downloaded += 1
        except Exception as e:
            failed.append({"key": annex["key"], "mst": item["serial_no"], "title": annex["title"], "error": str(e)[:300]})
            print(f"  [실패] {annex['key']} {annex['title']}: {str(e)[:100]}")
        if i % 100 == 0:
            print(f"  진행 {i}/{len(targets)} (신규 {downloaded}, 건너뜀 {skipped}, 실패 {len(failed)})")
        time.sleep(REQUEST_INTERVAL)

    atomic_write(FAILED_PATH, json.dumps(failed, ensure_ascii=False, indent=2).encode("utf-8"))
    print(f"\n완료: 신규 {downloaded}, 건너뜀 {skipped}, 실패 {len(failed)} → {ATTACHMENT_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(description="별표·서식 첨부파일 수집")
    parser.add_argument("--refresh", action="store_true", help="이미 받은 파일도 다시 받기")
    collect(parser.parse_args().refresh)


if __name__ == "__main__":
    main()
