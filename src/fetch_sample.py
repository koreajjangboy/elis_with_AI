"""자치법규 샘플 수집 및 구조 확인 스크립트.

실행:
    python -m src.fetch_sample                          # 서울특별시(본청) 조례 상위 3건
    python -m src.fetch_sample --org-name 수원시 --limit 5
    python -m src.fetch_sample --org-name "경기도 수원시" --query 주차장
    python -m src.fetch_sample --org 6410000 --sborg 3740000   # 기관코드 직접 지정
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from src.law_api import ORG_CODES, LawApiClient, LawApiError
from src.ordin_parser import parse_ordinance_detail, parse_search_result

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
OUTPUT_PATH = RAW_DIR / "sample_ordinance.json"
MAX_NAME_SEARCH_PAGES = 10  # 이름 검색 시 지자체명 필터링을 위해 넘겨볼 최대 페이지 수

# 여러 광역에 같은 이름이 있는 시·군·구. 검색 결과가 이름순이라 첫 페이지만으로는
# 중복을 감지할 수 없으므로 광역명을 포함한 전체 이름을 요구한다.
AMBIGUOUS_NAMES = {"중구", "동구", "서구", "남구", "북구", "강서구", "고성군"}


@dataclass
class OrgTarget:
    """조회 대상 지자체. 코드를 알면 코드로, 모르면 이름 검색 + 기관명 필터로 조회한다."""

    label: str
    org: str | None = None
    sborg: str | None = None
    name: str | None = None  # 지정 시 응답의 지자체기관명으로 필터링 (org는 검색 범위 축소용)

    @property
    def code(self) -> str:
        # 이름 필터 모드의 org는 광역 코드일 뿐 대상 지자체 코드가 아니다
        return "" if self.name else (self.sborg or self.org or "")


def resolve_org(org_name: str | None, org: str | None, sborg: str | None) -> OrgTarget:
    for code in (org, sborg):
        if code and not code.isdigit():
            raise LawApiError(f"기관코드는 숫자여야 합니다: '{code}'. 지자체 이름은 --org-name으로 지정하세요.")
    if org:
        return OrgTarget(label=f"기관코드 {org}" + (f"/{sborg}" if sborg else ""), org=org, sborg=sborg)
    if sborg:
        raise LawApiError("--sborg는 --org(광역 기관코드)와 함께 지정해야 합니다.")

    name = " ".join((org_name or "서울특별시").split())
    if name in ORG_CODES:  # 광역 본청: org=sborg=광역코드로 소속 시·군·구 조례를 제외
        code = ORG_CODES[name]
        return OrgTarget(label=name, org=code, sborg=code)
    if name in AMBIGUOUS_NAMES:
        raise LawApiError(f"'{name}'은 여러 광역에 있습니다. 광역명을 포함해 지정하세요. 예: --org-name \"서울특별시 {name}\"")
    # '서울특별시 중구'처럼 광역명이 붙어 있으면 광역 코드로 검색 범위를 좁힌다
    return OrgTarget(label=name, org=ORG_CODES.get(name.split()[0]) if " " in name else None, name=name)


def _matches_org(org_name: str, target: str) -> bool:
    # 응답 기관명은 '경기도 수원시' 형식이므로 '수원시'와 '경기도 수원시' 모두 허용
    return org_name == target or org_name.endswith(" " + target)


def search_target(client: LawApiClient, target: OrgTarget, query: str, limit: int) -> tuple[list[dict], str]:
    """대상 지자체의 조례 목록 상위 limit건과 검색 총건수를 반환."""
    if not target.name:
        root, xml = client.search_ordinances(query=query, org=target.org, sborg=target.sborg, display=limit)
        (RAW_DIR / "sample_list.xml").write_bytes(xml)
        result = parse_search_result(root)
        return result["items"][:limit], f"{result['total_count']}건"

    # 이름 검색: 검색어는 지자체명과도 매칭되며 공백은 AND로 동작한다
    # 광역명('경기도')은 검색어로 쓰면 매칭되지 않으므로 마지막 단어만 쓰고, 전체 이름은 필터에 쓴다
    short_name = target.name.split()[-1]
    full_query = short_name if query == "*" else f"{short_name} {query}"
    items, matched_orgs, total = [], set(), ""
    for page in range(1, MAX_NAME_SEARCH_PAGES + 1):
        root, xml = client.search_ordinances(query=full_query, org=target.org, display=100, page=page)
        if page == 1:
            (RAW_DIR / "sample_list.xml").write_bytes(xml)
        result = parse_search_result(root)
        total = total or result["total_count"]
        for item in result["items"]:
            if _matches_org(item["org_name"], target.name):
                matched_orgs.add(item["org_name"])
                items.append(item)
        if len(items) >= limit or len(result["items"]) < 100:
            break

    if len(matched_orgs) > 1:
        raise LawApiError(
            f"'{target.name}'에 해당하는 지자체가 여러 곳입니다: {', '.join(sorted(matched_orgs))}\n"
            "  --org-name에 광역명을 포함해 지정하세요. 예: --org-name \"서울특별시 중구\""
        )
    return items[:limit], f"{total}건(검색어 '{full_query}' 기준)"


def fetch_sample(client: LawApiClient, target: OrgTarget, query: str = "*", limit: int = 3) -> list[dict]:
    """대상 지자체 조례 목록 상위 limit건의 본문을 조회·파싱한다."""
    items, total = search_target(client, target, query, limit)
    print(f"[목록] {target.label} 검색 결과 총 {total} 중 {len(items)}건 조회")

    results = []
    for item in items:
        detail_root, detail_xml = client.get_ordinance(mst=item["serial_no"] or None, ordin_id=item["ordin_id"] or None)
        (RAW_DIR / f"sample_detail_{item['serial_no'] or item['ordin_id']}.xml").write_bytes(detail_xml)
        # 응답에 자치단체 코드가 없으므로 요청에 사용한 코드를 기록 (이름 검색이면 빈 값)
        detail = parse_ordinance_detail(detail_root, org_code=target.code)
        detail["list_meta"] = item
        results.append(detail)
    return results


def print_summary(ordinances: list[dict]) -> None:
    for i, o in enumerate(ordinances, 1):
        print("\n" + "=" * 70)
        print(f"[{i}] {o['name']}")
        print("-" * 70)
        print(f"  자치단체   : {o['org_name']} (코드 {o['org_code'] or '미상'})")
        print(f"  법규종류   : {o['kind']}")
        print(f"  공포일     : {o['promulgation_date']} (제{o['promulgation_no']}호)")
        print(f"  시행일     : {o['enforcement_date']}")
        print(f"  담당부서   : {o['department']}")
        print(f"  편/장/절   : {len(o['chapters'])}개, 조문 {len(o['articles'])}개, 부칙 {len(o['addenda'])}개")
        print(f"  인용 법령  : {', '.join(o['referenced_laws']) or '-'}")

        for art in o["articles"][:3]:
            print(f"\n  제{art['article_no']}조({art['title']})")
            for p in art["paragraphs"]:
                mark = f"{p['number']}항" if p["number"] else "본문"
                print(f"    [{mark}] {p['text'][:80]}")
                for it in p["items"]:
                    print(f"      {it['number']}. {it['text'][:70]}")
                    for sub in it["subitems"]:
                        print(f"        {sub['number']}. {sub['text'][:60]}")
        if len(o["articles"]) > 3:
            print(f"\n  ... 외 {len(o['articles']) - 3}개 조문")


def main() -> None:
    parser = argparse.ArgumentParser(description="자치법규 샘플 수집 및 구조 확인")
    parser.add_argument("--org-name", help="지자체 이름 (예: 서울특별시, 수원시, \"서울특별시 중구\")")
    parser.add_argument("--org", help="광역 기관코드 직접 지정 (예: 6410000)")
    parser.add_argument("--sborg", help="시·군·구 기관코드, --org와 함께 사용 (예: 3740000)")
    parser.add_argument("--query", default="*")
    parser.add_argument("--limit", type=int, default=3)
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    try:
        target = resolve_org(args.org_name, args.org, args.sborg)
        client = LawApiClient.from_env()
        ordinances = fetch_sample(client, target, args.query, args.limit)
    except LawApiError as e:
        raise SystemExit(f"[API 오류] {e}")

    if not ordinances:
        raise SystemExit("[결과 없음] 조회된 조례가 없어 파일을 저장하지 않았습니다. 지자체명·검색어를 확인하세요.")

    print_summary(ordinances)
    OUTPUT_PATH.write_text(json.dumps(ordinances, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n저장 완료: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
