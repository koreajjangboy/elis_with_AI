"""국가법령정보센터 Open API(DRF) 자치법규 클라이언트.

가이드: https://open.law.go.kr/LSO/openApi/guideList.do
- 목록: lawSearch.do?target=ordin
- 본문: lawService.do?target=ordin
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import requests
from dotenv import load_dotenv

DEFAULT_BASE_URL = "https://www.law.go.kr/DRF"

# 자치법규 종류 코드 (목록 조회 knd 파라미터)
ORDIN_KIND = {
    "조례": "30001",
    "규칙": "30002",
    "훈령": "30003",
    "예규": "30004",
    "기타": "30006",
    "고시": "30010",
    "의회규칙": "30011",
}

# 광역자치단체 기관코드 (목록 조회 org 파라미터). 2026-10 API 응답으로 검증한 값만 둔다.
# 광주광역시(6290000)·전라남도(6460000)는 전남광주통합특별시로 통합되어 옛 코드로는
# '(구)' 조례만 조회되므로 제외했다. 코드가 없는 지자체는 이름 검색으로 찾는다.
ORG_CODES = {
    "서울특별시": "6110000",
    "부산광역시": "6260000",
    "대구광역시": "6270000",
    "인천광역시": "6280000",
    "대전광역시": "6300000",
    "울산광역시": "6310000",
    "세종특별자치시": "5690000",
    "경기도": "6410000",
    "강원특별자치도": "6530000",
    "충청북도": "6430000",
    "충청남도": "6440000",
    "전북특별자치도": "6540000",
    "경상북도": "6470000",
    "경상남도": "6480000",
    "제주특별자치도": "6500000",
}


class LawApiError(RuntimeError):
    pass


@dataclass
class LawApiClient:
    oc: str
    base_url: str = DEFAULT_BASE_URL
    timeout: float = 15.0

    @classmethod
    def from_env(cls) -> "LawApiClient":
        load_dotenv()
        oc = os.getenv("LAW_API_OC", "").strip()
        if not oc:
            raise LawApiError(".env에 LAW_API_OC가 설정되어 있지 않습니다.")
        return cls(oc=oc, base_url=os.getenv("LAW_API_BASE_URL", DEFAULT_BASE_URL))

    def _get_xml(self, endpoint: str, params: dict) -> tuple[ET.Element, bytes]:
        query = {"OC": self.oc, "type": "XML", **{k: v for k, v in params.items() if v is not None}}
        resp = requests.get(f"{self.base_url}/{endpoint}", params=query, timeout=self.timeout)
        resp.raise_for_status()
        try:
            root = ET.fromstring(resp.content)
        except ET.ParseError as e:
            # 인증 실패(미등록 IP/도메인, 잘못된 OC) 시 XML이 아닌 HTML 안내 페이지가 올 수 있다
            snippet = resp.content[:300].decode("utf-8", errors="replace")
            raise LawApiError(f"XML 파싱 실패: {e}\n응답 앞부분: {snippet}") from e
        # 오류는 HTTP 200 + <Response><result>..</result><msg>..</msg></Response> 형태로 온다
        if root.tag == "Response" and root.find("result") is not None:
            raise LawApiError(f"{root.findtext('result', '').strip()} {root.findtext('msg', '').strip()}")
        return root, resp.content

    def search_ordinances(
        self,
        query: str = "*",
        org: str | None = None,
        sborg: str | None = None,
        kind: str | None = "조례",
        display: int = 20,
        page: int = 1,
        sort: str | None = None,
    ) -> tuple[ET.Element, bytes]:
        """자치법규 목록 조회."""
        return self._get_xml(
            "lawSearch.do",
            {
                "target": "ordin",
                "query": query,
                "org": org,
                "sborg": sborg,
                "knd": ORDIN_KIND.get(kind, kind) if kind else None,
                "display": display,
                "page": page,
                "sort": sort,
            },
        )

    def get_ordinance(self, mst: str | None = None, ordin_id: str | None = None) -> tuple[ET.Element, bytes]:
        """자치법규 본문 조회. MST(자치법규일련번호) 또는 ID(자치법규ID) 중 하나."""
        if not (mst or ordin_id):
            raise ValueError("mst 또는 ordin_id가 필요합니다.")
        return self._get_xml("lawService.do", {"target": "ordin", "MST": mst, "ID": ordin_id})

    # 법령 연계 API 조사 결과(2026-10): target=ordinLsCon은 빈 응답, lnkOrd는 연계 조례 목록만
    # 반환(연계 법령 정보 없음), lnkLsOrd는 '법령명 → 연계 조례' 방향 검색이다.
    # 조례별 상위법령은 우선 본문 「」 인용 추출(ordin_parser.extract_law_references)로 대체한다.
