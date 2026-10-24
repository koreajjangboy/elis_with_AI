"""별표·서식 첨부파일 텍스트 추출.

국가법령정보센터 자치법규 API는 별표·서식 본문(별표내용)을 비워 두고 첨부파일 URL만 준다
(수원시 기준 hwp 757건, hwpx 16건, txt 15건). 형식별 로컬 추출기로 텍스트를 뽑고,
텍스트가 거의 없는 문서(스캔 이미지·그림으로 된 표)는 OCR/멀티모달 추출기로 넘긴다.

    extract_text(data, file_type)  → ExtractResult(text, method, status)

이미지형 별표 대응 구조:
- 로컬 추출 결과가 MIN_TEXT_CHARS 미만이고 문서에 그림(BinData)이 있으면 status='needs_ocr'.
- needs_ocr 문서는 FallbackExtractor(프로토콜)로 처리한다. 기본 구현은 UpstageDocumentParse
  (POST /v1/document-digitization, HWP 직접 지원). 텍스트와 그림이 섞인 문서(has_images)는
  로컬 추출로도 본문은 나오지만 도면·그림 속 글자는 빠지므로, 필요하면 같은 추출기로 다시 처리한다.
- 다른 OCR(Clova OCR, Tesseract, GPT/Claude 비전 등)도 같은 프로토콜만 구현하면 교체 가능하다.
"""

from __future__ import annotations

import io
import os
import re
import struct
import zipfile
import zlib
from dataclasses import dataclass
from typing import Protocol
from xml.etree import ElementTree as ET

import olefile
import requests
from dotenv import load_dotenv

MIN_TEXT_CHARS = 30  # 이보다 짧으면 이미지형 문서로 의심


@dataclass
class ExtractResult:
    text: str
    method: str  # 'hwp5' | 'hwpx' | 'txt' | 'document-parse' | ...
    status: str  # 'ok' | 'needs_ocr' | 'unsupported' | 'error'
    has_images: bool = False
    error: str = ""


# ---------------------------------------------------------------- HWP 5.x (OLE)

HWPTAG_PARA_TEXT = 67  # HWPTAG_BEGIN(16) + 51
# 8 wchar(16바이트)를 차지하는 확장·인라인 제어문자 (표·그림 개체, 각주, 필드 등)
_HWP_WIDE_CONTROLS = {1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23}


def _hwp_records(data: bytes):
    pos = 0
    while pos + 4 <= len(data):
        header = struct.unpack_from("<I", data, pos)[0]
        tag, size = header & 0x3FF, (header >> 20) & 0xFFF
        pos += 4
        if size == 0xFFF:
            size = struct.unpack_from("<I", data, pos)[0]
            pos += 4
        yield tag, data[pos:pos + size]
        pos += size


def _hwp_para_text(payload: bytes) -> str:
    units = struct.unpack(f"<{len(payload) // 2}H", payload[: len(payload) // 2 * 2])
    out, i = [], 0
    while i < len(units):
        c = units[i]
        if c in _HWP_WIDE_CONTROLS:
            if c == 9:
                out.append("\t")
            i += 8
            continue
        if c in (10, 13):
            out.append("\n")
        elif c >= 32:
            out.append(chr(c))
        i += 1
    return "".join(out)


def extract_hwp5(data: bytes) -> ExtractResult:
    with olefile.OleFileIO(io.BytesIO(data)) as ole:
        header = ole.openstream("FileHeader").read()
        flags = struct.unpack_from("<I", header, 36)[0]
        if flags & 0x2:  # 암호화
            return ExtractResult("", "hwp5", "unsupported", error="암호화된 HWP")
        if flags & 0x4:  # 배포용 문서: 본문이 ViewText에 암호화되어 있다
            return ExtractResult("", "hwp5", "needs_ocr", error="배포용 HWP (본문 암호화)")
        compressed = bool(flags & 0x1)
        sections = sorted(
            (e for e in ole.listdir() if e[0] == "BodyText"), key=lambda e: int(re.sub(r"\D", "", e[1]) or 0)
        )
        has_images = any(e[0] == "BinData" for e in ole.listdir())
        paragraphs = []
        for entry in sections:
            raw = ole.openstream(entry).read()
            body = zlib.decompress(raw, -15) if compressed else raw
            paragraphs += [_hwp_para_text(p) for tag, p in _hwp_records(body) if tag == HWPTAG_PARA_TEXT]
    return _finish("\n".join(paragraphs), "hwp5", has_images)


# ---------------------------------------------------------------- HWPX (OWPML zip)

def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _hwpx_block(el: ET.Element) -> list[str]:
    """문단은 한 줄, 표는 행마다 '셀 | 셀' 한 줄로 펼친다."""
    name = _local(el.tag)
    if name == "tbl":
        rows = []
        for tr in (c for c in el.iter() if _local(c.tag) == "tr"):
            # 한 셀 안의 여러 문단은 공백으로 이어 붙인다
            cells = [
                " ".join(" ".join(_hwpx_text(p) for p in tc.iter() if _local(p.tag) == "p").split())
                for tc in tr if _local(tc.tag) == "tc"
            ]
            rows.append(" | ".join(cells))
        return rows
    if name == "p":
        if any(_local(c.tag) == "tbl" for c in el.iter()):
            return [line for c in el.iter() if _local(c.tag) == "tbl" for line in _hwpx_block(c)]
        return [_hwpx_text(el)]
    return [line for child in el for line in _hwpx_block(child)]


def _hwpx_text(el: ET.Element) -> str:
    return "".join(t.text or "" for t in el.iter() if _local(t.tag) == "t")


def extract_hwpx(data: bytes) -> ExtractResult:
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        names = z.namelist()
        sections = sorted(
            (n for n in names if re.match(r"Contents/section\d+\.xml$", n)), key=lambda n: int(re.sub(r"\D", "", n))
        )
        has_images = any(n.startswith("BinData/") for n in names)
        lines = []
        for name in sections:
            root = ET.fromstring(z.read(name))
            lines += _hwpx_block(root)
    return _finish("\n".join(lines), "hwpx", has_images)


# ---------------------------------------------------------------- TXT

def extract_txt(data: bytes) -> ExtractResult:
    for encoding in ("utf-8", "cp949"):
        try:
            return _finish(data.decode(encoding), "txt", False)
        except UnicodeDecodeError:
            continue
    return _finish(data.decode("cp949", errors="replace"), "txt", False)


# ---------------------------------------------------------------- 공통

def _finish(text: str, method: str, has_images: bool) -> ExtractResult:
    text = re.sub(r"[ \t　]+", " ", text)
    text = "\n".join(line.strip() for line in text.splitlines())
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    status = "needs_ocr" if len(text) < MIN_TEXT_CHARS and has_images else "ok"
    return ExtractResult(text, method, status, has_images)


EXTRACTORS = {"hwp": extract_hwp5, "hwpx": extract_hwpx, "txt": extract_txt}


def extract_text(data: bytes, file_type: str) -> ExtractResult:
    """로컬 추출. 형식은 확장자보다 파일 시그니처를 우선한다 (hwp로 표시된 hwpx 등 대비)."""
    if data[:8] == b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1":
        file_type = "hwp"
    elif data[:2] == b"PK":
        file_type = "hwpx"
    extractor = EXTRACTORS.get(file_type.lower())
    if extractor is None:
        return ExtractResult("", file_type, "unsupported", error=f"지원하지 않는 형식: {file_type}")
    try:
        return extractor(data)
    except Exception as e:  # 손상 파일 등 — 파이프라인은 계속 진행
        return ExtractResult("", file_type, "error", error=f"{type(e).__name__}: {e}"[:300])


# ---------------------------------------------------------------- OCR / 멀티모달 확장 지점

class FallbackExtractor(Protocol):
    """로컬 추출이 실패하거나 needs_ocr인 문서를 처리하는 추출기."""

    name: str

    def extract(self, data: bytes, filename: str) -> ExtractResult: ...


class UpstageDocumentParse:
    """Upstage Document Parse (레이아웃 분석 + OCR). 표는 markdown으로 받는다.

    - HWP 파일을 직접 받는 것을 확인했다 (2026-10-22, 수원시 별표 샘플). 그림 속 글자도 OCR된다.
    - chart_recognition은 끈다: 켜 두면 도면을 차트로 오인해 원문에 없는 수치표를 만들어 냈다.
      법령 데이터에서 없는 숫자가 생기는 것은 치명적이다.
    - 페이지당 과금이므로 기본적으로 needs_ocr·error 문서에만 사용한다.
    """

    name = "document-parse"
    url = "https://api.upstage.ai/v1/document-digitization"

    def __init__(self, api_key: str | None = None, timeout: float = 120.0):
        load_dotenv()
        self.api_key = api_key or os.getenv("UPSTAGE_API_KEY") or os.getenv("LLM_API_KEY", "")
        self.timeout = timeout

    def extract(self, data: bytes, filename: str) -> ExtractResult:
        try:
            resp = requests.post(
                self.url,
                headers={"Authorization": f"Bearer {self.api_key}"},
                files={"document": (filename, data)},
                data={
                    "model": "document-parse",
                    "ocr": "auto",
                    "output_formats": "['markdown']",
                    "chart_recognition": "false",
                    "coordinates": "false",
                },
                timeout=self.timeout,
            )
            if not resp.ok:
                return ExtractResult("", self.name, "error", error=f"HTTP {resp.status_code}: {resp.text[:200]}")
            markdown = resp.json().get("content", {}).get("markdown", "")
            return ExtractResult(markdown.strip(), self.name, "ok" if markdown.strip() else "needs_ocr", True)
        except requests.RequestException as e:
            return ExtractResult("", self.name, "error", error=str(e)[:300])
