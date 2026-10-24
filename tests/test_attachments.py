import io
import struct
import xml.etree.ElementTree as ET
import zipfile

from src.attachment_extract import _hwp_para_text, _hwp_records, extract_text
from src.chunker import build_attachment_chunks, split_lines
from src.load_attachments import wants_fallback
from src.attachment_extract import ExtractResult
from src.ordin_parser import annex_kind, parse_annexes

ANNEX_XML = """<LawService><별표><별표단위 별표키="22908689">
  <별표번호>0001</별표번호><별표가지번호>00</별표가지번호><별표구분>서식</별표구분>
  <별표제목>[별표 1] 시공기준(제3조의2 관련)</별표제목><별표첨부파일구분>hwp</별표첨부파일구분>
  <별표첨부파일명>http://www.law.go.kr/flDownload.do?flSeq=1</별표첨부파일명><별표내용></별표내용>
</별표단위><별표단위 별표키="2">
  <별표번호>0002</별표번호><별표가지번호>00</별표가지번호><별표구분>서식</별표구분>
  <별표제목>삭제 2018.11.16</별표제목><별표첨부파일구분></별표첨부파일구분><별표첨부파일명></별표첨부파일명>
</별표단위></별표></LawService>"""


def test_parse_annexes_and_kind():
    a, deleted = parse_annexes(ET.fromstring(ANNEX_XML))
    assert a["key"] == "22908689" and a["no"] == "1" and a["kind"] == "별표"
    assert a["related_article"] == "제3조의2" and a["file_type"] == "hwp"
    assert deleted["kind"] == "삭제"
    assert annex_kind("[별지 제1호서식] 신청서") == "서식"
    assert annex_kind("〔별표〕 시공기준") == "별표"
    assert annex_kind("<삭제 2025.03.11.>") == "삭제"
    assert annex_kind("[별표2] <삭제 2011.06.15>") == "삭제"
    assert annex_kind("별표2 <삭제 2023.07.12>") == "삭제"  # 괄호 없는 제목
    for title in ["[별표 1-1](삭제 2011.04.01)", "[별표 1]  삭제<2017.07.17>", "[별표 2] <2013.7.31 개정으로 삭제>",
                  "[별표 2]<삭제 2017.04.03 조례 제3655호>", "[별지 제28호서식]-2013. 7. 31 개정으로 삭제",
                  "[별지 제3호서식] 〈삭제 2017.04.28〉", "삭제 2018.11.16", "[별표2] 삭제"]:
        assert annex_kind(title) == "삭제", title
    assert annex_kind("[별표 2] 삭제할 사항의 처리기준") == "별표"  # 제목에 '삭제'가 든 실제 별표


def _record(tag: int, payload: bytes) -> bytes:
    return struct.pack("<I", tag | (len(payload) << 20)) + payload


def test_hwp_para_text_skips_controls():
    # '가' + 표 개체 제어문자(11, 8 wchar 차지) + '나' + 문단 끝(13)
    units = [ord("가"), 11, 0, 0, 0, 0, 0, 0, 11, ord("나"), 13]
    text = _hwp_para_text(struct.pack(f"<{len(units)}H", *units))
    assert text == "가나\n"
    records = list(_hwp_records(_record(67, "다".encode("utf-16-le")) + _record(66, b"\x00\x00")))
    assert [tag for tag, _ in records] == [67, 66]


def test_hwpx_table_rows_and_cells():
    section = (
        '<hs:sec xmlns:hs="s" xmlns:hp="p"><hp:p><hp:run><hp:t>【별표 2】</hp:t></hp:run></hp:p>'
        "<hp:p><hp:run><hp:tbl><hp:tr>"
        "<hp:tc><hp:subList><hp:p><hp:t>파장동</hp:t></hp:p></hp:subList></hp:tc>"
        "<hp:tc><hp:subList><hp:p><hp:t>19~23</hp:t></hp:p><hp:p><hp:t>산43</hp:t></hp:p></hp:subList></hp:tc>"
        "</hp:tr></hp:tbl></hp:run></hp:p></hs:sec>"
    )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("Contents/section0.xml", section)
    result = extract_text(buf.getvalue(), "hwpx")
    assert result.status == "ok"
    assert result.text == "【별표 2】\n파장동 | 19~23 산43"


def test_txt_cp949_and_signature_detection():
    assert extract_text("[별표 1] 시상기준".encode("cp949"), "txt").text == "[별표 1] 시상기준"
    # 확장자가 hwp여도 zip 시그니처면 hwpx로 처리 (손상 zip → error로 반환, 예외 없음)
    assert extract_text(b"PK\x03\x04broken", "hwp").status == "error"


def test_attachment_chunks_split_with_header():
    assert split_lines("a" * 10 + "\n" + "b" * 10, max_chars=15) == ["a" * 10, "b" * 10]
    assert split_lines("c" * 40, max_chars=15) == ["c" * 15, "c" * 15, "c" * 10]
    text = "\n".join(f"{i}통 관할구역" for i in range(400))
    chunks = build_attachment_chunks("수원시 통ㆍ반 설치 조례", "[별표 2]  통ㆍ반 명칭", text)
    assert len(chunks) > 1
    assert chunks[0].citation == f"수원시 통ㆍ반 설치 조례 [별표 2] 통ㆍ반 명칭 (1/{len(chunks)})"
    assert all(c.embed_text.startswith("[수원시 통ㆍ반 설치 조례] [별표 2] 통ㆍ반 명칭\n") for c in chunks)


def test_ocr_policy():
    ok_with_images = ExtractResult("본문" * 50, "hwp5", "ok", has_images=True)
    needs = ExtractResult("", "hwp5", "needs_ocr", has_images=True)
    assert not wants_fallback(needs, "none")
    assert wants_fallback(needs, "needs") and not wants_fallback(ok_with_images, "needs")
    assert wants_fallback(ok_with_images, "images")


def test_unextracted_attachment_gets_placeholder_chunk():
    from src.chunker import UNEXTRACTED_NOTE

    (chunk,) = build_attachment_chunks("수원시 주차장 조례", "[별표 3] 주차장 배치도", None)
    assert chunk.content == UNEXTRACTED_NOTE
    assert chunk.citation == "수원시 주차장 조례 [별표 3] 주차장 배치도"
    assert chunk.embed_text.startswith("[수원시 주차장 조례] [별표 3] 주차장 배치도\n")
