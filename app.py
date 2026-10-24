"""수원시 자치법규 질의응답 — Streamlit 웹 앱 (의원·정책지원관용).

FastAPI 백엔드(src/api.py)를 HTTP로 호출한다. 먼저 API 서버를 띄운 뒤 실행:
    uvicorn src.api:app --port 8000
    streamlit run app.py

API 주소는 환경변수 ELIS_API_URL로 바꿀 수 있다 (기본 http://localhost:8000).
"""

from __future__ import annotations

import html
import os
import re

import requests
import streamlit as st

from src.rag import split_quote

API_URL = os.getenv("ELIS_API_URL", "http://localhost:8000").rstrip("/")
UNVERIFIABLE = "확인 불가"
EXAMPLES = [
    "공영주차장 주차요금 감면 대상은?",
    "청년 창업 지원 대상과 지원 내용은?",
    "재난관리기금은 어디에 쓸 수 있나요?",
]

st.set_page_config(page_title="수원시 자치법규 AI", page_icon="⚖️", layout="wide")


# ---------------------------------------------------------------- API

def api_get(path: str, timeout: float = 10) -> dict:
    resp = requests.get(f"{API_URL}{path}", timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def api_post(path: str, body: dict, timeout: float = 300) -> dict | list:
    resp = requests.post(f"{API_URL}{path}", json=body, timeout=timeout)
    if not resp.ok:
        detail = resp.json().get("detail", resp.text) if resp.headers.get("content-type", "").startswith("application/json") else resp.text
        raise RuntimeError(f"API 오류 {resp.status_code}: {detail}")
    return resp.json()


# ---------------------------------------------------------------- 표시

def highlight(text: str, quotes: list[str]) -> str:
    """원문에서 인용 구절을 형광펜 표시 (공백 차이는 무시하고 찾는다). HTML 이스케이프 후 <mark>로 감싼다."""
    out = html.escape(text)
    for q in sorted(set(quotes), key=len, reverse=True):
        for piece in split_quote(q):  # 서버 검증과 같은 기준으로 생략 표기·감싼 따옴표를 뗀다
            words = html.escape(piece.strip()).split()
            if len(words) == 0:
                continue
            pattern = r"\s+".join(re.escape(w) for w in words)
            out = re.sub(f"({pattern})", r"<mark>\1</mark>", out, count=1)
    return out.replace("\n", "<br>")


def source_badges(src: dict) -> str:
    badges = ["📜 조문" if src["kind"] == "article" else f"📎 {src.get('attachment_kind') or '별표'}"]
    if src.get("related_to"):
        badges.append(f"🔗 {src['related_to']} 관련 별표")
    if src.get("extraction_method") == "document-parse":
        badges.append("🔍 OCR 추출본 — 원문 대조 필요")
    if src.get("has_images"):
        badges.append("🖼️ 그림 포함 — 그림 속 내용은 원문 확인")
    if not src.get("extracted", True):
        badges.append("⚠️ 본문 미추출")
    if src.get("partial"):
        badges.append("✂️ 일부 발췌")
    return " · ".join(badges)


def render_source(src: dict, quotes: list[str], expanded: bool) -> None:
    title = f"[{src['id']}] {src['label']}" + (f" ({src['title']})" if src["kind"] == "article" and src["title"] else "")
    with st.expander(title, expanded=expanded):
        st.caption(source_badges(src))
        st.markdown(
            f"<div style='font-size:0.92rem;line-height:1.7'>{highlight(src['content'], quotes)}</div>",
            unsafe_allow_html=True,
        )
        if src.get("image_description"):
            st.info(f"**AI 그림 설명 (원문 아님)**\n\n{src['image_description']}")
        if src.get("source_url"):
            st.link_button("원문 파일 내려받기", src["source_url"])


def render_answer(result: dict) -> None:
    if result["status"] == "unverifiable":
        st.warning(f"**{UNVERIFIABLE}** — 검색된 조문·별표에서 질문에 대한 근거를 찾지 못했습니다. "
                   "아래 검색 결과를 직접 확인하거나 질문을 구체적으로 바꿔 보세요.")
    else:
        st.markdown(result["answer"])

    for note in result.get("notes", []):
        st.caption(f"ℹ️ {note}")

    quotes_by_source: dict[str, list[str]] = {}
    for c in result["citations"]:
        quotes_by_source.setdefault(c["source_id"], []).append(c["quote"])

    cited = [s for s in result["sources"] if s["id"] in quotes_by_source]
    others = [s for s in result["sources"] if s["id"] not in quotes_by_source]
    if cited:
        st.markdown("**근거 조문·별표** (형광펜: 인용 구절)")
        for src in cited:
            render_source(src, quotes_by_source[src["id"]], expanded=False)
    if others:
        label = "검색된 조문·별표" if not cited else f"참고: 검색되었으나 인용되지 않은 자료 {len(others)}건"
        with st.expander(label, expanded=False):
            for src in others:
                st.markdown(f"**[{src['id']}] {src['label']}**  \n<small>{source_badges(src)}</small>", unsafe_allow_html=True)
                st.text(src["content"][:600] + ("…" if len(src["content"]) > 600 else ""))
                if src.get("source_url"):
                    st.markdown(f"[원문 파일]({src['source_url']})")
    st.caption(f"모델: {result['model']}")


# ---------------------------------------------------------------- 화면

with st.sidebar:
    st.header("⚖️ 수원시 자치법규 AI")
    try:
        health = api_get("/health")
        st.success(f"API 연결됨 · 현행 조례 {health['ordinances']:,}건 · 검색 단위 {health['chunks']:,}개")
        if health.get("llm"):
            st.caption(f"LLM: {health['llm']}")
        else:
            st.error(f"LLM 미설정: {health.get('llm_error')}")
    except requests.RequestException as e:
        st.error(f"API 서버에 연결할 수 없습니다 ({API_URL}).\n\n`uvicorn src.api:app --port 8000`으로 먼저 실행하세요.\n\n{e}")
    top_k = st.slider("검색할 조문·별표 수", 3, 12, 6, help="많을수록 근거 범위가 넓어지지만 답변이 느려집니다.")
    st.markdown(
        "**답변 원칙**\n"
        "- 검색된 조문·별표 원문만 근거로 답합니다.\n"
        f"- 근거가 없으면 **‘{UNVERIFIABLE}’**로 답하고 인용을 비웁니다.\n"
        "- 인용구는 서버에서 원문과 글자 단위로 대조해 검증합니다.\n"
        "- 질문마다 새로 검색합니다. 후속 질문에도 조례명·주제를 함께 적어 주세요."
    )
    if st.button("대화 지우기"):
        st.session_state.messages = []
        st.rerun()

st.title("수원시 자치법규 질의응답")
st.caption("조례·규칙의 조문과 별표를 근거로 답합니다. 답변은 참고용이며, 최종 판단 전 원문을 확인하세요.")

if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    cols = st.columns(len(EXAMPLES))
    for col, example in zip(cols, EXAMPLES):
        if col.button(example, use_container_width=True):
            st.session_state.pending = example
            st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg["content"])
        elif "result" in msg:
            render_answer(msg["result"])
        else:
            st.error(msg["content"])

question = st.chat_input("질문을 입력하세요 (예: 공영주차장 주차요금 감면 대상은?)") or st.session_state.pop("pending", None)
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("조문·별표를 검색하고 근거를 확인하는 중…"):
            try:
                result = api_post("/ask", {"question": question, "top_k": top_k})
            except (requests.RequestException, RuntimeError) as e:
                st.error(str(e))
                st.session_state.messages.append({"role": "assistant", "content": str(e)})
            else:
                render_answer(result)
                st.session_state.messages.append({"role": "assistant", "result": result})
