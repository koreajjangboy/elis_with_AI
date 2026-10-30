"""뉴스 기사 입력 처리: 질문 속 기사 URL을 읽어 제목·본문을 뽑거나, 붙여넣은 기사 본문을 그대로 쓴다.

    news = read_news(question)   # → NewsArticle | None (뉴스 입력이 아니면 None)

- URL: requests로 받아 BeautifulSoup으로 제목(og:title → h1 → title)과 본문(언론사 공통 본문 영역 → 긴 문단 모음)을 뽑는다.
  서버가 사용자 URL을 대신 여는 것이므로 http(s)만, 공인 IP로 풀리는 호스트만 연다(리디렉션마다 다시 확인 — 내부망 접근 방지).
  응답 크기·시간을 제한한다.
- URL을 못 읽으면(차단·시간 초과·본문 없음) 질문에 함께 붙여넣은 기사 본문이 있으면 그것을 쓰고, 없으면 error만 채워 돌려준다.
- URL이 없어도 기사처럼 긴 글(ARTICLE_MIN_CHARS 이상, 3줄·3문장 이상 또는 5문장 이상)이면 붙여넣은 기사로 본다.

    python -m src.news_parser https://www.segye.com/newsView/...   # 추출 결과 확인
"""

from __future__ import annotations

import ipaddress
import re
import socket
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse

import requests

URL_RE = re.compile(r"https?://[^\s<>\"'()\[\]]+")
ARTICLE_MIN_CHARS = 200
MAX_BODY_CHARS = 6000
MAX_DOWNLOAD_BYTES = 3_000_000
MAX_REDIRECTS = 3
TIMEOUT = 10
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) elis-with-ai/0.1 (+council research assistant)"

# 언론사 공통 본문 영역 (앞에서부터 시도, 200자 이상인 가장 긴 것)
BODY_SELECTORS = [
    "[itemprop=articleBody]", "#article_txt", "#articleBody", "#article-view-content-div", "#newsct_article", "#dic_area",
    "#articletxt", "#news_body_area", ".article_txt", ".article_body", ".article-body", ".news_view", ".view_con",
    ".news_content", ".article_view", "article",
]
NOISE_TAGS = ["script", "style", "noscript", "iframe", "form", "button", "header", "footer", "nav", "aside", "figure", "figcaption"]
NOISE_LINE_RE = re.compile(r"기자\s*페이지|무단\s*전재|재배포\s*금지|ⓒ|©|Copyright|구독|좋아요|기사\s*제보|[가-힣]{2,4}\s*기자\s*[\w.]+@")
_SENTENCE_RE = re.compile(r"[.!?。]\s|다\.\s?|요\.\s?")


class NewsError(RuntimeError):
    pass


@dataclass
class NewsArticle:
    title: str
    body: str
    url: str | None = None
    source: str = "text"  # 'url'(기사 URL에서 추출) | 'text'(붙여넣은 본문)
    error: str | None = None  # URL을 읽지 못한 사유

    @property
    def ok(self) -> bool:
        return bool(self.body.strip())


# ---------------------------------------------------------------- 입력 판별

def extract_url(text: str) -> str | None:
    m = URL_RE.search(text)
    return m.group(0).rstrip(".,;…") if m else None


def looks_like_article(text: str) -> bool:
    """붙여넣은 기사 본문으로 볼 만한 긴 글인지 (질문 한두 줄과 구분)."""
    text = text.strip()
    sentences = len(_SENTENCE_RE.findall(text))
    return len(text) >= ARTICLE_MIN_CHARS and ((text.count("\n") >= 2 and sentences >= 3) or sentences >= 5)


def article_from_text(text: str) -> NewsArticle:
    """붙여넣은 기사: 첫 줄이 짧으면 제목으로 본다."""
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    title = lines[0] if lines and len(lines[0]) <= 120 and len(lines) > 1 else ""
    body = "\n".join(lines[1:] if title else lines)
    return NewsArticle(title=title, body=clean_body(body), source="text")


def read_news(question: str, fetch=None) -> NewsArticle | None:
    """질문이 뉴스 입력(기사 URL 또는 붙여넣은 기사)이면 NewsArticle, 아니면 None."""
    fetch = fetch or fetch_article
    url = extract_url(question)
    rest = URL_RE.sub(" ", question).strip()
    if url:
        try:
            return fetch(url)
        except NewsError as e:
            if looks_like_article(rest):  # 크롤링이 막혀도 함께 붙여넣은 본문으로 분석
                article = article_from_text(rest)
                article.url, article.error = url, str(e)
                return article
            return NewsArticle(title="", body="", url=url, source="url", error=str(e))
    return article_from_text(question) if looks_like_article(question) else None


# ---------------------------------------------------------------- 크롤링

def check_public_url(url: str) -> None:
    """http(s)이고 호스트가 공인 IP로만 풀리는지 확인 (내부망·로컬 주소 차단)."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise NewsError(f"http(s) 기사 주소가 아닙니다: {url}")
    try:
        infos = socket.getaddrinfo(parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80))
    except socket.gaierror as e:
        raise NewsError(f"주소를 찾을 수 없습니다: {parsed.hostname}") from e
    for info in infos:
        ip = ipaddress.ip_address(info[4][0].split("%")[0])
        if not ip.is_global:
            raise NewsError(f"내부망·로컬 주소는 열 수 없습니다: {parsed.hostname}")


def fetch_article(url: str, timeout: float = TIMEOUT) -> NewsArticle:
    html = download(url, timeout)
    article = parse_article_html(html, url)
    if not article.ok:
        raise NewsError("기사 본문을 찾지 못했습니다 (본문을 직접 붙여넣어 주세요).")
    return article


def download(url: str, timeout: float = TIMEOUT) -> str:
    headers = {"User-Agent": USER_AGENT, "Accept-Language": "ko-KR,ko;q=0.9"}
    for _ in range(MAX_REDIRECTS + 1):
        check_public_url(url)
        try:
            resp = requests.get(url, headers=headers, timeout=timeout, stream=True, allow_redirects=False)
        except requests.RequestException as e:
            raise NewsError(f"기사를 불러오지 못했습니다: {type(e).__name__}") from e
        with resp:
            if resp.is_redirect:
                url = urljoin(url, resp.headers.get("Location", ""))
                continue
            if not resp.ok:
                raise NewsError(f"기사를 불러오지 못했습니다: HTTP {resp.status_code}")
            content = b""
            for chunk in resp.iter_content(64 * 1024):
                content += chunk
                if len(content) > MAX_DOWNLOAD_BYTES:
                    raise NewsError("기사 페이지가 너무 큽니다.")
            # 국내 언론사는 EUC-KR 페이지가 남아 있다: 헤더에 문자셋이 없으면 내용으로 추정
            encoding = resp.encoding if resp.encoding and resp.encoding.lower() != "iso-8859-1" else resp.apparent_encoding
            return content.decode(encoding or "utf-8", errors="replace")
    raise NewsError("리디렉션이 너무 많습니다.")


# ---------------------------------------------------------------- HTML 파싱 (네트워크 무관)

def parse_article_html(html: str, url: str | None = None) -> NewsArticle:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    title = _meta(soup, "og:title") or _text(soup.find("h1")) or _text(soup.title)
    for tag in soup(NOISE_TAGS):
        tag.decompose()

    candidates = [_block_text(el) for sel in BODY_SELECTORS for el in soup.select(sel)]
    body = max((c for c in candidates if len(c) >= 200), key=len, default="")
    if not body:  # 본문 영역을 못 찾으면 긴 문단을 모은다
        body = "\n".join(t for t in (_text(p) for p in soup.find_all("p")) if len(t) >= 40)
    if not body:
        body = _meta(soup, "og:description")
    return NewsArticle(title=title.strip(), body=clean_body(body), url=url, source="url")


def clean_body(text: str) -> str:
    lines = [re.sub(r"[ \t ]+", " ", ln).strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln and not NOISE_LINE_RE.search(ln)]
    return "\n".join(lines)[:MAX_BODY_CHARS]


def _block_text(el) -> str:
    # <br>·<p> 경계를 줄바꿈으로 보존
    for br in el.find_all("br"):
        br.replace_with("\n")
    return "\n".join(t.strip() for t in el.get_text("\n").splitlines() if t.strip())


def _text(el) -> str:
    return el.get_text(" ", strip=True) if el else ""


def _meta(soup, prop: str) -> str:
    tag = soup.find("meta", property=prop) or soup.find("meta", attrs={"name": prop})
    return (tag.get("content") or "").strip() if tag else ""


def main() -> None:
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="뉴스 기사 제목·본문 추출 확인")
    parser.add_argument("url")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    try:
        article = fetch_article(args.url)
    except NewsError as e:
        raise SystemExit(f"실패: {e}")
    print(f"제목: {article.title}\n본문 {len(article.body)}자\n{article.body}")


if __name__ == "__main__":
    main()
