"""PostgreSQL 연결."""

from __future__ import annotations

import os

import psycopg
from dotenv import load_dotenv


def connect() -> psycopg.Connection:
    """autocommit 연결. 원자성이 필요한 구간은 conn.transaction()으로 명시한다."""
    load_dotenv()
    url = os.getenv("DATABASE_URL", "").strip()
    if not url:
        raise SystemExit(".env에 DATABASE_URL이 설정되어 있지 않습니다. (예: postgresql://user:pw@host:5432/db?sslmode=require)")
    # Supabase Transaction pooler(6543)는 prepared statement를 지원하지 않는다
    prepare_threshold = None if ":6543/" in url else 5
    return psycopg.connect(url, autocommit=True, prepare_threshold=prepare_threshold)
