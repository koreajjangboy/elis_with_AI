"""PostgreSQL 연결."""

from __future__ import annotations

import os

import psycopg
from dotenv import load_dotenv
from psycopg_pool import ConnectionPool


def _connect_args() -> tuple[str, dict]:
    load_dotenv()
    url = os.getenv("DATABASE_URL", "").strip()
    if not url:
        raise SystemExit(".env에 DATABASE_URL이 설정되어 있지 않습니다. (예: postgresql://user:pw@host:5432/db?sslmode=require)")
    # Supabase Transaction pooler(6543)는 prepared statement를 지원하지 않는다
    prepare_threshold = None if ":6543/" in url else 5
    return url, {"autocommit": True, "prepare_threshold": prepare_threshold}


def connect() -> psycopg.Connection:
    """autocommit 연결. 원자성이 필요한 구간은 conn.transaction()으로 명시한다."""
    url, kwargs = _connect_args()
    return psycopg.connect(url, **kwargs)


def create_pool(min_size: int = 1, max_size: int = 5) -> ConnectionPool:
    """API 서버용 커넥션 풀. 요청마다 새로 연결하면 클라우드 DB TLS 핸드셰이크가 매번 든다."""
    url, kwargs = _connect_args()
    return ConnectionPool(url, kwargs=kwargs, min_size=min_size, max_size=max_size, open=True,
                          check=ConnectionPool.check_connection)
