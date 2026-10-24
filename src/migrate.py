"""SQL 파일을 하나의 트랜잭션으로 실행한다 (psql 없이 마이그레이션 적용).

실행:
    python -m src.migrate db/migrations/001_heading_boost.sql
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.db import connect


def main() -> None:
    parser = argparse.ArgumentParser(description="SQL 마이그레이션 적용")
    parser.add_argument("sql_file", type=Path)
    args = parser.parse_args()

    with connect() as conn, conn.transaction():
        conn.execute(args.sql_file.read_text(encoding="utf-8"))
    print(f"적용 완료: {args.sql_file}")


if __name__ == "__main__":
    main()
