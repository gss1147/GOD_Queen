import os
import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from config import AI_MEMORY_DIR

LOGS_DIR = AI_MEMORY_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
SQLITE_PATH = AI_MEMORY_DIR / "memory.sqlite3"

logger = logging.getLogger("HopeMemory")
if not logger.handlers:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / "memory.log"
    handler = logging.FileHandler(log_path, encoding="utf-8")
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] memory: %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


@dataclass
class MemoryRecord:
    kind: str
    content: str
    metadata: Dict[str, Any]

    def to_row(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "content": self.content,
            "metadata": json.dumps(self.metadata, ensure_ascii=False),
        }


class MemoryCore:
    """
    Simple, robust SQLite-based memory core.

    - Stores conversation turns, PDF summaries, datasets, and system events.
    - Uses a single table with json metadata.
    """

    def __init__(self) -> None:
        self.db_path = SQLITE_PATH
        self._conn: Optional[sqlite3.Connection] = None
        self._ensure_db()

    @property
    def _conn_handle(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
        return self._conn

    def _ensure_db(self) -> None:
        conn = self._conn_handle
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                kind TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata TEXT,
                created REAL NOT NULL
            )
            """
        )
        conn.commit()

    def store(self, rec: MemoryRecord) -> int:
        conn = self._conn_handle
        row = rec.to_row()
        cur = conn.execute(
            "INSERT INTO memory (kind, content, metadata, created) VALUES (?, ?, ?, ?)",
            (row["kind"], row["content"], row["metadata"], datetime.utcnow().timestamp()),
        )
        conn.commit()
        rec_id = int(cur.lastrowid)
        logger.info("Stored memory id=%s kind=%s", rec_id, rec.kind)
        return rec_id

    def fetch_recent(self, kind: str, limit: int = 50) -> List[MemoryRecord]:
        conn = self._conn_handle
        cur = conn.execute(
            "SELECT kind, content, metadata FROM memory "
            "WHERE kind=? ORDER BY id DESC LIMIT ?",
            (kind, limit),
        )
        rows = cur.fetchall()
        out: List[MemoryRecord] = []
        for k, content, metadata_json in rows:
            try:
                meta = json.loads(metadata_json) if metadata_json else {}
            except Exception:
                meta = {}
            out.append(MemoryRecord(kind=k, content=content, metadata=meta))
        return list(reversed(out))

    def vacuum(self) -> None:
        conn = self._conn_handle
        conn.execute("VACUUM")
        conn.commit()

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None