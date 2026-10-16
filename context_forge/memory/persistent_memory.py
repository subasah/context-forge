"""
ContextForge Persistent Memory Store.
SQLite-backed persistent memory for episodic facts, user profiles, and session entities.
"""

from __future__ import annotations
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class MemoryItem(BaseModel):
    id: Optional[int] = None
    user_id: str
    category: str  # 'profile', 'preference', 'entity', 'rule'
    key: str
    value: str
    confidence: float = 1.0
    updated_at: float = Field(default_factory=time.time)


class PersistentMemoryStore:
    """Thread-safe SQLite persistent memory store."""

    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        return self._conn

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memory_items (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    category TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    confidence REAL DEFAULT 1.0,
                    updated_at REAL NOT NULL,
                    UNIQUE(user_id, category, key)
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_user_cat ON memory_items (user_id, category)
            """)
            conn.commit()

    def set(self, user_id: str, category: str, key: str, value: str, confidence: float = 1.0) -> None:
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO memory_items (user_id, category, key, value, confidence, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id, category, key) DO UPDATE SET
                    value = excluded.value,
                    confidence = excluded.confidence,
                    updated_at = excluded.updated_at
            """, (user_id, category, key, value, confidence, time.time()))
            conn.commit()

    def get(self, user_id: str, category: str, key: str) -> Optional[MemoryItem]:
        with self._get_connection() as conn:
            row = conn.execute("""
                SELECT * FROM memory_items WHERE user_id = ? AND category = ? AND key = ?
            """, (user_id, category, key)).fetchone()
            if row:
                return MemoryItem(**dict(row))
        return None

    def get_all_for_user(self, user_id: str) -> List[MemoryItem]:
        with self._get_connection() as conn:
            rows = conn.execute("""
                SELECT * FROM memory_items WHERE user_id = ? ORDER BY category, key
            """, (user_id,)).fetchall()
            return [MemoryItem(**dict(r)) for r in rows]

    def search(self, user_id: str, keyword: str) -> List[MemoryItem]:
        with self._get_connection() as conn:
            rows = conn.execute("""
                SELECT * FROM memory_items
                WHERE user_id = ? AND (key LIKE ? OR value LIKE ?)
            """, (user_id, f"%{keyword}%", f"%{keyword}%")).fetchall()
            return [MemoryItem(**dict(r)) for r in rows]

    def format_for_context(self, user_id: str) -> str:
        items = self.get_all_for_user(user_id)
        if not items:
            return ""
        lines = [f"- {item.category.title()} [{item.key}]: {item.value}" for item in items]
        return "\n".join(lines)
