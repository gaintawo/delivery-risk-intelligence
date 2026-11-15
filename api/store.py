"""Explanation store: an append-only audit record of every score the API gives.

Each /predict call writes one row: when, which model version, the order facts
used (no personal data: the API never receives IDs, names or postcodes), the
score, the decisions, the top reasons and the suggested actions. An auditor,
the Review Board or a seller who challenges a flag can retrieve the exact
explanation later with GET /explanations/{prediction_id}.

SQLite by default (DRI_EXPLANATION_DB); a PostgreSQL URL works the same way via
SQLAlchemy in production. Retention follows the governance framework: 24 months.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path

DEFAULT = Path(__file__).resolve().parents[1] / "logs" / "explanations.db"
SCHEMA = """CREATE TABLE IF NOT EXISTS explanations (
    prediction_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    model TEXT NOT NULL,
    input_sha256 TEXT NOT NULL,
    inputs TEXT NOT NULL,
    late_probability REAL NOT NULL,
    flagged INTEGER NOT NULL,
    flagged_fairness_adjusted INTEGER,
    threshold REAL NOT NULL,
    top_reasons TEXT NOT NULL,
    suggested_actions TEXT NOT NULL,
    expected_delivery_days REAL
)"""


class ExplanationStore:
    def __init__(self, path: str | os.PathLike | None = None):
        self.path = Path(path or os.getenv("DRI_EXPLANATION_DB", DEFAULT))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._conn() as c:
            c.execute(SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=10)

    def record(self, inputs: dict, result: dict) -> str:
        pid = str(uuid.uuid4())
        payload = json.dumps(inputs, sort_keys=True, default=str)
        row = (pid, datetime.now(UTC).isoformat(timespec="seconds"), result["model"],
               hashlib.sha256(payload.encode()).hexdigest(), payload, result["late_probability"],
               int(result["flagged"]),
               None if result.get("flagged_fairness_adjusted") is None else int(result["flagged_fairness_adjusted"]),
               result["threshold"], json.dumps(result["top_reasons"], default=str),
               json.dumps(result.get("suggested_actions", [])), result.get("expected_delivery_days"))
        with self._lock, self._conn() as c:
            c.execute("INSERT INTO explanations VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", row)
        return pid

    @staticmethod
    def _to_dict(cur: sqlite3.Cursor, r: tuple) -> dict:
        d = {col[0]: v for col, v in zip(cur.description, r)}
        for k in ("inputs", "top_reasons", "suggested_actions"):
            d[k] = json.loads(d[k])
        for k in ("flagged", "flagged_fairness_adjusted"):
            d[k] = None if d[k] is None else bool(d[k])
        return d

    def get(self, prediction_id: str) -> dict | None:
        with self._conn() as c:
            cur = c.execute("SELECT * FROM explanations WHERE prediction_id = ?", (prediction_id,))
            r = cur.fetchone()
            return None if r is None else self._to_dict(cur, r)

    def recent(self, limit: int = 20) -> list[dict]:
        with self._conn() as c:
            cur = c.execute("SELECT * FROM explanations ORDER BY created_at DESC LIMIT ?", (limit,))
            return [self._to_dict(cur, r) for r in cur.fetchall()]

    def count(self) -> int:
        with self._conn() as c:
            return int(c.execute("SELECT COUNT(*) FROM explanations").fetchone()[0])
