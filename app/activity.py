"""
Real-time activity log: every time a recognized person moves into a new zone,
log a plain deterministic event ("Mia entered Espresso Station"). This is real
and instant, not LLM-generated. A separate narration layer (see review.py-style
Agent Router call in main.py) turns a window of these raw events into a natural
paragraph on request, so the "AI narrates" feature is honestly built on top of
real logged events, not invented.
"""
import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "attendance.db")

_last_zone = {}  # (name, source) -> zone_label, in-memory, resets on restart


def _conn():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS activity_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            source TEXT NOT NULL,
            ts TEXT NOT NULL,
            event TEXT NOT NULL
        )
    """)
    return conn


def maybe_log_zone_change(name: str, source: str, zone_label: str):
    key = (name, source)
    if _last_zone.get(key) == zone_label:
        return
    _last_zone[key] = zone_label
    conn = _conn()
    conn.execute(
        "INSERT INTO activity_log (name, source, ts, event) VALUES (?, ?, ?, ?)",
        (name, source, datetime.now().isoformat(), f"moved to {zone_label}"),
    )
    conn.commit()
    conn.close()


def recent_activity(name: str = None, source: str = None, limit: int = 30):
    conn = _conn()
    q = "SELECT name, source, ts, event FROM activity_log WHERE 1=1"
    params = []
    if name:
        q += " AND name=?"
        params.append(name)
    if source:
        q += " AND source=?"
        params.append(source)
    q += " ORDER BY id DESC LIMIT ?"
    params.append(limit)
    rows = conn.execute(q, params).fetchall()
    conn.close()
    return [{"name": r[0], "source": r[1], "ts": r[2], "event": r[3]} for r in rows]
