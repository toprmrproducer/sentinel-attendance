"""
Attendance ledger: every time a known face is recognized, log a check-in/check-out
event. Working hours = time between a person's first and last sighting in a day.
SQLite for the POC, swap for Postgres when this goes real.
"""
import sqlite3
import os
from datetime import datetime, date

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "attendance.db")


def _conn():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sightings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            ts TEXT NOT NULL,
            match_score REAL,
            source TEXT
        )
    """)
    return conn


def log_sighting(name: str, match_score: float, source: str = "camera"):
    if name == "unknown":
        return
    conn = _conn()
    conn.execute(
        "INSERT INTO sightings (name, ts, match_score, source) VALUES (?, ?, ?, ?)",
        (name, datetime.now().isoformat(), match_score, source),
    )
    conn.commit()
    conn.close()


def working_hours_today():
    """First-seen / last-seen / total span per person, for today."""
    today = date.today().isoformat()
    conn = _conn()
    rows = conn.execute(
        "SELECT name, MIN(ts), MAX(ts), COUNT(*) FROM sightings WHERE ts LIKE ? GROUP BY name",
        (f"{today}%",),
    ).fetchall()
    conn.close()
    out = []
    for name, first_ts, last_ts, count in rows:
        first = datetime.fromisoformat(first_ts)
        last = datetime.fromisoformat(last_ts)
        span_minutes = round((last - first).total_seconds() / 60, 1)
        out.append({
            "name": name,
            "first_seen": first.strftime("%H:%M:%S"),
            "last_seen": last.strftime("%H:%M:%S"),
            "span_minutes": span_minutes,
            "sightings": count,
        })
    out.sort(key=lambda r: r["first_seen"])
    return out


def sightings_for(name: str):
    conn = _conn()
    rows = conn.execute(
        "SELECT ts, match_score, source FROM sightings WHERE name=? ORDER BY ts ASC", (name,),
    ).fetchall()
    conn.close()
    return [{"ts": r[0], "match_score": r[1], "source": r[2]} for r in rows]


def all_identities():
    conn = _conn()
    rows = conn.execute("SELECT DISTINCT name FROM sightings").fetchall()
    conn.close()
    return [r[0] for r in rows]


def recent_sightings(limit: int = 50):
    conn = _conn()
    rows = conn.execute(
        "SELECT name, ts, match_score, source FROM sightings ORDER BY id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    conn.close()
    return [{"name": r[0], "ts": r[1], "match_score": r[2], "source": r[3]} for r in rows]
