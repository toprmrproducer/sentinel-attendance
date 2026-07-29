"""
Storage layer for the God's Eye system: every tracked object (person, car, bag, etc.)
gets a persistent track_id from YOLO's built-in tracker. This table is how we answer
"how many times has this track been seen" / "how long has it been on screen".
"""
import sqlite3
import os
import json
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "attendance.db")


def _conn():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS tracks (
            track_id INTEGER,
            source TEXT,
            class_name TEXT,
            identity_name TEXT,
            first_seen TEXT,
            last_seen TEXT,
            frame_count INTEGER DEFAULT 0,
            last_bbox TEXT,
            PRIMARY KEY (track_id, source)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS zones (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT,
            label TEXT,
            x1 INTEGER, y1 INTEGER, x2 INTEGER, y2 INTEGER,
            created_ts TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS zone_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            zone_id INTEGER,
            track_id INTEGER,
            source TEXT,
            ts TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS anomalies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT,
            kind TEXT,
            ts TEXT,
            severity REAL,
            note TEXT
        )
    """)
    return conn


def upsert_track(track_id: int, source: str, class_name: str, identity_name, bbox):
    conn = _conn()
    now = datetime.now().isoformat()
    row = conn.execute(
        "SELECT frame_count FROM tracks WHERE track_id=? AND source=?", (track_id, source)
    ).fetchone()
    if row is None:
        conn.execute(
            "INSERT INTO tracks (track_id, source, class_name, identity_name, first_seen, last_seen, frame_count, last_bbox) "
            "VALUES (?, ?, ?, ?, ?, ?, 1, ?)",
            (track_id, source, class_name, identity_name, now, now, json.dumps(bbox)),
        )
    else:
        conn.execute(
            "UPDATE tracks SET last_seen=?, frame_count=frame_count+1, last_bbox=?, "
            "identity_name=COALESCE(?, identity_name) WHERE track_id=? AND source=?",
            (now, json.dumps(bbox), identity_name, track_id, source),
        )
    conn.commit()
    conn.close()


def list_tracks(source: str, class_name: str = None):
    conn = _conn()
    q = "SELECT track_id, class_name, identity_name, first_seen, last_seen, frame_count, last_bbox FROM tracks WHERE source=?"
    params = [source]
    if class_name:
        q += " AND class_name=?"
        params.append(class_name)
    q += " ORDER BY last_seen DESC"
    rows = conn.execute(q, params).fetchall()
    conn.close()
    out = []
    for r in rows:
        first, last = datetime.fromisoformat(r[3]), datetime.fromisoformat(r[4])
        out.append({
            "track_id": r[0], "class_name": r[1], "identity_name": r[2],
            "first_seen": r[3], "last_seen": r[4],
            "dwell_seconds": round((last - first).total_seconds(), 1),
            "frame_count": r[5], "last_bbox": json.loads(r[6]) if r[6] else None,
        })
    return out


def track_profile(track_id: int, source: str):
    matches = [t for t in list_tracks(source) if t["track_id"] == track_id]
    return matches[0] if matches else None


def create_zone(source: str, label: str, x1: int, y1: int, x2: int, y2: int):
    conn = _conn()
    conn.execute(
        "INSERT INTO zones (source, label, x1, y1, x2, y2, created_ts) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (source, label, x1, y1, x2, y2, datetime.now().isoformat()),
    )
    conn.commit()
    zid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.close()
    return zid


def list_zones(source: str):
    conn = _conn()
    rows = conn.execute(
        "SELECT id, label, x1, y1, x2, y2 FROM zones WHERE source=?", (source,)
    ).fetchall()
    conn.close()
    return [{"id": r[0], "label": r[1], "bbox": [r[2], r[3], r[4], r[5]]} for r in rows]


def record_zone_event(zone_id: int, track_id: int, source: str):
    conn = _conn()
    conn.execute(
        "INSERT INTO zone_events (zone_id, track_id, source, ts) VALUES (?, ?, ?, ?)",
        (zone_id, track_id, source, datetime.now().isoformat()),
    )
    conn.commit()
    conn.close()


def zone_stats(source: str):
    conn = _conn()
    rows = conn.execute("""
        SELECT z.id, z.label, COUNT(DISTINCT e.track_id) as unique_tracks, COUNT(e.id) as total_events
        FROM zones z LEFT JOIN zone_events e ON e.zone_id = z.id AND e.source = z.source
        WHERE z.source=? GROUP BY z.id
    """, (source,)).fetchall()
    conn.close()
    return [{"id": r[0], "label": r[1], "unique_tracks": r[2], "total_events": r[3]} for r in rows]


def log_anomaly(source: str, kind: str, severity: float, note: str):
    conn = _conn()
    conn.execute(
        "INSERT INTO anomalies (source, kind, ts, severity, note) VALUES (?, ?, ?, ?, ?)",
        (source, kind, datetime.now().isoformat(), severity, note),
    )
    conn.commit()
    conn.close()


def throughput_per_minute(source: str, class_name: str = "cup"):
    """Proxy for output rate: counts distinct NEW tracks of a class per minute.
    Honest framing: this counts cup-shaped objects entering the frame/zone, not
    verified 'drink handed to customer' events. Good enough as a throughput signal,
    not a perfect POS-level count."""
    tracks = [t for t in list_tracks(source, class_name)]
    buckets = {}
    for t in tracks:
        minute_key = t["first_seen"][:16]  # YYYY-MM-DDTHH:MM
        buckets[minute_key] = buckets.get(minute_key, 0) + 1
    return [{"minute": k, "count": v} for k, v in sorted(buckets.items())]


def recent_anomalies(source: str = None, limit: int = 100):
    conn = _conn()
    if source:
        rows = conn.execute(
            "SELECT source, kind, ts, severity, note FROM anomalies WHERE source=? ORDER BY id DESC LIMIT ?",
            (source, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT source, kind, ts, severity, note FROM anomalies ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    conn.close()
    return [{"source": r[0], "kind": r[1], "ts": r[2], "severity": r[3], "note": r[4]} for r in rows]
