"""
LLM reasoning layer over real attendance data. Calls Agent Router (pooled Claude Opus 4.8
proxy, via the already-working ~/.claude/bin/ar CLI which handles its Claude-Code-user-agent
quirk) with a data-only summary of a person's logged hours/consistency and asks for a
plain-language review note. This is a draft aid for a human manager, not an automated
HR decision, the model never sees raw video, only the structured stats we computed.
"""
import os
import json
import subprocess
from datetime import datetime, timedelta

AR_CLI = os.path.expanduser("~/.claude/bin/ar")


def build_person_summary(name: str, sightings: list):
    """sightings: list of {ts, match_score, source} for this person, already filtered."""
    if not sightings:
        return None
    days = {}
    for s in sightings:
        d = s["ts"][:10]
        days.setdefault(d, []).append(s["ts"])
    day_rows = []
    weekend_days = 0
    for d, ts_list in sorted(days.items()):
        ts_list.sort()
        first, last = ts_list[0], ts_list[-1]
        dow = datetime.fromisoformat(d).strftime("%A")
        if dow in ("Saturday", "Sunday"):
            weekend_days += 1
        span_min = round((datetime.fromisoformat(last) - datetime.fromisoformat(first)).total_seconds() / 60, 1)
        day_rows.append({"date": d, "day_of_week": dow, "first_seen": first, "last_seen": last,
                          "span_minutes": span_min, "sightings": len(ts_list)})
    return {
        "name": name,
        "total_days_present": len(days),
        "weekend_days_worked": weekend_days,
        "avg_daily_span_minutes": round(sum(r["span_minutes"] for r in day_rows) / len(day_rows), 1),
        "per_day": day_rows,
    }


def request_review(summary: dict) -> dict:
    """Sends the STRUCTURED SUMMARY ONLY (no video, no images) to Agent Router / Opus 4.8
    via the `ar` CLI, and asks for a plain factual review note a manager could sanity-check."""
    prompt = f"""You are reviewing REAL, ALREADY-COMPUTED attendance data for one employee. \
Do not invent facts not present in the data. Do not make an employment decision, only a \
short, factual, data-grounded observation a manager could use as one input among many. \
If the data is too thin to say anything meaningful, say that plainly.

DATA:
{json.dumps(summary, indent=2)}

Write a 4-6 sentence plain-language note covering: attendance consistency, whether weekend \
work appears (and whether that's notable), and one honest caveat about what this data \
can't tell you (e.g. small sample, no context for absences). End with a one-line \
recommendation phrased as "worth a conversation about X" rather than a verdict."""

    try:
        proc = subprocess.run(
            [AR_CLI, "ask", prompt],
            capture_output=True, text=True, timeout=45,
        )
        if proc.returncode != 0:
            return {"ok": False, "error": proc.stderr.strip() or f"ar exited {proc.returncode}"}
        lines = proc.stdout.strip().split("\n")
        # first line is "[model, cost]" bracket metadata from the ar CLI, rest is the note
        model_line = lines[0] if lines and lines[0].startswith("[") else None
        note = "\n".join(lines[1:] if model_line else lines).strip()
        return {"ok": True, "note": note, "meta": model_line}
    except Exception as e:
        return {"ok": False, "error": str(e)}
