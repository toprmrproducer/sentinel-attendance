# Sentinel Attendance / God's Eye

Open-source, on-prem employee attendance + productivity + theft-aversion system.
Face recognition (InsightFace) + multi-object tracking (YOLOv8n + ByteTrack) fused
into one live pipeline, with a login-gated web UI for hover-to-identify / click-to-profile
/ zone drawing, plus an LLM (Claude Opus, via Agent Router) reasoning layer over the
computed attendance data.

## What's real here

- Face recognition: enroll a person from a photo, recognize them live off a camera feed.
- Multi-object tracking: every person/car/cup/etc gets a persistent track ID across frames.
- Attendance: check-in/out inferred from first/last sighting per day, working hours computed.
- Productivity proxy: cups/minute from "cup"-class detections in frame (not a verified POS count).
- Anomaly signals: lighting-drop and motion-spike, framed as "flag for human review", not
  a claim to detect intent or emotion from video.
- God's Eye UI: live annotated stream, hover tooltips, click-to-profile, zone drawing with
  live stats, an Opus 4.8 "review packet" over real attendance data.

## What's honestly still rough

- Camera input is recorded/pulled footage right now, not a persistent RTSP/webcam feed.
- "cups/min" and "cash collected" are detection-count proxies with a configurable
  assumed price-per-item, not integrated with a real POS.
- Face recognition works best on a front-facing camera; an overhead/pass-cam angle
  (good for productivity/zone tracking) is a poor angle for check-in accuracy.
- Theft "detection" never claims to read nervousness/intent from body language, that's
  not a reliable signal from video alone. It flags lighting/motion anomalies for a human
  to check, and can auto-dial a phone number on a flagged event via Vobiz telephony.

## Stack

Python 3.9, FastAPI + Starlette sessions, OpenCV, InsightFace (buffalo_l), Ultralytics
YOLOv8n + ByteTrack, SQLite. See `CLAUDE.md` for the full architecture map.

## Run it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8811
```

Then open `/godseye` (login-gated) or `/dashboard`.
