"""
The God's Eye per-frame pipeline: object+person detection with persistent track IDs
(YOLOv8n + built-in ByteTrack), face recognition fused onto person tracks, plus two
concrete anomaly signals (lighting drop, motion spike) that stand in for "something's off".

No claims of reading emotion/nervousness from posture here, that's not a reliable signal
from video alone. What IS real and useful: flagging an unusual event (lights cut, sudden
violent motion, someone lingering in a zone way past normal dwell) for a HUMAN to review,
optionally with an LLM description of the flagged clip. That's the honest version of
"theft detection" that doesn't overclaim.
"""
import os
import time
import cv2
from ultralytics import YOLO

from app import engine as face_engine
from app import tracking
from app import telephony

ALERT_CALL_NUMBER = os.environ.get("SENTINEL_ALERT_CALL_NUMBER")  # e.g. "919307512816", unset = disabled
ALERT_CALL_COOLDOWN_SEC = 120
_last_call_ts = 0

_yolo_by_source = {}  # each source needs its OWN model instance: ultralytics keeps
# ByteTrack state on the model object, so sharing one instance across concurrent
# videos corrupts both feeds' track IDs (this bit us in testing: two sources sharing
# one YOLO() gave empty/garbled detections on both).
_state = {}  # per-source: {"prev_gray": ndarray, "brightness_baseline": float}


def get_yolo(source: str):
    if source not in _yolo_by_source:
        _yolo_by_source[source] = YOLO("yolov8n.pt")
    return _yolo_by_source[source]


def _source_state(source: str):
    return _state.setdefault(source, {"prev_gray": None, "brightness_baseline": None})


def _maybe_call_alert(source: str, kind: str, note: str):
    global _last_call_ts
    if not ALERT_CALL_NUMBER:
        return
    now = time.time()
    if now - _last_call_ts < ALERT_CALL_COOLDOWN_SEC:
        return
    _last_call_ts = now
    telephony.trigger_alert_call(ALERT_CALL_NUMBER, f"{kind.replace('_', ' ')} on {source}. {note}")


def detect_anomalies(frame, source: str):
    st = _source_state(source)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_b = float(gray.mean())
    if st["brightness_baseline"] is None:
        st["brightness_baseline"] = mean_b
    else:
        prev_baseline = st["brightness_baseline"]
        st["brightness_baseline"] = 0.98 * prev_baseline + 0.02 * mean_b
        if mean_b < prev_baseline * 0.45 and prev_baseline > 15:
            note = f"brightness dropped from {prev_baseline:.1f} to {mean_b:.1f} (possible lights-out / camera tamper)"
            tracking.log_anomaly(source, "lighting_drop", severity=round(prev_baseline - mean_b, 1), note=note)
            _maybe_call_alert(source, "lighting_drop", note)
    if st["prev_gray"] is not None and st["prev_gray"].shape == gray.shape:
        diff = float(cv2.absdiff(gray, st["prev_gray"]).mean())
        if diff > 35:
            note = "sudden large frame-to-frame change (possible fast movement / struggle / camera knock)"
            tracking.log_anomaly(source, "motion_spike", severity=round(diff, 1), note=note)
            _maybe_call_alert(source, "motion_spike", note)
    st["prev_gray"] = gray


def _identify_person_crop(frame, bbox):
    x1, y1, x2, y2 = bbox
    x1, y1 = max(x1, 0), max(y1, 0)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    faces = face_engine.detect_faces(crop)
    if not faces:
        return None
    match = face_engine.recognize_face(faces[0].embedding)
    return match["name"] if match["name"] != "unknown" else None


def process_frame(frame, source: str, identify_faces: bool = True):
    """Runs detection+tracking+anomaly checks on one frame, returns a list of detections
    plus persists everything to the tracks/anomalies tables."""
    detect_anomalies(frame, source)
    yolo = get_yolo(source)
    results = yolo.track(frame, persist=True, verbose=False, tracker="bytetrack.yaml")[0]

    detections = []
    zones = tracking.list_zones(source)

    if results.boxes is not None and results.boxes.id is not None:
        boxes = results.boxes.xyxy.cpu().numpy()
        ids = results.boxes.id.cpu().numpy().astype(int)
        clss = results.boxes.cls.cpu().numpy().astype(int)
        confs = results.boxes.conf.cpu().numpy()

        for box, tid, cls, conf in zip(boxes, ids, clss, confs):
            bbox = [int(v) for v in box]
            cls_name = yolo.names[int(cls)]
            identity = None
            if identify_faces and cls_name == "person":
                identity = _identify_person_crop(frame, bbox)

            tracking.upsert_track(int(tid), source, cls_name, identity, bbox)

            cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
            for z in zones:
                zx1, zy1, zx2, zy2 = z["bbox"]
                if zx1 <= cx <= zx2 and zy1 <= cy <= zy2:
                    tracking.record_zone_event(z["id"], int(tid), source)

            detections.append({
                "track_id": int(tid),
                "class": cls_name,
                "identity": identity,
                "bbox": bbox,
                "conf": round(float(conf), 3),
            })
    return detections


def draw_detections(frame, detections):
    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        label = d["identity"] or f'{d["class"]} #{d["track_id"]}'
        color = (0, 200, 0) if d["identity"] else ((60, 160, 255) if d["class"] == "person" else (200, 200, 60))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, max(y1 - 6, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return frame
