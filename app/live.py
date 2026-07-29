"""
Runs god_eye.process_frame continuously in a background thread per video source,
so the web layer never blocks on inference. The stream endpoint and the detections
API both just read the latest shared result.
"""
import threading
import time
import cv2

from app import god_eye

_sources = {}  # name -> {"thread":..., "frame": jpeg_bytes, "detections": [...], "lock": Lock, "stop": bool}


def _worker(name: str, video_path: str, loop: bool, sample_every: int):
    state = _sources[name]
    cap = cv2.VideoCapture(video_path)
    frame_i = 0
    while not state["stop"]:
        ok, frame = cap.read()
        if not ok:
            if loop:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            break
        frame_i += 1
        if frame_i % sample_every == 0:
            dets = god_eye.process_frame(frame, name)
            annotated = god_eye.draw_detections(frame.copy(), dets)
            ok2, jpeg = cv2.imencode(".jpg", annotated)
            if ok2:
                with state["lock"]:
                    state["frame"] = jpeg.tobytes()
                    state["detections"] = dets
        time.sleep(0.01)
    cap.release()


def start_source(name: str, video_path: str, loop: bool = True, sample_every: int = 3):
    if name in _sources and _sources[name]["thread"].is_alive():
        return {"ok": True, "already_running": True}
    state = {"thread": None, "frame": None, "detections": [], "lock": threading.Lock(), "stop": False}
    _sources[name] = state
    t = threading.Thread(target=_worker, args=(name, video_path, loop, sample_every), daemon=True)
    state["thread"] = t
    t.start()
    return {"ok": True, "already_running": False}


def get_frame(name: str):
    state = _sources.get(name)
    if not state:
        return None
    with state["lock"]:
        return state["frame"]


def get_detections(name: str):
    state = _sources.get(name)
    if not state:
        return []
    with state["lock"]:
        return list(state["detections"])


def list_sources():
    return list(_sources.keys())
