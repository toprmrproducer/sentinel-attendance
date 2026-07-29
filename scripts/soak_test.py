"""
Long-feed verification. Runs the real detection+tracking+anomaly pipeline across
a real video file frame by frame (sampling every Nth frame to keep wall-clock sane),
and reports real throughput + real counts, not estimates.

Usage: python3 scripts/soak_test.py <video_path> <source_name> [sample_every] [max_processed_frames]
"""
import sys
import os
import time
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import cv2
from app import god_eye, tracking


def run(video_path, source_name, sample_every=5, max_processed_frames=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"ok": False, "error": f"could not open {video_path}"}

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_i = 0
    processed = 0
    t0 = time.time()
    class_counts = {}

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_i += 1
        if frame_i % sample_every != 0:
            continue
        dets = god_eye.process_frame(frame, source_name)
        for d in dets:
            class_counts[d["class"]] = class_counts.get(d["class"], 0) + 1
        processed += 1
        if max_processed_frames and processed >= max_processed_frames:
            break

    elapsed = time.time() - t0
    cap.release()

    tracks = tracking.list_tracks(source_name)
    unique_by_class = {}
    for t in tracks:
        unique_by_class.setdefault(t["class_name"], []).append(t)

    return {
        "ok": True,
        "video_duration_sec": round(total_frames / fps, 1),
        "video_duration_hms": time.strftime("%H:%M:%S", time.gmtime(total_frames / fps)),
        "frames_in_video": total_frames,
        "frames_processed": processed,
        "sample_every_nth_frame": sample_every,
        "wall_clock_sec": round(elapsed, 1),
        "processing_fps": round(processed / elapsed, 2) if elapsed else 0,
        "raw_detection_counts_by_class": class_counts,
        "unique_tracks_by_class": {k: len(v) for k, v in unique_by_class.items()},
        "top_dwell_person_tracks": sorted(
            [t for t in tracks if t["class_name"] == "person"],
            key=lambda t: t["dwell_seconds"], reverse=True
        )[:10],
        "anomalies_logged": len(tracking.recent_anomalies(source_name, limit=10000)),
    }


if __name__ == "__main__":
    video_path = sys.argv[1]
    source_name = sys.argv[2]
    sample_every = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    max_frames = int(sys.argv[4]) if len(sys.argv) > 4 else None
    result = run(video_path, source_name, sample_every, max_frames)
    print(json.dumps(result, indent=2, default=str))
