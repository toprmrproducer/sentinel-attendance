"""
Core face-recognition engine. No training here, just:
1. embed(): run a pretrained model over a face and get a 512-dim vector
2. enroll(): store a person's vector(s) under their name
3. recognize(): compare a new face's vector against everyone enrolled
"""
import os
import json
import numpy as np
import cv2
from insightface.app import FaceAnalysis

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
EMBED_STORE = os.path.join(DATA_DIR, "embeddings.json")
os.makedirs(DATA_DIR, exist_ok=True)

_face_app = None


def get_face_app():
    global _face_app
    if _face_app is None:
        _face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        _face_app.prepare(ctx_id=0, det_size=(640, 640))
    return _face_app


def _load_store():
    if os.path.exists(EMBED_STORE):
        with open(EMBED_STORE) as f:
            return json.load(f)
    return {}


def _save_store(store):
    with open(EMBED_STORE, "w") as f:
        json.dump(store, f)


def detect_faces(image_bgr):
    """Returns a list of insightface Face objects (has .bbox, .embedding, .det_score)."""
    app = get_face_app()
    return app.get(image_bgr)


def enroll_person(name: str, image_path: str) -> dict:
    img = cv2.imread(image_path)
    if img is None:
        return {"ok": False, "error": f"could not read image {image_path}"}
    faces = detect_faces(img)
    if not faces:
        return {"ok": False, "error": "no face detected in enrollment photo"}
    faces.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
    face = faces[0]
    store = _load_store()
    store.setdefault(name, [])
    store[name].append(face.embedding.tolist())
    _save_store(store)
    return {"ok": True, "name": name, "det_score": float(face.det_score), "num_enrolled_vectors": len(store[name])}


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a)
    b = np.asarray(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def recognize_face(embedding, threshold: float = 0.45):
    store = _load_store()
    best_name, best_score = "unknown", -1.0
    for name, vecs in store.items():
        for v in vecs:
            s = cosine_sim(embedding, v)
            if s > best_score:
                best_score, best_name = s, name
    if best_score < threshold:
        return {"name": "unknown", "score": best_score}
    return {"name": best_name, "score": best_score}


def recognize_in_frame(image_bgr):
    faces = detect_faces(image_bgr)
    results = []
    for f in faces:
        match = recognize_face(f.embedding)
        results.append({
            "bbox": [int(x) for x in f.bbox],
            "det_score": float(f.det_score),
            "match": match["name"],
            "match_score": round(match["score"], 4),
        })
    return results
