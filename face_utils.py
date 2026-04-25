import cv2
import numpy as np
import os
import json

FACE_DATA_DIR   = "./face_data"
FACE_MODEL_PATH = "./models/face_model.yml"
FACE_LABELS_PATH = "./models/face_labels.json"

_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
FACE_CASCADE  = cv2.CascadeClassifier(_cascade_path)


def detect_faces_haar(gray: np.ndarray) -> list:
    faces = FACE_CASCADE.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
    )
    return list(faces) if len(faces) > 0 else []


def save_training_images(name: str, pil_images: list) -> int:
    person_dir = os.path.join(FACE_DATA_DIR, name.strip())
    os.makedirs(person_dir, exist_ok=True)
    existing = len([f for f in os.listdir(person_dir) if f.endswith(".jpg")])
    saved = 0

    for pil_img in pil_images:
        arr = np.array(pil_img.convert("RGB"))
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        for (x, y, w, h) in detect_faces_haar(gray):
            crop = cv2.resize(gray[y:y+h, x:x+w], (100, 100))
            cv2.imwrite(os.path.join(person_dir, f"{existing+saved:04d}.jpg"), crop)
            saved += 1

    return saved


def train_lbph_model() -> tuple:
    if not os.path.exists(FACE_DATA_DIR):
        return False, "No training data directory found."

    persons = sorted([
        d for d in os.listdir(FACE_DATA_DIR)
        if os.path.isdir(os.path.join(FACE_DATA_DIR, d))
    ])
    if not persons:
        return False, "No persons found in training data."

    faces, labels, label_map = [], [], {}
    for lid, person in enumerate(persons):
        label_map[lid] = person
        pdir = os.path.join(FACE_DATA_DIR, person)
        for fname in sorted(os.listdir(pdir)):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                img = cv2.imread(os.path.join(pdir, fname), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    faces.append(img)
                    labels.append(lid)

    if not faces:
        return False, "No valid face images found."

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(labels, dtype=np.int32))
    os.makedirs("./models", exist_ok=True)
    recognizer.save(FACE_MODEL_PATH)

    with open(FACE_LABELS_PATH, "w") as f:
        json.dump({str(k): v for k, v in label_map.items()}, f)

    return True, (
        f"Trained on {len(faces)} images · "
        f"{len(persons)} person(s): {', '.join(persons)}"
    )


def load_recognizer():
    if not (os.path.exists(FACE_MODEL_PATH) and os.path.exists(FACE_LABELS_PATH)):
        return None, {}
    try:
        rec = cv2.face.LBPHFaceRecognizer_create()
        rec.read(FACE_MODEL_PATH)
        with open(FACE_LABELS_PATH) as f:
            raw = json.load(f)
        label_map = {int(k): v for k, v in raw.items()}
        return rec, label_map
    except Exception:
        return None, {}


def get_registered_persons() -> list:
    if not os.path.exists(FACE_DATA_DIR):
        return []
    return sorted([
        d for d in os.listdir(FACE_DATA_DIR)
        if os.path.isdir(os.path.join(FACE_DATA_DIR, d))
    ])


def recognize_faces(img_bgr: np.ndarray, recognizer, label_map: dict,
                    conf_threshold: float = 80.0) -> tuple:
    """Returns (annotated_bgr, list of {name, confidence, bbox})."""
    out = img_bgr.copy()
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    results = []

    for (x, y, w, h) in detect_faces_haar(gray):
        crop = cv2.resize(gray[y:y+h, x:x+w], (100, 100))
        name, conf_pct = "Unknown", 0.0

        if recognizer is not None:
            lid, dist = recognizer.predict(crop)
            if dist < conf_threshold:
                name     = label_map.get(lid, "Unknown")
                conf_pct = max(0.0, 100.0 - dist)

        color = (0, 220, 110) if name != "Unknown" else (0, 100, 255)
        cv2.rectangle(out, (x, y), (x+w, y+h), color, 2)
        banner_y = max(y - 28, 0)
        cv2.rectangle(out, (x, banner_y), (x+w, y), color, cv2.FILLED)
        cv2.putText(out, name, (x+4, y-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

        results.append({"name": name, "confidence": conf_pct, "bbox": (x, y, w, h)})

    return out, results
