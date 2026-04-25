# ARGUS INTELLIGENCE — System Documentation

## Overview

Argus Intelligence is a Kenyan license plate detection and recognition system with face recognition, watchlist alerting, and a 4-engine ensemble OCR pipeline. It runs as a Streamlit web application and processes both still images and video footage.

---

## Architecture

```
app.py          — Streamlit UI + all detection logic
db.py           — SQLite persistence (detections + watchlist)
util.py         — Kenyan plate format correction + deduplication
face_utils.py   — Face detection (Haar) + LBPH recognition
```

---

## Detection Pipeline

### 1. Vehicle Detection
Model: **YOLOv11s** (`yolo11s.pt`, COCO-trained)
Detects vehicles in the frame and classifies them as Car, Motorcycle, Bus, or Truck. Each detected plate is then associated with the nearest vehicle by Euclidean distance between centroids.

### 2. License Plate Detection
Model: **yolo-v9-t-640-license-plate-end2end** (ONNX, via fast-alpr)
Runs CoreML execution provider on Apple Silicon for accelerated inference. Outputs bounding boxes for plate regions. Each crop is padded by 8 pixels before being passed to OCR.

### 3. Image Preprocessing (per crop)
Before OCR, every plate crop goes through:
- **Blur detection** — Laplacian variance check. If below adaptive threshold, EDSR deblur runs first.
- **EDSR super-resolution** — EDSR-base x2 (eugenesiow/edsr-base via super-image). Upscales blurry crops 2× before OCR. Only triggers on crops that fail the sharpness test.
- **Upscaling** — Bicubic resize to at least 200×60px (minimum 2× scale).
- **Deskew** — MinAreaRect-based rotation correction (1°–15° range).
- **CLAHE** — Contrast-limited adaptive histogram equalization (clipLimit 2.0, 4×4 tiles).
- **Bilateral filter** — Edge-preserving noise reduction (d=9, σ=75).

### 4. Ensemble OCR (4 engines)

Every plate crop is read by all four engines independently:

| Engine | Model / Config | Strength |
|--------|---------------|----------|
| **fast-alpr** | cct-xs-v2-global-model (ONNX) | Fast, purpose-built for plates |
| **PaddleOCR** | PP-OCRv4 with angle classifier | Strong on rotated/angled text |
| **EasyOCR** | CRAFT + CRNN (en) | Robust on low-res crops |
| **Tesseract** | v5.5.2, PSM 7, LSTM + legacy OEM 3 | Strict whitelist `A-Z0-9`, single-line mode |

Each engine's raw text is passed through `process_alpr_text()` (util.py) which applies Kenyan-format positional correction:
- Status `valid` — exact match, no changes
- Status `corrected` — 1–2 character fixes (e.g. `0→O`, `1→I`)
- Status `guessed` — 3+ fixes or sliding-window match, confidence penalised
- Status `rejected` — no valid Kenyan plate producible

**Voting logic:**
1. All engines ranked by status (`valid > corrected > guessed > rejected`), then confidence.
2. If ≥2 engines reach the same top status, `vote_plate()` runs character-level majority voting across their results.
3. The voted consensus goes through format correction one final time.
4. Result: single best plate text + confidence.

### 5. Kenyan Plate Format
Format: `K[A-Z]{2} \d{3}[A-Z]` — e.g. `KDA 123B`, `KBZ 456Y`

Positional correction maps:
- Letter positions (0,1,2,6): digits corrected to visually similar letters (`0→O`, `1→I`, `8→B`, etc.)
- Digit positions (3,4,5): letters corrected to visually similar digits (`O→0`, `I→1`, `B→8`, etc.)

### 6. Video Frame Sampling
- Samples one frame per second (step = round(FPS)).
- Adaptive sharpness threshold: samples first 30 target frames, sets threshold at 35% of median Laplacian variance. Blurry frames are skipped.
- Deduplication via `PlateDeduplicator`: fuzzy matching (SequenceMatcher ≥0.82 similarity), 30-second cooldown per plate, 120-second re-entry window. Same plate reappearing after 120s is logged as a new visit.
- Post-processing voting pass: after all frames are processed, each saved plate's text is replaced by the character-voted consensus from all its OCR variants.

---

## OCR Ensemble Vote Board

Every detection renders a live vote board showing:
- The raw plate crop photo (base64-embedded JPEG)
- Each engine's result, normalised status, and confidence
- Which engine(s) were selected or went into the vote
- If voting triggered: which plates were compared and the consensus character-by-character result
- Yellow `VOTED` badge in the panel header when character voting fired

During video analysis the vote board appears inline in the live detection feed whenever a vote occurs.

---

## Persistence — SQLite Backend

Database: `csv_detections/argus.db`

Tables:
- **detections** — timestamp, source (image/video), vehicle_type, plate_text, ocr_confidence, status, image_file
- **watchlist** — plate_text (unique), label, added_at

Indices on `plate_text`, `timestamp`, and `source` for fast filtered queries.

Plate crop images are saved to `licenses_plates_imgs_detected/` as JPEG files (preprocessed version).

---

## Watchlist / Alert System

Plates can be added to the watchlist via the sidebar with an optional label (e.g. "Stolen vehicle").

On every detection — image or video — `insert_detections()` cross-references all detected plates against the watchlist. Hits are returned immediately and displayed as:
- Red alert banner at the top of the results section
- Plate badge turns red (`plate-badge-alert` styling) in the results grid
- Watchlist tab shows full detection history for every monitored plate

---

## Face Recognition

Engine: **OpenCV LBPH** (Local Binary Pattern Histogram)
Detection: Haar cascade (`haarcascade_frontalface_default.xml`)

Training: Upload ≥5 frontal images per subject via the Face Training tab. The system extracts face crops (100×100 grayscale), trains the LBPH model, and saves it to `models/face_model.yml`.

Recognition runs on every analysed image. Recognised faces are drawn with a green bounding box and name label; unknown faces get an orange box. Recognised names are listed below the detection results.

Confidence threshold: distance < 80 (LBPH distance metric, lower = more similar).

---

## User Interface

Framework: **Streamlit** with custom CSS (Inter font, light theme)

**Tabs:**
- **Image Analysis** — upload image, view annotated result, plate crops, vote boards
- **Video Analysis** — upload video, live frame view with bounding boxes, live detection feed, post-analysis vote boards
- **Detection Log** — filterable/searchable table of all detections with CSV export
- **Watchlist** — full watchlist management + detection history per monitored plate
- **Face Training** — enrol subjects, view enrolled persons

**Sidebar:**
- Detection confidence slider (0.10–0.95)
- Live statistics (saved / auto-fixed / guessed / rejected counts)
- Watchlist manager (add plate + label, remove individual entries)
- Clear All Detections button

**Live video annotations:**
Vehicle bounding boxes drawn in blue with class label. Plate bounding boxes drawn in green (saved) or orange (rejected). Plate text floats above each box as an overlay label.

**Persistent log box** at the bottom of every page — shows most recent 200 detections with search/filter and download.

---

## Configuration

**`.streamlit/config.toml`**
```toml
[theme]
base = "light"
primaryColor = "#1A56DB"
backgroundColor = "#F0F4F8"
secondaryBackgroundColor = "#FFFFFF"
textColor = "#1E293B"
font = "sans serif"

[server]
maxUploadSize = 500
```

Max upload: 500 MB (supports large video files).

---

## Dependencies

```
streamlit>=1.35.0          — UI framework
ultralytics>=8.0.0         — YOLOv11 vehicle detection
fast-alpr>=0.4.0           — ONNX plate detection + OCR engine 1
onnxruntime>=1.18.0        — ONNX inference (CoreML on Apple Silicon)
paddlepaddle>=3.0.0        — PaddleOCR backend
paddleocr>=3.0.0           — OCR engine 2
easyocr>=1.7.0             — OCR engine 3
pytesseract>=0.3.10        — OCR engine 4 (requires tesseract binary)
super-image>=0.2.0         — EDSR deblur model
torch>=2.0.0               — PyTorch (EDSR inference)
opencv-contrib-python>=4.9.0  — Image processing + face recognition
Pillow>=10.0.0             — Image I/O
numpy>=1.24.0              — Array operations
pandas>=2.0.0              — Tabular data
```

External binary: `tesseract` v5+ (installed via `brew install tesseract` on macOS).

---

## Running the App

```bash
venv/bin/python3 -m streamlit run app.py --server.port 8501
```

---

## Source Files

| File | Lines | Purpose |
|------|-------|---------|
| `app.py` | 1195 | Main application — UI, detection pipeline, ensemble OCR, vote board |
| `db.py` | 184 | SQLite backend — detections table, watchlist table, query helpers |
| `util.py` | 303 | Kenyan plate correction, `PlateDeduplicator`, `vote_plate` |
| `face_utils.py` | 127 | Haar face detection, LBPH training and recognition |
