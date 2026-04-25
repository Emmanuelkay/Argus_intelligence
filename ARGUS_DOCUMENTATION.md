# ARGUS INTELLIGENCE — System Documentation

## Overview

Argus Intelligence is a Kenyan license plate detection and recognition system built for real-world traffic scenes. It runs as a Streamlit web application and supports still image analysis, video file analysis, YouTube video input, and RTSP/IP camera streams. The system combines a deep-learning plate detector, a 3-engine ensemble OCR pipeline, face recognition, and a watchlist alert system — all persisted to a local SQLite database.

---

## Architecture

```
app.py              — Streamlit UI + full detection pipeline
db.py               — SQLite persistence (detections, watchlist, face data)
util.py             — Kenyan plate format validation, correction, deduplication
face_utils.py       — Haar face detection + LBPH recognition
llm_verifier.py     — (retained, not active in pipeline)
build_training_set.py     — Export confirmed plate crops as a labelled training set
collect_annotations.py    — Collect real video frames with YOLO pre-annotations
plates_training/          — Synthetic data generator + fine-tuning notebook
```

---

## Detection Pipeline

### 1. Vehicle Detection

Model: **YOLOv8n** (`yolov8n.pt`, COCO-trained)
Detects cars, motorcycles, buses, and trucks. Bounding boxes are stored per frame.

**Spatial gate:** Every plate detection is checked against vehicle bounding boxes. The plate center must fall within at least one vehicle bbox expanded by ±80px. Plates detected outside all vehicle regions are discarded before OCR runs. If no vehicles are detected in a frame the gate is bypassed.

### 2. Plate Detection

Model: **yolo-v9-t-640-license-plate-end2end** (ONNX via fast-alpr)
Runs ONNX Runtime (CoreML on Apple Silicon). Outputs plate bounding boxes. Each crop is padded by 8px before passing to OCR.

### 3. Plate Preprocessing (per crop)

Before OCR every plate crop is:
- **Upscaled** — bicubic resize to at least 200px wide (minimum 2× scale)
- **Deskewed** — MinAreaRect rotation correction (1°–15° range)
- **CLAHE** — contrast-limited adaptive histogram equalisation (clipLimit 2.0, 4×4 tiles)
- **Bilateral filter** — edge-preserving noise reduction (d=9, σ=75)

### 4. Ensemble OCR (3 engines)

| Engine | Model | Weight | Strength |
|--------|-------|--------|----------|
| **Fast-Plate-OCR** | cct-s-v2-global-model | 1.5 | Transformer, per-char confidence, primary engine |
| **PaddleOCR** | PP-OCRv4 + angle classifier | 1.2 | Strong on rotated/angled text |
| **fast-alpr CCT** | cct-xs-v2-global-model | 0.8 | Fast, plate-specific |

Each engine's raw text is passed through `process_alpr_text()` (util.py) for Kenyan-format positional correction:

| Status | Meaning |
|--------|---------|
| `valid` | Exact Kenyan plate match, no corrections |
| `corrected` | 1–2 character fixes (e.g. `0↔O`, `1↔I`, `S↔5`) |
| `guessed` | 3+ fixes or sliding-window match; confidence penalised |
| `rejected` | No valid Kenyan plate producible |

**Fast-path:** If fast-alpr returns conf ≥ 0.90 with status `valid` and a clean series, the other engines are skipped entirely.

**Voting logic:**
1. Engines ranked by status (`valid > corrected > guessed > rejected`), then confidence.
2. If ≥2 engines produce the same top status, `vote_plate()` runs character-level majority voting.
3. Voted consensus goes through format correction one final time.
4. Plates below 40% combined confidence are hidden from the dashboard.

### 5. Kenyan Plate Formats

| Type | Format | Example |
|------|--------|---------|
| Civilian | `K[A-Z]{2} \d{3}[A-Z]` | `KDA 123B` |
| Tuk-tuk | `KT[A-Z]{2} \d{3}[A-Z]` | `KTWA 123B` |
| Motorcycle | `KMC[A-Z] \d{3}[A-Z]` | `KMCA 123B` |

Positional OCR correction maps:
- Letter positions: digits corrected to visually similar letters (`0→O`, `1→I`, `8→B`, `6→G`)
- Digit positions: letters corrected to visually similar digits (`O→0`, `I→1`, `B→8`, `G→6`)

### 6. Video Frame Sampling

- Samples one frame per second (step = round(FPS)).
- **Adaptive sharpness:** samples first 30 target frames, sets threshold at 35% of median Laplacian variance. Blurry frames skipped.
- **Deduplication** via `PlateDeduplicator`: fuzzy match (SequenceMatcher ≥0.82), 30s cooldown per plate, 120s re-entry window. Same plate after 120s logged as new visit.

---

## Video Sources

| Source | How it works |
|--------|-------------|
| File Upload | Upload MP4/MOV/AVI/MKV up to 1 GB |
| YouTube URL | yt-dlp downloads best MP4 ≤720p to temp file; full seek/frame-skip pipeline runs normally |
| RTSP / IP Camera | cv2.VideoCapture accepts RTSP URLs directly; optional continuous live feed mode |

---

## OCR Ensemble Vote Board

After analysis, each detected plate has a collapsible **🔍 OCR Ensemble** expander showing:
- Plate crop image (rendered via `st.image()`)
- Each engine's result, corrected status, confidence, and latency
- Which engine(s) were selected or contributed to the vote
- Character-level vote breakdown when voting triggered
- Per-character confidence heatmap (green ≥90%, yellow ≥70%, red <70%)

Vote panels are hidden during live video processing and shown after analysis completes.

---

## Persistence — SQLite Backend

Database: `data/argus.db`

| Table | Contents |
|-------|----------|
| `detections` | timestamp, source, vehicle_type, plate_text, ocr_confidence, status, image_file |
| `watchlist` | plate_text (unique), label, added_at |
| `series_registry` | Known Kenyan plate series for validation |

Plate crop images saved to `plate_crops/` as JPEG (upscaled, colour).

---

## Watchlist / Alert System

Add any plate to the watchlist via the sidebar with an optional label (e.g. "Stolen").

On every detection `insert_detections()` cross-references all results against the watchlist. Hits trigger:
- Red alert banner at the top of results
- Red plate badge (`plate-badge-alert`) in the results grid
- Optional webhook fire to a configured endpoint
- Watchlist tab shows full detection history per monitored plate

---

## Face Recognition

Engine: **OpenCV LBPH** (Local Binary Pattern Histogram)
Detection: Haar cascade (`haarcascade_frontalface_default.xml`)

Training: Upload ≥5 frontal images per subject via Face Training tab. System extracts 100×100 grayscale crops, trains LBPH, saves to `models/face_model.yml`.

Recognition runs on every analysed image/frame. Recognised faces: green box + name. Unknown: orange box. Confidence threshold: LBPH distance < 80.

---

## User Interface

Framework: **Streamlit**, custom CSS (Inter font, light theme)

**Tabs:**

| Tab | Purpose |
|-----|---------|
| 📷 Image Analysis | Upload image, annotated result, plate crops, vote expanders |
| 🎬 Video Analysis | File / YouTube / RTSP input, live frame view, live detection table, post-run vote expanders |
| 📋 Detection Log | Filterable/searchable history with CSV export, per-plate crop viewer |
| 🔍 Rejection Audit | Review plates that were detected but rejected, with crop and reason |
| 🚨 Watchlist | Manage monitored plates, detection history per plate |
| 👤 Face Training | Enrol subjects, view enrolled persons, retrain model |
| 📊 Analytics | Detection volume over time, vehicle type breakdown, top plates chart |

**Sidebar:**
- Detection confidence slider (0.10–0.95)
- Live counters (saved / corrected / guessed / rejected)
- Watchlist manager (add plate + label, remove entries)
- Clear All Detections button

**Live video annotations:**
Vehicle bounding boxes in blue with class label. Plate bounding boxes in green (saved) or orange (rejected/unverified). Plate text floats above each box.

---

## Training Data Tools

### `collect_annotations.py`
Extracts frames from any dashcam/traffic video, runs the fast-alpr ONNX detector as a pre-annotator, and saves YOLO-format labels to `data/real_annotations/`. Output is ready to upload to Roboflow or Label Studio for human verification before fine-tuning.

```bash
python3 collect_annotations.py video.mp4 [--conf 0.25] [--every 10] [--limit 400]
```

### `build_training_set.py`
Pulls high-confidence SAVED/CORRECTED plate crops from the database, cross-references with a local Ollama LLM (llava-phi3) for blind verification, and exports agreed crops + labels to a training CSV.

```bash
python3 build_training_set.py [--conf 0.85] [--out training_data] [--limit 500]
```

### `plates_training/generate_synthetic_plates.py`
Generates 1500 synthetic Kenyan plate scenes (640×640) for YOLO fine-tuning. Features:
- Gradient backgrounds with distraction rectangles
- Yellow plate color jitter per image
- Shadow, perspective warp, blur, mud augmentation
- All three plate classes (car / tuk-tuk / motorcycle), 60/20/20 split
- Auto train/val split + `data.yaml`

```bash
python3 plates_training/generate_synthetic_plates.py
```

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
maxUploadSize = 1024
```

Max upload: **1 GB**.

---

## Running the App

```bash
# Activate venv
source venv/bin/activate

# Run
streamlit run app.py --server.port 8501
```

---

## Dependencies

See `requirements.txt` for pinned versions. Core packages:

| Package | Purpose |
|---------|---------|
| `streamlit` | UI framework |
| `ultralytics` | YOLOv8 vehicle detection |
| `fast-alpr` | ONNX plate detector + CCT OCR engine |
| `fast-plate-ocr` | Primary Transformer OCR engine |
| `paddleocr` / `paddlepaddle` | Secondary OCR engine |
| `onnxruntime` | ONNX inference runtime |
| `opencv-contrib-python` | Image processing + face recognition |
| `torch` / `torchvision` | PyTorch backend |
| `yt-dlp` | YouTube video download |
| `plotly` | Analytics charts |
| `fpdf2` | PDF incident report export |
| `fastapi` / `uvicorn` | REST API (`api.py`) |

---

## Source Files

| File | Purpose |
|------|---------|
| `app.py` | Main application — UI, pipeline, ensemble OCR, vote board |
| `db.py` | SQLite backend — detections, watchlist, query helpers |
| `util.py` | Kenyan plate correction, `PlateDeduplicator`, `vote_plate` |
| `face_utils.py` | Haar detection, LBPH training and recognition |
| `api.py` | FastAPI REST endpoints for headless integration |
| `report.py` | PDF incident report generator |
| `collect_annotations.py` | Real frame collection with pre-annotations |
| `build_training_set.py` | LLM-verified training set exporter |
| `plates_training/generate_synthetic_plates.py` | Synthetic scene generator |
