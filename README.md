# Argus Intelligence

Kenyan license plate recognition and watchlist system. Detects vehicles, reads plates, and alerts on watchlist matches in real time from images or video.

**Stack:** YOLOv11 · Fast-ALPR · Fast-Plate-OCR · PaddleOCR · llava-phi3 (Ollama) · Streamlit · FastAPI · SQLite

---

## Setup

### 1. System dependencies

**macOS**
```bash
brew install ollama
ollama pull llava-phi3
```

**Linux (Ubuntu/Debian)**
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llava-phi3
```

### 2. Python environment

```bash
python3 -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Model weights

The following files must be placed in the `models/` directory — they are not included in the repo:

| File | Purpose | Source |
|------|---------|--------|
| `models/yolo11s.pt` | Vehicle detection | [Ultralytics](https://github.com/ultralytics/ultralytics) — `yolo11s.pt` |
| `models/license_plate_detector.pt` | Plate bounding box detection | Transfer from original machine |

Fast-Plate-OCR and PaddleOCR models download automatically on first run.

### 4. Run

```bash
streamlit run app.py
```

The REST API starts automatically on port 8502. Docs at `http://localhost:8502/docs` — wait, the API runs on its own port. Check `api.py` for the port (default 8502).

---

## Directory structure

```
app.py              # Main Streamlit application
api.py              # FastAPI REST endpoints
db.py               # SQLite persistence layer
util.py             # Plate validation, OCR correction, ensemble voting
llm_verifier.py     # llava-phi3 LLM plate verification via Ollama
face_utils.py       # Face recognition (LBPH)
report.py           # PDF incident report generation
requirements.txt
.streamlit/         # Streamlit theme config
assets/             # UI screenshots
models/             # Model weights (not in repo — see Setup)
plate_crops/        # Saved plate crop images (runtime, gitignored)
face_data/          # Face training images (runtime, gitignored)
data/               # SQLite database (runtime, gitignored)
```

---

## Plate formats supported

| Type | Format | Example |
|------|--------|---------|
| Civilian | `K[A-Z]{2} \d{3}[A-Z]` | `KDA 123B` |
| Tuk-tuk | `KTW[A-Z] \d{3}[A-Z]` | `KTWA 123B` |
| Motorcycle | `KMC[A-Z] \d{3}[A-Z]` | `KMCA 123B` |

---

## REST API

Once running, the API is available at `http://localhost:8502`:

```
GET  /detections     List detections (filterable)
GET  /stats          Aggregate counts
GET  /watchlist      List watchlist plates
POST /watchlist      Add plate to watchlist
DELETE /watchlist/{plate}  Remove plate
GET  /health         Health check
```

Interactive docs: `http://localhost:8502/docs`
