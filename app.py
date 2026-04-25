#!/usr/bin/env python3
"""ARGUS INTELLIGENCE — Streamlit UI (light theme)"""

from __future__ import annotations
import os, re, tempfile, uuid, subprocess, time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import requests
import plotly.express as px
import streamlit as st
from PIL import Image

from util import process_alpr_text, PlateDeduplicator, vote_plate, validate_kenyan_series, get_plate_category
from face_utils import (
    save_training_images, train_lbph_model,
    load_recognizer, recognize_faces, get_registered_persons, FACE_DATA_DIR,
)
from db import (
    init_db, insert_detections,
    query_detections,
    get_watchlist, add_to_watchlist, remove_from_watchlist,
    check_watchlist, get_stats, clear_all_detections, get_distinct_values,
    update_detection_dwell, override_plate, mark_ground_truth, get_shift_report, record_exit,
)

# ── Ollama auto-start ─────────────────────────────────────────────────────────
def _ensure_ollama() -> None:
    import requests as _req
    try:
        _req.get("http://localhost:11434/api/tags", timeout=2)
        return  # already running
    except Exception:
        pass
    subprocess.Popen(
        ["ollama", "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    for _ in range(15):  # wait up to 15s
        time.sleep(1)
        try:
            _req.get("http://localhost:11434/api/tags", timeout=1)
            return
        except Exception:
            continue

_ensure_ollama()

# Start background REST API
try:
    from api import start_api_background as _start_api
    _start_api()
except Exception:
    pass

def _fire_webhook(plate: str, source: str) -> None:
    """POST watchlist-hit JSON to configured webhook URL (fire-and-forget)."""
    url = st.session_state.get("webhook_url", "").strip()
    if not url:
        return
    try:
        requests.post(
            url,
            json={"plate": plate, "timestamp": datetime.now().isoformat(), "source": source},
            timeout=3,
        )
    except Exception:
        pass


# ── Constants ─────────────────────────────────────────────────────────────────
PLATES_DIR    = Path("./plate_crops")
MODEL_COCO    = "./models/yolo11s.pt"
VEHICLE_CLS   = {2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}
_CROP_PAD     = 8
_SHARP_THRESH  = 80.0
_STATUS_RANK   = {"valid": 4, "corrected": 3, "rejected": 1}
_MIN_DISP_CONF = 0.40   # plates below 40% confidence hidden from dashboard


_PLATE_VEHICLE_OVERRIDE = {"KMC": "Motorcycle"}  # tuk-tuk resolved via get_plate_category (KT prefix)
# Civilian plates (K[A-Z][A-Z]) cannot belong to these YOLO classes
_CIVILIAN_FORBIDDEN_LABELS = {"Motorcycle", "Tuk Tuk"}

def _resolve_vehicle_type(plate_text: str, yolo_vehicle: str) -> str:
    """Derive vehicle type from plate prefix. KT** = Tuk Tuk, KMC* = Motorcycle."""
    clean = re.sub(r'\s+', '', (plate_text or "").upper())
    # Tuk-tuk: KT + 2 any letters (8-char plate)
    if len(clean) >= 4 and clean[:2] == "KT" and clean[2].isalpha() and clean[3].isalpha():
        return "Tuk Tuk"
    if clean[:3] in _PLATE_VEHICLE_OVERRIDE:
        return _PLATE_VEHICLE_OVERRIDE[clean[:3]]
    # Civilian plate on a YOLO-detected motorcycle → must be Car
    if clean.startswith("K") and yolo_vehicle in _CIVILIAN_FORBIDDEN_LABELS:
        return "Car"
    return yolo_vehicle

for _d in [PLATES_DIR, Path(FACE_DATA_DIR), Path("./models")]:
    _d.mkdir(parents=True, exist_ok=True)

init_db()

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Argus Intelligence",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    background: #EEF2F7 !important;
    font-family: 'Inter', sans-serif !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #1E3A5F 0%, #1A2F4E 100%) !important;
    border-right: none !important;
    box-shadow: 4px 0 20px rgba(0,0,0,0.15);
}
[data-testid="stSidebar"] * { color: #CBD5E1 !important; }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #F1F5F9 !important; }
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stMarkdown p { color: #94A3B8 !important; }
[data-testid="stSidebar"] [data-testid="stMetricValue"] { color: #38BDF8 !important; }
[data-testid="stSidebar"] [data-testid="stMetricLabel"] { color: #94A3B8 !important; }
[data-testid="stSidebar"] [data-testid="metric-container"] {
    background: rgba(255,255,255,0.07) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
}
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.1) !important; }

/* ── Header ── */
[data-testid="stHeader"] {
    background: #FFFFFF !important;
    border-bottom: 1px solid #E2E8F0 !important;
    box-shadow: 0 1px 8px rgba(0,0,0,0.06);
}

/* ── Main content ── */
.main .block-container {
    padding-top: 1.5rem !important;
    max-width: 1400px !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #FFFFFF;
    border-radius: 12px;
    padding: 5px;
    border: 1px solid #E2E8F0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px !important;
    font-weight: 600;
    font-size: 0.85rem;
    color: #64748B !important;
    padding: 8px 18px !important;
    transition: all 0.2s;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #1A56DB, #2563EB) !important;
    color: #FFFFFF !important;
    box-shadow: 0 2px 8px rgba(26,86,219,0.35) !important;
}

/* ── Cards / panels ── */
[data-testid="stExpander"] {
    background: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 12px !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05) !important;
}

/* ── Metrics ── */
[data-testid="metric-container"] {
    background: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 12px !important;
    padding: 16px !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05) !important;
}
[data-testid="stMetricValue"] { color: #1E3A5F !important; font-weight: 700 !important; }
[data-testid="stMetricLabel"] { color: #64748B !important; }

/* ── Buttons ── */
.stButton > button {
    border-radius: 9px !important;
    font-weight: 600 !important;
    font-size: 0.875rem !important;
    padding: 0.5rem 1.2rem !important;
    transition: all 0.2s !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #1A56DB, #2563EB) !important;
    border: none !important;
    color: #FFFFFF !important;
    box-shadow: 0 4px 14px rgba(26,86,219,0.35) !important;
}
.stButton > button[kind="primary"]:hover {
    box-shadow: 0 6px 20px rgba(26,86,219,0.45) !important;
    transform: translateY(-1px) !important;
}

/* ── File uploader ── */
[data-testid="stFileUploadDropzone"] {
    background: #FFFFFF !important;
    border: 2px dashed #93C5FD !important;
    border-radius: 12px !important;
    transition: border-color 0.2s !important;
}
[data-testid="stFileUploadDropzone"]:hover {
    border-color: #2563EB !important;
}

/* ── Progress bar ── */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #1A56DB, #38BDF8) !important;
    border-radius: 99px !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border: 1px solid #E2E8F0 !important;
    border-radius: 12px !important;
    overflow: hidden !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05) !important;
}

/* ── Success / warning / error ── */
[data-testid="stAlert"] { border-radius: 10px !important; }

/* ── Plate badge ── */
.plate-badge {
    display: inline-block;
    background: linear-gradient(135deg, #EFF6FF, #DBEAFE);
    color: #1E40AF;
    border: 1.5px solid #93C5FD;
    border-radius: 8px;
    padding: 5px 14px;
    font-family: 'Courier New', monospace;
    font-size: 1.1rem;
    font-weight: 800;
    letter-spacing: 0.12em;
    box-shadow: 0 2px 6px rgba(37,99,235,0.15);
}
.plate-badge-alert {
    display: inline-block;
    background: linear-gradient(135deg, #FEF2F2, #FEE2E2);
    color: #991B1B;
    border: 2px solid #F87171;
    border-radius: 8px;
    padding: 5px 14px;
    font-family: 'Courier New', monospace;
    font-size: 1.1rem;
    font-weight: 800;
    letter-spacing: 0.12em;
    box-shadow: 0 2px 6px rgba(220,38,38,0.25);
}

/* ── Status colours ── */
.st-saved     { color: #15803D; font-weight: 700; }
.st-corrected { color: #1D4ED8; font-weight: 700; }
.st-unverified { color: #B45309; font-weight: 700; }
.st-rejected  { color: #DC2626; font-weight: 700; }

/* ── Subheaders ── */
h2, h3 {
    color: #1E3A5F !important;
    font-weight: 700 !important;
}
</style>
""", unsafe_allow_html=True)

# ── Model loading ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading vehicle detector…")
def get_coco():
    from ultralytics import YOLO
    return YOLO(MODEL_COCO)

_DETECTOR_LABEL_TO_VEHICLE = {
    "car_plate":       "Car",
    "tuktuk_plate":    "Tuk Tuk",
    "motorbike_plate": "Motorcycle",
}

@st.cache_resource(show_spinner="Loading plate detector…")
def get_alpr():
    from fast_alpr import ALPR
    return ALPR(
        detector_model="yolo-v9-t-640-license-plate-end2end",
        ocr_model="cct-xs-v2-global-model",
    )

@st.cache_resource(show_spinner="Loading PaddleOCR…")
def get_paddle_ocr():
    from paddleocr import PaddleOCR
    return PaddleOCR(use_angle_cls=True, lang="en", show_log=False)

@st.cache_resource(show_spinner="Loading Fast-Plate-OCR (cct-s-v2)…")
def get_fast_plate_ocr():
    from fast_plate_ocr import LicensePlateRecognizer
    return LicensePlateRecognizer("cct-s-v2-global-model")

@st.cache_resource(show_spinner="Loading face recogniser…")
def get_face_rec():
    return load_recognizer()

@st.cache_resource(show_spinner="Loading deblur model…")
def get_esrgan():
    try:
        from super_image import EdsrModel
        model = EdsrModel.from_pretrained("eugenesiow/edsr-base", scale=2)
        model.eval()
        return model
    except Exception:
        return None

def deblur_crop(crop_bgr: np.ndarray) -> np.ndarray:
    model = get_esrgan()
    if model is None:
        return crop_bgr
    try:
        import torch
        from super_image import ImageLoader
        from PIL import Image as _PILImage
        pil = _PILImage.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
        inputs = ImageLoader.load_image(pil)
        with torch.no_grad():
            preds = model(inputs)
        out_arr = preds.squeeze().permute(1, 2, 0).clamp(0, 1).numpy()
        out_bgr = cv2.cvtColor((out_arr * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        return out_bgr
    except Exception:
        return crop_bgr

# ── Image helpers ─────────────────────────────────────────────────────────────
def _deskew(gray: np.ndarray) -> np.ndarray:
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) < 10:
        return gray
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:  angle += 90
    elif angle > 45: angle -= 90
    if abs(angle) < 1 or abs(angle) > 15:
        return gray
    h, w = gray.shape
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)

def preprocess_plate(crop_bgr: np.ndarray) -> np.ndarray:
    h, w = crop_bgr.shape[:2]
    if w == 0 or h == 0:
        return crop_bgr
    gray_check = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    if cv2.Laplacian(gray_check, cv2.CV_64F).var() < _SHARP_THRESH:
        crop_bgr = deblur_crop(crop_bgr)
        h, w = crop_bgr.shape[:2]
    scale = max(200 / max(w, 1), 60 / max(h, 1), 2.0)
    up = cv2.resize(crop_bgr, (int(w * scale), int(h * scale)),
                    interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
    gray = _deskew(gray)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(gray)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

def upscale_crop_color(crop_bgr: np.ndarray) -> np.ndarray:
    """Upscale to min 200px wide keeping color — for disk save and LLM input."""
    h, w = crop_bgr.shape[:2]
    if w == 0 or h == 0:
        return crop_bgr
    scale = max(200 / max(w, 1), 60 / max(h, 1), 2.0)
    return cv2.resize(crop_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

def is_sharp(frame: np.ndarray, threshold: float = _SHARP_THRESH) -> bool:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() >= threshold

# ── OCR engines ───────────────────────────────────────────────────────────────

def _ocr_paddle(crop_bgr: np.ndarray) -> tuple[str, float]:
    if crop_bgr is None or crop_bgr.size == 0:
        return "", 0.0
    processed = preprocess_plate(crop_bgr)
    try:
        results = get_paddle_ocr().ocr(processed, cls=True)
    except Exception:
        return "", 0.0
    if not results or not results[0]:
        return "", 0.0
    best_text, best_conf = "", 0.0
    for line in (results[0] or []):
        if not line or len(line) < 2 or not line[1]:
            continue
        raw_t = str(line[1][0]).upper().strip()
        t = re.sub(r"[^A-Z0-9]", "", raw_t)
        c = float(line[1][1])
        if 5 <= len(t) <= 9 and c > best_conf:
            if len(t) == 8:
                best_text = t[:4] + " " + t[4:]
            elif len(t) == 7:
                best_text = t[:3] + " " + t[3:]
            else:
                best_text = raw_t
            best_conf = c
    return best_text, best_conf

def _ocr_fast_plate(crop_bgr: np.ndarray) -> tuple[str, float, list | None]:
    """Fast-Plate-OCR (cct-s-v2-global-model) — primary Transformer engine.
    Returns (text, conf, char_probs_list_or_None).
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return "", 0.0, None
    try:
        processed = preprocess_plate(crop_bgr)
        rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        preds = get_fast_plate_ocr().run(rgb, return_confidence=True)
        if not preds:
            return "", 0.0, None
        pred = preds[0]
        raw_t = re.sub(r"[^A-Z0-9]", "", (pred.plate or "").upper())
        if not 5 <= len(raw_t) <= 9:
            return "", 0.0, None
        char_probs = [float(v) for v in pred.char_probs] if pred.char_probs is not None else None
        conf = float(np.mean(pred.char_probs)) if pred.char_probs is not None else 0.5
        if len(raw_t) == 8:
            text = raw_t[:4] + " " + raw_t[4:]
        elif len(raw_t) == 7:
            text = raw_t[:3] + " " + raw_t[3:]
        else:
            text = raw_t
        # Trim padding tokens — model outputs fixed-length, clip to actual plate length
        if char_probs:
            char_probs = char_probs[:len(raw_t)]
        return text, conf, char_probs
    except Exception:
        return "", 0.0, None


# ── Vote board ────────────────────────────────────────────────────────────────
_STATUS_COLOR = {
    "valid":      ("#166534", "#DCFCE7"),
    "corrected":  ("#1E40AF", "#DBEAFE"),
    "unverified": ("#92400E", "#FEF3C7"),
    "rejected":   ("#991B1B", "#FEE2E2"),
}
_ENGINE_ICON = {
    "fast-alpr":      "⚡",
    "PaddleOCR":      "🏓",
    "Fast-Plate-OCR": "🔷",
}
# Engine weights for weighted voting
_ENGINE_WEIGHT = {
    "Fast-Plate-OCR": 1.5,
    "PaddleOCR":      1.2,
    "fast-alpr":      0.8,
}

def _plate_heatmap_html(plate: str, char_probs: list[float]) -> str:
    """Render plate text as colored inline spans based on per-char confidence."""
    if not char_probs:
        return f'<span style="font-family:monospace;font-weight:800;">{plate}</span>'
    chars = list(re.sub(r"\s", "", plate))
    spans = []
    for idx, ch in enumerate(chars):
        prob = char_probs[idx] if idx < len(char_probs) else 0.5
        if prob > 0.90:
            fg, bg = "#15803D", "#DCFCE7"
        elif prob >= 0.60:
            fg, bg = "#92400E", "#FEF3C7"
        else:
            fg, bg = "#991B1B", "#FEE2E2"
        spans.append(
            f'<span style="background:{bg};color:{fg};border-radius:3px;'
            f'padding:1px 3px;font-family:monospace;font-weight:800;">{ch}</span>'
        )
    return "".join(spans)


def _render_vote_panel(vote_info: dict, skip_crop: bool = False) -> str:
    """Return HTML string for the ensemble vote board."""
    # ── Fast-path: simplified badge, no engine table ──────────────────────────
    if vote_info.get("fast_path"):
        final    = vote_info.get("final_plate", "—")
        conf_val = vote_info.get("combined_conf", 0.0)
        return f"""
    <div style="background:#F0FDF4;border:1.5px solid #86EFAC;border-radius:10px;
                padding:10px 16px;margin:6px 0;display:flex;align-items:center;gap:12px;">
      <span style="font-family:monospace;font-size:1.2rem;font-weight:900;color:#166534;">
        {final}</span>
      <span style="font-size:0.85rem;color:#166534;">{conf_val:.0%}</span>
      <span style="background:#1D4ED8;color:#FFFFFF;font-size:0.72rem;font-weight:800;
                   border-radius:20px;padding:3px 10px;margin-left:auto;">
        ⚡ HIGH CONF — FAST PATH</span>
    </div>"""

    engines  = vote_info.get("engines", [])
    voted    = vote_info.get("voted", False)
    final    = vote_info.get("final_plate", "—")
    inputs   = vote_info.get("voted_from", [])
    crop_b64 = vote_info.get("crop_b64", "")

    rows_html = ""
    for eng in engines:
        name    = eng["name"]
        plate   = eng.get("plate") or "—"
        status  = eng.get("status", "rejected")
        conf    = eng.get("conf", 0.0)
        chosen  = eng.get("chosen", False)
        ms      = eng.get("latency_ms")
        ms_str  = f"{ms} ms" if ms is not None else "—"
        fg, bg  = _STATUS_COLOR.get(status, ("#374151", "#F3F4F6"))
        icon    = _ENGINE_ICON.get(name, "🔲")
        weight  = _ENGINE_WEIGHT.get(name, 1.0)
        chosen_style = "border-left:3px solid #2563EB;" if chosen else "border-left:3px solid transparent;"
        rows_html += f"""
        <tr style="background:{'#EFF6FF' if chosen else '#FFFFFF'};{chosen_style}">
          <td style="padding:6px 10px;font-weight:600;color:#374151;white-space:nowrap;">
            {icon} {name}
            <span style="font-size:0.68rem;color:#94A3B8;margin-left:4px;">w={weight}</span></td>
          <td style="padding:6px 10px;font-family:monospace;font-weight:800;
                     font-size:1rem;color:#1E3A5F;">{plate}</td>
          <td style="padding:6px 10px;">
            <span style="background:{bg};color:{fg};border-radius:20px;
                         padding:2px 10px;font-size:0.72rem;font-weight:700;">
              {status.upper()}</span></td>
          <td style="padding:6px 10px;color:#64748B;font-size:0.85rem;">
            {float(conf or 0):.0%}</td>
          <td style="padding:6px 10px;color:#94A3B8;font-size:0.8rem;font-family:monospace;">
            {ms_str}</td>
          <td style="padding:6px 10px;font-size:0.8rem;color:#2563EB;font-weight:700;">
            {'✦ SELECTED' if chosen and not voted else ('✦ IN VOTE' if chosen else '')}</td>
        </tr>"""

    vote_badge = ""
    if voted:
        chars_html = " · ".join(
            f"<b>{c}</b>" for c in re.sub(r"\s", "", final)
        )
        inputs_str = " vs ".join(f"<code>{p}</code>" for p in inputs)
        vote_badge = f"""
        <div style="margin-top:10px;padding:8px 14px;
                    background:linear-gradient(135deg,#EFF6FF,#DBEAFE);
                    border:1.5px solid #93C5FD;border-radius:8px;font-size:0.8rem;">
          <span style="font-weight:700;color:#1E40AF;">🗳 CHARACTER VOTE TRIGGERED</span>
          <span style="margin-left:8px;color:#64748B;">{inputs_str}</span>
          <div style="margin-top:4px;color:#1E3A5F;font-family:monospace;font-size:0.95rem;">
            Consensus → {chars_html}
          </div>
        </div>"""

    crop_html = ""  # crop shown via st.image() in the expander, not inline HTML

    # ── Char-probs heatmap ────────────────────────────────────────────────────
    char_probs  = vote_info.get("char_probs")
    heatmap_html = ""
    if char_probs:
        plate_chars = list(re.sub(r"\s", "", final))
        boxes = []
        for idx, prob in enumerate(char_probs[:len(plate_chars)]):
            ch = plate_chars[idx]
            if prob > 0.9:
                bg, fg = "#DCFCE7", "#166534"
            elif prob >= 0.7:
                bg, fg = "#FEF9C3", "#854D0E"
            else:
                bg, fg = "#FEE2E2", "#991B1B"
            boxes.append(
                f'<div style="background:{bg};color:{fg};border-radius:5px;'
                f'padding:4px 8px;font-family:monospace;font-weight:800;'
                f'font-size:0.9rem;text-align:center;min-width:28px;'
                f'border:1px solid rgba(0,0,0,0.08);">'
                f'{ch}<br><span style="font-size:0.6rem;font-weight:500;">'
                f'{float(prob or 0):.0%}</span></div>'
            )
        heatmap_html = (
            '<div style="padding:8px 14px;display:flex;gap:4px;align-items:center;">'
            '<span style="font-size:0.65rem;color:#94A3B8;text-transform:uppercase;'
            'letter-spacing:0.05em;margin-right:6px;">Char Conf</span>'
            + "".join(boxes)
            + "</div>"
        )

    return f"""
    <div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:10px;
                overflow:hidden;margin:6px 0;">
      <div style="background:linear-gradient(135deg,#1E3A5F,#2563EB);
                  padding:7px 14px;display:flex;align-items:center;gap:8px;">
        <span style="font-size:0.8rem;font-weight:800;color:#FFFFFF;letter-spacing:0.05em;">
          🗳 ENSEMBLE VOTE</span>
        <span style="font-family:monospace;font-size:1rem;font-weight:800;
                     color:#BFDBFE;">{final}</span>
        {'<span style="margin-left:auto;background:#FCD34D;color:#92400E;font-size:0.68rem;font-weight:800;border-radius:20px;padding:2px 8px;">VOTED</span>' if voted else ''}
      </div>
      {crop_html}
      <table style="width:100%;border-collapse:collapse;{'margin-top:6px;' if not crop_html else ''}">
        <thead>
          <tr style="background:#F1F5F9;font-size:0.72rem;color:#94A3B8;text-transform:uppercase;letter-spacing:0.05em;">
            <th style="padding:5px 10px;text-align:left;">Engine</th>
            <th style="padding:5px 10px;text-align:left;">Result</th>
            <th style="padding:5px 10px;text-align:left;">Status</th>
            <th style="padding:5px 10px;text-align:left;">Conf</th>
            <th style="padding:5px 10px;text-align:left;">Time</th>
            <th style="padding:5px 10px;text-align:left;"></th>
          </tr>
        </thead>
        <tbody>{rows_html}</tbody>
      </table>
      {vote_badge}
      {heatmap_html}
    </div>"""


def _show_vote_expander(vote_info: dict, label: str) -> None:
    """Render a collapsible ensemble vote expander using native st.image for the crop."""
    import base64 as _b64m
    with st.expander(f"🔍 OCR Ensemble — {label}"):
        crop_b64 = vote_info.get("crop_b64", "")
        if crop_b64:
            try:
                crop_bytes = _b64m.b64decode(crop_b64)
                crop_arr   = np.frombuffer(crop_bytes, np.uint8)
                crop_bgr   = cv2.imdecode(crop_arr, cv2.IMREAD_COLOR)
                if crop_bgr is not None:
                    st.image(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB),
                             caption=label, width=220)
            except Exception:
                pass
        st.markdown(_render_vote_panel(vote_info), unsafe_allow_html=True)


def _timed_ocr(fn, *args) -> tuple[str, float, int]:
    """Run an OCR function, return (text, conf, latency_ms)."""
    t0 = time.perf_counter()
    text, conf = fn(*args)
    return text, conf, round((time.perf_counter() - t0) * 1000)


def _ensemble_ocr(crop_bgr, alpr_text: str, alpr_conf: float):
    """
    Weighted ensemble: Fast-Plate-OCR (1.5) + PaddleOCR (1.2) + fast-alpr (0.8).
    LLM referee fires only when top-2 engines disagree OR weighted conf < 60%.
    Returns (plate, conf, n_subs, status, raw_text, vote_info, needs_llm).

    Fast-path: if alpr_conf >= 0.85 and alpr result is 'valid', skip other engines.
    """
    # ── Fast-path: skip ensemble on high-confidence ALPR read ────────────────
    _fp_plate, _fp_conf, _fp_nsubs, _fp_status = process_alpr_text(alpr_text, alpr_conf)
    _fp_series = validate_kenyan_series(_fp_plate) if _fp_plate else {"valid": False, "uncertainty": "flagged", "flags": []}

    def _encode_crop():
        _b64 = ""
        if crop_bgr is not None and crop_bgr.size > 0:
            try:
                _ok, _buf = cv2.imencode(".jpg", crop_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
                if _ok:
                    import base64 as _b64m
                    _b64 = _b64m.b64encode(_buf.tobytes()).decode()
            except Exception:
                pass
        return _b64

    # Tier 1: conf >= 0.90, valid series, status == "valid" → full fast path, no LLM
    if (alpr_conf >= 0.90 and _fp_status == "valid" and _fp_plate
            and _fp_series["valid"] and _fp_series["uncertainty"] == "none"):
        _crop_b64 = _encode_crop()
        vote_info = {
            "engines": [{"name": "fast-alpr", "raw": alpr_text, "plate": _fp_plate,
                         "conf": _fp_conf, "status": _fp_status, "n_subs": _fp_nsubs,
                         "chosen": True, "latency_ms": 0}],
            "voted": False,
            "final_plate": _fp_plate,
            "voted_from": [],
            "crop_b64":    _crop_b64,
            "needs_llm":   False,
            "combined_conf": round(alpr_conf, 3),
            "fast_path":   True,
            "char_probs":  None,
            "series_flags": _fp_series["flags"],
        }
        return _fp_plate, _fp_conf, _fp_nsubs, _fp_status, alpr_text, vote_info

    # Tier 2: conf >= 0.75, valid/corrected, series uncertainty not high → Fast-Plate-OCR only
    if (alpr_conf >= 0.75 and _fp_status in ("valid", "corrected") and _fp_plate
            and _fp_series.get("uncertainty") != "high"):
        _crop_b64 = _encode_crop()
        _t0 = time.perf_counter()
        fp_text, fp_conf, fp_char_probs = _ocr_fast_plate(crop_bgr)
        fp_ms = round((time.perf_counter() - _t0) * 1000)
        # Use fast-plate confirmation if it agrees, else stick with alpr result
        _engines = [{"name": "fast-alpr", "raw": alpr_text, "plate": _fp_plate,
                     "conf": _fp_conf, "status": _fp_status, "n_subs": _fp_nsubs,
                     "chosen": True, "latency_ms": 0, "char_probs": None}]
        _final_plate, _final_conf, _final_n, _final_s = _fp_plate, _fp_conf, _fp_nsubs, _fp_status
        _winner_probs = None
        if fp_text:
            fp_p, fp_c2, fp_n, fp_s = process_alpr_text(fp_text, fp_conf)
            _engines.append({"name": "Fast-Plate-OCR", "raw": fp_text, "plate": fp_p,
                              "conf": fp_c2, "status": fp_s, "n_subs": fp_n,
                              "chosen": True, "latency_ms": fp_ms, "char_probs": fp_char_probs})
            if fp_p and _STATUS_RANK.get(fp_s, 1) >= _STATUS_RANK.get(_fp_status, 1):
                _final_plate, _final_conf, _final_n, _final_s = fp_p, fp_c2, fp_n, fp_s
                _winner_probs = fp_char_probs
        vote_info = {
            "engines":      _engines,
            "voted":        False,
            "final_plate":  _final_plate or "—",
            "voted_from":   [],
            "crop_b64":     _crop_b64,
            "needs_llm":    False,
            "combined_conf": round(alpr_conf, 3),
            "fast_path":    False,
            "char_probs":   _winner_probs,
            "series_flags": _fp_series["flags"],
        }
        return _final_plate, _final_conf, _final_n, _final_s, alpr_text, vote_info

    engine_results = []

    def _run(name, raw_text, raw_conf, latency_ms=0, char_probs=None):
        p, c, n, s = process_alpr_text(raw_text, raw_conf)
        engine_results.append({
            "name": name, "raw": raw_text, "plate": p,
            "conf": c, "status": s, "n_subs": n,
            "chosen": False, "latency_ms": latency_ms,
            "char_probs": char_probs,
        })
        return p, c, n, s

    # fast-alpr (already run upstream, no extra timing needed)
    _run("fast-alpr", alpr_text, alpr_conf, latency_ms=0)

    # Fast-Plate-OCR — primary Transformer engine (returns 3-tuple)
    _t0 = time.perf_counter()
    fp_text, fp_conf, fp_char_probs = _ocr_fast_plate(crop_bgr)
    fp_ms = round((time.perf_counter() - _t0) * 1000)
    if fp_text:
        _run("Fast-Plate-OCR", fp_text, fp_conf, latency_ms=fp_ms, char_probs=fp_char_probs)

    # PaddleOCR — secondary engine (handles skew/complex lighting)
    pp_text, pp_conf, pp_ms = _timed_ocr(_ocr_paddle, crop_bgr)
    if pp_text:
        _run("PaddleOCR", pp_text, pp_conf, latency_ms=pp_ms)

    # Encode crop for vote panel display
    _crop_b64 = ""
    if crop_bgr is not None and crop_bgr.size > 0:
        try:
            _ok, _buf = cv2.imencode(".jpg", crop_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
            if _ok:
                import base64 as _b64
                _crop_b64 = _b64.b64encode(_buf.tobytes()).decode()
        except Exception:
            pass

    valid_candidates = [e for e in engine_results if e["plate"]]
    if not valid_candidates:
        vote_info = {"engines": engine_results, "voted": False,
                     "final_plate": "—", "voted_from": [], "crop_b64": _crop_b64,
                     "needs_llm": True}
        return None, 0.0, 0, "rejected", alpr_text, vote_info

    # ── Weighted scoring ──────────────────────────────────────────────────────
    for e in valid_candidates:
        w = _ENGINE_WEIGHT.get(e["name"], 1.0)
        e["weighted_score"] = e["conf"] * w * (_STATUS_RANK.get(e["status"], 1) / 4.0)

    valid_candidates.sort(key=lambda x: x["weighted_score"], reverse=True)

    # ── Smart LLM trigger ────────────────────────────────────────────────────
    top1 = valid_candidates[0]
    top2 = valid_candidates[1] if len(valid_candidates) > 1 else None
    plates_disagree = top2 and (
        re.sub(r"\s", "", top1["plate"] or "") != re.sub(r"\s", "", top2["plate"] or "")
    )
    combined_conf = sum(e["weighted_score"] for e in valid_candidates[:2]) / max(
        sum(_ENGINE_WEIGHT.get(e["name"], 1.0) for e in valid_candidates[:2]), 1
    )
    needs_llm = bool(plates_disagree or combined_conf < 0.60)

    # ── Final selection via weighted vote ─────────────────────────────────────
    voted, voted_from = False, []
    final_plate = top1["plate"]
    final_conf  = top1["conf"]
    final_n     = top1["n_subs"]
    final_s     = top1["status"]

    if top2:
        voted_from = [e["plate"] for e in valid_candidates[:2]]
        voted_raw  = vote_plate(voted_from)
        if voted_raw:
            pv, cv, nv, sv = process_alpr_text(voted_raw, top1["conf"])
            if pv:
                voted = True
                final_plate, final_conf, final_n, final_s = pv, top1["conf"], nv, sv
                for e in valid_candidates[:2]:
                    e["chosen"] = True
    else:
        top1["chosen"] = True

    # Include char_probs from the winning/selected engine if available
    _winner_char_probs = None
    for _e in engine_results:
        if _e.get("chosen") and _e.get("char_probs"):
            _winner_char_probs = _e["char_probs"]
            break

    _final_series = validate_kenyan_series(final_plate) if final_plate else {"flags": []}
    vote_info = {
        "engines":     engine_results,
        "voted":       voted,
        "final_plate": final_plate or "—",
        "voted_from":  voted_from,
        "crop_b64":    _crop_b64,
        "needs_llm":   needs_llm,
        "combined_conf": round(combined_conf, 3),
        "char_probs":  _winner_char_probs,
        "series_flags": _final_series.get("flags", []),
    }
    return final_plate, final_conf, final_n, final_s, alpr_text, vote_info

def _associate_vehicle(px1, py1, px2, py2, vehicles):
    if not vehicles:
        return "Unknown"
    pcx, pcy = (px1 + px2) / 2, (py1 + py2) / 2
    best, best_d = "Unknown", float("inf")
    for vx1, vy1, vx2, vy2, vcls in vehicles:
        d = ((pcx - (vx1 + vx2) / 2) ** 2 + (pcy - (vy1 + vy2) / 2) ** 2) ** 0.5
        if d < best_d:
            best_d, best = d, VEHICLE_CLS.get(vcls, "Vehicle")
    return best


def _plate_near_vehicle(px1, py1, px2, py2, vehicles, margin: int = 80) -> bool:
    """Plate center must fall inside at least one vehicle bbox (expanded by margin px).
    If no vehicles detected, pass through — COCO may have missed them at current conf."""
    if not vehicles:
        return True
    pcx, pcy = (px1 + px2) / 2, (py1 + py2) / 2
    for vx1, vy1, vx2, vy2, _ in vehicles:
        if vx1 - margin <= pcx <= vx2 + margin and vy1 - margin <= pcy <= vy2 + margin:
            return True
    return False


# ── Detection functions ───────────────────────────────────────────────────────
def run_detection(img_rgb: np.ndarray, conf: float):
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    h_bgr, w_bgr = bgr.shape[:2]
    vehicles = []
    for det in get_coco()(bgr, conf=conf)[0].boxes.data.tolist():
        x1, y1, x2, y2, _, cls = det
        if int(cls) in VEHICLE_CLS:
            vehicles.append((int(x1), int(y1), int(x2), int(y2), int(cls)))

    plates, all_dets = [], []
    for result in get_alpr().predict(bgr):
        if result.ocr is None:
            continue
        bb = result.detection.bounding_box
        x1, y1, x2, y2 = int(bb.x1), int(bb.y1), int(bb.x2), int(bb.y2)
        if not _plate_near_vehicle(x1, y1, x2, y2, vehicles):
            continue  # plate outside all vehicle bboxes — discard
        cx1 = max(0, x1 - _CROP_PAD);  cy1 = max(0, y1 - _CROP_PAD)
        cx2 = min(w_bgr, x2 + _CROP_PAD); cy2 = min(h_bgr, y2 + _CROP_PAD)
        crop = bgr[cy1:cy2, cx1:cx2]
        if not crop.size:
            continue
        alpr_text = (result.ocr.text or "").strip()
        _rc = result.ocr.confidence
        alpr_conf = float(sum(_rc) / len(_rc) if isinstance(_rc, list) else (_rc or 0.0))
        vehicle = _associate_vehicle(x1, y1, x2, y2, vehicles)
        crop_ocr = upscale_crop_color(crop)
        corrected, final_conf, n_subs, plate_status, raw_text, vote_info = _ensemble_ocr(
            crop_ocr, alpr_text, alpr_conf
        )
        if not corrected or final_conf < _MIN_DISP_CONF:
            all_dets.append({"text": alpr_text, "score": alpr_conf,
                             "vehicle_type": _resolve_vehicle_type(alpr_text, vehicle), "status": "REJECTED", "image_file": ""})
            continue
        if plate_status == "guessed":
            # Show in UI but never persist — no DB write, no watchlist check
            entry = {"text": corrected, "score": final_conf, "vehicle_type": _resolve_vehicle_type(corrected, vehicle),
                     "image_file": "", "status": "UNVERIFIED", "n_subs": n_subs, "vote_info": vote_info}
            if vote_info.get("char_probs") and corrected:
                st.session_state[f"charprobs_{corrected}"] = vote_info["char_probs"]
            plates.append(entry)
            continue
        if plate_status not in ("valid", "corrected"):
            all_dets.append({"text": corrected, "score": final_conf,
                             "vehicle_type": _resolve_vehicle_type(corrected, vehicle), "status": "REJECTED", "image_file": ""})
            continue
        fname = f"{uuid.uuid1()}.jpg"
        cv2.imwrite(str(PLATES_DIR / fname), upscale_crop_color(crop))
        shown_status = "SAVED" if plate_status == "valid" else plate_status.upper()
        entry = {"text": corrected, "score": final_conf, "vehicle_type": _resolve_vehicle_type(corrected, vehicle),
                 "image_file": fname, "status": shown_status, "n_subs": n_subs,
                 "vote_info": vote_info}
        if vote_info.get("char_probs") and corrected:
            st.session_state[f"charprobs_{corrected}"] = vote_info["char_probs"]
        plates.append(entry)
        all_dets.append(entry)

    # Face recognition
    try:
        rec, lmap = get_face_rec()
        bgr_ann, faces = recognize_faces(bgr, rec, lmap)
        annotated = cv2.cvtColor(bgr_ann, cv2.COLOR_BGR2RGB)
    except Exception:
        annotated = img_rgb
        faces = []

    wl_hits = insert_detections(all_dets, source="image")
    _now_s = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for _det in all_dets:
        if _det.get("text") and _det.get("status") not in ("REJECTED",):
            record_exit(_det["text"], _now_s)
    return annotated, plates, all_dets, faces, wl_hits


def _draw_annotations(frame: np.ndarray, vehicles: list, plate_boxes: list) -> np.ndarray:
    out = frame.copy()
    for vx1, vy1, vx2, vy2, vcls in vehicles:
        cv2.rectangle(out, (vx1, vy1), (vx2, vy2), (30, 100, 220), 2)
        label = VEHICLE_CLS.get(vcls, "Vehicle")
        cv2.rectangle(out, (vx1, vy1 - 22), (vx1 + len(label) * 10 + 8, vy1), (30, 100, 220), -1)
        cv2.putText(out, label, (vx1 + 4, vy1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    for (px1, py1, px2, py2, plate_text, saved) in plate_boxes:
        color = (0, 200, 80) if saved else (0, 180, 255)
        cv2.rectangle(out, (px1, py1), (px2, py2), color, 2)
        if plate_text:
            cv2.rectangle(out, (px1, py1 - 24), (px1 + len(plate_text) * 11 + 8, py1), color, -1)
            cv2.putText(out, plate_text, (px1 + 4, py1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 1)
    return out


def process_video(path: str, conf: float, progress_bar, status_ph, frame_ph, live_ph):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return [], [], []
    fps    = cap.get(cv2.CAP_PROP_FPS) or 25
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step   = max(1, int(round(fps)))
    targets = list(range(0, total, step))
    n       = len(targets)

    status_ph.info("⏱ Sampling video quality…")
    _sample_vars = []
    for _si in targets[:min(30, n)]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, _si)
        ok, sf = cap.read()
        if ok:
            sg = cv2.cvtColor(sf, cv2.COLOR_BGR2GRAY)
            _sample_vars.append(cv2.Laplacian(sg, cv2.CV_64F).var())
    sharp_thresh = max(30.0, float(np.median(_sample_vars) * 0.35)) if _sample_vars else _SHARP_THRESH
    status_ph.info("🔍 Analysing frames…")

    dets, all_dets, live_rows, all_wl_hits = [], [], [], []
    dedup = PlateDeduplicator()
    alpr, coco = get_alpr(), get_coco()

    try:
        for i, fidx in enumerate(targets):
                pct = (i + 1) / n
                progress_bar.progress(pct,
                    text=f"Frame {fidx:,} / {total:,}  ·  {len(dets)} plate(s) saved  ·  {pct*100:.0f}%")
                cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
                ret, frame = cap.read()
                if not ret:
                    continue
                if not is_sharp(frame, sharp_thresh):
                    continue

                ts = round(fidx / fps, 2)
                h_f, w_f = frame.shape[:2]
                vehicles = []
                for det in coco(frame, conf=conf)[0].boxes.data.tolist():
                    vx1, vy1, vx2, vy2, _, vcls = det
                    if int(vcls) in VEHICLE_CLS:
                        vehicles.append((int(vx1), int(vy1), int(vx2), int(vy2), int(vcls)))

                plate_boxes = []
                for result in alpr.predict(frame):
                    if result.ocr is None:
                        continue
                    bb = result.detection.bounding_box
                    x1, y1, x2, y2 = int(bb.x1), int(bb.y1), int(bb.x2), int(bb.y2)
                    if not _plate_near_vehicle(x1, y1, x2, y2, vehicles):
                        continue  # plate outside all vehicle bboxes — discard
                    cx1 = max(0, x1 - _CROP_PAD);  cy1 = max(0, y1 - _CROP_PAD)
                    cx2 = min(w_f, x2 + _CROP_PAD); cy2 = min(h_f, y2 + _CROP_PAD)
                    crop = frame[cy1:cy2, cx1:cx2]
                    if not crop.size:
                        continue
                    alpr_text = (result.ocr.text or "").strip()
                    _rc = result.ocr.confidence
                    alpr_conf = float(sum(_rc) / len(_rc) if isinstance(_rc, list) else (_rc or 0.0))
                    vehicle = _associate_vehicle(x1, y1, x2, y2, vehicles)
                    crop_ocr = upscale_crop_color(crop)
                    corrected, final_conf, n_subs, plate_status, raw_text, vote_info = _ensemble_ocr(
                        crop_ocr, alpr_text, alpr_conf
                    )
                    now_s = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    if not corrected or final_conf < _MIN_DISP_CONF:
                        plate_boxes.append((x1, y1, x2, y2, alpr_text, False))
                        all_dets.append({"text": alpr_text, "score": alpr_conf,
                                         "vehicle_type": _resolve_vehicle_type(raw_text, vehicle), "status": "REJECTED", "image_file": ""})
                        continue
                    if plate_status == "guessed":
                        # Display in live feed only — not saved to DB
                        plate_boxes.append((x1, y1, x2, y2, corrected, False))
                        live_rows.append({
                            "Time": now_s, "Plate": corrected,
                            "Confidence": f"{final_conf:.0%}",
                            "Vehicle": _resolve_vehicle_type(corrected, vehicle),
                            "Status": "⚠️ UNVERIFIED", "Voted": "—",
                        })
                        live_ph.markdown(
                            pd.DataFrame(live_rows).to_html(
                                index=False, border=0, classes="", table_id="live-table"
                            ),
                            unsafe_allow_html=True,
                        )
                        continue
                    if plate_status not in ("valid", "corrected"):
                        plate_boxes.append((x1, y1, x2, y2, corrected, False))
                        all_dets.append({"text": corrected, "score": final_conf,
                                         "vehicle_type": _resolve_vehicle_type(corrected, vehicle), "status": "REJECTED", "image_file": ""})
                        continue
                    dup_status, best_text, best_conf = dedup.process(corrected, final_conf, fidx, ts)
                    plate_boxes.append((x1, y1, x2, y2, best_text or corrected, True))
                    if dup_status == "duplicate":
                        continue
                    fname = f"{uuid.uuid1()}.jpg"
                    cv2.imwrite(str(PLATES_DIR / fname), upscale_crop_color(crop))
                    shown_status = "SAVED" if plate_status == "valid" else plate_status.upper()
                    row = {"text": best_text or corrected,
                           "score": round(best_conf or final_conf, 4),
                           "vehicle_type": _resolve_vehicle_type(best_text or corrected, vehicle), "image_file": fname,
                           "status": shown_status, "timestamp": now_s,
                           "vote_info": vote_info}
                    dets.append(row)
                    all_dets.append(row)
                    live_rows.append({
                        "Time":       now_s,
                        "Plate":      row["text"],
                        "Confidence": f"{row['score']:.0%}",
                        "Vehicle":    vehicle,
                        "Status":     shown_status,
                        "Voted":      "🗳 YES" if vote_info.get("voted") else "—",
                    })
                    _feed_html = pd.DataFrame(live_rows).to_html(
                        index=False, border=0, classes="", table_id="live-table"
                    )
                    live_ph.markdown(_feed_html, unsafe_allow_html=True)

                annotated = _draw_annotations(frame, vehicles, plate_boxes)
                frame_ph.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                               use_container_width=True)
    finally:
        cap.release()

    # Voting pass + dwell time recording
    _end_ts = time.time()
    for row in dets:
        voted = dedup.get_voted_text(row["text"])
        if voted and voted != row["text"]:
            row["text"] = voted
        # Record dwell time
        try:
            _dwell = dedup.get_dwell_seconds(row["text"], _end_ts)
            _fs    = dedup.get_first_seen_str(row["text"])
            if _dwell > 0 and _fs:
                update_detection_dwell(row["text"], str(_fs), _dwell)
        except Exception:
            pass

    wl_hits = insert_detections(all_dets, source="video")
    all_wl_hits.extend(wl_hits)
    _now_s = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for _det in all_dets:
        if _det.get("text") and _det.get("status") not in ("REJECTED",):
            record_exit(_det["text"], _now_s)
    return dets, all_dets, all_wl_hits


# ── Page header ──────────────────────────────────────────────────────────────
st.markdown("""
<div style="background:linear-gradient(135deg,#1E3A5F 0%,#2563EB 100%);
            border-radius:14px;padding:24px 32px;margin-bottom:1.5rem;
            box-shadow:0 4px 20px rgba(26,86,219,0.25);">
  <div style="display:flex;align-items:center;gap:14px;">
    <span style="font-size:2rem;">🔍</span>
    <div>
      <div style="font-size:1.6rem;font-weight:800;color:#FFFFFF;
                  letter-spacing:0.04em;line-height:1.1;">ARGUS INTELLIGENCE</div>
      <div style="font-size:0.78rem;color:#93C5FD;font-weight:500;
                  letter-spacing:0.12em;text-transform:uppercase;margin-top:2px;">
        Kenyan License Plate Detection · YOLOv11 · Fast-ALPR · Fast-Plate-OCR · PaddleOCR · llava-phi3</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:20px 0 8px;text-align:center;">
      <div style="font-size:1.3rem;font-weight:800;color:#F1F5F9;
                  letter-spacing:0.06em;">🔍 ARGUS</div>
      <div style="font-size:0.65rem;color:#64748B;letter-spacing:0.1em;
                  text-transform:uppercase;margin-top:2px;">Intelligence v4.0</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    st.subheader("⚙️ Settings")
    conf = st.slider("Detection confidence", 0.10, 0.95, 0.35, 0.05)

    st.divider()
    st.subheader("📊 Statistics")
    try:
        _stats = get_stats()
        c1, c2 = st.columns(2)
        c1.metric("Saved",      _stats["saved"])
        c2.metric("Auto-Fixed", _stats["fixed"])
        c1.metric("Rejected",   _stats["rejected"])
    except Exception:
        st.info("No detections yet.")

    st.divider()

    # ── Watchlist manager ─────────────────────────────────────────────────────
    st.subheader("🚨 Watchlist")
    with st.expander("⚙️ Webhook Settings", expanded=False):
        _wh_url = st.text_input(
            "Webhook URL", value=st.session_state.get("webhook_url", ""),
            placeholder="https://hooks.example.com/…", key="webhook_url_input",
        )
        if _wh_url != st.session_state.get("webhook_url", ""):
            st.session_state["webhook_url"] = _wh_url
    with st.expander("Add plate to watchlist", expanded=False):
        wl_plate = st.text_input("Plate number", placeholder="KDA 123B",
                                  key="wl_add_plate")
        wl_label = st.text_input("Label / reason", placeholder="Stolen vehicle",
                                  key="wl_add_label")
        if st.button("➕ Add to Watchlist", use_container_width=True):
            if wl_plate.strip():
                if add_to_watchlist(wl_plate, wl_label):
                    st.success(f"Added {wl_plate.upper().strip()}")
                    st.rerun()
                else:
                    st.warning("Already on watchlist.")
            else:
                st.error("Enter a plate number.")

    _wl_df = get_watchlist()
    if not _wl_df.empty:
        st.caption(f"{len(_wl_df)} plate(s) monitored")
        for _, _wl_row in _wl_df.iterrows():
            _wcol1, _wcol2 = st.columns([3, 1])
            _wcol1.markdown(
                f"<span style='font-family:monospace;font-weight:700;"
                f"color:#F87171;font-size:0.85rem;'>{_wl_row['plate_text']}</span>"
                + (f"<br><span style='font-size:0.7rem;color:#94A3B8;'>"
                   f"{_wl_row['label']}</span>" if _wl_row.get("label") else ""),
                unsafe_allow_html=True,
            )
            if _wcol2.button("✕", key=f"rm_{_wl_row['plate_text']}"):
                remove_from_watchlist(_wl_row["plate_text"])
                st.rerun()
    else:
        st.caption("No plates on watchlist.")

    st.divider()
    st.sidebar.caption("API: http://localhost:8502/docs")
    if st.button("🗑️ Clear All Detections", use_container_width=True):
        deleted = 0
        if PLATES_DIR.exists():
            for f in PLATES_DIR.iterdir():
                if f.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    f.unlink(missing_ok=True)
                    deleted += 1
        clear_all_detections()
        st.success(f"Cleared {deleted} image(s) and database.")
        st.rerun()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_img, tab_vid, tab_log, tab_rejection, tab_wl, tab_faces, tab_analytics = st.tabs([
    "📷  Image Analysis",
    "🎬  Video Analysis",
    "📋  Detection Log",
    "🔍  Rejection Audit",
    "🚨  Watchlist",
    "👤  Face Training",
    "📊  Analytics",
])

# ══ IMAGE TAB ════════════════════════════════════════════════════════════════
with tab_img:
    st.subheader("Image Analysis")
    uploaded = st.file_uploader(
        "Upload an image", type=["jpg", "jpeg", "png", "bmp", "webp"],
        label_visibility="collapsed",
    )
    if uploaded:
        img_arr = np.frombuffer(uploaded.read(), np.uint8)
        img_bgr = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        col_in, col_out = st.columns(2)
        with col_in:
            st.image(img_rgb, caption="Input image", use_container_width=True)

        with st.spinner("Detecting plates…"):
            annotated, plates, all_dets, faces, wl_hits = run_detection(img_rgb, conf)

        with col_out:
            st.image(annotated, caption="Annotated", use_container_width=True)

        # Watchlist alerts
        if wl_hits:
            for _hit in wl_hits:
                st.error(
                    f"🚨 **WATCHLIST ALERT** — plate **{_hit}** detected!",
                    icon="🚨",
                )
                st.toast(f"🚨 Watchlist hit: {_hit}", icon="🚨")
                _fire_webhook(_hit, "image")

        if plates:
            st.success(f"**{len(plates)}** plate(s) detected")
            cols = st.columns(min(4, len(plates)))
            for i, p in enumerate(plates):
                fp = PLATES_DIR / p["image_file"]
                with cols[i % 4]:
                    if fp.exists():
                        crop_rgb = cv2.cvtColor(cv2.imread(str(fp)), cv2.COLOR_BGR2RGB)
                        st.image(crop_rgb, use_container_width=True)
                    badge_cls = "plate-badge-alert" if p["text"] in wl_hits else "plate-badge"
                    st.markdown(
                        f'<span class="{badge_cls}">{p["text"]}</span>',
                        unsafe_allow_html=True,
                    )
                    status_cls = f"st-{p['status'].lower()}"
                    _cat = get_plate_category(p["text"])
                    _cat_label = f" · {_cat}" if _cat not in ("Civilian", "Unknown") else ""
                    st.markdown(
                        f'<span class="{status_cls}">{p["status"]}</span> · '
                        f'{p["score"]:.0%} · {p["vehicle_type"]}{_cat_label}',
                        unsafe_allow_html=True,
                    )

            # Vote board — collapsible per detection
            for p in plates:
                vi = p.get("vote_info")
                if vi:
                    try:
                        _show_vote_expander(vi, p.get("text", "—"))
                    except Exception:
                        pass
        else:
            st.warning("No valid Kenyan plates detected.")

        if faces:
            st.info(f"Faces recognised: {', '.join(f['name'] for f in faces)}")

        with st.expander("All detections (including rejected)"):
            st.dataframe(pd.DataFrame(all_dets), use_container_width=True, hide_index=True)

# ══ VIDEO TAB ════════════════════════════════════════════════════════════════
with tab_vid:
    st.subheader("Video Analysis")

    # ── Source selection ──────────────────────────────────────────────────────
    _vid_src = st.radio("Video source", ["📁 File Upload", "📡 RTSP / IP Camera", "▶️ YouTube URL"],
                        horizontal=True, key="vid_source_radio")

    vid_file   = None
    rtsp_url   = ""
    yt_url     = ""
    _live_feed = False

    if _vid_src == "📁 File Upload":
        vid_file = st.file_uploader(
            "Upload a video", type=["mp4", "mov", "avi", "mkv", "m4v"],
            label_visibility="collapsed", key="vid_upload",
        )
    elif _vid_src == "▶️ YouTube URL":
        st.markdown("##### ▶️ YouTube")
        yt_url = st.text_input(
            "YouTube URL",
            placeholder="https://www.youtube.com/watch?v=...",
            key="yt_url_input",
        )
        if yt_url:
            st.info("Video will be downloaded to a temp file before analysis.")
    else:
        st.markdown("##### 📡 RTSP / IP Camera")
        rtsp_url = st.text_input(
            "RTSP stream URL",
            placeholder="rtsp://user:pass@192.168.1.1/stream  or  http://…/video",
            key="rtsp_url_input",
        )
        _live_feed = st.toggle("🔴 Live Feed (continuous)", value=False, key="live_feed_toggle")
        if rtsp_url:
            st.info(f"Stream: `{rtsp_url}`")

    # Determine video path (file or RTSP)
    _use_rtsp    = bool(rtsp_url and _vid_src == "📡 RTSP / IP Camera")
    _use_yt      = bool(yt_url and _vid_src == "▶️ YouTube URL")
    _can_analyse = vid_file is not None or _use_rtsp or _use_yt

    if vid_file:
        st.video(vid_file)

    if _can_analyse:
        _btn_label = "🔴 Start Live Feed" if _live_feed else "🚀  Analyse Footage"
        if st.button(_btn_label, type="primary", use_container_width=True, key="analyse_btn"):
            if vid_file:
                suffix = Path(vid_file.name).suffix or ".mp4"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(vid_file.read())
                    tmp_path = tmp.name
            elif _use_yt:
                with st.spinner("Downloading YouTube video…"):
                    try:
                        import yt_dlp
                        _yt_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                        _yt_tmp.close()
                        ydl_opts = {
                            "format": "bestvideo[ext=mp4][height<=720]+bestaudio[ext=m4a]/best[ext=mp4][height<=720]/best",
                            "outtmpl": _yt_tmp.name,
                            "quiet": True,
                            "no_warnings": True,
                            "merge_output_format": "mp4",
                        }
                        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                            ydl.download([yt_url])
                        tmp_path = _yt_tmp.name
                        st.success("Download complete — starting analysis…")
                    except Exception as _yt_err:
                        st.error(f"YouTube download failed: {_yt_err}")
                        st.stop()
            else:
                tmp_path = rtsp_url  # cv2.VideoCapture accepts RTSP URLs directly

            progress_bar = st.progress(0, text="Starting…")
            status_ph    = st.empty()
            st.markdown("#### Live Frame View")
            frame_ph     = st.empty()
            st.markdown("#### Detection Feed")
            live_ph      = st.empty()

            _stop_ph = st.empty()
            if _live_feed:
                _stop_ph.info("Live feed running — close the tab or reload to stop.")

            try:
                dets, all_dets, wl_hits = process_video(
                    tmp_path, conf, progress_bar, status_ph, frame_ph, live_ph
                )
                progress_bar.progress(1.0, text="Complete!")
                status_ph.empty()
                _stop_ph.empty()
            finally:
                if vid_file:
                    try: os.unlink(tmp_path)
                    except OSError: pass

            # Watchlist alerts
            if wl_hits:
                for _hit in wl_hits:
                    st.error(f"🚨 **WATCHLIST ALERT** — plate **{_hit}** detected in footage!", icon="🚨")
                    st.toast(f"🚨 Watchlist hit: {_hit}", icon="🚨")
                    _fire_webhook(_hit, "video")

            n_saved = len(dets)
            if n_saved:
                st.success(
                    f"Analysis complete — **{n_saved}** unique plate(s) saved · "
                    f"{len(all_dets)} total detections"
                )
            else:
                st.warning(
                    f"Analysis complete — no valid plates found "
                    f"({len(all_dets)} detections checked)"
                )

            if dets:
                st.markdown("#### Saved Plates")
                cols = st.columns(min(4, n_saved))
                for i, d in enumerate(dets[:16]):
                    fp = PLATES_DIR / d.get("image_file", "")
                    with cols[i % 4]:
                        if fp.exists():
                            crop_rgb = cv2.cvtColor(cv2.imread(str(fp)), cv2.COLOR_BGR2RGB)
                            st.image(crop_rgb, use_container_width=True)
                        badge_cls = "plate-badge-alert" if d["text"] in wl_hits else "plate-badge"
                        st.markdown(
                            f'<span class="{badge_cls}">{d["text"]}</span>',
                            unsafe_allow_html=True,
                        )
                        _cat = get_plate_category(d["text"])
                        _cat_label = f" · {_cat}" if _cat not in ("Civilian", "Unknown") else ""
                        st.caption(f'{d["status"]} · {d["score"]:.0%} · {d["vehicle_type"]}{_cat_label}')

                # Vote boards — collapsible per detection
                voted_dets = [d for d in dets if d.get("vote_info")]
                for d in voted_dets:
                    try:
                        _show_vote_expander(d["vote_info"], d.get("text", "—"))
                    except Exception:
                        pass

# ══ LOG TAB ══════════════════════════════════════════════════════════════════
with tab_log:
    st.subheader("Detection Log")

    # ── Ground Truth Mode toggle ──────────────────────────────────────────────
    _gt_mode = st.toggle("🏷️ Ground Truth Mode", key="gt_mode", value=False)
    if _gt_mode:
        st.markdown(
            '<div style="background:#DCFCE7;border:1.5px solid #86EFAC;border-radius:8px;'
            'padding:10px 16px;color:#166534;font-weight:700;margin-bottom:8px;">'
            '✅ Ground Truth Mode Active — overrides will be flagged as verified training data.</div>',
            unsafe_allow_html=True,
        )

    _status_opts = get_distinct_values("status")
    _source_opts = get_distinct_values("source")

    if _status_opts or _source_opts:
        fc1, fc2, fc3 = st.columns(3)
        fc1.multiselect("Status", _status_opts, default=_status_opts, key="log_status")
        fc2.multiselect("Source", _source_opts, default=_source_opts, key="log_source")
        fc3.text_input("Search plate", "", key="log_search")

        @st.fragment(run_every=4)
        def _log_table_fragment():
            _sf   = st.session_state.get("log_status") or None
            _src  = st.session_state.get("log_source") or None
            _srch = st.session_state.get("log_search", "")
            df_log = query_detections(status_filter=_sf, source_filter=_src, plate_search=_srch)
            if "verified_ground_truth" in df_log.columns:
                df_log["GT"] = df_log["verified_ground_truth"].apply(
                    lambda v: "✅ GT" if (v and int(v) > 0) else ""
                )
            try:
                from db import DB_PATH as _DB_PATH
                import sqlite3 as _sq3
                with _sq3.connect(str(_DB_PATH)) as _gc:
                    _gt_count = _gc.execute(
                        "SELECT COUNT(*) FROM detections WHERE verified_ground_truth > 0"
                    ).fetchone()[0]
                st.caption(f"{len(df_log):,} records · 🏷️ {_gt_count} ground truth verified · ↻ auto-refresh 4s")
            except Exception:
                st.caption(f"{len(df_log):,} records · ↻ auto-refresh 4s")
            st.dataframe(df_log, use_container_width=True, hide_index=True)
            st.download_button(
                "⬇️  Download CSV",
                data=df_log.to_csv(index=False),
                file_name="argus_plate_log.csv",
                mime="text/csv",
                key="log_dl",
            )

        _log_table_fragment()

        # ── Feature 2: Manual Override ────────────────────────────────────────
        st.divider()
        with st.expander("✏️ Manual Override"):
            _ov_col1, _ov_col2, _ov_col3 = st.columns([1, 2, 2])
            _ov_id   = _ov_col1.number_input("Detection ID", min_value=1, step=1, key="override_id")
            _ov_text = _ov_col2.text_input("Corrected Plate", placeholder="KDA 123B", key="override_text")
            _ov_note = _ov_col3.text_input("Note (optional)", placeholder="Operator note…", key="override_note")
            _ov_gt   = st.checkbox(
                "Mark as verified ground truth",
                value=_gt_mode,
                key="override_gt_check",
            )
            if st.button("Apply Override", key="override_btn"):
                if _ov_text.strip():
                    override_plate(int(_ov_id), _ov_text.strip(), _ov_note.strip())
                    if _ov_gt:
                        mark_ground_truth(int(_ov_id), is_correct=True)
                    st.success("Override saved." + (" Marked as ground truth." if _ov_gt else ""))
                    st.rerun()
                else:
                    st.error("Enter a corrected plate text.")

        # ── Feature 5: Crop viewer by plate search ────────────────────────────
        st.divider()
        st.markdown("#### 🖼️ Detection Crop Viewer")
        _crop_search = st.text_input("Enter plate to view crop", placeholder="KDA 123B",
                                     key="crop_viewer_search")
        if _crop_search.strip():
            _crop_df = query_detections(plate_search=_crop_search.strip(), limit=5)
            if not _crop_df.empty:
                _crop_row = _crop_df.iloc[0]
                _cv_col_l, _cv_col_r = st.columns(2)
                _img_file = str(_crop_row.get("image_file", "") or "")
                _img_path = PLATES_DIR / _img_file if _img_file else None
                with _cv_col_l:
                    if _img_path and _img_path.exists():
                        _crop_rgb = cv2.cvtColor(cv2.imread(str(_img_path)), cv2.COLOR_BGR2RGB)
                        st.image(_crop_rgb, caption="Saved crop", use_container_width=True)
                    else:
                        st.markdown(
                            '<div style="background:#F1F5F9;border:2px dashed #CBD5E1;'
                            'border-radius:10px;height:120px;display:flex;align-items:center;'
                            'justify-content:center;color:#94A3B8;font-size:0.85rem;">'
                            'No crop image saved</div>',
                            unsafe_allow_html=True,
                        )
                with _cv_col_r:
                    _plate_disp = str(_crop_row.get("plate_text", "—"))
                    _status_disp = str(_crop_row.get("status", "—"))
                    _status_colors = {"SAVED": "#15803D", "CORRECTED": "#1D4ED8",
                                      "UNVERIFIED": "#B45309", "REJECTED": "#DC2626",
                                      "OVERRIDE": "#7C3AED"}
                    _sc = _status_colors.get(_status_disp, "#374151")
                    # Build heatmap if char_probs available in session state
                    _hm_key = f"charprobs_{_plate_disp}"
                    _stored_probs = st.session_state.get(_hm_key)
                    _heatmap_span = (
                        _plate_heatmap_html(_plate_disp, _stored_probs)
                        if _stored_probs
                        else f'<span style="font-family:monospace;font-size:2rem;font-weight:900;'
                             f'color:#1E3A5F;letter-spacing:0.12em;">{_plate_disp}</span>'
                    )
                    st.markdown(
                        f'<div style="margin-top:20px;font-size:2rem;">{_heatmap_span}</div>'
                        f'<div style="margin-top:10px;">'
                        f'<span style="background:{_sc};color:#FFF;border-radius:20px;'
                        f'padding:3px 14px;font-size:0.8rem;font-weight:700;">{_status_disp}</span>'
                        f'</div>'
                        f'<div style="margin-top:8px;color:#64748B;font-size:0.8rem;">'
                        f'{_crop_row.get("timestamp","—")} · {_crop_row.get("vehicle_type","—")}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.info("No detection found for that plate.")

        # Incident report generator
        st.divider()
        st.markdown("#### 📄 Generate Incident Report")
        _rep_plate = st.text_input("Plate number for report", placeholder="KDA 123B",
                                   key="rep_plate_input")
        if st.button("📄 Generate PDF Report", key="gen_report_btn"):
            if _rep_plate.strip():
                _rep_df = query_detections(plate_search=_rep_plate.strip())
                if _rep_df.empty:
                    st.warning(f"No detections found for plate: {_rep_plate}")
                else:
                    try:
                        from report import generate_incident_pdf
                        _pdf_bytes = generate_incident_pdf(_rep_plate.strip(), _rep_df)
                        st.download_button(
                            label=f"⬇️ Download Report — {_rep_plate.upper().strip()}",
                            data=_pdf_bytes,
                            file_name=f"argus_incident_{_rep_plate.strip().replace(' ', '_')}.pdf",
                            mime="application/pdf",
                            key="incident_pdf_dl",
                        )
                        st.success(f"Report ready: {len(_rep_df)} detection(s) included.")
                    except Exception as _re:
                        st.error(f"PDF generation failed: {_re}")
            else:
                st.error("Enter a plate number.")
    else:
        st.info("No detections yet. Run an image or video analysis first.")

# ══ REJECTION AUDIT TAB ══════════════════════════════════════════════════════
with tab_rejection:
    st.subheader("Rejection Audit")
    st.caption("Review all REJECTED detections and correct missed reads.")

    _rej_df = query_detections(status_filter=["REJECTED"], limit=200)

    if _rej_df.empty:
        st.info("No rejected detections found.")
    else:
        # Summary metrics
        _rej_total = len(_rej_df)
        _rej_with_img = _rej_df["image_file"].apply(lambda v: bool(v and str(v).strip() and str(v) != "nan")).sum()
        _rej_pct_img = (_rej_with_img / _rej_total * 100) if _rej_total else 0
        _rej_dates = _rej_df["timestamp"].dropna()
        _rej_date_range = f"{_rej_dates.min()[:10]} → {_rej_dates.max()[:10]}" if not _rej_dates.empty else "—"

        _rm1, _rm2, _rm3 = st.columns(3)
        _rm1.metric("Total Rejected", _rej_total)
        _rm2.metric("With Saved Images", f"{_rej_pct_img:.0f}%")
        _rm3.metric("Date Range", _rej_date_range)

        st.divider()
        st.markdown("#### Rejected Detections — Override by ID")

        # Show table with key columns
        _rej_cols = ["id", "timestamp", "vehicle_type", "plate_text", "ocr_confidence", "status"]
        _rej_display = _rej_df[[c for c in _rej_cols if c in _rej_df.columns]].copy()
        _rej_display.rename(columns={
            "id": "ID", "timestamp": "Time", "vehicle_type": "Vehicle",
            "plate_text": "Raw OCR", "ocr_confidence": "Conf", "status": "Status",
        }, inplace=True)
        st.dataframe(_rej_display, use_container_width=True, hide_index=True, height=300)

        # Show raw OCR text with heatmap coloring for selected ID
        st.markdown("#### Raw OCR Heatmap")
        _ra_hm_id = st.number_input("Detection ID to preview heatmap", min_value=1, step=1, key="rej_hm_id")
        _hm_row = _rej_df[_rej_df["id"] == int(_ra_hm_id)] if not _rej_df.empty else pd.DataFrame()
        if not _hm_row.empty:
            _hm_plate = str(_hm_row.iloc[0].get("plate_text", ""))
            _hm_stored_probs = st.session_state.get(f"charprobs_{_hm_plate}")
            if _hm_stored_probs:
                st.markdown(
                    f'<div style="margin:8px 0;">'
                    + _plate_heatmap_html(_hm_plate, _hm_stored_probs)
                    + f' <span style="color:#64748B;font-size:0.75rem;margin-left:8px;">'
                      f'(char confidence heatmap)</span></div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div style="color:#94A3B8;font-family:monospace;font-size:1.1rem;">'
                    f'{_hm_plate} <span style="font-size:0.75rem;">(no char probs — greyed)</span></div>',
                    unsafe_allow_html=True,
                )

        # Inline override form for rejection audit
        st.markdown("#### Correct a Missed Read")
        _ra_col1, _ra_col2, _ra_col3 = st.columns([1, 2, 2])
        _ra_id      = _ra_col1.number_input("Detection ID", min_value=1, step=1, key="rej_audit_id")
        _ra_plate   = _ra_col2.text_input("Correct Plate", placeholder="KDA 123B", key="rej_audit_plate")
        _ra_note    = _ra_col3.text_input("Note (optional)", placeholder="Operator note…", key="rej_audit_note")
        if st.button("✅ Mark as Missed Read", key="rej_audit_btn"):
            if _ra_plate.strip():
                override_plate(int(_ra_id), _ra_plate.strip(), _ra_note.strip())
                st.success(f"Detection {int(_ra_id)} corrected to '{_ra_plate.strip().upper()}' and marked OVERRIDE.")
                st.rerun()
            else:
                st.error("Enter the correct plate text.")

# ══ WATCHLIST TAB ════════════════════════════════════════════════════════════
with tab_wl:
    st.subheader("Watchlist Management")
    st.caption("Plates on this list trigger alerts whenever they are detected.")

    _wl_full = get_watchlist()
    if _wl_full.empty:
        st.info("Watchlist is empty. Add plates via the sidebar.")
    else:
        st.dataframe(_wl_full, use_container_width=True, hide_index=True)

        # Show detection history for watchlisted plates
        st.divider()
        st.subheader("Detection History (Watchlist Plates)")
        _wl_plates = _wl_full["plate_text"].tolist()
        _wl_history_rows = []
        for _wp in _wl_plates:
            _df_wp = query_detections(plate_search=_wp)
            _wl_history_rows.append(_df_wp)
        if _wl_history_rows:
            _wl_hist = pd.concat(_wl_history_rows, ignore_index=True)
            if not _wl_hist.empty:
                _wl_hist = _wl_hist.sort_values("timestamp", ascending=False)
                st.dataframe(_wl_hist, use_container_width=True, hide_index=True)
                st.download_button(
                    "⬇️  Download Watchlist History",
                    data=_wl_hist.to_csv(index=False),
                    file_name="argus_watchlist_history.csv",
                    mime="text/csv",
                    key="wl_hist_dl",
                )
            else:
                st.info("No detections yet for watchlisted plates.")

# ══ PERSISTENT CSV LOG BOX ═══════════════════════════════════════════════════
st.markdown("""
<div style="background:linear-gradient(135deg,#1E3A5F,#2563EB);
            border-radius:14px 14px 0 0;padding:16px 24px;margin-top:2rem;">
  <div style="display:flex;align-items:center;justify-content:space-between;">
    <div>
      <span style="font-size:1rem;font-weight:800;color:#FFFFFF;
                   letter-spacing:0.05em;">📋 LIVE DETECTION LOG</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<style>
.csv-box {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-top: none;
    border-radius: 0 0 14px 14px;
    padding: 20px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.07);
    margin-bottom: 2rem;
}
</style>
<div class="csv-box">
""", unsafe_allow_html=True)

_log_df = query_detections(limit=200)
if not _log_df.empty:
    _lc1, _lc2, _lc3 = st.columns([2, 2, 2])
    _search_log = _lc1.text_input("🔍 Search plate", "", key="csv_search")
    _status_opts2 = get_distinct_values("status")
    _source_opts2 = get_distinct_values("source")
    _status_sel2  = _lc2.multiselect("Status", _status_opts2,
                                      default=_status_opts2, key="csv_status")
    _source_sel2  = _lc3.multiselect("Source", _source_opts2,
                                      default=_source_opts2, key="csv_source")

    _df_show = query_detections(
        status_filter=_status_sel2 or None,
        source_filter=_source_sel2 or None,
        plate_search=_search_log,
        limit=200,
    )
    _total = get_stats()["total"]
    st.caption(f"Showing **{len(_df_show):,}** of **{_total:,}** records (most recent 200)")
    st.dataframe(_df_show, use_container_width=True, hide_index=True, height=320)

    _dl1, _ = st.columns([1, 4])
    _dl1.download_button(
        "⬇️ Download CSV",
        data=_df_show.to_csv(index=False),
        file_name="argus_plate_log.csv",
        mime="text/csv",
        key="csv_download_main",
    )
else:
    st.info("No detections yet. Run an image or video analysis to populate the log.")

st.markdown("</div>", unsafe_allow_html=True)

# ══ FACE TRAINING TAB ════════════════════════════════════════════════════════
with tab_faces:
    st.subheader("Face Recognition Training")
    st.caption("Enrol subjects into the biometric database. Minimum 5 frontal images.")

    registered = get_registered_persons()
    if registered:
        st.success(f"**{len(registered)}** subject(s) enrolled: {', '.join(registered)}")
    else:
        st.info("No subjects enrolled yet.")

    st.divider()
    with st.form("face_train_form", clear_on_submit=True):
        person_name  = st.text_input("Subject name", placeholder="Full name…")
        face_uploads = st.file_uploader(
            "Training images (JPG/PNG, min 5)",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
        )
        c1, c2 = st.columns(2)
        save_btn   = c1.form_submit_button("Enrol Subject", type="primary", use_container_width=True)
        cancel_btn = c2.form_submit_button("Cancel", use_container_width=True)

    if save_btn:
        name = re.sub(r"[^a-zA-Z0-9_\- ]", "", person_name).strip()
        if not name:
            st.error("Invalid name.")
        elif not face_uploads:
            st.error("No images uploaded.")
        else:
            pil_imgs = [Image.open(f) for f in face_uploads]
            with st.spinner("Saving training images…"):
                n_saved = save_training_images(name, pil_imgs)
            if n_saved == 0:
                st.warning("No faces detected in uploaded images.")
            else:
                st.success(f"Saved {n_saved} face image(s) for '{name}'.")
                with st.spinner("Training model…"):
                    ok, msg = train_lbph_model()
                if ok:
                    get_face_rec.clear()
                    st.success(f"Model trained: {msg}")
                else:
                    st.error(f"Training failed: {msg}")


# ══ ANALYTICS TAB ════════════════════════════════════════════════════════════
with tab_analytics:
    st.subheader("Analytics Dashboard")

    try:
        from db import DB_PATH
        import sqlite3 as _sqlite3

        with _sqlite3.connect(str(DB_PATH)) as _ac:
            _ac.row_factory = _sqlite3.Row

            # Detections per hour
            _hr_rows = _ac.execute(
                "SELECT strftime('%H', timestamp) AS hour, COUNT(*) AS count "
                "FROM detections GROUP BY hour ORDER BY hour"
            ).fetchall()
            if _hr_rows:
                _hr_df = pd.DataFrame([dict(r) for r in _hr_rows])
                _hr_df["hour"] = _hr_df["hour"].apply(lambda h: f"{h}:00")
                fig_hr = px.bar(_hr_df, x="hour", y="count",
                                title="Detections per Hour of Day",
                                labels={"hour": "Hour", "count": "Detections"},
                                color_discrete_sequence=["#2563EB"])
                fig_hr.update_layout(plot_bgcolor="white", paper_bgcolor="white")
                st.plotly_chart(fig_hr, use_container_width=True)
            else:
                st.info("No detections yet for hourly chart.")

            _ac1, _ac2 = st.columns(2)

            # Vehicle type breakdown
            _vt_rows = _ac.execute(
                "SELECT vehicle_type, COUNT(*) AS count FROM detections "
                "WHERE vehicle_type IS NOT NULL GROUP BY vehicle_type"
            ).fetchall()
            if _vt_rows:
                _vt_df = pd.DataFrame([dict(r) for r in _vt_rows])
                fig_vt = px.pie(_vt_df, names="vehicle_type", values="count",
                                title="Vehicle Type Breakdown",
                                color_discrete_sequence=px.colors.qualitative.Set2)
                _ac1.plotly_chart(fig_vt, use_container_width=True)

            # Status breakdown
            _st_rows = _ac.execute(
                "SELECT status, COUNT(*) AS count FROM detections "
                "WHERE status IS NOT NULL GROUP BY status"
            ).fetchall()
            if _st_rows:
                _st_df = pd.DataFrame([dict(r) for r in _st_rows])
                fig_st = px.pie(_st_df, names="status", values="count",
                                title="Detection Status Breakdown",
                                color_discrete_sequence=["#15803D", "#1D4ED8", "#B45309", "#DC2626"])
                _ac2.plotly_chart(fig_st, use_container_width=True)

            # Top 10 most seen plates
            _tp_rows = _ac.execute(
                "SELECT plate_text, COUNT(*) AS count FROM detections "
                "WHERE plate_text IS NOT NULL GROUP BY plate_text "
                "ORDER BY count DESC LIMIT 10"
            ).fetchall()
            if _tp_rows:
                _tp_df = pd.DataFrame([dict(r) for r in _tp_rows])
                fig_tp = px.bar(_tp_df, x="plate_text", y="count",
                                title="Top 10 Most Seen Plates",
                                labels={"plate_text": "Plate", "count": "Count"},
                                color_discrete_sequence=["#1E3A5F"])
                fig_tp.update_layout(plot_bgcolor="white", paper_bgcolor="white")
                st.plotly_chart(fig_tp, use_container_width=True)


    except Exception as _ae:
        st.warning(f"Analytics unavailable: {_ae}")
