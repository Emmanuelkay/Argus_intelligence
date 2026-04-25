"""
Auto-label high-confidence plate crops using llava-phi3, export training dataset.

Usage:
    python3 build_training_set.py [--conf 0.85] [--out training_data] [--limit 500]

Logic:
    1. Pull SAVED/CORRECTED detections with ocr_confidence >= threshold
    2. Run LLM on each crop independently (no OCR hint — blind read)
    3. Keep only where LLM agrees with OCR label exactly
    4. Export: training_data/images/*.jpg + training_data/labels.csv
"""

import argparse
import csv
import re
import shutil
import sqlite3
import sys
import time
from pathlib import Path

import cv2
import base64
import requests
from io import BytesIO
from PIL import Image

DB_PATH        = Path("data/argus.db")
PLATES_DIR     = Path("plate_crops")
OLLAMA_URL     = "http://localhost:11434/api/chat"
OLLAMA_MODEL   = "llava-phi3"

_OPTIONS = {
    "temperature":    0.0,
    "num_predict":    20,
    "num_ctx":        512,
    "top_k":          1,
    "repeat_penalty": 1.5,
    "stop":           ["\n", "sorry", "Sorry", "I ", "cannot", "None", "none"],
}
_SYSTEM_PROMPT = (
    "You are a Kenyan license plate character extractor. "
    "Civilian plates: K[A-Z][A-Z] [0-9][0-9][0-9][A-Z] — 8 chars including space. Example: KDA 123B. "
    "Tuk-tuk plates: KT[A-Z][A-Z] [0-9][0-9][0-9][A-Z] — 9 chars including space. Example: KTWA 123B. "
    "Motorcycle plates: KMC[A-Z] [0-9][0-9][0-9][A-Z] — 9 chars including space. Example: KMCA 123B. "
    "Fix OCR errors by position: O↔0, I↔1, S↔5, B↔8, G↔6, T↔7, Z↔2. "
    "Output ONLY the plate string. Nothing else. No words. No sentences. "
    "If you cannot read a valid plate, output: None"
)


def upscale_color(img):
    h, w = img.shape[:2]
    if w == 0 or h == 0:
        return img
    scale = max(200 / max(w, 1), 60 / max(h, 1), 2.0)
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)


def llm_read_blind(img) -> str | None:
    """Send crop to LLM with no OCR hint. Returns parsed plate or None."""
    buf = BytesIO()
    Image.fromarray(img[..., ::-1]).save(buf, format="JPEG", quality=95)
    b64 = base64.b64encode(buf.getvalue()).decode()
    try:
        resp = requests.post(OLLAMA_URL, json={
            "model":      OLLAMA_MODEL,
            "stream":     False,
            "keep_alive": -1,
            "options":    _OPTIONS,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": "Extract the license plate number.", "images": [b64]},
            ],
        }, timeout=25)
        raw = resp.json().get("message", {}).get("content", "").strip().upper()
    except Exception:
        return None

    # Parse — extract first valid Kenyan plate pattern
    clean = re.sub(r'[^A-Z0-9 ]', '', raw)
    for pat in (r'KT[A-Z]{2}\s*\d{3}[A-Z]', r'KMC[A-Z]\s*\d{3}[A-Z]', r'K[A-Z]{2}\s*\d{3}[A-Z]'):
        m = re.search(pat, clean)
        if m:
            t = re.sub(r'\s+', '', m.group())
            return (t[:4] + " " + t[4:]) if len(t) == 8 else (t[:3] + " " + t[3:])
    return None


def normalize(p: str) -> str:
    return re.sub(r'\s+', '', (p or "").upper().strip())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf",  type=float, default=0.85, help="Min OCR confidence")
    parser.add_argument("--out",   type=str,   default="training_data")
    parser.add_argument("--limit", type=int,   default=500)
    args = parser.parse_args()

    out_dir    = Path(args.out)
    img_dir    = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(DB_PATH))
    rows = conn.execute(
        "SELECT plate_text, image_file, ocr_confidence FROM detections "
        "WHERE image_file != '' AND ocr_confidence >= ? AND status IN ('SAVED','CORRECTED') "
        "ORDER BY ocr_confidence DESC LIMIT ?",
        (args.conf, args.limit),
    ).fetchall()
    conn.close()

    print(f"Candidates: {len(rows)} (conf >= {args.conf:.0%})")
    print(f"Output: {out_dir}/\n")

    confirmed = skipped_crop = skipped_aspect = llm_disagree = llm_none = 0
    labels = []

    for i, (ocr_plate, fname, conf) in enumerate(rows):
        img_path = PLATES_DIR / fname
        img = cv2.imread(str(img_path))
        if img is None:
            skipped_crop += 1
            continue

        # Check aspect ratio before upscaling
        h, w = img.shape[:2]
        if w < 40 or h < 10:
            skipped_aspect += 1
            continue
        aspect = w / h
        if not (1.5 <= aspect <= 8.0):
            img = upscale_color(img)
            h, w = img.shape[:2]
            if not (1.5 <= w / h <= 8.0):
                skipped_aspect += 1
                continue

        img = upscale_color(img)

        llm_plate = llm_read_blind(img)

        print(f"[{i+1:>3}/{len(rows)}] OCR={ocr_plate:<12} LLM={str(llm_plate or '—'):<12} conf={conf:.0%}", end="  ")

        if llm_plate is None:
            print("SKIP — LLM no read")
            llm_none += 1
            continue

        if normalize(llm_plate) != normalize(ocr_plate):
            print(f"SKIP — disagree")
            llm_disagree += 1
            continue

        # Both agree — confirmed label
        dest = img_dir / fname
        shutil.copy2(str(img_path), str(dest))
        labels.append({"image": fname, "plate": ocr_plate, "ocr_conf": round(conf, 4)})
        confirmed += 1
        print("✓ CONFIRMED")

    # Write labels.csv
    csv_path = out_dir / "labels.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "plate", "ocr_conf"])
        writer.writeheader()
        writer.writerows(labels)

    print(f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 Training set built
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 Confirmed (LLM + OCR agree) : {confirmed}
 LLM no read                 : {llm_none}
 LLM disagrees with OCR      : {llm_disagree}
 Bad crops (missing/tiny)    : {skipped_crop + skipped_aspect}
 Labels saved to             : {csv_path}
 Images saved to             : {img_dir}
""")


if __name__ == "__main__":
    main()
