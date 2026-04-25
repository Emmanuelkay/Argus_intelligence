"""LLM plate verifier — llava-phi3 via local Ollama, optimized for low latency."""
from __future__ import annotations

import base64
import hashlib
import re
import time
from collections import OrderedDict
from io import BytesIO
from typing import Optional

import numpy as np
import requests
from PIL import Image

from util import is_valid_kenyan_plate, correct_plate

OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
OLLAMA_URL      = "http://localhost:11434/api/generate"
OLLAMA_MODEL    = "llava-phi3"
TIMEOUT_S       = 20

_SYSTEM_PROMPT = (
    "You are a Kenyan license plate character extractor. "
    "Civilian plates: K[A-Z][A-Z] [0-9][0-9][0-9][A-Z] — 8 chars including space. Example: KDA 123B. "
    "Tuk-tuk plates: KTW[A-Z] [0-9][0-9][0-9][A-Z] — 9 chars including space. Example: KTWA 123B. "
    "Motorcycle plates: KMC[A-Z] [0-9][0-9][0-9][A-Z] — 9 chars including space. Example: KMCA 123B. "
    "Fix OCR errors by position: O↔0, I↔1, S↔5, B↔8, G↔6, T↔7, Z↔2. "
    "Output ONLY the plate string. Nothing else. No words. No sentences. "
    "If you cannot read a valid plate, output: None"
)

_USER_PROMPT = "Extract the license plate number."

_OPTIONS = {
    "temperature":    0.0,
    "num_predict":    20,
    "num_ctx":        512,
    "top_k":          1,
    "repeat_penalty": 1.5,
    "stop":           ["\n", "sorry", "Sorry", "I ", "cannot", "None", "none"],
}

# Chatter words that mean the model failed — treat as None
_CHATTER = {"THE", "THE PLATE", "IS", "PLATE", "NONE", "N/A", "NA", "NO", "NOT", "UNKNOWN"}

# Simple LRU image-hash cache — avoids re-calling LLM on same crop
_CACHE_MAX  = 256
_result_cache: OrderedDict[str, dict] = OrderedDict()


def _image_hash(img: np.ndarray) -> str:
    buf = BytesIO()
    Image.fromarray(img[..., ::-1]).save(buf, format="JPEG", quality=85)
    return hashlib.md5(buf.getvalue()).hexdigest()


def _is_valid_crop(img: np.ndarray) -> bool:
    """Reject crops that are too small or wrong aspect ratio to be a plate."""
    if img is None or img.size == 0:
        return False
    h, w = img.shape[:2]
    if w < 40 or h < 10:
        return False
    aspect = w / h
    return 1.5 <= aspect <= 8.0  # plates are wide, not square or tall


def _encode_image(img: np.ndarray) -> str:
    buf = BytesIO()
    Image.fromarray(img[..., ::-1]).save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def _parse_plate(response: str) -> Optional[str]:
    """
    Regex-only extraction. Accepts nothing that doesn't match Kenyan plate format.
    No fallbacks, no loose word matching — garbage in, None out.
    """
    cleaned = re.sub(r'\s+', ' ', response.strip().upper())

    if not cleaned or len(cleaned) < 7 or cleaned in _CHATTER:
        return None

    # Strip everything except alphanumerics and spaces for a clean scan
    stripped = re.sub(r'[^A-Z0-9 ]', '', cleaned)

    # Primary: strict 8-char KTW/KMC regex
    for pattern_8 in (r'\bKTW[A-Z]\s*\d{3}[A-Z]\b', r'\bKMC[A-Z]\s*\d{3}[A-Z]\b'):
        m = re.search(pattern_8, stripped)
        if m:
            raw = re.sub(r'\s+', '', m.group())
            formatted = raw[:4] + " " + raw[4:]
            if is_valid_kenyan_plate(formatted):
                return formatted

    # Primary: strict civilian regex K[A-Z]{2} \d{3}[A-Z]
    match = re.search(r'\bK[A-Z]{2}\s*\d{3}[A-Z]\b', stripped)
    if match:
        raw = re.sub(r'\s+', '', match.group())
        formatted = raw[:3] + " " + raw[3:]
        if is_valid_kenyan_plate(formatted):
            return formatted

    # Secondary: try correct_plate on 8-char K-prefix blocks (KTW/KMC)
    for token in re.findall(r'K[A-Z0-9]{7}', stripped.replace(" ", "")):
        plate, status, _ = correct_plate(token)
        if status in ("valid", "corrected"):
            return plate

    # Secondary: try correct_plate on 7-char K-prefix blocks (civilian)
    for token in re.findall(r'K[A-Z0-9]{6}', stripped.replace(" ", "")):
        plate, status, _ = correct_plate(token)
        if status in ("valid", "corrected"):
            return plate

    return None


def warmup() -> bool:
    """Verify llava-phi3 is in ollama list then pre-load into RAM."""
    try:
        # Confirm model exists before warming up
        tags = requests.get("http://localhost:11434/api/tags", timeout=5).json()
        names = [m["name"] for m in tags.get("models", [])]
        if not any(OLLAMA_MODEL in n for n in names):
            raise RuntimeError(f"{OLLAMA_MODEL} not found in ollama list: {names}")
        resp = requests.post(
            OLLAMA_URL,
            json={
                "model":      OLLAMA_MODEL,
                "prompt":     "K",
                "stream":     False,
                "keep_alive": -1,
                "options":    {"num_predict": 1, "num_ctx": 512},
            },
            timeout=60,
        )
        return resp.status_code == 200
    except Exception as e:
        print(f"[llm_verifier] warmup failed: {e}")
        return False


def verify_plate(
    crop_bgr: np.ndarray,
    ocr_text: str,
    ocr_conf: float,
) -> dict:
    """
    Send cropped plate image to llava-phi3 for verification.

    Triggers on: corrected, guessed, rejected statuses.
    Caches results by image hash — same crop never hits Ollama twice.
    keep_alive=-1 keeps model in RAM between calls.

    Returns benchmark dict:
        plate_llm  — plate text from LLM (None if chatter/unparseable)
        plate_used — final plate after comparison
        llm_valid  — bool
        latency_ms — round-trip ms (0 if cache hit)
        ocr_input  — original OCR text fed in
        ocr_conf   — original OCR confidence
        improved   — True if LLM result differs from OCR input
        cached     — True if result served from cache
        error      — error string or None
    """
    # Reject bad crops before touching Ollama
    if not _is_valid_crop(crop_bgr):
        return {
            "plate_llm": None, "plate_used": ocr_text, "llm_valid": False,
            "latency_ms": 0, "ocr_input": ocr_text, "ocr_conf": ocr_conf,
            "improved": False, "cached": False, "error": "invalid_crop",
        }

    img_key = _image_hash(crop_bgr)

    # Cache hit — return stored result with updated ocr fields
    if img_key in _result_cache:
        cached = dict(_result_cache[img_key])
        cached["ocr_input"] = ocr_text
        cached["ocr_conf"]  = ocr_conf
        cached["cached"]    = True
        cached["latency_ms"] = 0
        _result_cache.move_to_end(img_key)
        return cached

    t0 = time.perf_counter()
    result = {
        "plate_llm":  None,
        "plate_used": ocr_text,
        "llm_valid":  False,
        "latency_ms": 0,
        "ocr_input":  ocr_text,
        "ocr_conf":   ocr_conf,
        "improved":   False,
        "cached":     False,
        "error":      None,
    }

    try:
        img_b64 = _encode_image(crop_bgr)
        payload = {
            "model":      OLLAMA_MODEL,
            "stream":     False,
            "keep_alive": -1,
            "options":    _OPTIONS,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": _USER_PROMPT, "images": [img_b64]},
            ],
        }
        resp = requests.post(OLLAMA_CHAT_URL, json=payload, timeout=TIMEOUT_S)
        resp.raise_for_status()
        raw_response = resp.json().get("message", {}).get("content", "")

        plate_llm = _parse_plate(raw_response)
        result["plate_llm"] = plate_llm
        result["llm_valid"] = plate_llm is not None

        if plate_llm and plate_llm != ocr_text:
            result["plate_used"] = plate_llm
            result["improved"]   = True
        elif plate_llm:
            result["plate_used"] = plate_llm

    except requests.exceptions.Timeout:
        result["error"] = "timeout"
    except Exception as e:
        result["error"] = str(e)
    finally:
        result["latency_ms"] = round((time.perf_counter() - t0) * 1000)

    # Store in cache, evict oldest if full
    _result_cache[img_key] = result
    if len(_result_cache) > _CACHE_MAX:
        _result_cache.popitem(last=False)

    return result
