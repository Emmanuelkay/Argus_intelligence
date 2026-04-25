"""ARGUS Intelligence REST API — FastAPI in background thread."""
from __future__ import annotations

import threading
from datetime import datetime
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from db import (
    query_detections, get_watchlist, add_to_watchlist,
    remove_from_watchlist, get_stats,
)

# ── Pydantic schemas ──────────────────────────────────────────────────────────

class Detection(BaseModel):
    timestamp: str
    source: str
    vehicle_type: Optional[str] = None
    plate_text: Optional[str] = None
    ocr_confidence: Optional[float] = None
    status: Optional[str] = None
    image_file: Optional[str] = None

class WatchlistEntry(BaseModel):
    plate_text: str
    label: str = ""
    added_at: str

class WatchlistAddRequest(BaseModel):
    plate: str = Field(..., min_length=3, max_length=20, description="Plate text e.g. KDA 123B")
    label: str = Field("", max_length=100, description="Optional label e.g. 'Stolen vehicle'")

class WatchlistAddResponse(BaseModel):
    success: bool
    plate: str
    label: str
    message: str

class StatsResponse(BaseModel):
    total: int
    saved: int
    fixed: int
    guessed: int
    rejected: int

class DetectionListResponse(BaseModel):
    count: int
    detections: list[Detection]

class WatchlistResponse(BaseModel):
    count: int
    entries: list[WatchlistEntry]


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Argus Intelligence ALPR API",
    description=(
        "**Professional Kenyan License Plate Recognition & Watchlist Management System**\n\n"
        "Integrate with the Argus Intelligence live feed to query detections, "
        "manage the watchlist, and retrieve system statistics.\n\n"
        "Plates added via `POST /watchlist` are immediately active in the live Streamlit feed."
    ),
    version="2.0.0",
    contact={"name": "Argus Intelligence", "email": "aimoran2026@gmail.com"},
    license_info={"name": "Proprietary"},
)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get(
    "/detections",
    response_model=DetectionListResponse,
    summary="List detections",
    tags=["Detections"],
)
def get_detections(
    limit: int = Query(100, ge=1, le=5000, description="Max records to return"),
    plate_text: Optional[str] = Query(None, description="Filter by plate substring e.g. 'KDA'"),
    status: Optional[str] = Query(None, description="Filter by status: SAVED, CORRECTED, GUESSED, REJECTED"),
    source: Optional[str] = Query(None, description="Filter by source: image, video"),
    start_date: Optional[str] = Query(None, description="Start datetime filter (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS)"),
    end_date: Optional[str] = Query(None, description="End datetime filter (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS)"),
):
    status_filter = [status] if status else None
    source_filter = [source] if source else None

    df = query_detections(
        status_filter=status_filter,
        source_filter=source_filter,
        plate_search=plate_text or "",
        limit=limit,
    )

    if start_date:
        df = df[df["timestamp"] >= start_date]
    if end_date:
        df = df[df["timestamp"] <= end_date]

    records = df.where(df.notna(), other=None).to_dict(orient="records")
    return DetectionListResponse(count=len(records), detections=records)


@app.get(
    "/stats",
    response_model=StatsResponse,
    summary="Detection statistics",
    tags=["System"],
)
def get_detection_stats():
    """Aggregate counts: total, saved, corrected, guessed, rejected."""
    return get_stats()


@app.get(
    "/watchlist",
    response_model=WatchlistResponse,
    summary="List watchlist plates",
    tags=["Watchlist"],
)
def list_watchlist():
    """Return all plates currently on the watchlist."""
    df = get_watchlist()
    records = df.where(df.notna(), other=None).to_dict(orient="records")
    return WatchlistResponse(count=len(records), entries=records)


@app.post(
    "/watchlist",
    response_model=WatchlistAddResponse,
    summary="Add plate to watchlist",
    tags=["Watchlist"],
    status_code=201,
)
def add_to_watchlist_endpoint(body: WatchlistAddRequest):
    """
    Add a plate to the watchlist. Takes effect immediately in the live Streamlit feed —
    the next detection of this plate will trigger an alert and webhook notification.
    """
    ok = add_to_watchlist(body.plate, body.label)
    if not ok:
        raise HTTPException(status_code=400, detail=f"Could not add plate '{body.plate}' — may already exist or be invalid.")
    return WatchlistAddResponse(
        success=True,
        plate=body.plate.strip().upper(),
        label=body.label,
        message="Plate added. Live feed will alert on next detection.",
    )


@app.delete(
    "/watchlist/{plate_text}",
    summary="Remove plate from watchlist",
    tags=["Watchlist"],
)
def remove_from_watchlist_endpoint(plate_text: str):
    """Remove a plate from the watchlist. URL-encode spaces as %20 (e.g. KDA%20123B)."""
    remove_from_watchlist(plate_text)
    return {"success": True, "plate": plate_text.strip().upper(), "message": "Plate removed from watchlist."}


@app.get(
    "/health",
    summary="Health check",
    tags=["System"],
)
def health():
    """Returns API status and current timestamp."""
    return {"status": "ok", "timestamp": datetime.now().isoformat(), "version": "2.0.0"}


# ── Runner ────────────────────────────────────────────────────────────────────

def start_api(port: int = 8502) -> None:
    try:
        uvicorn.run(app, host="0.0.0.0", port=port, log_level="error")
    except Exception:
        pass


def start_api_background(port: int = 8502) -> None:
    threading.Thread(target=start_api, args=(port,), daemon=True).start()
