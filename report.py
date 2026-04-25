"""PDF incident report generator for ARGUS Intelligence."""
from __future__ import annotations

from datetime import datetime
from io import BytesIO

import pandas as pd


def generate_incident_pdf(plate_text: str, detections_df: pd.DataFrame) -> bytes:
    """
    Generate a PDF incident report for a given plate.

    Parameters
    ----------
    plate_text : str
        The plate number being reported.
    detections_df : pd.DataFrame
        All detections for that plate (timestamp, vehicle_type, ocr_confidence, status, …).

    Returns
    -------
    bytes
        Raw PDF bytes suitable for st.download_button.
    """
    from fpdf import FPDF

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    # ── Header ──────────────────────────────────────────────────────────────
    pdf.set_fill_color(30, 58, 95)   # dark blue
    pdf.rect(0, 0, 210, 30, "F")
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(255, 255, 255)
    pdf.set_xy(10, 8)
    pdf.cell(0, 12, "ARGUS INTELLIGENCE -- Incident Report", ln=True)

    # ── Metadata ────────────────────────────────────────────────────────────
    pdf.set_xy(10, 36)
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(30, 58, 95)
    pdf.cell(0, 8, f"Plate: {plate_text.upper()}", ln=True)

    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 6, f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.cell(0, 6, f"Total detections: {len(detections_df)}", ln=True)
    pdf.ln(4)

    # ── Table header ────────────────────────────────────────────────────────
    _cols = ["timestamp", "vehicle_type", "ocr_confidence", "status", "source"]
    _labels = ["Timestamp", "Vehicle Type", "Confidence", "Status", "Source"]
    _widths = [52, 36, 28, 26, 28]

    pdf.set_fill_color(37, 99, 235)   # blue
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 9)

    for lbl, w in zip(_labels, _widths):
        pdf.cell(w, 8, lbl, border=1, fill=True, align="C")
    pdf.ln()

    # ── Table rows ──────────────────────────────────────────────────────────
    pdf.set_font("Helvetica", "", 8)
    fill = False
    for _, row in detections_df.head(200).iterrows():
        pdf.set_fill_color(239, 246, 255) if fill else pdf.set_fill_color(255, 255, 255)
        pdf.set_text_color(30, 30, 30)
        for col, w in zip(_cols, _widths):
            val = row.get(col, "")
            if val is None or (isinstance(val, float) and __import__("math").isnan(val)):
                val = "—"
            elif col == "ocr_confidence":
                try:
                    val = f"{float(val):.0%}"
                except Exception:
                    val = str(val)
            else:
                val = str(val)
            pdf.cell(w, 7, val[:22], border=1, fill=True, align="C")
        pdf.ln()
        fill = not fill

    # ── Footer ──────────────────────────────────────────────────────────────
    pdf.set_y(-20)
    pdf.set_fill_color(30, 58, 95)
    pdf.rect(0, pdf.get_y(), 210, 20, "F")
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 10, "CONFIDENTIAL — ARGUS INTELLIGENCE — For Authorised Use Only", align="C")

    return pdf.output()
