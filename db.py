"""SQLite persistence layer for Argus Intelligence."""
from __future__ import annotations

import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import pandas as pd

DB_PATH = Path("./data/argus.db")


def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS detections (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp     TEXT    NOT NULL,
                source        TEXT    NOT NULL DEFAULT 'image',
                vehicle_type  TEXT,
                plate_text    TEXT,
                ocr_confidence REAL,
                status        TEXT,
                image_file    TEXT
            );
            CREATE TABLE IF NOT EXISTS llm_benchmarks (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp     TEXT    NOT NULL,
                ocr_input     TEXT,
                ocr_conf      REAL,
                plate_llm     TEXT,
                plate_used    TEXT,
                llm_valid     INTEGER,
                improved      INTEGER,
                latency_ms    INTEGER,
                cached        INTEGER DEFAULT 0,
                error         TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_bench_ts ON llm_benchmarks(timestamp);
        """)
        # Migrate: add cached column if missing (for existing DBs)
        existing = {row[1] for row in conn.execute("PRAGMA table_info(llm_benchmarks)").fetchall()}
        if "cached" not in existing:
            conn.execute("ALTER TABLE llm_benchmarks ADD COLUMN cached INTEGER DEFAULT 0")
        # Migrate: add dwell tracking columns to detections if missing
        det_cols = {row[1] for row in conn.execute("PRAGMA table_info(detections)").fetchall()}
        if "first_seen" not in det_cols:
            conn.execute("ALTER TABLE detections ADD COLUMN first_seen TEXT")
        if "dwell_seconds" not in det_cols:
            conn.execute("ALTER TABLE detections ADD COLUMN dwell_seconds REAL")
        if "entry_timestamp" not in det_cols:
            conn.execute("ALTER TABLE detections ADD COLUMN entry_timestamp TEXT")
        if "exit_timestamp" not in det_cols:
            conn.execute("ALTER TABLE detections ADD COLUMN exit_timestamp TEXT")
        if "verified_ground_truth" not in det_cols:
            conn.execute("ALTER TABLE detections ADD COLUMN verified_ground_truth INTEGER DEFAULT 0")
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS watchlist (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                plate_text TEXT    UNIQUE NOT NULL,
                label      TEXT    DEFAULT '',
                added_at   TEXT    NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_det_plate ON detections(plate_text);
            CREATE INDEX IF NOT EXISTS idx_det_ts    ON detections(timestamp);
            CREATE INDEX IF NOT EXISTS idx_det_src   ON detections(source);
        """)


@contextmanager
def _conn():
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def insert_detections(rows: list[dict], source: str = "image") -> list[str]:
    """Persist detection rows; return list of plate texts that matched the watchlist."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    hits: list[str] = []
    with _conn() as conn:
        watchlist = {
            r[0] for r in conn.execute("SELECT plate_text FROM watchlist").fetchall()
        }
        for row in rows:
            plate = str(row.get("text") or "").strip()
            if not plate or plate in ("—", "nan"):
                continue
            if str(row.get("status", "")).upper() == "VERIFYING":
                continue  # already inserted via insert_detection_verifying
            ts = row.get("timestamp", now)
            conn.execute(
                "INSERT INTO detections "
                "(timestamp, entry_timestamp, source, vehicle_type, plate_text, "
                " ocr_confidence, status, image_file) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    ts,
                    ts,  # entry_timestamp = same as detection timestamp
                    source,
                    row.get("vehicle_type", "Unknown"),
                    plate,
                    round(float(row.get("score") or 0), 4),
                    row.get("status", "SAVED"),
                    row.get("image_file", ""),
                ),
            )
            if plate.upper() in watchlist and str(row.get("status", "")).upper() != "GUESSED":
                hits.append(plate)
    return hits


def insert_detection_verifying(row: dict, source: str = "video") -> int:
    """Insert a single VERIFYING detection. Returns new row ID."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ts = row.get("timestamp", now)
    with _conn() as conn:
        cur = conn.execute(
            "INSERT INTO detections "
            "(timestamp, entry_timestamp, source, vehicle_type, plate_text, "
            " ocr_confidence, status, image_file) "
            "VALUES (?, ?, ?, ?, ?, ?, 'VERIFYING', ?)",
            (
                ts, ts, source,
                row.get("vehicle_type", "Unknown"),
                str(row.get("text") or "").strip(),
                round(float(row.get("score") or 0), 4),
                row.get("image_file", ""),
            ),
        )
        return cur.lastrowid


def update_llm_result(detection_id: int, llm_plate: str | None, error: str | None = None) -> None:
    """Resolve a VERIFYING detection after LLM completes."""
    if detection_id is None:
        return
    with _conn() as conn:
        existing = conn.execute(
            "SELECT plate_text FROM detections WHERE id=?", (detection_id,)
        ).fetchone()
        if existing is None:
            return
        ocr_plate = existing["plate_text"]
        if llm_plate:
            clean = re.sub(r'\s+', '', llm_plate.upper().strip())
            if len(clean) == 8:
                formatted = clean[:4] + " " + clean[4:]
            elif len(clean) == 7:
                formatted = clean[:3] + " " + clean[3:]
            else:
                formatted = llm_plate.strip().upper()
            new_status = "CORRECTED" if formatted != ocr_plate else "SAVED"
            conn.execute(
                "UPDATE detections SET plate_text=?, status=? WHERE id=?",
                (formatted, new_status, detection_id),
            )
        else:
            conn.execute(
                "UPDATE detections SET status=? WHERE id=?",
                ("OCR_ONLY" if error else "SAVED", detection_id),
            )


def override_plate(detection_id: int, corrected_plate: str, operator_note: str = "") -> None:
    """Update a detection's plate_text and mark status as OVERRIDE."""
    with _conn() as conn:
        conn.execute(
            "UPDATE detections SET plate_text=?, status='OVERRIDE', verified_ground_truth=2, "
            "image_file=COALESCE(image_file||'|note:'||?, image_file) WHERE id=?",
            (corrected_plate.strip().upper(), operator_note, detection_id),
        )


def record_exit(plate_text: str, exit_ts: str, min_gap_minutes: float = 5.0) -> bool:
    """
    If plate was seen before and last detection is > min_gap_minutes ago,
    treat this as an exit event. Update exit_timestamp and dwell_seconds.
    Returns True if exit recorded.
    """
    plate_text = plate_text.strip().upper()
    with _conn() as conn:
        row = conn.execute(
            "SELECT id, timestamp FROM detections WHERE UPPER(plate_text)=? "
            "ORDER BY id DESC LIMIT 1",
            (plate_text,),
        ).fetchone()
        if row is None:
            return False
        last_ts_str = row["timestamp"]
        try:
            last_dt  = datetime.strptime(last_ts_str, "%Y-%m-%d %H:%M:%S")
            exit_dt  = datetime.strptime(exit_ts, "%Y-%m-%d %H:%M:%S")
            gap_mins = (exit_dt - last_dt).total_seconds() / 60.0
        except Exception:
            return False
        if gap_mins < min_gap_minutes:
            return False
        dwell = (exit_dt - last_dt).total_seconds()
        conn.execute(
            "UPDATE detections SET exit_timestamp=?, dwell_seconds=? WHERE id=?",
            (exit_ts, round(dwell, 2), row["id"]),
        )
        return True


def get_shift_report(date: str | None = None) -> dict:
    """
    Summary for a shift (default: today).
    Returns: total_vehicles, unique_plates, watchlist_hits, avg_dwell_seconds,
             peak_hour (str), first_detection, last_detection.
    """
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")
    with _conn() as conn:
        total_vehicles = conn.execute(
            "SELECT COUNT(*) FROM detections WHERE DATE(timestamp)=?", (date,)
        ).fetchone()[0]

        unique_plates = conn.execute(
            "SELECT COUNT(DISTINCT plate_text) FROM detections WHERE DATE(timestamp)=?", (date,)
        ).fetchone()[0]

        wl_plates = {r[0] for r in conn.execute("SELECT plate_text FROM watchlist").fetchall()}
        watchlist_hits = 0
        if wl_plates:
            placeholders = ",".join("?" * len(wl_plates))
            watchlist_hits = conn.execute(
                f"SELECT COUNT(*) FROM detections WHERE DATE(timestamp)=? "
                f"AND UPPER(plate_text) IN ({placeholders})",
                (date, *wl_plates),
            ).fetchone()[0]

        avg_dwell_row = conn.execute(
            "SELECT AVG(dwell_seconds) FROM detections "
            "WHERE DATE(timestamp)=? AND dwell_seconds IS NOT NULL",
            (date,),
        ).fetchone()
        avg_dwell_seconds = round(avg_dwell_row[0] or 0.0, 2)

        peak_row = conn.execute(
            "SELECT strftime('%H', timestamp) AS hr, COUNT(*) AS cnt "
            "FROM detections WHERE DATE(timestamp)=? "
            "GROUP BY hr ORDER BY cnt DESC LIMIT 1",
            (date,),
        ).fetchone()
        peak_hour = peak_row["hr"] if peak_row else None

        first_row = conn.execute(
            "SELECT MIN(timestamp) FROM detections WHERE DATE(timestamp)=?", (date,)
        ).fetchone()
        last_row = conn.execute(
            "SELECT MAX(timestamp) FROM detections WHERE DATE(timestamp)=?", (date,)
        ).fetchone()

    return {
        "total_vehicles":    total_vehicles,
        "unique_plates":     unique_plates,
        "watchlist_hits":    watchlist_hits,
        "avg_dwell_seconds": avg_dwell_seconds,
        "peak_hour":         peak_hour,
        "first_detection":   first_row[0] if first_row else None,
        "last_detection":    last_row[0] if last_row else None,
    }


def mark_ground_truth(detection_id: int, is_correct: bool) -> None:
    """Set verified_ground_truth=1 if correct, 2 if manually corrected."""
    with _conn() as conn:
        conn.execute(
            "UPDATE detections SET verified_ground_truth=? WHERE id=?",
            (1 if is_correct else 2, detection_id),
        )


def query_detections(
    status_filter: list[str] | None = None,
    source_filter: list[str] | None = None,
    plate_search: str = "",
    limit: int = 5000,
) -> pd.DataFrame:
    sql = (
        "SELECT id, timestamp, source, vehicle_type, plate_text, ocr_confidence, "
        "status, image_file, verified_ground_truth FROM detections WHERE 1=1"
    )
    params: list = []
    if status_filter:
        sql += f" AND status IN ({','.join('?'*len(status_filter))})"
        params.extend(status_filter)
    if source_filter:
        sql += f" AND source IN ({','.join('?'*len(source_filter))})"
        params.extend(source_filter)
    if plate_search:
        sql += " AND UPPER(plate_text) LIKE ?"
        params.append(f"%{plate_search.upper()}%")
    sql += " ORDER BY timestamp DESC LIMIT ?"
    params.append(limit)
    with _conn() as conn:
        rows = conn.execute(sql, params).fetchall()
    return pd.DataFrame([dict(r) for r in rows])


def get_stats() -> dict:
    with _conn() as conn:
        total    = conn.execute("SELECT COUNT(*) FROM detections").fetchone()[0]
        saved    = conn.execute(
            "SELECT COUNT(*) FROM detections WHERE status IN ('SAVED','CORRECTED','GUESSED')"
        ).fetchone()[0]
        fixed    = conn.execute(
            "SELECT COUNT(*) FROM detections WHERE status='CORRECTED'"
        ).fetchone()[0]
        guessed  = conn.execute(
            "SELECT COUNT(*) FROM detections WHERE status='GUESSED'"
        ).fetchone()[0]
        rejected = conn.execute(
            "SELECT COUNT(*) FROM detections WHERE status='REJECTED'"
        ).fetchone()[0]
    return {"total": total, "saved": saved, "fixed": fixed,
            "guessed": guessed, "rejected": rejected}


def clear_all_detections() -> None:
    with _conn() as conn:
        conn.execute("DELETE FROM detections")


# ── LLM Benchmarks ────────────────────────────────────────────────────────────

def insert_llm_benchmark(result: dict) -> None:
    """Persist one llm_verifier result dict to llm_benchmarks table."""
    from datetime import datetime
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with _conn() as conn:
        conn.execute(
            "INSERT INTO llm_benchmarks "
            "(timestamp, ocr_input, ocr_conf, plate_llm, plate_used, "
            " llm_valid, improved, latency_ms, cached, error) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                now,
                result.get("ocr_input"),
                result.get("ocr_conf"),
                result.get("plate_llm"),
                result.get("plate_used"),
                int(bool(result.get("llm_valid"))),
                int(bool(result.get("improved"))),
                result.get("latency_ms"),
                int(bool(result.get("cached"))),
                result.get("error"),
            ),
        )


def get_llm_benchmarks(limit: int = 500) -> pd.DataFrame:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT timestamp, ocr_input, ocr_conf, plate_llm, plate_used, "
            "llm_valid, improved, latency_ms, cached, error "
            "FROM llm_benchmarks ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return pd.DataFrame([dict(r) for r in rows])


def get_llm_benchmark_stats() -> dict:
    with _conn() as conn:
        total     = conn.execute("SELECT COUNT(*) FROM llm_benchmarks").fetchone()[0]
        valid     = conn.execute("SELECT COUNT(*) FROM llm_benchmarks WHERE llm_valid=1").fetchone()[0]
        improved  = conn.execute("SELECT COUNT(*) FROM llm_benchmarks WHERE improved=1").fetchone()[0]
        errors    = conn.execute("SELECT COUNT(*) FROM llm_benchmarks WHERE error IS NOT NULL").fetchone()[0]
        cached    = conn.execute("SELECT COUNT(*) FROM llm_benchmarks WHERE cached=1").fetchone()[0]
        avg_ms    = conn.execute(
            "SELECT AVG(latency_ms) FROM llm_benchmarks WHERE error IS NULL AND cached=0"
        ).fetchone()[0]
    return {
        "total":    total,
        "valid":    valid,
        "improved": improved,
        "errors":   errors,
        "cached":   cached,
        "avg_ms":   round(avg_ms or 0),
    }


def get_distinct_values(column: str) -> list:
    _allowed = {"status", "source", "vehicle_type"}
    if column not in _allowed:
        return []
    with _conn() as conn:
        rows = conn.execute(
            f"SELECT DISTINCT {column} FROM detections WHERE {column} IS NOT NULL ORDER BY {column}"
        ).fetchall()
    return [r[0] for r in rows]


# ── Watchlist ─────────────────────────────────────────────────────────────────

def get_watchlist() -> pd.DataFrame:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT plate_text, label, added_at FROM watchlist ORDER BY added_at DESC"
        ).fetchall()
    return pd.DataFrame([dict(r) for r in rows])


def add_to_watchlist(plate_text: str, label: str = "") -> bool:
    plate_text = plate_text.strip().upper()
    if not plate_text:
        return False
    try:
        with _conn() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO watchlist (plate_text, label, added_at) VALUES (?, ?, ?)",
                (plate_text, label.strip(), datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            )
        return True
    except Exception:
        return False


def remove_from_watchlist(plate_text: str) -> None:
    with _conn() as conn:
        conn.execute("DELETE FROM watchlist WHERE plate_text = ?",
                     (plate_text.strip().upper(),))


def update_detection_dwell(plate_text: str, first_seen: str, dwell_seconds: float) -> None:
    """Update first_seen and dwell_seconds for the most recent record matching plate_text."""
    plate_text = plate_text.strip().upper()
    with _conn() as conn:
        conn.execute(
            "UPDATE detections SET first_seen=?, dwell_seconds=? "
            "WHERE id=(SELECT MAX(id) FROM detections WHERE UPPER(plate_text)=?)",
            (first_seen, round(dwell_seconds, 2), plate_text),
        )


def check_watchlist(plates: list[str]) -> list[str]:
    """Return subset of plates present in the watchlist."""
    if not plates:
        return []
    with _conn() as conn:
        wl = {r[0] for r in conn.execute("SELECT plate_text FROM watchlist").fetchall()}
    return [p for p in plates if p.upper() in wl]
