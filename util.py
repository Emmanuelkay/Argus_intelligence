import re
import difflib

# ---------------------------------------------------------------------------
# Kenyan plate formats:
#   Civilian:    K[A-Z]{2} \d{3}[A-Z]          e.g. KDA 123B   (7 chars)
#   Tuk Tuk:     KT[A-Z]{2} \d{3}[A-Z]         e.g. KTWA 123B  (8 chars, KT + any 2 letters)
#   Motorcycle:  KMC[A-Z] \d{3}[A-Z]           e.g. KMCA 123B  (8 chars)
# ---------------------------------------------------------------------------

_KE_PLATE_RE   = re.compile(r'^K[A-Z]{2} \d{3}[A-Z]$')        # civilian  (with space)
_KE_TUKTUK_RE  = re.compile(r'^KT[A-Z]{2} \d{3}[A-Z]$')       # tuk-tuk   (with space)
_KE_MOTO_RE    = re.compile(r'^KMC[A-Z] \d{3}[A-Z]$')         # motorcycle (with space)

_SPECIAL_PREFIXES = {"KT", "KMC"}   # 8-char plate prefixes (KT covers all tuk-tuk series)

_LETTER_POSITIONS   = {0, 1, 2, 6}          # civilian 7-char letter positions (no space)
_DIGIT_POSITIONS    = {3, 4, 5}             # civilian 7-char digit positions
_LETTER_POSITIONS_8 = {0, 1, 2, 3, 7}      # 8-char plate letter positions
_DIGIT_POSITIONS_8  = {4, 5, 6}            # 8-char plate digit positions

# Legacy maps (used by format_kenyan_plate / guess_kenyan_plate)
_DIGIT_TO_LETTER = {
    "0": "O", "1": "I", "2": "Z", "3": "J",
    "4": "A", "5": "S", "6": "G", "8": "B",
}
_LETTER_TO_DIGIT = {
    "O": "0", "I": "1", "Z": "2", "J": "3",
    "A": "4", "S": "5", "G": "6", "B": "8",
}
_SYMBOL_TO_LETTER = {
    "|": "I", "!": "I", "(": "C", "[": "C", "{": "C",
    ")": "D", "]": "D", "#": "H", "@": "A", "&": "B",
}
_SYMBOL_TO_DIGIT = {
    "?": "7", "@": "0", "!": "1", "|": "1",
    "$": "5", "&": "8", "#": "4",
}

# ---------------------------------------------------------------------------
# Positional correction maps (includes 7→T which serifs make ambiguous)
# ---------------------------------------------------------------------------
_LETTER_FIXES = {"0": "O", "1": "I", "2": "Z", "5": "S", "6": "G", "7": "T", "8": "B"}
_DIGIT_FIXES  = {"B": "8", "G": "6", "I": "1", "O": "0", "S": "5", "T": "7", "Z": "2"}
# O and I are banned at letter positions on Kenyan plates; remap to visual lookalikes
_BANNED_LETTER_FIXES = {"O": "D", "I": "J"}
_POS_EXPECT   = ["L", "L", "L", "D", "D", "D", "L"]          # 7-char: L=letter D=digit
_POS_EXPECT_8 = ["L", "L", "L", "L", "D", "D", "D", "L"]    # 8-char: first 4 = letters

_SUB_PENALTY = 0.06


def get_plate_category(plate: str) -> str:
    """Return human-readable vehicle category from plate prefix."""
    clean = re.sub(r'\s+', '', (plate or "").upper())
    if clean.startswith("KT") and len(clean) >= 4 and clean[2].isalpha() and clean[3].isalpha():
        return "Tuk Tuk"
    if clean.startswith("KMC"):
        return "Motorcycle"
    if clean.startswith("K"):
        return "Civilian"
    return "Unknown"


def _correct_char(ch: str, position: int,
                  letter_pos: set = _LETTER_POSITIONS) -> tuple[str, int]:
    if position in letter_pos:
        if ch.isalpha():
            # O and I are banned at letter positions — remap to visual lookalike
            if ch in _BANNED_LETTER_FIXES:
                return _BANNED_LETTER_FIXES[ch], 1
            return ch, 0
        new = _DIGIT_TO_LETTER.get(ch) or _SYMBOL_TO_LETTER.get(ch)
        if new and new in _BANNED_LETTER_FIXES:
            new = _BANNED_LETTER_FIXES[new]
        return (new, 1) if new else ("", 0)
    else:
        if ch.isdigit():
            return ch, 0
        new = _LETTER_TO_DIGIT.get(ch) or _SYMBOL_TO_DIGIT.get(ch)
        return (new, 1) if new else ("", 0)


def _apply_pos_fixes(chars: list, pos_expect: list) -> int:
    """Apply letter/digit positional fixes in-place. Returns number of changes."""
    changes = 0
    for i, (ch, expect) in enumerate(zip(chars, pos_expect)):
        if expect == "L":
            if ch.isdigit():
                fix = _LETTER_FIXES.get(ch)
                if fix:
                    # If the digit-to-letter mapping lands on a banned letter, remap again
                    fix = _BANNED_LETTER_FIXES.get(fix, fix)
                    chars[i] = fix
                    changes += 1
            elif ch in _BANNED_LETTER_FIXES:
                # O or I at a letter position — remap to visual lookalike
                chars[i] = _BANNED_LETTER_FIXES[ch]
                changes += 1
        elif expect == "D" and ch.isalpha():
            fix = _DIGIT_FIXES.get(ch)
            if fix:
                chars[i] = fix
                changes += 1
    return changes


def correct_plate(raw: str) -> tuple[str, str, int]:
    """
    Positional OCR correction for civilian (7-char) and special (8-char) Kenyan plates.

    Returns (corrected_plate, status, n_changes).
    status: "valid" | "corrected" | "guessed" | "rejected"
    """
    clean = re.sub(r'\s+', '', raw.upper().strip())

    if len(clean) == 7:
        chars = list(clean)
        changes = _apply_pos_fixes(chars, _POS_EXPECT)
        result = "".join(chars)
        if result[0] != "K":
            return raw, "rejected", 0
        formatted = result[:3] + " " + result[3:]
        if _KE_PLATE_RE.match(formatted):
            if changes == 0:   return formatted, "valid", 0
            if changes <= 2:   return formatted, "corrected", changes
            return formatted, "guessed", changes
        return raw, "rejected", 0

    if len(clean) == 8:
        chars = list(clean)
        changes = _apply_pos_fixes(chars, _POS_EXPECT_8)
        result = "".join(chars)
        formatted = result[:4] + " " + result[4:]
        if _KE_TUKTUK_RE.match(formatted) or _KE_MOTO_RE.match(formatted):
            if changes == 0:   return formatted, "valid", 0
            if changes <= 2:   return formatted, "corrected", changes
            return formatted, "guessed", changes
        return raw, "rejected", 0

    return raw, "rejected", 0


def format_kenyan_plate(text: str) -> str | None:
    raw = re.sub(r'\s+', '', text.upper())
    if len(raw) == 7:
        corrected = []
        for i, ch in enumerate(raw):
            fixed, _ = _correct_char(ch, i, _LETTER_POSITIONS)
            if not fixed:
                return None
            corrected.append(fixed)
        if corrected[0] != "K":
            return None
        formatted = f"{''.join(corrected[:3])} {''.join(corrected[3:6])}{corrected[6]}"
        return formatted if _KE_PLATE_RE.match(formatted) else None
    if len(raw) == 8:
        corrected = []
        for i, ch in enumerate(raw):
            fixed, _ = _correct_char(ch, i, _LETTER_POSITIONS_8)
            if not fixed:
                return None
            corrected.append(fixed)
        formatted = f"{''.join(corrected[:4])} {''.join(corrected[4:7])}{corrected[7]}"
        return formatted if (_KE_TUKTUK_RE.match(formatted) or _KE_MOTO_RE.match(formatted)) else None
    return None


def guess_kenyan_plate(text: str) -> tuple[str | None, int]:
    raw = re.sub(r'\s+', '', text.upper())
    best_plate, best_subs = None, 999

    # Try 7-char windows (civilian)
    if len(raw) >= 7:
        for window in [raw[i:i+7] for i in range(len(raw) - 6)]:
            corrected, total_subs, ok = [], 0, True
            for i, ch in enumerate(window):
                if i == 0:
                    corrected.append("K"); total_subs += 0 if ch == "K" else 1; continue
                fixed, subs = _correct_char(ch, i, _LETTER_POSITIONS)
                if not fixed: ok = False; break
                corrected.append(fixed); total_subs += subs
            if not ok: continue
            formatted = f"{''.join(corrected[:3])} {''.join(corrected[3:6])}{corrected[6]}"
            if _KE_PLATE_RE.match(formatted) and total_subs < best_subs:
                best_subs, best_plate = total_subs, formatted

    # Try 8-char windows (tuk-tuk / motorcycle)
    if len(raw) >= 8:
        for window in [raw[i:i+8] for i in range(len(raw) - 7)]:
            corrected, total_subs, ok = [], 0, True
            for i, ch in enumerate(window):
                if i == 0:
                    corrected.append("K"); total_subs += 0 if ch == "K" else 1; continue
                fixed, subs = _correct_char(ch, i, _LETTER_POSITIONS_8)
                if not fixed: ok = False; break
                corrected.append(fixed); total_subs += subs
            if not ok: continue
            formatted = f"{''.join(corrected[:4])} {''.join(corrected[4:7])}{corrected[7]}"
            if (_KE_TUKTUK_RE.match(formatted) or _KE_MOTO_RE.match(formatted)) and total_subs < best_subs:
                best_subs, best_plate = total_subs, formatted

    return (best_plate, best_subs) if best_plate else (None, 0)


def is_valid_kenyan_plate(text: str) -> bool:
    if not text:
        return False
    return bool(
        _KE_PLATE_RE.match(text) or
        _KE_TUKTUK_RE.match(text) or
        _KE_MOTO_RE.match(text)
    )


def vote_plate(plates: list[str]) -> str:
    """
    Character-level majority vote across multiple OCR reads of the same plate.
    e.g. ["KCL 033T", "KCL 0337", "KCL 033T"] → "KCL 033T"
    """
    from collections import Counter
    clean = [re.sub(r'\s+', '', p.upper()) for p in plates if p and p.strip()]
    if not clean:
        return plates[0] if plates else ""
    target_len = Counter(len(v) for v in clean).most_common(1)[0][0]
    clean = [v for v in clean if len(v) == target_len]
    if not clean:
        return plates[0]
    voted = "".join(
        Counter(v[i] for v in clean if i < len(v)).most_common(1)[0][0]
        for i in range(target_len)
    )
    if len(voted) == 7:
        return voted[:3] + " " + voted[3:]
    if len(voted) == 8:
        return voted[:4] + " " + voted[4:]
    return voted


def validate_kenyan_series(plate: str) -> dict:
    """
    Validate a Kenyan civilian plate against series rules.

    Returns:
        {
          "valid": bool,
          "uncertainty": str,   # "none" | "high" | "flagged"
          "flags": list[str],
          "corrected": str|None,
        }
    """
    clean = re.sub(r'\s+', '', plate.upper().strip())

    valid       = True
    uncertainty = "none"
    flags: list[str] = []
    corrected_chars: list[str] | None = None

    # Accept 7-char (civilian) or 8-char (tuk-tuk/motorcycle)
    if len(clean) not in (7, 8):
        return {"valid": False, "uncertainty": "flagged",
                "flags": ["wrong_length"], "corrected": None}

    chars = list(clean)
    is_8char = len(clean) == 8
    prefix2  = clean[:2]  # "KT" = tuk-tuk, "KM" = motorcycle prefix start

    # Rule A: first char must be 'K'
    if chars[0] != 'K':
        flags.append("not_civilian")
        valid = False

    # Rule B: series constraint — only for civilian plates (not KT.../KMC)
    # Tuk-tuk (KT**) and motorcycle (KMC*) series letters are unconstrained
    if not is_8char:
        if len(chars) > 1 and chars[1].isalpha() and chars[1] > 'D':
            flags.append("series_beyond_D")
            uncertainty = "high"

    # Rule C: letter positions must NOT be 'I' or 'O'
    # Civilian: positions 0,1,2,6 | 8-char: positions 0,1,2,3,7
    _letter_pos_no_IO = _LETTER_POSITIONS_8 if is_8char else {0, 1, 2, 6}
    for pos in _letter_pos_no_IO:
        if pos < len(chars) and chars[pos] in ('I', 'O'):
            flags.append("excluded_letter")
            valid = False
            break

    # Rule D: positional forcing
    # Digit positions (civilian 3,4,5 / 8-char 4,5,6): 'B' → '8'
    # Letter positions: '0' cannot be forced to 'O' — O is excluded by Rule C.
    # Flag as invalid with digit_at_letter_pos if '0' appears at a letter position.
    _forced = False
    working = chars[:]
    _digit_pos = _DIGIT_POSITIONS_8 if is_8char else {3, 4, 5}
    _letter_pos_all = _LETTER_POSITIONS_8 if is_8char else {0, 1, 2, 6}

    # Letter positions with '0' → cannot recover, flag invalid
    for pos in _letter_pos_all:
        if pos < len(working) and working[pos] == '0':
            if "digit_at_letter_pos" not in flags:
                flags.append("digit_at_letter_pos")
            valid = False

    # Digit positions with 'B' → '8' (unambiguous)
    for pos in _digit_pos:
        if pos < len(working) and working[pos] == 'B':
            working[pos] = '8'
            if "forced_B_to_8" not in flags:
                flags.append("forced_B_to_8")
            _forced = True

    if _forced:
        if is_8char:
            forced_str = (working[0] + working[1] + working[2] + working[3] + " " +
                          working[4] + working[5] + working[6] + working[7])
        else:
            forced_str = (working[0] + working[1] + working[2] + " " +
                          working[3] + working[4] + working[5] + working[6])
        _matched = (_KE_TUKTUK_RE.match(forced_str) or _KE_MOTO_RE.match(forced_str)
                    if is_8char else _KE_PLATE_RE.match(forced_str))
        if _matched:
            forced_clean = re.sub(r'\s+', '', forced_str)
            _rc_ok = all(
                forced_clean[pos] not in ('I', 'O')
                for pos in _letter_pos_no_IO if pos < len(forced_clean)
            )
            if _rc_ok and forced_clean[0] == 'K' and valid:
                corrected_chars = forced_str

    if uncertainty == "none" and flags:
        uncertainty = "flagged"

    return {
        "valid":       valid,
        "uncertainty": uncertainty,
        "flags":       flags,
        "corrected":   corrected_chars,
    }


def process_alpr_text(raw_text: str, confidence: float) -> tuple[str | None, float, int, str]:
    """
    Apply Kenyan format correction on Fast-ALPR output.

    Returns (corrected_plate, final_confidence, n_substitutions, status).
    status: "valid" | "corrected" | "guessed" | "rejected"
      valid     — exact match, confidence unchanged
      corrected — 1–2 char fixes, confidence unchanged
      guessed   — 3+ fixes or sliding-window match, confidence penalised
      rejected  — no valid plate could be produced
    """
    if not raw_text:
        return None, 0.0, 0, "rejected"

    text = raw_text.upper().strip()

    # Primary: positional correction on exactly 7 chars
    plate, status, n_changes = correct_plate(text)
    if status != "rejected":
        if status == "guessed":
            penalised = max(0.0, confidence - n_changes * _SUB_PENALTY)
            plate, confidence, n_changes, status = plate, penalised, n_changes, status

        # Apply Kenyan series validation; downgrade status if needed
        if plate:
            _series = validate_kenyan_series(plate)
            if not _series["valid"] and _series.get("corrected"):
                # Rule D auto-corrected it
                plate  = _series["corrected"]
                status = "corrected"
            elif _series["uncertainty"] == "high":
                status = "guessed"
                confidence = max(0.0, confidence - _SUB_PENALTY)
            elif not _series["valid"]:
                status = "rejected"
                return None, 0.0, 0, "rejected"

        return plate, confidence, n_changes, status

    # Fallback: sliding window for longer OCR strings
    guessed, n_subs = guess_kenyan_plate(text)
    if guessed:
        penalised = max(0.0, confidence - n_subs * _SUB_PENALTY)
        g_status  = "guessed" if n_subs > 0 else "valid"

        # Series validation on sliding-window result
        _series = validate_kenyan_series(guessed)
        if not _series["valid"] and _series.get("corrected"):
            guessed   = _series["corrected"]
            g_status  = "corrected"
        elif _series["uncertainty"] == "high":
            g_status  = "guessed"
        elif not _series["valid"]:
            return None, 0.0, 0, "rejected"

        return guessed, penalised, n_subs, g_status

    return None, 0.0, 0, "rejected"


# ---------------------------------------------------------------------------
# Plate deduplication across video frames
# ---------------------------------------------------------------------------

class PlateDeduplicator:
    """
    One saved record per unique plate per vehicle visit.
    Fuzzy matching catches OCR inconsistency (KCB835U vs KCB8350).
    Cooldown window suppresses repeated reads of the same vehicle.
    Re-entry gap treats the same plate reappearing after a long absence
    as a new vehicle visit and saves it again.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.82,
        cooldown_seconds: float = 30.0,
        min_confidence: float = 0.20,
        time_gap_for_reentry: float = 120.0,
    ):
        self.sim_thresh = similarity_threshold
        self.cooldown   = cooldown_seconds
        self.min_conf   = min_confidence
        self.reentry_gap = time_gap_for_reentry
        self._records: dict = {}

    def _normalize(self, text: str) -> str:
        return re.sub(r'\s+', '', text.strip()).upper()

    def _find_match(self, normalized: str) -> str | None:
        candidates = [
            (difflib.SequenceMatcher(None, k, normalized).ratio(), k)
            for k in self._records
        ]
        valid = [(r, k) for r, k in candidates if r >= self.sim_thresh]
        return max(valid)[1] if valid else None

    def process(
        self, plate_text: str, confidence: float, _frame: int, timestamp_s: float
    ) -> tuple[str, str | None, float | None]:
        """
        Returns (status, best_text, best_conf).
        status: "new" | "duplicate" | "reentry" | "low_conf"
        """
        if confidence < self.min_conf:
            return "low_conf", None, None

        normalized = self._normalize(plate_text)
        match_key  = self._find_match(normalized)

        if match_key is None:
            self._records[normalized] = {
                "last_seen":   timestamp_s,
                "first_seen":  timestamp_s,
                "best_conf":   confidence,
                "best_text":   plate_text,
                "variants":    [plate_text],
                "entry_count": 1,
            }
            return "new", plate_text, confidence

        rec        = self._records[match_key]
        time_since = timestamp_s - rec["last_seen"]

        if confidence > rec["best_conf"]:
            rec["best_conf"] = confidence
            rec["best_text"] = plate_text
        if plate_text not in rec["variants"]:
            rec["variants"].append(plate_text)
        rec["last_seen"] = timestamp_s

        if time_since >= self.reentry_gap:
            rec["entry_count"] += 1
            rec["first_seen"] = timestamp_s  # reset first_seen on re-entry
            return "reentry", rec["best_text"], rec["best_conf"]

        return "duplicate", rec["best_text"], rec["best_conf"]

    def get_dwell_seconds(self, plate_text: str, current_ts: float) -> float:
        """Return how long (seconds) a plate has been visible since first_seen."""
        normalized = self._normalize(plate_text)
        match_key  = self._find_match(normalized)
        if match_key is None:
            return 0.0
        return max(0.0, current_ts - self._records[match_key].get("first_seen", current_ts))

    def get_first_seen_str(self, plate_text: str) -> str | None:
        """Return ISO timestamp string of first_seen for a plate (seconds offset as string)."""
        normalized = self._normalize(plate_text)
        match_key  = self._find_match(normalized)
        if match_key is None:
            return None
        return str(self._records[match_key].get("first_seen", ""))

    def get_voted_text(self, plate_text: str) -> str:
        """Return character-voted consensus text for a plate, or original if only one read."""
        normalized = self._normalize(plate_text)
        match_key  = self._find_match(normalized)
        if match_key is None or len(self._records[match_key]["variants"]) <= 1:
            return plate_text
        return vote_plate(self._records[match_key]["variants"])

    def get_final_results(self) -> dict:
        records = list(self._records.values())
        return {
            "unique_plates": len(records),
            "total_visits":  sum(r["entry_count"] for r in records),
            "records": [
                {
                    "plate":      vote_plate(r["variants"]) if len(r["variants"]) > 1 else r["best_text"],
                    "confidence": r["best_conf"],
                    "variants":   r["variants"],
                    "visits":     r["entry_count"],
                }
                for r in records
            ],
        }
