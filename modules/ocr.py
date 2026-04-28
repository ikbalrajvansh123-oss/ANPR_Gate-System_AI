import easyocr
import re
import difflib

# ─────────────────────────────────────────────
#  Indian Number Plate format patterns
#  e.g.  MH12AB1234  |  DL3CAB1234  |  PB10CG0001
# ─────────────────────────────────────────────
INDIAN_PLATE_PATTERNS = [
    r'^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}$',   # Standard: MH12AB1234
    r'^[A-Z]{2}[0-9]{1,2}[A-Z]{1,2}[0-9]{4}$',   # 2-letter series: DL3CAB1234
    r'^[A-Z]{2}[0-9]{2}[A-Z]{2}[0-9]{4}$',        # Common: PB10CG0001
    r'^[A-Z]{2}[0-9]{1,2}[0-9]{4}$',              # Old format: DL 01 1234
]

# Minimum confidence score — 0.0 to 1.0
# Tune this: 0.6 = moderate, 0.75 = strict
MIN_CONFIDENCE = 0.50

# Minimum plate length after cleaning
MIN_PLATE_LEN = 6

# Similarity threshold for fuzzy dedup (0.0–1.0)
# If two plates are 85%+ similar they are treated as same plate
FUZZY_THRESHOLD = 0.85


class OCR:
    def __init__(self):
        self.reader = easyocr.Reader(['en'], gpu=False)
        self._recent_plates: list[str] = []   # short memory for dedup
        self._recent_limit = 20

    # ── raw read returning (text, confidence) ──
    def read_with_confidence(self, plate_img):
        """Returns (text, avg_confidence) or (None, 0)"""
        results = self.reader.readtext(plate_img)
        if not results:
            return None, 0.0

        # Concat all detected text regions, weighted by confidence
        texts, confidences = [], []
        for (_, text, conf) in results:
            texts.append(text.strip())
            confidences.append(conf)

        combined_text = ''.join(texts)
        avg_conf = sum(confidences) / len(confidences)
        return combined_text, avg_conf

    # ── main read (kept compatible with app.py) ──
    def read(self, plate_img):
        """Returns cleaned plate text if confident enough, else None."""
        raw, conf = self.read_with_confidence(plate_img)
        if not raw:
            return None

        # Low confidence → discard entirely
        if conf < MIN_CONFIDENCE:
            return None

        return raw   # cleaning done separately in clean()

    # ── cleaning & validation ──
    def clean(self, text: str) -> str | None:
        """
        1. Upper-case, remove spaces/special chars
        2. Common OCR character substitutions
        3. Regex format check
        Returns cleaned plate string or None if invalid.
        """
        if not text:
            return None

        text = text.upper().strip()

        # Remove everything except A-Z and 0-9
        text = re.sub(r'[^A-Z0-9]', '', text)

        # ── OCR confusion fixes ──
        # First 2 chars should be letters 
        fixed = list(text)
        for i in range(min(2, len(fixed))):
            fixed[i] = self._digit_to_letter(fixed[i])

        # Chars 2-4 are usually digits 
        for i in range(2, min(4, len(fixed))):
            fixed[i] = self._letter_to_digit(fixed[i])

        text = ''.join(fixed)

        if len(text) < MIN_PLATE_LEN:
            return None

        # ── Format validation ──
        if not self._is_valid_plate(text):
            return None

        return text

    # ── fuzzy duplicate check ──
    def is_duplicate(self, plate: str) -> bool:
        """
        Returns True if plate is very similar to a recently seen plate.
        Prevents DOVCE / DOVICE / DEVICE being logged as 3 separate entries.
        """
        for seen in self._recent_plates:
            ratio = difflib.SequenceMatcher(None, plate, seen).ratio()
            if ratio >= FUZZY_THRESHOLD:
                return True

        # Add to memory
        self._recent_plates.append(plate)
        if len(self._recent_plates) > self._recent_limit:
            self._recent_plates.pop(0)

        return False

    # ── helpers ──
    def _is_valid_plate(self, text: str) -> bool:
        for pattern in INDIAN_PLATE_PATTERNS:
            if re.match(pattern, text):
                return True
        return False

    @staticmethod
    def _digit_to_letter(ch: str) -> str:
        """Fix digits OCR'd in letter positions."""
        return {'0': 'O', '1': 'I', '5': 'S', '8': 'B'}.get(ch, ch)

    @staticmethod
    def _letter_to_digit(ch: str) -> str:
        """Fix letters OCR'd in digit positions."""
        return {'O': '0', 'I': '1', 'S': '5', 'B': '8', 'Z': '2',
                'G': '6', 'T': '7'}.get(ch, ch)