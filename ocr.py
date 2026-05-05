import easyocr
import re
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class OCR:
    def __init__(self):
        try:
            self.reader = easyocr.Reader(['en'], gpu=False)
            logger.info("✅ EasyOCR initialized")
        except Exception as e:
            logger.error(f"❌ OCR init failed: {e}")
            raise
        
        self.memory = defaultdict(list)

    def read(self, img) -> str | None:
        """Read text from image crop"""
        try:
            if img is None or img.size == 0:
                return None
            
            result = self.reader.readtext(img)
            if not result:
                return None
            
            # Join all detected text
            text = ''.join([r[1] for r in result])
            return text if text else None
            
        except Exception as e:
            logger.debug(f"OCR read error: {e}")
            return None

    def clean(self, text: str) -> str | None:
        """Clean and validate plate text"""
        if not text:
            return None
        
        # Convert to uppercase
        text = text.upper().strip()
        
        # Remove special characters, keep only alphanumeric
        text = re.sub(r'[^A-Z0-9]', '', text)
        
        # Indian plates: typically 6-10 chars (DL01AB1234, etc)
        if len(text) < 6 or len(text) > 10:
            logger.debug(f"Text '{text}' too short/long, rejected")
            return None
        
        # Should have both letters and numbers
        has_alpha = any(c.isalpha() for c in text)
        has_digit = any(c.isdigit() for c in text)
        
        if not (has_alpha and has_digit):
            logger.debug(f"Text '{text}' missing letters or digits")
            return None
        
        return text

    def vote(self, tid: int, text: str) -> str | None:
        """
        Voting system: need 3 out of last 5 reads to match
        This prevents false positives from single bad reads
        """
        if not text:
            return None
        
        mem = self.memory[tid]
        mem.append(text)
        
        # Keep only last 5 reads
        if len(mem) > 5:
            mem.pop(0)
        
        # Count occurrences
        from collections import Counter
        counts = Counter(mem)
        most_common = counts.most_common(1)[0]
        candidate, count = most_common
        
        # Need at least 3 matches out of 5
        if count >= 3:
            logger.info(f"✅ Plate CONFIRMED: {candidate} ({count}/5 votes)")
            return candidate
        
        logger.debug(f"Plate '{text}' - {count}/5 votes (need 3), history: {mem}")
        return None