import easyocr
import re
from collections import defaultdict

class OCR:
    def __init__(self):
        self.reader = easyocr.Reader(['en'], gpu=False)
        self.memory = defaultdict(list)

    def read(self, img):
        result = self.reader.readtext(img)
        if not result:
            return None
        return ''.join([r[1] for r in result])

    def clean(self, text):
        text = text.upper()
        text = re.sub(r'[^A-Z0-9]', '', text)
        if len(text) < 6:
            return None
        return text

    def vote(self, tid, text):
        self.memory[tid].append(text)
        if len(self.memory[tid]) > 5:
            self.memory[tid].pop(0)

        votes = self.memory[tid]
        final = max(set(votes), key=votes.count)

        if votes.count(final) >= 3:
            return final
        return None