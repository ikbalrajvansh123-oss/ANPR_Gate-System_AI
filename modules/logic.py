import time

class Logic:
    def __init__(self):
        # plate -> last status ("IN" or "OUT")
        self.status = {}

        # plate -> last seen time
        self.last_seen = {}

        # seconds gap for new event
        self.COOLDOWN = 5   # avoid rapid flicker

    def process(self, plate):
        now = time.time()

        # prevent too fast repeat (same frame spam)
        if plate in self.last_seen:
            if now - self.last_seen[plate] < self.COOLDOWN:
                return None

        self.last_seen[plate] = now

        # NEW PLATE → ENTRY
        if plate not in self.status:
            self.status[plate] = "IN"
            return "ENTRY"

        # TOGGLE LOGIC
        if self.status[plate] == "IN":
            self.status[plate] = "OUT"
            return ("EXIT",)

        else:
            self.status[plate] = "IN"
            return "ENTRY"