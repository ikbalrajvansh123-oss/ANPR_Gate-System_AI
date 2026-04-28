from deep_sort_realtime.deepsort_tracker import DeepSort

class Tracker:
    def __init__(self):
        self.tracker = DeepSort(max_age=30)

    def update(self, detections, frame):
        ds_input = []

        for x1, y1, x2, y2, conf in detections:
            w, h = x2 - x1, y2 - y1
            ds_input.append(([x1, y1, w, h], conf, "car"))

        tracks = self.tracker.update_tracks(ds_input, frame=frame)

        results = []
        for t in tracks:
            if not t.is_confirmed():
                continue

            x1, y1, x2, y2 = map(int, t.to_ltrb())
            results.append((x1, y1, x2, y2, t.track_id))

        return results