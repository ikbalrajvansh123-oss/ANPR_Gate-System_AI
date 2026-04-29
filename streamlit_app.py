import streamlit as st
import av
import cv2
import pandas as pd
import time
import threading

from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

from detector import Detector
from ocr import OCR
from database import DB
from utils import preprocess

st.title("📱 Stable ANPR System (Crash-Free)")

# ---------------- INIT ONLY ONCE ----------------
@st.cache_resource
def load_models():
    detector = Detector("models/best.pt")
    ocr = OCR()
    db = DB()
    return detector, ocr, db

detector, ocr, db = load_models()

# ---------------- GLOBAL SAFE STORAGE ----------------
last_seen = {}
lock = threading.Lock()

# ---------------- PROCESSOR ----------------
class VideoProcessor(VideoProcessorBase):

    def __init__(self):
        self.frame_count = 0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.frame_count += 1

        # 🔥 PROCESS ONLY EVERY 5th FRAME (VERY IMPORTANT)
        if self.frame_count % 5 != 0:
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        img = preprocess(img)

        try:
            boxes = detector.detect(img)
        except:
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        for i, (x1, y1, x2, y2) in enumerate(boxes):

            # safe bounding box
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)

            if x2 <= x1 or y2 <= y1:
                continue

            crop = img[y1:y2, x1:x2]

            try:
                raw = ocr.read(crop)
                text = ocr.clean(raw)
                final = ocr.vote(i, text)
            except:
                continue

            if not final:
                continue

            now = time.time()

            # 🔥 THREAD SAFE DUPLICATE CHECK
            with lock:
                if final in last_seen and now - last_seen[final] < 10:
                    continue
                last_seen[final] = now

            dt = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")

            # 🔥 NON-BLOCKING DB INSERT
            try:
                threading.Thread(target=db.insert, args=(final, dt)).start()
            except:
                pass

            # draw box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, final, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ---------------- WEBCAM ----------------
webrtc_streamer(
    key="stable-anpr",
    video_processor_factory=VideoProcessor,
    media_stream_constraints={
        "video": True,
        "audio": False
    },
    async_processing=True,
)

# ---------------- DATABASE UI ----------------
st.subheader("📋 Entry Logs")

data = db.fetch()
df = pd.DataFrame(data, columns=["Plate", "DateTime"])
st.dataframe(df, use_container_width=True)

# ---------------- CLEAR DATA ----------------
if st.button("🗑 Clear Database"):
    db.delete_all()
    st.success("Database cleared")