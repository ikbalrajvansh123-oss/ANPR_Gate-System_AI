import streamlit as st
import av
import cv2
import pandas as pd
import time

from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

from detector import Detector
from ocr import OCR
from database import DB
from utils import preprocess

st.title("📱 ANPR WebRTC System (ENTRY ONLY)")

detector = Detector("models/best.pt")
ocr = OCR()
db = DB()

# prevent duplicate entry within 10 sec
last_seen = {}

class VideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        img = preprocess(img)
        boxes = detector.detect(img)

        for i, (x1, y1, x2, y2) in enumerate(boxes):

            crop = img[y1:y2, x1:x2]

            raw = ocr.read(crop)
            if not raw:
                continue

            text = ocr.clean(raw)
            if not text:
                continue

            final = ocr.vote(i, text)
            if not final:
                continue

            now = time.time()

            # duplicate control
            if final in last_seen and now - last_seen[final] < 10:
                continue

            last_seen[final] = now

            dt = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            db.insert(final, dt)

            # draw
            cv2.rectangle(img, (x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(img, final, (x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0),2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ---------------- CAMERA ----------------
webrtc_streamer(
    key="anpr",
    video_processor_factory=VideoProcessor,
    media_stream_constraints={
        "video": True,
        "audio": False
    },
    async_processing=True
)

# ---------------- TABLE ----------------
st.subheader("📋 Entry Logs")

data = db.fetch()
df = pd.DataFrame(data, columns=["Plate", "DateTime"])
st.dataframe(df, use_container_width=True)

# ---------------- DELETE ----------------
if st.button("🗑 Delete All"):
    db.delete_all()
    st.success("All data deleted")