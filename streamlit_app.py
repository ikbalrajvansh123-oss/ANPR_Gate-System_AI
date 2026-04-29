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

st.title("📱 ANPR WebRTC System")

detector = Detector("models/best.pt")
ocr = OCR()
db = DB()

last_seen = {}
frame_skip = 0   # 🔥 ADD THIS

class VideoProcessor(VideoProcessorBase):

    def recv(self, frame):
        global frame_skip

        img = frame.to_ndarray(format="bgr24")

        frame_skip += 1

        # 🔥 PROCESS ONLY EVERY 3rd FRAME
        if frame_skip % 3 != 0:
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        img = preprocess(img)
        boxes = detector.detect(img)

        for i, (x1, y1, x2, y2) in enumerate(boxes):

            # safety crop fix
            x1, y1 = max(0,x1), max(0,y1)
            x2, y2 = min(img.shape[1],x2), min(img.shape[0],y2)

            if x2 <= x1 or y2 <= y1:
                continue

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

            # 🔥 duplicate control safe
            if final in last_seen and now - last_seen[final] < 10:
                continue

            last_seen[final] = now

            dt = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")

            # 🔥 avoid blocking crash (IMPORTANT)
            try:
                db.insert(final, dt)
            except:
                pass

            # draw
            cv2.rectangle(img, (x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(img, final, (x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0),2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


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