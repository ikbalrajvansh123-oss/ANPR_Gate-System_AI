import streamlit as st
import cv2
import pandas as pd
import threading
import time
from queue import Queue
from collections import defaultdict

from modules.detector import Detector
from modules.ocr import OCR
from modules.logic import Logic
from modules.database import DB
from modules.utils import preprocess, save_plate
from modules.voice import Voice

# CONFIG 
st.set_page_config(layout="wide")
st.title("🚗 ANPR Gate System")

# SESSION 
if "run" not in st.session_state:
    st.session_state.run = False

# MODULES 
detector = Detector("models/best.pt")
ocr = OCR()
logic = Logic()
voice = Voice()
db = DB("data.db")

# VOICE 
voice_queue = Queue()

def voice_worker():
    while True:
        text = voice_queue.get()
        voice.speak(text)
        voice_queue.task_done()

threading.Thread(target=voice_worker, daemon=True).start()

def speak_async(text):
    voice_queue.put(text)

#  ALERT SYSTEM 
plate_history = defaultdict(list)
alert_triggered = {}

ALERT_LIMIT = 7
TIME_WINDOW = 3600

def check_alert(plate):
    now = time.time()

    plate_history[plate].append(now)
    plate_history[plate] = [t for t in plate_history[plate] if now - t <= TIME_WINDOW]

    if len(plate_history[plate]) >= ALERT_LIMIT:
        if plate not in alert_triggered or now - alert_triggered[plate] > TIME_WINDOW:
            alert_triggered[plate] = now
            return True
    return False

#  BUTTONS 
col1, col2, col3 = st.columns(3)

if col1.button("▶ Start"):
    st.session_state.run = True

if col2.button("⛔ Stop"):
    st.session_state.run = False

if col3.button("🗑 Delete All"):
    db.delete_all()
    st.success("All data deleted")

# UI
frame_box = st.empty()
table_box = st.empty()

# PROCESS 
def process(frame):
    frame = preprocess(frame)
    boxes = detector.detect(frame)

    for (x1, y1, x2, y2) in boxes:

        plate = frame[y1:y2, x1:x2]

        if plate.size == 0:
            continue

        # OCR
        raw_text = ocr.read(plate)
        if not raw_text:
            continue

        text = ocr.clean(raw_text)
        if not text:
            continue

        final = text

        result = logic.process(final)
        now = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")

        # ENTRY
        if result == "ENTRY":
            db.insert_entry(final, now)
            save_plate(plate, final)

            if check_alert(final):
                speak_async(f"Alert! Vehicle {final} entered multiple times")

        # EXIT
        elif result and result[0] == "EXIT":
            db.update_exit(final, now)

        # DRAW
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, final, (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    return frame

# CAMERA RUN 
if st.session_state.run:

    cap = cv2.VideoCapture(0)

    while st.session_state.run:

        ret, frame = cap.read()
        if not ret:
            break

        frame = process(frame)

        frame_box.image(frame, channels="BGR")

        # REAL-TIME TABLE
        data = db.fetch()
        df = pd.DataFrame(data, columns=["Plate","Entry","Exit"])
        table_box.dataframe(df, use_container_width=True)

        time.sleep(0.03)

    cap.release()

# SEARCH & DELETE 
st.subheader("🔍 Search & Delete Plate")

colA, colB = st.columns([3,1])

search_input = colA.text_input("Enter Plate Number")

if colA.button("Search"):
    results = db.search_plate(search_input)

    if results:
        df_search = pd.DataFrame(results, columns=["Plate","Entry","Exit"])
        st.success(f"Found {len(results)} records")
        st.dataframe(df_search, use_container_width=True)
    else:
        st.warning("No record found")

if colB.button("Delete Plate"):
    if search_input:
        db.delete_plate(search_input)
        st.success(f"{search_input} deleted")
    else:
        st.warning("Enter plate first")

# FINAL TABLE 
st.subheader("📊 Logs")

data = db.fetch()
df = pd.DataFrame(data, columns=["Plate","Entry","Exit"])

st.dataframe(df, use_container_width=True)