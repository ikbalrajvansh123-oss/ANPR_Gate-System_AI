
import asyncio
import base64
import time
import cv2
import numpy as np
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from detector import Detector
from ocr import OCR
from database import DB
from utils import preprocess

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)

# Thread pool for blocking operations (YOLO, EasyOCR)
executor = ThreadPoolExecutor(max_workers=2)

# ── Global singletons ─────────────────────────────────────────────────────────
detector: Optional[Detector] = None
ocr_engine: Optional[OCR] = None
db_conn: Optional[DB] = None


# ── Lifespan (startup/shutdown) ───────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global detector, ocr_engine, db_conn

    logger.info("=" * 60)
    logger.info("🚀 ANPR SYSTEM STARTING...")
    logger.info("=" * 60)

    # Load YOLO detector
    try:
        detector = Detector("models/best.pt")
    except Exception as e:
        logger.error(f"❌ YOLO model failed: {e}")
        detector = None

    # Load OCR engine
    try:
        ocr_engine = OCR()
    except Exception as e:
        logger.error(f"❌ OCR engine failed: {e}")
        ocr_engine = None

    # Connect to database
    try:
        db_conn = DB()
        logger.info("✅ Database connected")
    except Exception as e:
        logger.error(f"❌ Database failed: {e}")
        db_conn = None

    logger.info("=" * 60)
    logger.info("✅ ANPR SYSTEM READY")
    logger.info("=" * 60)

    yield

    # Shutdown
    logger.info("🛑 Shutting down...")
    if db_conn:
        db_conn.close()
    executor.shutdown(wait=False)


# ── FastAPI App ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="ANPR API",
    description="Automatic Number Plate Recognition",
    version="1.0.0",
    lifespan=lifespan
)

# CORS for localhost + production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic Models ───────────────────────────────────────────────────────────
class LogEntry(BaseModel):
    plate: str
    datetime: str


class DeleteResponse(BaseModel):
    success: bool
    message: str


# ── Notification Manager ──────────────────────────────────────────────────────
class NotificationManager:
    def __init__(self):
        self.clients: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.clients.append(ws)
        logger.info(f"Client connected. Total: {len(self.clients)}")

    def disconnect(self, ws: WebSocket):
        if ws in self.clients:
            self.clients.remove(ws)
            logger.info(f"Client disconnected. Total: {len(self.clients)}")

    async def broadcast(self, message: dict):
        dead = []
        for client in self.clients:
            try:
                await client.send_json(message)
            except Exception as e:
                logger.debug(f"Broadcast error: {e}")
                dead.append(client)
        
        for client in dead:
            self.disconnect(client)


notify_manager = NotificationManager()


# ── Frame Processing (blocking, runs in ThreadPoolExecutor) ──────────────────
def _process_frame(
    frame_bytes: bytes,
    last_seen: dict,
    det: Optional[Detector],
    ocr: Optional[OCR],
) -> list[dict]:
    """
    Process single frame:
    1. Decode JPEG
    2. Detect plates (YOLO)
    3. Read text (EasyOCR)
    4. Vote & confirm
    5. Check duplicates
    """
    
    # Decode
    np_arr = np.frombuffer(frame_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    
    if frame is None:
        logger.warning("Frame decode failed")
        return []
    
    # Preprocess
    frame = preprocess(frame)
    h, w = frame.shape[:2]
    
    detections = []
    
    # Return if detector not loaded
    if det is None:
        return detections
    
    # ── YOLO Detection ────────────────────────────────────────────────────────
    try:
        boxes = det.detect(frame)
        logger.debug(f"📍 Detected {len(boxes)} bounding box(es)")
    except Exception as e:
        logger.error(f"Detection failed: {e}")
        return []
    
    # ── Process each detection ────────────────────────────────────────────────
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        # Clip to frame boundaries
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        
        # Extract crop
        crop = frame[y1:y2, x1:x2]
        
        if crop.size == 0:
            logger.debug(f"Box {i}: Empty crop, skipping")
            continue
        
        if ocr is None:
            logger.warning("OCR not loaded")
            continue
        
        # ── OCR Reading ───────────────────────────────────────────────────────
        try:
            raw_text = ocr.read(crop)
            logger.debug(f"Box {i}: Raw OCR = '{raw_text}'")
            
            if not raw_text:
                logger.debug(f"Box {i}: OCR returned empty")
                continue
            
            # Clean text
            cleaned_text = ocr.clean(raw_text)
            logger.debug(f"Box {i}: Cleaned = '{cleaned_text}'")
            
            if not cleaned_text:
                logger.debug(f"Box {i}: Cleaning failed")
                continue
            
            # Vote (3 out of 5 matches needed)
            confirmed_text = ocr.vote(i, cleaned_text)
            logger.debug(f"Box {i}: Voted = '{confirmed_text}'")
            
        except Exception as e:
            logger.error(f"Box {i} processing error: {e}")
            continue
        
        # ── Build detection result ────────────────────────────────────────────
        now = time.time()
        
        if not confirmed_text:
            # Not yet confirmed (needs more votes)
            detections.append({
                "box": [x1, y1, x2, y2],
                "plate": cleaned_text,
                "confirmed": False,
                "saved": False,
                "_new": False,
            })
            logger.debug(f"Box {i}: Added as unconfirmed '{cleaned_text}'")
            continue
        
        # Check if this is a new plate (not seen in last 10 seconds)
        is_new = confirmed_text not in last_seen or (now - last_seen[confirmed_text]) >= 10
        
        if is_new:
            last_seen[confirmed_text] = now
            logger.info(f"🚗 NEW PLATE: {confirmed_text}")
        else:
            logger.debug(f"Duplicate: {confirmed_text} (seen {now - last_seen[confirmed_text]:.1f}s ago)")
        
        detections.append({
            "box": [x1, y1, x2, y2],
            "plate": confirmed_text,
            "confirmed": True,
            "saved": is_new,
            "_new": is_new,
        })
    
    return detections


# ── REST API Endpoints ────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Check system health"""
    return {
        "status": "ok",
        "detector_loaded": detector is not None,
        "ocr_loaded": ocr_engine is not None,
        "db_connected": db_conn is not None,
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/logs", response_model=List[LogEntry])
async def get_logs():
    """Get all plate logs"""
    if not db_conn:
        raise HTTPException(status_code=500, detail="Database not available")
    
    try:
        rows = db_conn.fetch()
        return [{"plate": r[0], "datetime": r[1]} for r in rows]
    except Exception as e:
        logger.error(f"Fetch logs error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/logs", response_model=DeleteResponse)
async def delete_logs():
    """Delete all logs"""
    if not db_conn:
        raise HTTPException(status_code=500, detail="Database not available")
    
    try:
        db_conn.delete_all()
        return {"success": True, "message": "All logs deleted"}
    except Exception as e:
        logger.error(f"Delete logs error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/logs/count")
async def log_count():
    """Get log count"""
    if not db_conn:
        return {"count": 0}
    
    return {"count": len(db_conn.fetch())}


# ── WebSocket: Video Stream Processing ────────────────────────────────────────

@app.websocket("/ws/video")
async def websocket_video(websocket: WebSocket):
    """
    WebSocket for real-time video processing
    Client sends: base64 JPEG frames
    Server sends: detection results
    """
    await websocket.accept()
    logger.info("📹 Video client connected")
    
    last_seen: dict[str, float] = {}
    frame_count = 0
    loop = asyncio.get_event_loop()
    
    try:
        while True:
            # Receive base64 frame
            data = await websocket.receive_text()
            
            if not data.startswith("data:image"):
                continue
            
            # Parse base64
            try:
                _, b64_data = data.split(",", 1)
                frame_bytes = base64.b64decode(b64_data)
            except Exception as e:
                logger.warning(f"Base64 decode error: {e}")
                continue
            
            frame_count += 1
            logger.debug(f"📊 Frame {frame_count} received ({len(frame_bytes)} bytes)")
            
            # Process in thread pool (non-blocking)
            try:
                detections = await loop.run_in_executor(
                    executor,
                    _process_frame,
                    frame_bytes,
                    last_seen,
                    detector,
                    ocr_engine,
                )
            except Exception as e:
                logger.error(f"Frame processing error: {e}")
                detections = []
            
            # Save new plates to DB and broadcast
            for det in detections:
                if det.pop("_new", False):
                    plate = det["plate"]
                    dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    
                    try:
                        db_conn.insert(plate, dt)
                        logger.info(f"💾 Saved: {plate} at {dt}")
                    except Exception as e:
                        logger.error(f"DB insert failed: {e}")
                    
                    # Broadcast to all notify clients
                    await notify_manager.broadcast({
                        "type": "new_plate",
                        "plate": plate,
                        "datetime": dt,
                    })
            
            # Send detections back to client
            try:
                await websocket.send_json({
                    "type": "detections",
                    "frame": frame_count,
                    "detections": detections,
                })
            except Exception as e:
                logger.error(f"Send detections error: {e}")
                break
    
    except WebSocketDisconnect:
        logger.info("📹 Video client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


# ── WebSocket: Live Notifications ─────────────────────────────────────────────

@app.websocket("/ws/notify")
async def websocket_notify(websocket: WebSocket):
    """
    WebSocket for receiving live plate notifications
    Server sends: new_plate events when plates are detected
    """
    await notify_manager.connect(websocket)
    logger.info("🔔 Notify client connected")
    
    try:
        while True:
            # Keep-alive ping every 30 seconds
            await asyncio.sleep(30)
            try:
                await websocket.send_json({"type": "ping"})
            except Exception:
                break
    
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"Notify WS error: {e}")
    finally:
        notify_manager.disconnect(websocket)
        logger.info("🔔 Notify client disconnected")