from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Annotated

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "weights" / "best.pt"
IMAGE_SIZE = 640
MAX_BYTES = 10 * 1024 * 1024
SETTINGS = {
    "device": "cpu",
    "imgsz": IMAGE_SIZE,
    "rect": False,
    "iou": 0.70,
    "max_det": 300,
    "verbose": False,
    "save": False
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.ready = False
    
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    
    model = YOLO(str(MODEL_PATH))
    
    if model.task != "detect":
        raise ValueError("A detection checkpoint is required.")
    
    dummy = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    model.predict(dummy, conf=0.25, **SETTINGS)
    
    app.state.model = model
    app.state.inference_lock = Lock()
    app.state.ready = True
    
    print("[startup] Model loaded once; warm-up complete.", flush=True)
    
    try:
        yield
    finally:
        app.state.ready = False
        app.state.model = None


app = FastAPI(title="RDD2022 Detection API", lifespan=lifespan)


@app.get("/health")
def health(request: Request):
    if not getattr(request.app.state, "ready", False):
        raise HTTPException(status_code=503, detail="Model is not ready.")
    
    return {
        "status": "ready",
        "model": MODEL_PATH.name,
        "device": SETTINGS["device"],
        "imgsz": IMAGE_SIZE,
        "classes": request.app.state.model.names
    }


def decode_image(data: bytes) -> np.ndarray:
    if not data:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty."
        )
    
    if len(data) > MAX_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Image must be at most 10 MiB.",
        )
    
    try:
        array = np.frombuffer(data, dtype=np.uint8)
        image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    
    except cv2.error:
        image = None
    
    if image is None:
        raise HTTPException(
            status_code=400,
            detail="Cannot decode file as an image.",
        )
    
    return image