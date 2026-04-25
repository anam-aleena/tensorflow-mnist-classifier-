"""
MNIST Classifier — REST API
Author: Aleena Anam | github.com/anam-aleena

FastAPI endpoints for digit prediction.
Accepts base64-encoded images or raw pixel arrays.

Run:
    uvicorn api.app:app --reload
    → http://localhost:8000/docs
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import io
import base64
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
from PIL import Image

from src.inference import load_model, predict


app = FastAPI(
    title="MNIST Digit Classifier API",
    description=(
        "Deep learning REST API for handwritten digit recognition. "
        "Built with TensorFlow/Keras CNN achieving ~99.4% test accuracy.\n\n"
        "Author: Aleena Anam | github.com/anam-aleena"
    ),
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Load model once at startup
model = None

@app.on_event("startup")
def startup_event():
    global model
    try:
        model = load_model()
    except FileNotFoundError:
        print("[API] Warning: Model not found. Run src/train.py first.")


# ─── SCHEMAS ─────────────────────────────────────────────────────────────────

class PixelRequest(BaseModel):
    pixels: List[float]  # 784 float values (28x28 flattened)

    class Config:
        json_schema_extra = {
            "example": {
                "pixels": [0.0] * 784  # 784 pixel values 0.0–1.0
            }
        }

class Base64Request(BaseModel):
    image_base64: str  # Base64 encoded PNG/JPEG image

    class Config:
        json_schema_extra = {
            "example": {
                "image_base64": "iVBORw0KGgoAAAANSUhEUgAAAA..."
            }
        }


# ─── ENDPOINTS ───────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    return {
        "service": "MNIST Digit Classifier API",
        "model": "CNN — TensorFlow/Keras",
        "accuracy": "~99.4% on MNIST test set",
        "author": "Aleena Anam",
        "github": "github.com/anam-aleena/tensorflow-mnist-classifier",
        "endpoints": {
            "POST /predict/pixels":    "Predict from 784 pixel values (flattened 28x28)",
            "POST /predict/image":     "Predict from base64-encoded image",
            "GET  /health":            "Health check",
            "GET  /docs":              "Interactive API documentation",
        }
    }


@app.get("/health", tags=["Health"])
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "service": "mnist-digit-classifier"
    }


@app.post("/predict/pixels", tags=["Prediction"])
def predict_from_pixels(request: PixelRequest):
    """
    Predict digit from 784 raw pixel values (flattened 28x28 image).
    Values should be normalised to [0.0, 1.0].
    """
    if model is None:
        raise HTTPException(status_code=503,
                            detail="Model not loaded. Run src/train.py first.")
    if len(request.pixels) != 784:
        raise HTTPException(status_code=400,
                            detail=f"Expected 784 pixel values, got {len(request.pixels)}.")

    image = np.array(request.pixels, dtype="float32").reshape(28, 28)
    result = predict(image, model)
    return result


@app.post("/predict/image", tags=["Prediction"])
def predict_from_image(request: Base64Request):
    """
    Predict digit from a base64-encoded image (PNG or JPEG).
    Image will be resized to 28x28 and converted to grayscale.
    """
    if model is None:
        raise HTTPException(status_code=503,
                            detail="Model not loaded. Run src/train.py first.")
    try:
        image_bytes = base64.b64decode(request.image_base64)
        image = Image.open(io.BytesIO(image_bytes)).convert("L")
        image = image.resize((28, 28), Image.LANCZOS)
        image_array = np.array(image, dtype="float32")
    except Exception as e:
        raise HTTPException(status_code=400,
                            detail=f"Invalid image: {str(e)}")

    result = predict(image_array, model)
    return result
