"""FastAPI upload endpoint for the four-source YAMNet + PANNs classifier."""

import os
import tempfile
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .inference import GenrePredictor

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
ALLOWED_SUFFIXES = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".au"}
MODEL_NAME = "four_source"
MODEL_DIRECTORY = Path(
    os.getenv("FOUR_SOURCE_MODEL_DIR", "artifacts/musify_four_dataset_hybrid_v2")
)

app = FastAPI(title="Musify Four-source Music Genre Classifier API", version="1.2.0")
allowed_origins = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
    ).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@lru_cache(maxsize=1)
def get_predictor() -> GenrePredictor:
    try:
        return GenrePredictor(MODEL_DIRECTORY)
    except (FileNotFoundError, ValueError, RuntimeError, ImportError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/")
def root():
    return {
        "application": "Musify",
        "model": MODEL_NAME,
        "health": "/health",
        "docs": "/docs",
        "prediction": "POST /predict",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "artifact_directory": str(MODEL_DIRECTORY),
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),  # noqa: B008 - FastAPI declares uploads this way.
):
    suffix = Path(file.filename or "upload").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported audio format: {suffix or 'none'}",
        )
    payload = await file.read(MAX_UPLOAD_BYTES + 1)
    if not payload:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Audio uploads must be 25 MB or smaller")

    predictor = get_predictor()
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp:
            temp.write(payload)
            temp_path = temp.name
        return predictor.predict(temp_path)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Audio inference failed. Check server logs.",
        ) from exc
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)
