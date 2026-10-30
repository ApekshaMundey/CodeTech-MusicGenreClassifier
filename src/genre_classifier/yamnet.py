"""YAMNet audio loading and fixed-size track embedding extraction."""

from pathlib import Path

import librosa
import numpy as np
import tensorflow_hub as hub

YAMNET_SAMPLE_RATE = 16_000
YAMNET_TF_HUB_HANDLE = "https://tfhub.dev/google/yamnet/1"
YAMNET_EMBEDDING_SIZE = 1_024
YAMNET_POOLING = ("mean", "std", "max")
YAMNET_FEATURE_SIZE = YAMNET_EMBEDDING_SIZE * len(YAMNET_POOLING)


def load_yamnet(model_dir: Path):
    """Load local YAMNet, or fetch the pinned TF Hub module on first use."""
    model_dir = model_dir.expanduser().resolve()
    if model_dir.is_dir() and (model_dir / "saved_model.pb").is_file():
        try:
            return hub.load(str(model_dir))
        except Exception as exc:
            raise RuntimeError(f"Could not load local YAMNet SavedModel at {model_dir}: {exc}") from exc

    try:
        # TensorFlow Hub downloads the versioned module into its local cache.
        return hub.load(YAMNET_TF_HUB_HANDLE)
    except Exception as exc:
        raise RuntimeError(
            "YAMNet is not available locally and its automatic download failed. "
            "Check your internet connection, or download and extract the module into "
            f"{model_dir}. You can choose a cache location with TFHUB_CACHE_DIR."
        ) from exc


def load_yamnet_audio(source: str | Path) -> np.ndarray:
    """Decode an entire track as finite, mono, 16 kHz float32 audio."""
    try:
        audio, _ = librosa.load(str(source), sr=YAMNET_SAMPLE_RATE, mono=True)
    except Exception as exc:
        raise ValueError(f"Could not decode audio {source}: {exc}") from exc
    if audio.size == 0 or not np.isfinite(audio).all():
        raise ValueError(f"Audio is empty or contains invalid values: {source}")
    return audio.astype(np.float32, copy=False)


def pool_frame_embeddings(frame_embeddings: np.ndarray) -> np.ndarray:
    """Summarize frame embeddings with mean, standard deviation, and maximum."""
    frames = np.asarray(frame_embeddings, dtype=np.float32)
    if frames.ndim != 2 or frames.shape[0] == 0:
        raise ValueError("YAMNet embeddings must be a non-empty 2D frame-by-feature array")
    if frames.shape[1] != YAMNET_EMBEDDING_SIZE or not np.isfinite(frames).all():
        raise ValueError(
            f"Expected finite YAMNet frames with {YAMNET_EMBEDDING_SIZE} features per frame"
        )
    pooled = np.concatenate(
        (frames.mean(axis=0), frames.std(axis=0), frames.max(axis=0)),
        axis=0,
    )
    return pooled.astype(np.float32, copy=False)


def extract_track_embedding_from_audio(model, audio: np.ndarray) -> np.ndarray:
    """Extract a fixed-length feature vector from decoded mono 16 kHz audio."""
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 1 or audio.size == 0 or not np.isfinite(audio).all():
        raise ValueError("YAMNet input must be a non-empty finite mono waveform")
    try:
        outputs = model(audio)
        if not isinstance(outputs, (tuple, list)) or len(outputs) < 2:
            raise ValueError("YAMNet returned an unexpected output structure")
        frame_embeddings = np.asarray(outputs[1], dtype=np.float32)
    except Exception as exc:
        raise RuntimeError(f"YAMNet failed to process the audio: {exc}") from exc
    return pool_frame_embeddings(frame_embeddings)


def extract_track_embedding(model, source: str | Path) -> np.ndarray:
    """Extract a single fixed-length feature vector from one complete track."""
    return extract_track_embedding_from_audio(model, load_yamnet_audio(source))
