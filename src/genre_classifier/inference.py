"""Load a trained Musify model and predict one uploaded audio track."""

import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from .audio import load_audio, mel_spectrogram, visualization_data
from .config import AudioConfig


def _resolve_project_path(value: str, config_path: Path) -> Path:
    """Resolve metadata paths relative to the project, cwd, or model configuration."""
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    project_root = Path(__file__).resolve().parents[2]
    candidates = (Path.cwd() / path, project_root / path, config_path.parent / path)
    return next((candidate for candidate in candidates if candidate.exists()), candidates[1])


class GenrePredictor:
    """Load either the hybrid transfer model or the from-scratch CNN."""

    def __init__(self, artifact_dir: Path):
        config_path = artifact_dir / "preprocessing.json"
        if not config_path.is_file():
            raise FileNotFoundError(
                f"Trained model artifacts are missing in {artifact_dir}. "
                "Check the selected model directory."
            )
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        self.model_type = payload.get("model_type", "crnn")
        self.class_hierarchy = payload.get("class_hierarchy", {})
        if not isinstance(self.class_hierarchy, dict) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in self.class_hierarchy.items()
        ):
            raise ValueError(f"Invalid class hierarchy in {config_path}")
        self.audio_window_seconds = payload.get("audio_window_seconds")
        self.audio_hop_seconds = payload.get("audio_hop_seconds", self.audio_window_seconds)
        self.max_inference_windows = payload.get("max_inference_windows")
        self.selected_features = payload.get("selected_features", "hybrid")
        if self.selected_features not in {"hybrid", "yamnet", "panns"}:
            raise ValueError(f"Invalid selected feature set in {config_path}")
        if self.audio_window_seconds is not None and (
            not isinstance(self.audio_window_seconds, (int, float))
            or self.audio_window_seconds <= 0
            or not isinstance(self.audio_hop_seconds, (int, float))
            or self.audio_hop_seconds <= 0
            or self.audio_hop_seconds > self.audio_window_seconds
        ):
            raise ValueError(f"Invalid audio window settings in {config_path}")
        if self.max_inference_windows is not None and (
            not isinstance(self.max_inference_windows, int)
            or isinstance(self.max_inference_windows, bool)
            or self.max_inference_windows < 1
        ):
            raise ValueError(f"Invalid max_inference_windows in {config_path}")
        self.model = None
        self.yamnet = None
        self.panns = None

        if self.model_type in {"yamnet_embedding_mlp", "yamnet_panns_logistic"}:
            self.classes = payload.get("classes")
            if not isinstance(self.classes, list) or not self.classes or not all(
                isinstance(label, str) for label in self.classes
            ):
                raise ValueError(f"Invalid or missing class list in {config_path}")
            self.config = AudioConfig()
            yamnet_dir = _resolve_project_path(
                payload.get("yamnet_model_path", "artifacts/pretrained/yamnet"),
                config_path,
            )
            if self.model_type == "yamnet_embedding_mlp" or self.selected_features in {
                "hybrid",
                "yamnet",
            }:
                try:
                    from .yamnet import load_yamnet

                    self.yamnet = load_yamnet(yamnet_dir)
                except ImportError as exc:
                    raise RuntimeError(
                        "YAMNet inference dependencies are missing. Install the hybrid extra "
                        "from the README setup instructions."
                    ) from exc

            if self.model_type == "yamnet_embedding_mlp":
                model_path = artifact_dir / "best_model.keras"
                if not model_path.is_file():
                    raise FileNotFoundError(f"Keras model is missing: {model_path}")
                self.model = tf.keras.models.load_model(model_path)
            else:
                model_path = artifact_dir / "best_model.joblib"
                checkpoint = _resolve_project_path(
                    payload.get("panns_checkpoint_path", ""),
                    config_path,
                )
                if not model_path.is_file():
                    raise FileNotFoundError(f"Hybrid classifier is missing: {model_path}")
                try:
                    import joblib

                    from .panns import load_panns

                    self.panns = load_panns(checkpoint)
                    self.model = joblib.load(model_path)
                except ImportError as exc:
                    raise RuntimeError(
                        "PANNs inference dependencies are missing. Install the hybrid extra "
                        "from the README setup instructions."
                    ) from exc
                except Exception as exc:
                    raise RuntimeError(
                        f"Could not load the hybrid classifier or PANNs checkpoint: {exc}"
                    ) from exc
            training_sources = set(payload.get("training_sources", []))
            self.model_name = (
                "Four-source YAMNet/PANNs classifier (GTZAN + FMA-Small + Saraga Hindustani + Carnatic)"
                if training_sources
                == {"GTZAN", "FMA-Small", "Saraga Hindustani", "Saraga Carnatic"}
                else
                "Three-source PANNs embedding classifier (GTZAN + FMA-Small + Saraga)"
                if training_sources
                == {"GTZAN", "FMA-Small", "Saraga Hindustani"}
                else
                "Saraga-aware hybrid (experimental)"
                if "Hindustani classical" in self.classes
                else "Hybrid transfer model (YAMNet + PANNs CNN14)"
                if self.model_type == "yamnet_panns_logistic"
                else "YAMNet embedding classifier"
            )
        elif self.model_type in {
            "crnn",
            "musify_multiscale_cnn",
            "musify_attention_crnn",
        }:
            self.config, self.classes = AudioConfig.load(config_path)
            model_path = artifact_dir / "best_model.keras"
            if not model_path.is_file():
                raise FileNotFoundError(f"Keras model is missing: {model_path}")
            self.model = tf.keras.models.load_model(model_path)
            self.model_name = (
                "Multi-Scale Residual CNN"
                if self.model_type == "musify_multiscale_cnn"
                else "Residual Attention CRNN"
                if self.model_type == "musify_attention_crnn"
                else "CRNN"
            )
        else:
            raise ValueError(f"Unsupported model type in {config_path}: {self.model_type}")

    def predict(self, source: str) -> dict:
        audio = load_audio(source, self.config)
        if self.model_type in {"yamnet_embedding_mlp", "yamnet_panns_logistic"}:
            from .panns import extract_panns_embedding
            from .yamnet import extract_track_embedding_from_audio, load_yamnet_audio

            audio_16k = load_yamnet_audio(source)
            window_seconds = self.audio_window_seconds
            if window_seconds is None:
                windows = [audio_16k]
            else:
                window_size = round(window_seconds * 16_000)
                hop_size = round(self.audio_hop_seconds * 16_000)
                if audio_16k.size < window_size:
                    left = (window_size - audio_16k.size) // 2
                    windows = [
                        np.pad(audio_16k, (left, window_size - audio_16k.size - left))
                    ]
                else:
                    last_start = audio_16k.size - window_size
                    starts = list(range(0, last_start + 1, hop_size))
                    if not starts or starts[-1] != last_start:
                        starts.append(last_start)
                    if self.max_inference_windows is not None and len(starts) > self.max_inference_windows:
                        selected = np.linspace(
                            0, len(starts) - 1, num=self.max_inference_windows
                        ).round().astype(int)
                        starts = [starts[int(index)] for index in np.unique(selected)]
                    windows = [
                        audio_16k[start : start + window_size] for start in starts
                    ]

            window_probabilities = []
            for window in windows:
                if self.model_type == "yamnet_embedding_mlp":
                    yamnet_features = extract_track_embedding_from_audio(self.yamnet, window)
                    prediction = self.model.predict(
                        yamnet_features[np.newaxis, :], verbose=0
                    )[0]
                else:
                    if self.selected_features == "yamnet":
                        features = extract_track_embedding_from_audio(self.yamnet, window)
                    elif self.selected_features == "panns":
                        features = extract_panns_embedding(self.panns, window)
                    else:
                        yamnet_features = extract_track_embedding_from_audio(self.yamnet, window)
                        panns_features = extract_panns_embedding(self.panns, window)
                        features = np.concatenate((yamnet_features, panns_features))
                    features = features[np.newaxis, :]
                    prediction = self.model.predict_proba(features)[0]
                window_probabilities.append(np.asarray(prediction, dtype=np.float32))
            probabilities = np.mean(window_probabilities, axis=0)
        else:
            clip_samples = self.config.samples_per_clip
            starts = (
                [0]
                if audio.size <= clip_samples
                else list(range(0, audio.size - clip_samples + 1, clip_samples // 2))
            )
            if audio.size > clip_samples and starts[-1] != audio.size - clip_samples:
                starts.append(audio.size - clip_samples)
            batches = np.stack(
                [mel_spectrogram(audio[start : start + clip_samples], self.config) for start in starts]
            )
            probabilities = np.asarray(self.model.predict(batches, verbose=0)).mean(axis=0)

        if probabilities.shape != (len(self.classes),) or not np.isfinite(probabilities).all():
            raise RuntimeError("Classifier returned invalid genre scores")
        order = np.argsort(probabilities)[::-1]
        predicted_genre = self.classes[int(order[0])]
        return {
            "model_name": self.model_name,
            "predicted_genre": predicted_genre,
            "parent_category": self.class_hierarchy.get(predicted_genre),
            "confidence": float(probabilities[order[0]]),
            "probabilities": [
                {
                    "genre": self.classes[int(index)],
                    "probability": float(probabilities[index]),
                    "parent_category": self.class_hierarchy.get(self.classes[int(index)]),
                }
                for index in order
            ],
            "visualization": visualization_data(audio, self.config),
        }
