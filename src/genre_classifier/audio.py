"""Librosa audio loading, fixed-duration mel preprocessing, and visual summaries."""

import librosa
import numpy as np

from .config import AudioConfig


def load_audio(source: str, config: AudioConfig) -> np.ndarray:
    try:
        audio, _ = librosa.load(source, sr=config.sample_rate, mono=True)
    except Exception as exc:
        raise ValueError(f"Could not decode audio: {exc}") from exc
    if audio.size == 0 or not np.isfinite(audio).all():
        raise ValueError("Audio is empty or contains invalid values")
    return audio.astype(np.float32, copy=False)


def crop_or_pad(audio: np.ndarray, config: AudioConfig, *, training: bool = False,
                rng: np.random.Generator | None = None) -> np.ndarray:
    target = config.samples_per_clip
    if audio.size > target:
        start = int(rng.integers(0, audio.size - target + 1)) if training and rng else (audio.size - target) // 2
        audio = audio[start:start + target]
    elif audio.size < target:
        left = (target - audio.size) // 2
        audio = np.pad(audio, (left, target - audio.size - left))
    return np.asarray(audio, dtype=np.float32)


def mel_spectrogram(audio: np.ndarray, config: AudioConfig, *, training: bool = False,
                    rng: np.random.Generator | None = None) -> np.ndarray:
    clip = crop_or_pad(audio, config, training=training, rng=rng)
    power = librosa.feature.melspectrogram(y=clip, sr=config.sample_rate,
        n_fft=config.n_fft, hop_length=config.hop_length, n_mels=config.n_mels,
        fmin=config.fmin, fmax=config.fmax, power=2.0)
    db = librosa.power_to_db(power, ref=np.max)
    normalized = np.clip((db - config.db_floor) / (config.db_ceiling - config.db_floor), 0, 1)
    return normalized.astype(np.float32)[..., np.newaxis]


def visualization_data(audio: np.ndarray, config: AudioConfig, points: int = 1200) -> dict:
    """Bound payload size for the browser while retaining useful plotting detail."""
    clip = crop_or_pad(audio, config)
    stride = max(1, int(np.ceil(audio.size / points)))
    waveform = audio[::stride]
    mel = mel_spectrogram(clip, config)[..., 0]
    # Return a bounded, display-oriented 96-column dB image.
    source_x = np.linspace(0.0, 1.0, mel.shape[1])
    target_x = np.linspace(0.0, 1.0, 96)
    image = np.stack([np.interp(target_x, source_x, row) for row in mel])
    return {"waveform": waveform.tolist(), "waveform_sample_rate": config.sample_rate / stride,
            "spectrogram": image.tolist(), "duration_seconds": len(audio) / config.sample_rate}
