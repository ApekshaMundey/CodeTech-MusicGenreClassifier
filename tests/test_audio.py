import numpy as np

from genre_classifier.audio import crop_or_pad, mel_spectrogram
from genre_classifier.config import AudioConfig


def test_preprocessing_has_fixed_finite_shape():
    config = AudioConfig()
    result = mel_spectrogram(np.zeros(config.sample_rate * 3, dtype=np.float32), config)
    assert result.shape == (config.n_mels, config.frames_per_clip, 1)
    assert np.isfinite(result).all()
    assert result.min() >= 0 and result.max() <= 1


def test_crop_and_pad_produce_expected_length():
    config = AudioConfig()
    assert len(crop_or_pad(np.ones(12), config)) == config.samples_per_clip
