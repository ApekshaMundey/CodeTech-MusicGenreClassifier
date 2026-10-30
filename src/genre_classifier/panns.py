"""PANNs CNN14-16k feature extraction for track-level classification.

The network layout follows the Cnn14_16k implementation by Qiuqiang Kong et al.
under the MIT license. See THIRD_PARTY_LICENSES.md for attribution and terms.
"""

import hashlib
import logging
import os
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchlibrosa.augmentation import SpecAugmentation
from torchlibrosa.stft import LogmelFilterBank, Spectrogram

PANNs_SAMPLE_RATE = 16_000
PANNs_EMBEDDING_SIZE = 2_048
PANNs_CHECKPOINT_NAME = "Cnn14_16k_mAP_0.438.pth"
PANNs_CHECKPOINT_MD5 = "362fc5ff18f1d6ad2f6d464b45893f2c"
PANNs_CHECKPOINT_URL = (
    "https://zenodo.org/records/3987831/files/"
    "Cnn14_16k_mAP%3D0.438.pth?download=1"
)
logger = logging.getLogger(__name__)


class ConvBlock(nn.Module):
    """Two-convolution block used by the pretrained PANNs CNN14."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, (3, 3), padding=(1, 1), bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, (3, 3), padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, inputs: torch.Tensor, pool_size=(2, 2)) -> torch.Tensor:
        x = F.relu_(self.bn1(self.conv1(inputs)))
        x = F.relu_(self.bn2(self.conv2(x)))
        return F.avg_pool2d(x, kernel_size=pool_size)


class Cnn14_16k(nn.Module):
    """AudioSet-pretrained CNN14 with the original 16 kHz input configuration."""

    def __init__(self, classes_num: int = 527):
        super().__init__()
        self.spectrogram_extractor = Spectrogram(
            n_fft=512,
            hop_length=160,
            win_length=512,
            window="hann",
            center=True,
            pad_mode="reflect",
            freeze_parameters=True,
        )
        self.logmel_extractor = LogmelFilterBank(
            sr=PANNs_SAMPLE_RATE,
            n_fft=512,
            n_mels=64,
            fmin=50,
            fmax=8_000,
            ref=1.0,
            amin=1e-10,
            top_db=None,
            freeze_parameters=True,
        )
        self.spec_augmenter = SpecAugmentation(
            time_drop_width=64,
            time_stripes_num=2,
            freq_drop_width=8,
            freq_stripes_num=2,
        )
        self.bn0 = nn.BatchNorm2d(64)
        self.conv_block1 = ConvBlock(1, 64)
        self.conv_block2 = ConvBlock(64, 128)
        self.conv_block3 = ConvBlock(128, 256)
        self.conv_block4 = ConvBlock(256, 512)
        self.conv_block5 = ConvBlock(512, 1_024)
        self.conv_block6 = ConvBlock(1_024, 2_048)
        self.fc1 = nn.Linear(2_048, 2_048)
        self.fc_audioset = nn.Linear(2_048, classes_num)

    def forward(self, waveform: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return AudioSet scores and the 2,048-value track embedding."""
        x = self.spectrogram_extractor(waveform)
        x = self.logmel_extractor(x)
        x = self.bn0(x.transpose(1, 3)).transpose(1, 3)
        if self.training:
            x = self.spec_augmenter(x)

        for index in range(1, 6):
            x = getattr(self, f"conv_block{index}")(x, pool_size=(2, 2))
            x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(1, 1))
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)
        x = torch.max(x, dim=2).values + torch.mean(x, dim=2)
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        clipwise_output = torch.sigmoid(self.fc_audioset(x))
        return {"clipwise_output": clipwise_output, "embedding": embedding}


def ensure_panns_checkpoint(checkpoint_path: Path) -> Path:
    """Return the verified checkpoint, downloading it atomically when absent."""
    checkpoint_path = checkpoint_path.expanduser().resolve()
    if checkpoint_path.is_file():
        digest = hashlib.md5()
        with checkpoint_path.open("rb") as checkpoint_file:
            for block in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() == PANNs_CHECKPOINT_MD5:
            return checkpoint_path

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = checkpoint_path.with_name(checkpoint_path.name + ".download")
    request = Request(
        PANNs_CHECKPOINT_URL,
        headers={"User-Agent": "Musify/1.0 (PANNs checkpoint setup)"},
    )
    digest = hashlib.md5()
    downloaded = 0
    last_reported_mib = -1
    try:
        logger.info("Downloading the PANNs CNN14 checkpoint (about 343 MiB)...")
        with urlopen(request, timeout=60) as response, temporary_path.open("wb") as output:
            total = int(response.headers.get("Content-Length", "0"))
            while block := response.read(8 * 1024 * 1024):
                output.write(block)
                digest.update(block)
                downloaded += len(block)
                downloaded_mib = downloaded // (1024 * 1024)
                if downloaded_mib >= last_reported_mib + 32:
                    if total:
                        logger.info(
                            "PANNs checkpoint download: %d / %d MiB",
                            downloaded_mib,
                            (total + 1024 * 1024 - 1) // (1024 * 1024),
                        )
                    else:
                        logger.info("PANNs checkpoint download: %d MiB", downloaded_mib)
                    last_reported_mib = downloaded_mib

        if digest.hexdigest() != PANNs_CHECKPOINT_MD5:
            raise ValueError("Downloaded PANNs checkpoint checksum did not match the expected MD5.")
        os.replace(temporary_path, checkpoint_path)
        logger.info("PANNs checkpoint is ready: %s", checkpoint_path)
        return checkpoint_path
    except Exception as exc:
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError(
            "Could not obtain the PANNs checkpoint automatically. Check your internet "
            "connection and available disk space, or run scripts/download_panns_checkpoint.ps1."
        ) from exc


def load_panns(checkpoint_path: Path) -> Cnn14_16k:
    """Load the checksum-verified published checkpoint onto CPU."""
    checkpoint_path = ensure_panns_checkpoint(checkpoint_path)
    model = Cnn14_16k(classes_num=527)
    try:
        # The official checkpoint stores NumPy metadata with the state dict. Its pinned
        # checksum is verified above before loading this trusted published pickle.
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = checkpoint.get("model", checkpoint)
        model.load_state_dict(state, strict=True)
    except Exception as exc:
        raise RuntimeError(f"Could not load the PANNs checkpoint {checkpoint_path}: {exc}") from exc
    model.eval()
    return model


def extract_panns_embedding(model: Cnn14_16k, audio: np.ndarray) -> np.ndarray:
    """Compute one finite, fixed-size CNN14 embedding from a mono 16 kHz track."""
    waveform = np.asarray(audio, dtype=np.float32)
    if waveform.ndim != 1 or waveform.size == 0 or not np.isfinite(waveform).all():
        raise ValueError("PANNs input must be a non-empty finite mono waveform")
    batch = torch.from_numpy(np.ascontiguousarray(waveform)).unsqueeze(0)
    try:
        with torch.inference_mode():
            embedding = model(batch)["embedding"].squeeze(0).cpu().numpy()
    except Exception as exc:
        raise RuntimeError(f"PANNs failed to process the audio: {exc}") from exc
    if embedding.shape != (PANNs_EMBEDDING_SIZE,) or not np.isfinite(embedding).all():
        raise ValueError(f"PANNs returned an invalid embedding with shape {embedding.shape}")
    return embedding.astype(np.float32, copy=False)
