"""Shared, serializable preprocessing and training settings."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = 22050
    clip_seconds: float = 10.0
    n_mels: int = 128
    n_fft: int = 2048
    hop_length: int = 512
    fmin: int = 30
    fmax: int = 11025
    db_floor: float = -80.0
    db_ceiling: float = 0.0

    @property
    def samples_per_clip(self) -> int:
        return int(self.sample_rate * self.clip_seconds)

    @property
    def frames_per_clip(self) -> int:
        return 1 + self.samples_per_clip // self.hop_length

    def save(self, path: Path, classes: list[str]) -> None:
        payload = {**asdict(self), "classes": classes, "format_version": 1}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> tuple["AudioConfig", list[str]]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        allowed = set(cls.__dataclass_fields__)
        config = cls(**{k: v for k, v in payload.items() if k in allowed})
        classes = payload.get("classes")
        if not isinstance(classes, list) or not classes or not all(isinstance(x, str) for x in classes):
            raise ValueError(f"Invalid or missing class list in {path}")
        return config, classes
