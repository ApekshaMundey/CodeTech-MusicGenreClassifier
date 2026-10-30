"""GTZAN-style discovery and leakage-safe, reproducible track splits."""

import hashlib
import json
import os
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

SUPPORTED_AUDIO = {".wav", ".au", ".mp3", ".flac", ".ogg", ".m4a"}


@dataclass(frozen=True)
class Track:
    path: Path
    label: str
    track_id: str


def discover_tracks(root: Path) -> list[Track]:
    """Find genre-folder audio and skip byte-identical duplicate files."""
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")

    found: list[Track] = []
    seen: dict[str, Track] = {}
    conflicting_hashes: set[str] = set()

    for genre_dir in sorted(
        p for p in root.iterdir()
        if p.is_dir() and not p.name.startswith(".")
    ):
        label = genre_dir.name.lower()

        for path in sorted(genre_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in SUPPORTED_AUDIO:
                continue

            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            previous = seen.get(digest)

            if previous is not None:
                if previous.label != label:
                    conflicting_hashes.add(digest)
                continue

            track = Track(path, label, digest)
            seen[digest] = track
            found.append(track)

    if conflicting_hashes:
        found = [
            track for track in found
            if track.track_id not in conflicting_hashes
        ]
        warnings.warn(
            f"Excluded {len(conflicting_hashes)} exact-audio group(s) "
            "with conflicting genre labels.",
            RuntimeWarning,
        )

    if not found:
        raise ValueError(
            f"No supported audio found under {root}. "
            "Expected genre-named folders."
        )

    labels = {track.label for track in found}
    if len(labels) < 2:
        raise ValueError(
            f"At least two genre folders are required; found {sorted(labels)}"
        )

    return found


def split_tracks(tracks: list[Track], seed: int = 42, test_fraction: float = 0.15,
                 validation_fraction: float = 0.15) -> dict[str, list[Track]]:
    """Stratify full tracks first. No segments are created until after this step."""
    if not tracks:
        raise ValueError("Cannot split an empty track list")
    if test_fraction <= 0 or validation_fraction <= 0 or test_fraction + validation_fraction >= 1:
        raise ValueError("Test and validation fractions must be positive and sum to less than one")
    ids = np.arange(len(tracks))
    labels = np.asarray([t.label for t in tracks])
    train_val, test = train_test_split(ids, test_size=test_fraction, random_state=seed,
                                       stratify=labels)
    val_ratio = validation_fraction / (1 - test_fraction)
    train, validation = train_test_split(train_val, test_size=val_ratio, random_state=seed,
                                         stratify=labels[train_val])
    result = {name: [tracks[int(i)] for i in ix]
              for name, ix in (("train", train), ("validation", validation), ("test", test))}
    assert_split_integrity(result)
    return result


def assert_split_integrity(splits: dict[str, list[Track]]) -> None:
    seen: dict[str, str] = {}
    for name, tracks in splits.items():
        for track in tracks:
            if track.track_id in seen:
                raise ValueError(f"Track {track.track_id} occurs in both {seen[track.track_id]} and {name}")
            seen[track.track_id] = name


def save_manifest(splits: dict[str, list[Track]], path: Path, seed: int) -> None:
    all_tracks = [track for rows in splits.values() for track in rows]
    dataset_root = Path(os.path.commonpath([str(track.path.parent) for track in all_tracks]))
    payload = {"seed": seed, "paths_relative_to": "the directory containing the genre folders",
               "splits": {name: [
        {"path": t.path.relative_to(dataset_root).as_posix(), "label": t.label,
         "track_id": t.track_id} for t in rows
    ] for name, rows in splits.items()}}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
