"""Train a release-disjoint GTZAN + FMA-Small + Saraga hybrid classifier.

This experiment preserves the existing three-source split assignments and adds
Saraga Carnatic as a twelfth class. Carnatic recordings are grouped by their
release directory before splitting. Long recordings contribute every overlapping
30-second window at a 15-second hop; each track keeps the same total training
weight, and validation/test scores are averaged at the track level.
The existing model artifacts are never overwritten.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import io
import json
import shutil
import warnings
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from zipfile import BadZipFile, ZipFile

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import GroupShuffleSplit

from .panns import extract_panns_embedding, load_panns
from .train_three_dataset_hybrid import (
    FEATURE_DIMENSION,
    FEATURE_SETS,
    Track,
    _feature_view,
    _fit_classifier,
    _load_feature_cache,
)
from .train_three_dataset_hybrid import (
    SOURCES as THREE_SOURCES,
)
from .yamnet import extract_track_embedding_from_audio, load_yamnet, load_yamnet_audio

SARAGA_LABEL = "Carnatic classical"
SUPPORTED_SUFFIXES = {".wav", ".au", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}
SPEECH_MARKERS = (
    "introductory speech",
    "introduction about",
    "shrutinandan concept - an introduction",
)
WINDOW_SECONDS = 30.0
WINDOW_HOP_SECONDS = 15.0
FEATURE_CACHE_VERSION = 1
SARAGA_INVENTORY_FORMAT = 2
SOURCES = (*THREE_SOURCES, "saraga_carnatic")


@dataclass(frozen=True)
class SaragaRecord:
    track: Track
    member: str
    filepath: str
    release_group: str


def _read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Required manifest is missing: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read {path}: {exc}") from exc


def _digest_from_track_id(track_id: str) -> str:
    return track_id.split(":", 1)[-1]


def _base_tracks(manifest_path: Path) -> list[Track]:
    payload = _read_json(manifest_path)
    rows = payload.get("tracks")
    if not isinstance(rows, list):
        raise TypeError(f"Expected a tracks list in {manifest_path}")
    tracks: list[Track] = []
    for row in rows:
        try:
            track = Track(**row)
        except (TypeError, KeyError) as exc:
            raise ValueError(f"Malformed track in {manifest_path}: {row}") from exc
        if track.source not in THREE_SOURCES or track.split not in {
            "train", "validation", "test"
        }:
            raise ValueError(f"Unexpected source/split in {manifest_path}: {row}")
        tracks.append(track)
    for source in THREE_SOURCES:
        for split in ("train", "validation", "test"):
            if not any(t.source == source and t.split == split for t in tracks):
                raise ValueError(f"The existing manifest lacks {source}/{split} tracks")
    if len({(t.source, t.track_id) for t in tracks}) != len(tracks):
        raise ValueError("The existing manifest contains duplicate source/track IDs")
    group_splits: dict[tuple[str, str], set[str]] = defaultdict(set)
    for track in tracks:
        group_splits[(track.source, track.group_id)].add(track.split)
    leaked = [key for key, splits in group_splits.items() if len(splits) > 1]
    if leaked:
        raise ValueError(f"An existing track/release group crosses splits: {leaked[0]}")
    return tracks


def _member_for_filepath(filepath: str, names: set[str], root: str) -> str | None:
    normalized = filepath.replace("\\", "/").strip("/")
    base = f"{root}/{normalized}"
    candidates = [base]
    if not PurePosixPath(normalized).suffix:
        candidates.extend(base + ext for ext in (".mp3.mp3", ".wav", ".au", ".mp3", ".flac", ".ogg", ".m4a", ".aac"))
    else:
        candidates.extend(base + ext for ext in (".mp3", ".wav", ".au", ".flac", ".ogg", ".m4a", ".aac"))
        candidates.append(f"{root}/{normalized}.mp3")
    for candidate in candidates:
        if candidate in names:
            return candidate
    return None


def _sha256_member(archive: ZipFile, member: str) -> str:
    digest = hashlib.sha256()
    with archive.open(member) as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _inventory_saraga(
    archive_path: Path,
    tradition: str,
    expected_tracks: list[Track] | None = None,
    cache_path: Path | None = None,
) -> tuple[list[SaragaRecord], dict]:
    """Index and hash eligible audio without unpacking the complete ZIP to disk."""
    root = f"saraga1.5_{tradition}"
    if not archive_path.is_file():
        raise FileNotFoundError(f"Saraga {tradition.title()} archive not found: {archive_path}")
    csv_member = f"{root}/file_paths.csv"
    quick_signature = {
        "archive_size_bytes": archive_path.stat().st_size,
        "archive_mtime_ns": archive_path.stat().st_mtime_ns,
    }
    if cache_path is not None and cache_path.is_file():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            with ZipFile(archive_path) as archive:
                index_hash = hashlib.sha256(archive.read(csv_member)).hexdigest()
            saved_signature = cached.get("signature", {})
            if (
                (tradition != "carnatic" or cached.get("format_version") == SARAGA_INVENTORY_FORMAT)
                and all(saved_signature.get(key) == value for key, value in quick_signature.items())
                and saved_signature.get("index_sha256") == index_hash
            ):
                restored = [
                    SaragaRecord(
                        track=Track(**row["track"]),
                        member=row["member"],
                        filepath=row["filepath"],
                        release_group=row["release_group"],
                    )
                    for row in cached["records"]
                ]
                expected_by_id = (
                    {row.track_id: row for row in expected_tracks}
                    if expected_tracks is not None
                    else None
                )
                cache_matches_manifest = expected_by_id is None or (
                    {row.track.track_id for row in restored} == set(expected_by_id)
                    and all(
                        row.track == expected_by_id[row.track.track_id]
                        for row in restored
                    )
                )
                if cache_matches_manifest:
                    print(f"Reusing {tradition} Saraga inventory: {cache_path}", flush=True)
                    return restored, saved_signature
        except (BadZipFile, KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
            warnings.warn(f"Saraga {tradition} inventory cache will be rebuilt.")
    try:
        with ZipFile(archive_path) as archive:
            names = set(archive.namelist())
            if csv_member not in names:
                raise ValueError(f"Missing Saraga index in archive: {csv_member}")
            audio_by_directory: dict[str, list[str]] = defaultdict(list)
            for name in names:
                path = PurePosixPath(name)
                if (
                    name.startswith(f"{root}/")
                    and path.suffix.lower() in SUPPORTED_SUFFIXES
                    and "multitrack" not in name.casefold()
                ):
                    audio_by_directory[path.parent.as_posix()].append(name)
            csv_bytes = archive.read(csv_member)
            index_hash = hashlib.sha256(csv_bytes).hexdigest()
            text = csv_bytes.decode("utf-8-sig", errors="replace")
            records: list[SaragaRecord] = []
            missing: list[str] = []
            metadata_count = 0
            expected_by_digest = (
                {_digest_from_track_id(track.track_id): track for track in expected_tracks}
                if expected_tracks is not None
                else None
            )
            for row_index, row in enumerate(csv.DictReader(io.StringIO(text)), start=1):
                filepath = (row.get("filepath") or "").strip().replace("\\", "/").strip("/")
                if not filepath:
                    continue
                metadata_count += 1
                if any(marker in filepath.casefold() for marker in SPEECH_MARKERS):
                    continue
                member = _member_for_filepath(filepath, names, root)
                if member is None:
                    # Some index paths use only a composition title, while the
                    # sole mixed recording is named "artist - title". Accept
                    # that fallback only when the folder has one non-multitrack
                    # audio file, so we never pick an instrument stem arbitrarily.
                    recording_directory = f"{root}/{PurePosixPath(filepath).parent.as_posix()}"
                    candidates = audio_by_directory.get(recording_directory, [])
                    if len(candidates) == 1:
                        member = candidates[0]
                if member is None:
                    missing.append(filepath)
                    continue
                if PurePosixPath(member).suffix.lower() not in SUPPORTED_SUFFIXES:
                    missing.append(filepath)
                    continue
                digest = _sha256_member(archive, member)
                parts = PurePosixPath(filepath).parts
                # Saraga stores performances under their release directory. Keeping
                # the first path component together makes the split release-disjoint.
                release = parts[0] if len(parts) > 1 else PurePosixPath(filepath).stem
                if expected_by_digest is not None:
                    track = expected_by_digest.get(digest)
                    if track is None:
                        continue
                    if track.group_id.split(":", 1)[-1] != release:
                        # The same recording may be referenced under more than one
                        # release path. Keep the path belonging to the saved group.
                        continue
                else:
                    track = Track(
                        source="saraga_carnatic",
                        split="",
                        label=SARAGA_LABEL,
                        track_id=f"saraga_carnatic:{digest}",
                        group_id=f"saraga_carnatic:{release}",
                    )
                records.append(SaragaRecord(track, member, filepath, release))
                if len(records) % 25 == 0:
                    print(
                        f"Saraga {tradition.title()} audio indexed and hashed: {len(records)}",
                        flush=True,
                    )
    except BadZipFile as exc:
        raise ValueError(f"Saraga ZIP is unreadable: {archive_path}") from exc

    # Remove exact duplicates within Saraga deterministically. Cross-source exact
    # duplicates are removed below before they can enter any split.
    by_digest: dict[str, list[SaragaRecord]] = defaultdict(list)
    for record in records:
        by_digest[_digest_from_track_id(record.track.track_id)].append(record)
    deduplicated = [
        min(group, key=lambda item: item.filepath)
        for group in by_digest.values()
    ]
    if expected_tracks is not None:
        found = {_digest_from_track_id(row.track.track_id) for row in deduplicated}
        expected = {_digest_from_track_id(track.track_id) for track in expected_tracks}
        if found != expected:
            raise ValueError(
                "Saraga Hindustani archive does not exactly match the saved track manifest "
                f"(missing={sorted(expected - found)[:3]}, extra={sorted(found - expected)[:3]})"
            )
    signature = {
        **quick_signature,
        "index_sha256": index_hash,
        "inventory_format_version": SARAGA_INVENTORY_FORMAT,
        "metadata_rows": metadata_count,
        "indexed_audio_records": len(records),
        "exact_duplicate_records_removed": len(records) - len(deduplicated),
        "missing_audio_records": missing,
    }
    sorted_records = sorted(deduplicated, key=lambda item: item.filepath)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps({
                "format_version": SARAGA_INVENTORY_FORMAT,
                "signature": signature,
                "records": [
                    {
                        "track": asdict(row.track),
                        "member": row.member,
                        "filepath": row.filepath,
                        "release_group": row.release_group,
                    }
                    for row in sorted_records
                ],
            }, indent=2),
            encoding="utf-8",
        )
    return sorted_records, signature


def _split_carnatic(
    records: list[SaragaRecord], seed: int, test_fraction: float, validation_fraction: float
) -> list[SaragaRecord]:
    groups = np.asarray([row.track.group_id for row in records], dtype=str)
    indices = np.arange(len(records))
    first = GroupShuffleSplit(n_splits=1, test_size=test_fraction, random_state=seed)
    development, test = next(first.split(indices, groups=groups))
    validation_relative_fraction = validation_fraction / (1.0 - test_fraction)
    second = GroupShuffleSplit(
        n_splits=1,
        test_size=validation_relative_fraction,
        random_state=seed + 1,
    )
    train_relative, validation_relative = next(
        second.split(development, groups=groups[development])
    )
    assignments = {
        "train": development[train_relative],
        "validation": development[validation_relative],
        "test": test,
    }
    result: list[SaragaRecord] = []
    seen_groups: dict[str, str] = {}
    for split, row_indices in assignments.items():
        for index in row_indices:
            row = records[int(index)]
            prior = seen_groups.setdefault(row.track.group_id, split)
            if prior != split:
                raise ValueError(f"Saraga release leaked across splits: {row.track.group_id}")
            result.append(SaragaRecord(row.track.__class__(
                source=row.track.source,
                split=split,
                label=row.track.label,
                track_id=row.track.track_id,
                group_id=row.track.group_id,
            ), row.member, row.filepath, row.release_group))
    for split in ("train", "validation", "test"):
        if not any(row.track.split == split for row in result):
            raise ValueError(f"Saraga Carnatic split {split} is empty")
    group_counts = Counter(row.track.split for row in result)
    if min(group_counts.values()) < 10:
        warnings.warn(f"A Carnatic split is small at track level: {dict(group_counts)}")
    return sorted(result, key=lambda item: (item.track.split, item.filepath))


def _load_existing_features(args, tracks: list[Track]) -> dict[str, np.ndarray]:
    base_cache = _load_feature_cache(args.gtzan_saraga_features, FEATURE_DIMENSION)
    fma_dev = _load_feature_cache(args.fma_dev_features, FEATURE_DIMENSION)
    fma_test = _load_feature_cache(args.fma_test_features, FEATURE_DIMENSION)
    if set(fma_dev) & set(fma_test):
        raise ValueError("FMA development and test embedding caches overlap")
    expected_base = {t.track_id for t in tracks if t.source in {"gtzan", "saraga"}}
    expected_gtzan = {t.track_id for t in tracks if t.source == "gtzan"}
    expected_fma = {t.track_id for t in tracks if t.source == "fma_small"}
    if set(base_cache) != expected_base:
        raise ValueError(
            "GTZAN/Saraga feature IDs do not exactly match the existing split manifest "
            f"(missing={sorted(expected_base - set(base_cache))[:3]}, "
            f"extra={sorted(set(base_cache) - expected_base)[:3]})"
        )
    fma = fma_dev | fma_test
    if set(fma) != expected_fma:
        raise ValueError(
            "FMA feature IDs do not exactly match the existing split manifest "
            f"(missing={sorted(expected_fma - set(fma))[:3]}, "
            f"extra={sorted(set(fma) - expected_fma)[:3]})"
        )
    return {track_id: base_cache[track_id] for track_id in expected_gtzan} | fma


def _window_starts(
    audio_samples: int, window_samples: int, hop_samples: int, max_windows: int | None
) -> list[int]:
    if audio_samples <= window_samples:
        return [0]
    last = audio_samples - window_samples
    starts = list(range(0, last + 1, hop_samples))
    if starts[-1] != last:
        starts.append(last)
    if max_windows is not None and len(starts) > max_windows:
        chosen = np.linspace(0, len(starts) - 1, num=max_windows).round().astype(int)
        starts = [starts[int(index)] for index in np.unique(chosen)]
    return starts


def _load_partial_features(
    path: Path,
    signature: dict,
    *,
    allowed_track_ids: set[str] | None = None,
) -> tuple[dict[str, list[np.ndarray]], set[str]]:
    metadata_path = path.with_suffix(".json")
    if not path.is_file() or not metadata_path.is_file():
        return {}, set()
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        with np.load(path, allow_pickle=False) as cache:
            ids = cache["track_ids"].astype(str).tolist()
            windows = cache["window_indices"].astype(int).tolist()
            features = np.asarray(cache["features"], dtype=np.float32)
            version = int(cache["format_version"].item())
        saved_signature = metadata.get("archive_signature", {})
        archive_keys = ("archive_size_bytes", "archive_mtime_ns", "index_sha256")
        feature_keys = (
            "window_seconds", "window_hop_seconds", "max_windows", "feature_dimension"
        )
        signature_matches = saved_signature == signature or (
            all(saved_signature.get(key) == signature.get(key) for key in archive_keys)
            and all(saved_signature.get(key) == signature.get(key) for key in feature_keys)
        )
        if (
            not signature_matches
            or version != FEATURE_CACHE_VERSION
            or features.shape != (len(ids), FEATURE_DIMENSION)
            or len(ids) != len(windows)
            or not np.isfinite(features).all()
        ):
            raise ValueError("partial feature cache metadata or dimensions do not match")
        result: dict[str, list[tuple[int, np.ndarray]]] = defaultdict(list)
        for track_id, window_index, feature in zip(ids, windows, features, strict=True):
            result[track_id].append((window_index, feature))
        ordered = {
            track_id: [item[1] for item in sorted(rows, key=lambda item: item[0])]
            for track_id, rows in result.items()
        }
        if allowed_track_ids is not None:
            ordered = {key: value for key, value in ordered.items() if key in allowed_track_ids}
        return ordered, set(ordered)
    except (EOFError, KeyError, OSError, ValueError, json.JSONDecodeError) as exc:
        warnings.warn(f"Ignoring incomplete Carnatic feature checkpoint: {exc}")
        return {}, set()


def _save_partial_features(path: Path, feature_rows: dict[str, list[np.ndarray]], signature: dict) -> None:
    ids: list[str] = []
    window_indices: list[int] = []
    features: list[np.ndarray] = []
    for track_id in sorted(feature_rows):
        for window_index, feature in enumerate(feature_rows[track_id]):
            ids.append(track_id)
            window_indices.append(window_index)
            features.append(feature)
    matrix = np.stack(features).astype(np.float32, copy=False) if features else np.empty((0, FEATURE_DIMENSION), np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(
        temporary,
        track_ids=np.asarray(ids, dtype="U80"),
        window_indices=np.asarray(window_indices, dtype=np.int32),
        features=matrix,
        format_version=np.asarray(FEATURE_CACHE_VERSION, dtype=np.int32),
    )
    temporary.replace(path)
    path.with_suffix(".json").write_text(
        json.dumps({"archive_signature": signature, "completed_tracks": len(feature_rows)}, indent=2),
        encoding="utf-8",
    )


def _extract_saraga_features(
    archive_path: Path,
    records: list[SaragaRecord],
    yamnet,
    panns,
    cache_path: Path,
    signature: dict,
    max_windows: int | None,
    tradition: str,
) -> dict[str, list[np.ndarray]]:
    feature_signature = {
        **signature,
        "window_seconds": WINDOW_SECONDS,
        "window_hop_seconds": WINDOW_HOP_SECONDS,
        "max_windows": max_windows,
        "feature_dimension": FEATURE_DIMENSION,
    }
    feature_rows, completed = _load_partial_features(
        cache_path,
        feature_signature,
        allowed_track_ids={row.track.track_id for row in records},
    )
    pending = [row for row in records if row.track.split == "train" and row.track.track_id not in completed]
    # Validation/test are also needed for fair selection and final metrics.
    pending.extend(row for row in records if row.track.split != "train" and row.track.track_id not in completed)
    clip_samples = round(WINDOW_SECONDS * 16_000)
    hop_samples = round(WINDOW_HOP_SECONDS * 16_000)
    cache_interval = 10
    cache_dir = cache_path.parent
    cache_dir.mkdir(parents=True, exist_ok=True)
    audio_path = cache_dir / f"saraga_{tradition.casefold()}_audio.tmp.mp3"
    window_description = (
        "every 30-second window per recording"
        if max_windows is None
        else f"up to {max_windows} evenly spaced windows per recording"
    )
    print(
        f"Saraga {tradition} embeddings: {len(completed)}/{len(records)} tracks already cached; "
        f"extracting {window_description} on CPU.",
        flush=True,
    )
    try:
        with ZipFile(archive_path) as archive:
            for count, record in enumerate(pending, start=1):
                try:
                    with archive.open(record.member) as source, audio_path.open("wb") as target:
                        shutil.copyfileobj(source, target, length=1024 * 1024)
                    audio = load_yamnet_audio(audio_path)
                    starts = _window_starts(audio.size, clip_samples, hop_samples, max_windows)
                    windows: list[np.ndarray] = []
                    for start in starts:
                        clip = audio[start : start + clip_samples]
                        if clip.size < clip_samples:
                            left = (clip_samples - clip.size) // 2
                            clip = np.pad(clip, (left, clip_samples - clip.size - left))
                        yamnet_feature = extract_track_embedding_from_audio(yamnet, clip)
                        panns_feature = extract_panns_embedding(panns, clip)
                        vector = np.concatenate((yamnet_feature, panns_feature)).astype(np.float32, copy=False)
                        if vector.shape != (FEATURE_DIMENSION,) or not np.isfinite(vector).all():
                            raise ValueError(f"Invalid embedding vector shape/value: {vector.shape}")
                        windows.append(vector)
                    feature_rows[record.track.track_id] = windows
                except (OSError, ValueError, RuntimeError, BadZipFile) as exc:
                    warnings.warn(f"Skipping {record.filepath}: {exc}")
                finally:
                    audio_path.unlink(missing_ok=True)
                if count % 5 == 0 or count == len(pending):
                    print(
                        f"Saraga {tradition} tracks embedded: {len(completed) + count}/{len(records)} "
                        f"(valid cached {len(feature_rows)})",
                        flush=True,
                    )
                if count % cache_interval == 0 or count == len(pending):
                    _save_partial_features(cache_path, feature_rows, feature_signature)
    finally:
        audio_path.unlink(missing_ok=True)
    missing = [row.track.track_id for row in records if row.track.track_id not in feature_rows]
    if missing:
        raise RuntimeError(
            f"Could not embed {len(missing)} Saraga {tradition} tracks; examples: {missing[:3]}. "
            "See the warnings above and correct the audio decode problem before training."
        )
    return feature_rows


def _track_weights(tracks: list[Track], source_order: tuple[str, ...]) -> dict[str, float]:
    counts = Counter((track.source, track.label) for track in tracks)
    classes = sorted({track.label for track in tracks})
    sources_per_class = {
        label: sum(1 for source in source_order if counts[(source, label)])
        for label in classes
    }
    if any(sources_per_class[label] == 0 for label in classes):
        raise ValueError("Every class must have at least one training example")
    return {
        track.track_id: 1.0 / (
            len(classes)
            * sources_per_class[track.label]
            * counts[(track.source, track.label)]
        )
        for track in tracks
    }


def _rows_for_tracks(
    tracks: list[Track],
    base_features: dict[str, np.ndarray],
    saraga_features: dict[str, list[np.ndarray]],
    classes: list[str],
) -> tuple[list[Track], np.ndarray, np.ndarray, list[str]]:
    class_to_id = {label: index for index, label in enumerate(classes)}
    output_tracks: list[Track] = []
    feature_rows: list[np.ndarray] = []
    labels: list[int] = []
    row_track_ids: list[str] = []
    for track in tracks:
        if track.source.startswith("saraga"):
            vectors = saraga_features.get(track.track_id)
            if vectors is None:
                raise ValueError(f"Missing full-track Saraga windows for {track.track_id}")
        else:
            vector = base_features.get(track.track_id)
            if vector is None:
                raise ValueError(f"Missing feature vector for {track.track_id}")
            vectors = [vector]
        for vector in vectors:
            feature_rows.append(vector)
            labels.append(class_to_id[track.label])
            row_track_ids.append(track.track_id)
            output_tracks.append(track)
    return output_tracks, np.stack(feature_rows).astype(np.float32), np.asarray(labels, np.int32), row_track_ids


def _row_weights(row_tracks: list[Track], row_track_ids: list[str]) -> np.ndarray:
    track_weights = _track_weights(list({track.track_id: track for track in row_tracks}.values()), SOURCES)
    window_counts = Counter(row_track_ids)
    weights = np.asarray(
        [track_weights[track_id] / window_counts[track_id] for track_id in row_track_ids],
        dtype=np.float64,
    )
    weights *= len(weights) / weights.sum()
    return weights.astype(np.float32)


def _track_probabilities(
    row_tracks: list[Track], row_track_ids: list[str], probabilities: np.ndarray
) -> tuple[list[Track], np.ndarray]:
    positions: dict[str, list[int]] = defaultdict(list)
    track_by_id: dict[str, Track] = {}
    for index, (track, track_id) in enumerate(zip(row_tracks, row_track_ids, strict=True)):
        positions[track_id].append(index)
        track_by_id[track_id] = track
    ids = sorted(positions)
    tracks = [track_by_id[track_id] for track_id in ids]
    matrix = np.stack([probabilities[positions[track_id]].mean(axis=0) for track_id in ids])
    return tracks, matrix


def _evaluate_source(tracks: list[Track], truth: np.ndarray, predicted: np.ndarray, classes: list[str]) -> dict:
    active = sorted({track.label for track in tracks})
    class_to_id = {label: index for index, label in enumerate(classes)}
    ids = [class_to_id[label] for label in active]
    precision, recall, f1, _ = precision_recall_fscore_support(
        truth, predicted, labels=ids, zero_division=0
    )
    return {
        "track_count": len(tracks),
        "release_group_count": len({track.group_id for track in tracks}) if tracks and tracks[0].source.startswith("saraga") else None,
        "classes_scored": active,
        "accuracy": float(accuracy_score(truth, predicted)),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "predictions_outside_source_labels": int(np.sum(~np.isin(predicted, ids))),
        "classification_report": classification_report(
            truth, predicted, labels=ids, target_names=active, output_dict=True, zero_division=0
        ),
    }


def _score_validation(
    tracks: list[Track], row_track_ids: list[str], truths: np.ndarray, matrix: np.ndarray, classes: list[str]
) -> tuple[dict, dict[str, np.ndarray]]:
    track_rows, track_probs = _track_probabilities(tracks, row_track_ids, matrix)
    predicted = track_probs.argmax(axis=1)
    source_reports: dict[str, dict] = {}
    source_predictions: dict[str, np.ndarray] = {}
    track_truth_by_id = {track.track_id: int(truths[index]) for index, track in enumerate(tracks)}
    for source in SOURCES:
        indices = [i for i, track in enumerate(track_rows) if track.source == source]
        if not indices:
            continue
        source_tracks = [track_rows[i] for i in indices]
        source_y = np.asarray([track_truth_by_id[track.track_id] for track in source_tracks], dtype=np.int32)
        source_pred = predicted[indices]
        source_reports[source] = _evaluate_source(source_tracks, source_y, source_pred, classes)
        source_predictions[source] = source_pred
    return {
        "source_validation": source_reports,
        "mean_source_macro_f1": float(np.mean([row["macro_f1"] for row in source_reports.values()])),
        "mean_source_accuracy": float(np.mean([row["accuracy"] for row in source_reports.values()])),
    }, source_predictions


def _plot_confusion_matrix(matrix: np.ndarray, classes: list[str], output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(max(10, len(classes) * 0.9), max(8, len(classes) * 0.75)))
    image = axis.imshow(matrix, cmap="magma")
    axis.set(
        xticks=range(len(classes)), yticks=range(len(classes)),
        xticklabels=classes, yticklabels=classes,
        xlabel="Predicted genre", ylabel="True genre",
        title="Four-source release-disjoint held-out test confusion matrix",
    )
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(len(classes)):
        for column in range(len(classes)):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center",
                      color="white" if matrix[row, column] < threshold else "black", fontsize=7)
    figure.colorbar(image, ax=axis, label="Track count")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def train(args: argparse.Namespace) -> dict:
    if args.all_windows:
        args.max_windows = None
    if args.output.exists():
        raise FileExistsError(f"Output already exists; choose another --output path: {args.output}")
    if not 0 < args.test_fraction < 0.5 or not 0 < args.validation_fraction < 0.5:
        raise ValueError("Test and validation fractions must each be between 0 and 0.5")
    if args.test_fraction + args.validation_fraction >= 1:
        raise ValueError("Test and validation fractions must sum to less than one")
    if args.max_windows is not None and args.max_windows < 1:
        raise ValueError("--max-windows must be at least one")
    if not args.c_values or any(value <= 0 for value in args.c_values):
        raise ValueError("--c-values must contain positive values")

    args.work_dir.mkdir(parents=True, exist_ok=True)
    base_tracks = _base_tracks(args.base_manifest)
    hindustani_tracks = [track for track in base_tracks if track.source == "saraga"]
    hindustani_records, hindustani_signature = _inventory_saraga(
        args.hindustani_archive,
        "hindustani",
        hindustani_tracks,
        args.work_dir / "hindustani_inventory.json",
    )
    carnatic_records, carnatic_signature = _inventory_saraga(
        args.carnatic_archive,
        "carnatic",
        cache_path=args.work_dir / "carnatic_inventory.json",
    )
    existing_digests = {_digest_from_track_id(track.track_id) for track in base_tracks}
    before_cross_source = len(carnatic_records)
    carnatic_records = [
        row for row in carnatic_records
        if _digest_from_track_id(row.track.track_id) not in existing_digests
    ]
    cross_source_duplicates_removed = before_cross_source - len(carnatic_records)
    if not carnatic_records:
        raise ValueError("No unique Carnatic recordings remain after deduplication")
    group_count = len({row.track.group_id for row in carnatic_records})
    if group_count < 8:
        raise ValueError(
            f"Only {group_count} release groups were found; at least 8 are needed for a useful group split"
        )
    carnatic_records = _split_carnatic(
        carnatic_records, args.seed, args.test_fraction, args.validation_fraction
    )
    tracks = base_tracks + [row.track for row in carnatic_records]
    classes = sorted({track.label for track in tracks})
    if SARAGA_LABEL not in classes or len(classes) < 2:
        raise ValueError("The combined training set does not include Carnatic and other genres")
    base_features = _load_existing_features(args, base_tracks)

    split_manifest = {
        "split_policy": (
            "Existing GTZAN/FMA-Small/Saraga-Hindustani track assignments retained; "
            "Saraga Carnatic deduplicated and split by release directory before any windows are extracted. "
            "Both Saraga traditions use every 30-second window with a 15-second hop after track-level splitting."
        ),
        "seed": args.seed,
        "window_policy": {
            "window_seconds": WINDOW_SECONDS,
            "hop_seconds": WINDOW_HOP_SECONDS,
            "max_evenly_spaced_windows_per_track": args.max_windows,
            "all_windows_per_track": args.max_windows is None,
            "track_level_split_before_windowing": True,
        },
        "saraga_hindustani_archive": str(args.hindustani_archive),
        "saraga_hindustani_inventory": hindustani_signature,
        "saraga_carnatic_archive": str(args.carnatic_archive),
        "saraga_carnatic_inventory": carnatic_signature,
        "saraga_carnatic_exact_duplicates_removed_across_sources": cross_source_duplicates_removed,
        "tracks": [asdict(track) for track in tracks],
    }
    (args.work_dir / "split_manifest.json").write_text(json.dumps(split_manifest, indent=2), encoding="utf-8")

    yamnet = load_yamnet(args.yamnet_dir)
    panns = load_panns(args.panns_checkpoint)
    hindustani_feature_rows = _extract_saraga_features(
        args.hindustani_archive,
        hindustani_records,
        yamnet,
        panns,
        args.work_dir / "hindustani_window_features_partial.npz",
        hindustani_signature,
        args.max_windows,
        "Hindustani",
    )
    carnatic_feature_rows = _extract_saraga_features(
        args.carnatic_archive,
        carnatic_records,
        yamnet,
        panns,
        args.work_dir / "carnatic_window_features_partial.npz",
        carnatic_signature,
        args.max_windows,
        "Carnatic",
    )
    saraga_feature_rows = hindustani_feature_rows | carnatic_feature_rows
    # The encoders are frozen feature extractors. Release them before fitting the
    # dense linear classifier so that this CPU-only 16 GB machine has headroom.
    del yamnet, panns
    gc.collect()

    split_tracks: dict[str, list[Track]] = {
        split: [track for track in tracks if track.split == split]
        for split in ("train", "validation", "test")
    }
    split_rows = {}
    for split, split_rows_tracks in split_tracks.items():
        rows, matrix, labels, row_track_ids = _rows_for_tracks(
            split_rows_tracks, base_features, saraga_feature_rows, classes
        )
        split_rows[split] = (rows, matrix, labels, row_track_ids)

    train_rows, x_train, y_train, train_row_ids = split_rows["train"]
    validation_rows, x_validation, _, validation_row_ids = split_rows["validation"]
    test_rows, x_test, _, test_row_ids = split_rows["test"]
    training_weights = _row_weights(train_rows, train_row_ids)

    candidate_scores: list[dict] = []
    for feature_name in args.feature_sets:
        x_train_view = _feature_view(x_train, feature_name)
        x_validation_view = _feature_view(x_validation, feature_name)
        for c_value in args.c_values:
            candidate = _fit_classifier(x_train_view, y_train, training_weights, float(c_value))
            validation_probabilities = candidate.predict_proba(x_validation_view)
            class_to_id = {label: index for index, label in enumerate(classes)}
            validation_labels_by_track = {
                track.track_id: class_to_id[track.label] for track in validation_rows
            }
            row_truths = np.asarray([validation_labels_by_track[i] for i in validation_row_ids], dtype=np.int32)
            score, _ = _score_validation(
                validation_rows, validation_row_ids, row_truths, validation_probabilities, classes
            )
            row = {
                "feature_set": feature_name,
                "C": float(c_value),
                **score,
            }
            candidate_scores.append(row)
            print(
                f"Validation {feature_name} C={c_value:g}: "
                f"equal-source macro-F1={row['mean_source_macro_f1']:.4f}, "
                f"equal-source accuracy={row['mean_source_accuracy']:.4f}",
                flush=True,
            )
    selected = max(candidate_scores, key=lambda row: (row["mean_source_macro_f1"], row["mean_source_accuracy"]))
    selected_features = selected["feature_set"]
    selected_c = selected["C"]
    print(f"Selected {selected_features} features with C={selected_c:g} on validation only.", flush=True)

    development_tracks = split_tracks["train"] + split_tracks["validation"]
    development_rows, x_development, y_development, development_row_ids = _rows_for_tracks(
        development_tracks, base_features, saraga_feature_rows, classes
    )
    development_weights = _row_weights(development_rows, development_row_ids)
    final_model = _fit_classifier(
        _feature_view(x_development, selected_features), y_development,
        development_weights, selected_c,
    )
    test_window_probabilities = final_model.predict_proba(_feature_view(x_test, selected_features))
    test_track_rows, test_track_probabilities = _track_probabilities(test_rows, test_row_ids, test_window_probabilities)
    class_to_id = {label: index for index, label in enumerate(classes)}
    test_truth = np.asarray([class_to_id[track.label] for track in test_track_rows], dtype=np.int32)
    test_predictions = test_track_probabilities.argmax(axis=1)
    source_metrics: dict[str, dict] = {}
    for source in SOURCES:
        indices = [index for index, track in enumerate(test_track_rows) if track.source == source]
        if indices:
            source_metrics[source] = _evaluate_source(
                [test_track_rows[i] for i in indices], test_truth[indices], test_predictions[indices], classes
            )
    baseline_comparison = None
    if args.baseline_evaluation.is_file():
        baseline = _read_json(args.baseline_evaluation)
        baseline_sources = baseline.get("source_metrics", {})
        baseline_comparison = {}
        for source in sorted(set(source_metrics) & set(baseline_sources)):
            baseline_comparison[source] = {
                "same_held_out_track_count": (
                    source_metrics[source]["track_count"] == baseline_sources[source].get("track_count")
                ),
                "baseline_accuracy": baseline_sources[source].get("accuracy"),
                "four_source_accuracy": source_metrics[source]["accuracy"],
                "accuracy_delta": source_metrics[source]["accuracy"] - baseline_sources[source].get("accuracy", 0),
                "baseline_macro_f1": baseline_sources[source].get("macro_f1"),
                "four_source_macro_f1": source_metrics[source]["macro_f1"],
                "macro_f1_delta": source_metrics[source]["macro_f1"] - baseline_sources[source].get("macro_f1", 0),
            }
    labels = list(range(len(classes)))
    precision, recall, f1, _ = precision_recall_fscore_support(
        test_truth, test_predictions, labels=labels, zero_division=0
    )
    matrix = confusion_matrix(test_truth, test_predictions, labels=labels)
    train_track_counts = Counter(track.source for track in development_tracks)
    train_class_counts = {
        source: dict(Counter(track.label for track in development_tracks if track.source == source))
        for source in SOURCES
    }
    group_counts = {
        split: len({row.track.group_id for row in carnatic_records if row.track.split == split})
        for split in ("train", "validation", "test")
    }
    report = {
        "experiment": "GTZAN + FMA-Small + Saraga Hindustani + Saraga Carnatic 12-label hybrid",
        "model_type": "yamnet_panns_logistic",
        "selected_candidate": selected_features,
        "selected_C": float(selected_c),
        "candidate_selection": (
            "Grid search on validation tracks only; equal-weight mean of source-specific macro-F1 "
            "for the four sources. Long-track windows are averaged to one track prediction."
        ),
        "candidate_scores": candidate_scores,
        "test_accuracy_pooled": float(accuracy_score(test_truth, test_predictions)),
        "test_macro_precision_pooled": float(precision.mean()),
        "test_macro_recall_pooled": float(recall.mean()),
        "test_macro_f1_pooled": float(f1.mean()),
        "test_accuracy_equal_source_mean": float(np.mean([row["accuracy"] for row in source_metrics.values()])),
        "test_macro_f1_equal_source_mean": float(np.mean([row["macro_f1"] for row in source_metrics.values()])),
        "classes": classes,
        "class_hierarchy": {
            "Western classical": "Classical",
            "Hindustani classical": "Classical",
            "Carnatic classical": "Classical",
        },
        "test_track_count": len(test_track_rows),
        "test_tracks_by_source": dict(Counter(track.source for track in test_track_rows)),
        "source_metrics": source_metrics,
        "three_source_baseline_comparison": baseline_comparison,
        "confusion_matrix_labels": classes,
        "confusion_matrix": matrix.tolist(),
        "training_tracks_by_source": dict(train_track_counts),
        "training_tracks_by_source_and_label": train_class_counts,
        "training_windows_by_source": dict(Counter(track.source for track in development_rows)),
        "train_split_windows_by_source": dict(Counter(track.source for track in train_rows)),
        "validation_tracks_by_source": dict(Counter(track.source for track in split_tracks["validation"])),
        "saraga_carnatic_release_groups_by_split": group_counts,
        "saraga_carnatic_tracks_by_split": dict(Counter(row.track.split for row in carnatic_records)),
        "saraga_hindustani_archive_inventory": hindustani_signature,
        "saraga_carnatic_archive_inventory": carnatic_signature,
        "saraga_carnatic_cross_source_exact_duplicates_removed": cross_source_duplicates_removed,
        "sample_weight_policy": (
            "Equal total weight per class; equal source share among sources containing that class; "
            "within a source/class each track has equal total weight divided over its windows."
        ),
        "feature_extractors": "Frozen YAMNet and PANNs CNN14 track-window embeddings; only the weighted StandardScaler and LogisticRegression are fitted locally.",
        "test_set_caveat": (
            "GTZAN/FMA-Small/Hindustani use the previously fixed held-out splits; Saraga Carnatic is "
            "held out by release directory. Carnatic remains one source and one label, so this tests "
            "release-disjoint recognition within Saraga rather than independent-source generalization."
        ),
        "confidence_note": "Logistic-regression probabilities are not calibrated confidence values.",
        "window_policy": {
            "window_seconds": WINDOW_SECONDS,
            "hop_seconds": WINDOW_HOP_SECONDS,
            "max_evenly_spaced_windows_per_track": args.max_windows,
            "all_windows_per_track": args.max_windows is None,
            "evaluation_unit": "track; average window probabilities before scoring",
        },
        "split_manifests": {"base": str(args.base_manifest), "combined": str(args.work_dir / "split_manifest.json")},
    }

    args.output.mkdir(parents=True, exist_ok=False)
    joblib.dump(final_model, args.output / "best_model.joblib")
    (args.output / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (args.output / "split_manifest.json").write_text(json.dumps(split_manifest, indent=2), encoding="utf-8")
    config = {
        "model_type": "yamnet_panns_logistic",
        "model_variant": "gtzan_fma_saraga_hindustani_carnatic_four_source",
        "classes": classes,
        "class_hierarchy": report["class_hierarchy"],
        "feature_order": ["yamnet_mean_std_max_3072", "panns_cnn14_16k_embedding_2048"],
        "feature_dimension": FEATURE_DIMENSION,
        "selected_features": selected_features,
        "sample_rate": 16_000,
        "audio_window_seconds": WINDOW_SECONDS,
        "audio_hop_seconds": WINDOW_HOP_SECONDS,
        "max_inference_windows": args.max_windows,
        "yamnet_model_path": "artifacts/pretrained/yamnet",
        "panns_checkpoint_path": "artifacts/pretrained/panns/Cnn14_16k_mAP_0.438.pth",
        "panns_architecture": "Cnn14_16k",
        "classifier": f"track-balanced StandardScaler + LogisticRegression(C={selected_c}, track-balanced)",
        "training_sources": ["GTZAN", "FMA-Small", "Saraga Hindustani", "Saraga Carnatic"],
        "training_protocol": report["candidate_selection"],
        "format_version": 1,
    }
    (args.output / "preprocessing.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    _plot_confusion_matrix(matrix, classes, args.output / "confusion_matrix.png")
    print(
        json.dumps({
            "selected_features": selected_features,
            "selected_C": selected_c,
            "test_accuracy_pooled": report["test_accuracy_pooled"],
            "test_macro_f1_pooled": report["test_macro_f1_pooled"],
            "test_accuracy_equal_source_mean": report["test_accuracy_equal_source_mean"],
            "test_macro_f1_equal_source_mean": report["test_macro_f1_equal_source_mean"],
            "source_metrics": {
                source: {"accuracy": metric["accuracy"], "macro_f1": metric["macro_f1"], "tracks": metric["track_count"]}
                for source, metric in source_metrics.items()
            },
            "three_source_baseline_comparison": baseline_comparison,
            "saraga_carnatic_release_groups_by_split": group_counts,
        }, indent=2),
        flush=True,
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hindustani-archive", type=Path, default=Path("data/raw/saraga_hindustani/saraga1.5_hindustani.zip"))
    parser.add_argument("--carnatic-archive", type=Path, default=Path("data/raw/saraga_carnatic/saraga1.5_carnatic.zip"))
    parser.add_argument("--base-manifest", type=Path, default=Path("artifacts/musify_three_dataset_hybrid_v1/split_manifest.json"))
    parser.add_argument("--baseline-evaluation", type=Path, default=Path("artifacts/musify_three_dataset_hybrid_v1/evaluation.json"))
    parser.add_argument("--gtzan-saraga-features", type=Path, default=Path("artifacts/musify_saraga_hybrid_v1/hybrid_features.npz"))
    parser.add_argument("--fma-dev-features", type=Path, default=Path("artifacts/pretrained/fma_small_hybrid_dev_features_v1.npz"))
    parser.add_argument("--fma-test-features", type=Path, default=Path("artifacts/pretrained/fma_small_hybrid_test_features_v1.npz"))
    parser.add_argument("--yamnet-dir", type=Path, default=Path("artifacts/pretrained/yamnet"))
    parser.add_argument("--panns-checkpoint", type=Path, default=Path("artifacts/pretrained/panns/Cnn14_16k_mAP_0.438.pth"))
    parser.add_argument("--work-dir", type=Path, default=Path("artifacts/work_musify_four_dataset_hybrid_v1"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/musify_four_dataset_hybrid_v1"))
    parser.add_argument("--seed", type=int, default=20261022)
    parser.add_argument("--test-fraction", type=float, default=0.20)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--max-windows", type=int, default=12)
    parser.add_argument(
        "--all-windows",
        action="store_true",
        help="Use every 30-second window at the configured hop instead of capping windows per track.",
    )
    parser.add_argument("--c-values", type=float, nargs="+", default=[0.001, 0.01, 0.1, 1.0])
    parser.add_argument(
        "--feature-sets",
        nargs="+",
        choices=FEATURE_SETS,
        default=list(FEATURE_SETS),
        help=(
            "Feature representations to compare. Use only 'hybrid' to train the "
            "combined YAMNet + PANNs submission model."
        ),
    )
    args = parser.parse_args()
    try:
        train(args)
    except (FileNotFoundError, TypeError, ValueError, RuntimeError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
