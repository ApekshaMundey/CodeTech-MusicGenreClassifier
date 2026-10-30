"""Train an optional GTZAN + Saraga hierarchical-label hybrid classifier.

The existing Musify checkpoints are not overwritten. GTZAN's broad classical label is
renamed Western classical; Saraga contributes a separate Hindustani classical class.
Features are extracted from one centered 30-second window per track using frozen
YAMNet/PANNs encoders, then model selection uses group-aware CV.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import shutil
import tempfile
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from zipfile import BadZipFile, ZipFile

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import Track, discover_tracks, split_tracks
from .panns import (
    PANNs_CHECKPOINT_MD5,
    PANNs_EMBEDDING_SIZE,
    PANNs_SAMPLE_RATE,
    extract_panns_embedding,
    load_panns,
)
from .yamnet import (
    YAMNET_EMBEDDING_SIZE,
    extract_track_embedding_from_audio,
    load_yamnet,
    load_yamnet_audio,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SARAGA_ROOT_IN_ARCHIVE = "saraga1.5_hindustani"
SUPPORTED_SUFFIXES = {".wav", ".au", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}
SPEECH_MARKERS = (
    "introductory speech",
    "introduction about",
    "shrutinandan concept - an introduction",
)
CLIP_SECONDS = 30.0
FEATURE_CACHE_VERSION = 1
YAMNET_FEATURE_SIZE = YAMNET_EMBEDDING_SIZE * 3
HYBRID_FEATURE_SIZE = YAMNET_FEATURE_SIZE + PANNs_EMBEDDING_SIZE


@dataclass(frozen=True)
class Sample:
    source: str
    label: str
    sample_id: str
    audio_hash: str
    group_id: str
    location: str
    path: Path


def _clip_center(audio: np.ndarray, samples: int) -> np.ndarray:
    if audio.size > samples:
        start = (audio.size - samples) // 2
        return np.asarray(audio[start : start + samples], dtype=np.float32)
    if audio.size < samples:
        left = (samples - audio.size) // 2
        return np.pad(audio, (left, samples - audio.size - left)).astype(np.float32)
    return np.asarray(audio, dtype=np.float32)


def _saraga_member_for_path(filepath: str, names: set[str]) -> str | None:
    base = f"{SARAGA_ROOT_IN_ARCHIVE}/{filepath}"
    suffixes = (".mp3.mp3", ".wav", ".au", ".mp3", ".flac", ".ogg", ".m4a", ".aac")
    for suffix in suffixes:
        candidate = base + suffix
        if candidate in names:
            return candidate
    return None


def _copy_and_hash_member(
    archive: ZipFile, member: str, destination: Path
) -> str:
    digest = hashlib.sha256()
    with archive.open(member) as source, destination.open("wb") as target:
        while block := source.read(1024 * 1024):
            digest.update(block)
            target.write(block)
    return digest.hexdigest()


def _load_saraga_tracks(
    archive_path: Path, work_dir: Path
) -> tuple[list[Sample], list[str], list[str], int]:
    if not archive_path.is_file():
        raise FileNotFoundError(f"Saraga archive not found: {archive_path}")

    tracks: list[Sample] = []
    excluded: list[str] = []
    errors: list[str] = []
    metadata_count = 0
    try:
        with ZipFile(archive_path) as archive:
            names = set(archive.namelist())
            csv_member = f"{SARAGA_ROOT_IN_ARCHIVE}/file_paths.csv"
            if csv_member not in names:
                raise ValueError(f"Missing Saraga index in archive: {csv_member}")
            raw = archive.read(csv_member).decode("utf-8-sig", errors="replace")
            for row_index, row in enumerate(csv.DictReader(io.StringIO(raw)), start=1):
                filepath = (row.get("filepath") or "").strip().replace("\\", "/").strip("/")
                if not filepath:
                    errors.append(f"Index row {row_index}: missing filepath")
                    continue
                metadata_count += 1
                lowered = filepath.casefold()
                if any(marker in lowered for marker in SPEECH_MARKERS):
                    excluded.append(filepath)
                    continue
                member = _saraga_member_for_path(filepath, names)
                if member is None:
                    errors.append(f"Index row {row_index}: no audio member for {filepath}")
                    continue
                suffix = PurePosixPath(member).suffix.lower()
                if suffix not in SUPPORTED_SUFFIXES:
                    errors.append(f"Unsupported audio suffix for {member}")
                    continue
                target = work_dir / f"saraga_{row_index:04d}{suffix}"
                try:
                    audio_hash = _copy_and_hash_member(archive, member, target)
                except (OSError, BadZipFile, RuntimeError) as exc:
                    target.unlink(missing_ok=True)
                    errors.append(f"{member}: archive read failed: {exc}")
                    continue
                album = PurePosixPath(filepath).parts[0]
                tracks.append(
                    Sample(
                        source="saraga",
                        label="Hindustani classical",
                        sample_id=f"saraga:{audio_hash}",
                        audio_hash=audio_hash,
                        group_id=f"saraga:{album}",
                        location=member,
                        path=target,
                    )
                )
    except BadZipFile as exc:
        raise ValueError(f"Saraga ZIP is unreadable: {archive_path}") from exc

    if not tracks:
        raise ValueError("No eligible audio tracks were found in the Saraga archive")
    return tracks, excluded, errors, metadata_count


def _deduplicate(
    gtzan_tracks: list[Track], saraga_tracks: list[Sample]
) -> tuple[list[Sample], dict[str, int]]:
    candidates = [
        Sample(
            source="gtzan",
            label="Western classical" if track.label == "classical" else track.label,
            sample_id=f"gtzan:{track.track_id}",
            audio_hash=track.track_id,
            group_id=f"gtzan:{track.track_id}",
            location=track.path.as_posix(),
            path=track.path,
        )
        for track in gtzan_tracks
    ] + saraga_tracks

    by_hash: dict[str, list[Sample]] = defaultdict(list)
    for sample in candidates:
        by_hash[sample.audio_hash].append(sample)

    kept: list[Sample] = []
    duplicate_groups = 0
    duplicate_tracks = 0
    conflicting_groups = 0
    for same_audio in by_hash.values():
        if len(same_audio) == 1:
            kept.append(same_audio[0])
            continue
        duplicate_groups += 1
        duplicate_tracks += len(same_audio) - 1
        if len({sample.label for sample in same_audio}) > 1:
            conflicting_groups += 1
            continue
        kept.append(min(same_audio, key=lambda sample: (sample.source, sample.sample_id)))

    stats = {
        "exact_duplicate_groups": duplicate_groups,
        "exact_duplicate_tracks_removed": duplicate_tracks,
        "conflicting_label_groups_excluded": conflicting_groups,
    }
    return sorted(kept, key=lambda sample: (sample.source, sample.sample_id)), stats


def _load_cached_gtzan_features(samples: list[Sample]) -> dict[str, np.ndarray]:
    """Reuse the project's previously verified per-track GTZAN embeddings when aligned."""
    gtzan_samples = [sample for sample in samples if sample.source == "gtzan"]
    expected_hashes = {sample.audio_hash for sample in gtzan_samples}
    yamnet_path = PROJECT_ROOT / "artifacts" / "yamnet_v1" / "yamnet_embeddings.npz"
    panns_path = PROJECT_ROOT / "artifacts" / "panns_v1" / "panns_embeddings.npz"
    if not gtzan_samples or not yamnet_path.is_file() or not panns_path.is_file():
        return {}

    try:
        with np.load(yamnet_path, allow_pickle=False) as cache:
            y_ids = cache["track_ids"].astype(str).tolist()
            y_features = np.asarray(cache["features"], dtype=np.float32)
            y_version = int(cache["format_version"].item())
        with np.load(panns_path, allow_pickle=False) as cache:
            p_ids = cache["track_ids"].astype(str).tolist()
            p_features = np.asarray(cache["features"], dtype=np.float32)
            p_version = int(cache["format_version"].item())
    except (EOFError, KeyError, OSError, ValueError) as exc:
        warnings.warn(f"Could not read reusable GTZAN embeddings: {exc}")
        return {}

    if (
        y_version != 1
        or p_version != 1
        or y_ids != p_ids
        or set(y_ids) != expected_hashes
        or y_features.shape != (len(y_ids), YAMNET_FEATURE_SIZE)
        or p_features.shape != (len(p_ids), PANNs_EMBEDDING_SIZE)
        or not np.isfinite(y_features).all()
        or not np.isfinite(p_features).all()
    ):
        warnings.warn("GTZAN embedding cache IDs or dimensions do not match; recomputing them.")
        return {}

    y_by_hash = dict(zip(y_ids, y_features, strict=True))
    p_by_hash = dict(zip(p_ids, p_features, strict=True))
    result = {
        sample.sample_id: np.concatenate(
            (y_by_hash[sample.audio_hash], p_by_hash[sample.audio_hash])
        ).astype(np.float32, copy=False)
        for sample in gtzan_samples
    }
    print(
        f"Reusing existing YAMNet/PANNs cache for {len(result)} GTZAN tracks "
        "(all content hashes matched).",
        flush=True,
    )
    return result


def _extract_features(
    samples: list[Sample],
    yamnet,
    panns,
    cache_path: Path,
) -> tuple[dict[str, np.ndarray], list[dict[str, str]]]:
    ids = [sample.sample_id for sample in samples]
    if cache_path.is_file():
        try:
            with np.load(cache_path, allow_pickle=False) as cache:
                cached_ids = cache["track_ids"].astype(str).tolist()
                features = np.asarray(cache["features"], dtype=np.float32)
                version = int(cache["format_version"].item())
            if (
                cached_ids == ids
                and version == FEATURE_CACHE_VERSION
                and features.shape == (len(samples), HYBRID_FEATURE_SIZE)
                and np.isfinite(features).all()
            ):
                print(f"Reusing combined feature cache: {cache_path}", flush=True)
                return dict(zip(ids, features, strict=True)), []
        except (EOFError, KeyError, OSError, ValueError) as exc:
            warnings.warn(f"Combined feature cache will be rebuilt: {exc}")

    features_by_id = _load_cached_gtzan_features(samples)
    pending = [sample for sample in samples if sample.sample_id not in features_by_id]
    errors: list[dict[str, str]] = []
    clip_samples = int(CLIP_SECONDS * PANNs_SAMPLE_RATE)
    for index, sample in enumerate(pending, start=1):
        try:
            audio = load_yamnet_audio(sample.path)
            audio = _clip_center(audio, clip_samples)
            yamnet_features = extract_track_embedding_from_audio(yamnet, audio)
            panns_features = extract_panns_embedding(panns, audio)
            features = np.concatenate((yamnet_features, panns_features)).astype(
                np.float32, copy=False
            )
            if features.shape != (HYBRID_FEATURE_SIZE,) or not np.isfinite(features).all():
                raise ValueError(f"Invalid hybrid feature shape: {features.shape}")
            features_by_id[sample.sample_id] = features
        except (ValueError, RuntimeError, OSError) as exc:
            errors.append(
                {"source": sample.source, "location": sample.location, "error": str(exc)}
            )
        if index % 10 == 0 or index == len(pending):
            print(
                f"New tracks embedded: {index}/{len(pending)} "
                f"(total valid {len(features_by_id)}, errors {len(errors)})",
                flush=True,
            )

    valid_samples = [sample for sample in samples if sample.sample_id in features_by_id]
    if not valid_samples:
        raise RuntimeError("No tracks could be converted into hybrid features")
    valid_ids = [sample.sample_id for sample in valid_samples]
    feature_matrix = np.stack([features_by_id[sample_id] for sample_id in valid_ids])
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_name(cache_path.stem + ".tmp.npz")
    np.savez_compressed(
        temporary,
        track_ids=np.asarray(valid_ids, dtype="U80"),
        features=feature_matrix.astype(np.float32, copy=False),
        format_version=np.asarray(FEATURE_CACHE_VERSION, dtype=np.int32),
    )
    temporary.replace(cache_path)
    return features_by_id, errors

def _split_tracks(
    samples: list[Sample], seed: int, test_fraction: float, validation_fraction: float
) -> dict[str, list[Sample]]:
    if test_fraction <= 0 or validation_fraction <= 0 or test_fraction + validation_fraction >= 1:
        raise ValueError("Test and validation fractions must be positive and sum to less than one")

    by_source = {source: [sample for sample in samples if sample.source == source]
                 for source in ("gtzan", "saraga")}
    if not by_source["gtzan"] or not by_source["saraga"]:
        raise ValueError("Both GTZAN and Saraga tracks are required")

    gtzan_splits = split_tracks(
        [
            Track(sample.path, sample.label, sample.sample_id)
            for sample in by_source["gtzan"]
        ],
        seed=seed,
        test_fraction=test_fraction,
        validation_fraction=validation_fraction,
    )
    gtzan_by_id = {sample.sample_id: sample for sample in by_source["gtzan"]}
    result = {
        name: [gtzan_by_id[track.track_id] for track in rows]
        for name, rows in gtzan_splits.items()
    }

    saraga = by_source["saraga"]
    groups = np.asarray([sample.group_id for sample in saraga], dtype=str)
    indices = np.arange(len(saraga))
    first_split = GroupShuffleSplit(
        n_splits=1, test_size=test_fraction, random_state=seed
    )
    development_idx, test_idx = next(first_split.split(indices, groups=groups))
    validation_relative_fraction = validation_fraction / (1.0 - test_fraction)
    second_split = GroupShuffleSplit(
        n_splits=1,
        test_size=validation_relative_fraction,
        random_state=seed + 1,
    )
    train_relative, validation_relative = next(
        second_split.split(development_idx, groups=groups[development_idx])
    )
    saraga_splits = {
        "train": [saraga[int(i)] for i in development_idx[train_relative]],
        "validation": [saraga[int(i)] for i in development_idx[validation_relative]],
        "test": [saraga[int(i)] for i in test_idx],
    }
    for name, rows in result.items():
        rows.extend(saraga_splits[name])
        rows.sort(key=lambda sample: sample.sample_id)

    owners: dict[str, str] = {}
    track_owners: dict[str, str] = {}
    for split_name, rows in result.items():
        for sample in rows:
            if sample.sample_id in track_owners:
                raise ValueError(f"Track leakage across splits: {sample.location}")
            track_owners[sample.sample_id] = split_name
            if sample.source == "saraga":
                previous = owners.setdefault(sample.group_id, split_name)
                if previous != split_name:
                    raise ValueError(f"Saraga release leakage across splits: {sample.group_id}")
    classes = {sample.label for sample in samples}
    for split_name, rows in result.items():
        present = {sample.label for sample in rows}
        if present != classes:
            raise ValueError(
                f"{split_name} split is missing classes: {sorted(classes - present)}"
            )
    return result


def _feature_set(matrix: np.ndarray, name: str) -> np.ndarray:
    if name == "yamnet":
        return matrix[:, :YAMNET_FEATURE_SIZE]
    if name == "panns":
        return matrix[:, YAMNET_FEATURE_SIZE:]
    return matrix


def _candidate_models() -> dict[str, tuple[str, object]]:
    return {
        "yamnet_logistic": (
            "yamnet",
            make_pipeline(
                StandardScaler(),
                LogisticRegression(C=0.01, max_iter=3_000, class_weight="balanced"),
            ),
        ),
        "panns_logistic": (
            "panns",
            make_pipeline(
                StandardScaler(),
                LogisticRegression(C=0.1, max_iter=3_000, class_weight="balanced"),
            ),
        ),
        "hybrid_logistic": (
            "hybrid",
            make_pipeline(
                StandardScaler(),
                LogisticRegression(C=0.01, max_iter=3_000, class_weight="balanced"),
            ),
        ),
    }


def _select_candidate(
    matrix: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    classes: list[str],
    folds: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    if len(set(groups.tolist())) < folds:
        raise ValueError(f"Need at least {folds} distinct development groups for CV")
    splitter = GroupKFold(n_splits=folds)
    split_indices = list(splitter.split(matrix, labels, groups))
    scores: dict[str, dict[str, float]] = {}
    for candidate_name, (feature_name, estimator) in _candidate_models().items():
        accuracies: list[float] = []
        macro_f1s: list[float] = []
        for train_idx, validation_idx in split_indices:
            model = clone(estimator)
            model.fit(
                _feature_set(matrix[train_idx], feature_name),
                labels[train_idx],
            )
            predictions = model.predict(
                _feature_set(matrix[validation_idx], feature_name)
            )
            accuracies.append(float(accuracy_score(labels[validation_idx], predictions)))
            macro_f1s.append(
                float(
                    precision_recall_fscore_support(
                        labels[validation_idx],
                        predictions,
                        labels=np.arange(len(classes)),
                        average="macro",
                        zero_division=0,
                    )[2]
                )
            )
        scores[candidate_name] = {
            "accuracy_mean": float(np.mean(accuracies)),
            "accuracy_std": float(np.std(accuracies)),
            "macro_f1_mean": float(np.mean(macro_f1s)),
            "macro_f1_std": float(np.std(macro_f1s)),
        }
        print(
            f"{candidate_name}: group-CV macro F1 "
            f"{scores[candidate_name]['macro_f1_mean']:.4f} "
            f"+/- {scores[candidate_name]['macro_f1_std']:.4f}",
            flush=True,
        )
    return scores


def _plot_confusion_matrix(matrix: np.ndarray, classes: list[str], output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(max(9, len(classes) * 0.85), max(8, len(classes) * 0.75)))
    image = axis.imshow(matrix, cmap="magma")
    axis.set(
        xticks=range(len(classes)),
        yticks=range(len(classes)),
        xticklabels=classes,
        yticklabels=classes,
        xlabel="Predicted label",
        ylabel="True label",
        title="GTZAN + Saraga held-out test confusion matrix",
    )
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.setp(axis.get_yticklabels(), rotation=0)
    figure.colorbar(image, ax=axis, label="Track count")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _write_manifest(
    splits: dict[str, list[Sample]],
    path: Path,
    seed: int,
    gtzan_root: Path,
    saraga_archive: Path,
    duplicate_stats: dict[str, int],
) -> None:
    payload = {
        "seed": seed,
        "split_policy": (
            "GTZAN stratified by track; Saraga grouped by top-level release; "
            "all exact byte-level audio duplicates removed before splitting."
        ),
        "source_roots": {
            "gtzan": str(gtzan_root),
            "saraga_archive": str(saraga_archive),
        },
        "duplicate_handling": duplicate_stats,
        "splits": {
            split: [
                {
                    "source": sample.source,
                    "path_or_archive_member": sample.location,
                    "label": sample.label,
                    "sample_id": sample.sample_id,
                    "group_id": sample.group_id,
                }
                for sample in rows
            ]
            for split, rows in splits.items()
        },
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def train(args: argparse.Namespace) -> dict:
    if args.cv_folds < 2:
        raise ValueError("--cv-folds must be at least 2")
    args.output.mkdir(parents=True, exist_ok=True)
    work_dir = Path(tempfile.mkdtemp(prefix="_saraga_audio_", dir=args.output))
    try:
        print("Discovering and de-duplicating GTZAN tracks...", flush=True)
        gtzan_tracks = discover_tracks(args.gtzan_data)
        saraga_tracks, excluded_speech, archive_errors, metadata_count = _load_saraga_tracks(
            args.saraga_archive, work_dir
        )
        print(
            f"GTZAN tracks: {len(gtzan_tracks)} | Saraga index rows: {metadata_count} | "
            f"eligible Saraga performances: {len(saraga_tracks)} | "
            f"spoken items excluded: {len(excluded_speech)}",
            flush=True,
        )
        samples, duplicate_stats = _deduplicate(gtzan_tracks, saraga_tracks)
        gtzan_count = sum(sample.source == "gtzan" for sample in samples)
        saraga_count = sum(sample.source == "saraga" for sample in samples)
        if gtzan_count < 100 or saraga_count < 20:
            raise ValueError(
                f"Unexpected source counts after exact deduplication: "
                f"GTZAN={gtzan_count}, Saraga={saraga_count}"
            )
        print(
            f"After exact deduplication: GTZAN={gtzan_count}, Saraga={saraga_count}; "
            f"cross-label duplicate groups excluded="
            f"{duplicate_stats['conflicting_label_groups_excluded']}",
            flush=True,
        )

        torch.set_num_threads(min(4, torch.get_num_threads()))
        try:
            import tensorflow as tf

            tf.config.threading.set_intra_op_parallelism_threads(4)
            tf.config.threading.set_inter_op_parallelism_threads(2)
        except RuntimeError:
            pass
        yamnet = load_yamnet(args.yamnet_model)
        panns = load_panns(args.panns_checkpoint)
        features_by_id, feature_errors = _extract_features(
            samples,
            yamnet,
            panns,
            args.output / "hybrid_features.npz",
        )
        valid_samples = [sample for sample in samples if sample.sample_id in features_by_id]
        split_errors = feature_errors + [
            {"source": "saraga_index", "location": error, "error": "Index/archive mapping issue"}
            for error in archive_errors
        ]
        if not valid_samples:
            raise RuntimeError("No valid tracks remain after audio preprocessing")

        splits = _split_tracks(
            valid_samples,
            seed=args.seed,
            test_fraction=args.test_fraction,
            validation_fraction=args.validation_fraction,
        )
        _write_manifest(
            splits,
            args.output / "split_manifest.json",
            args.seed,
            args.gtzan_data,
            args.saraga_archive,
            duplicate_stats,
        )
        (args.output / "excluded_saraga_speech.json").write_text(
            json.dumps(excluded_speech, indent=2), encoding="utf-8"
        )
        (args.output / "data_errors.json").write_text(
            json.dumps(split_errors, indent=2), encoding="utf-8"
        )

        classes = sorted({sample.label for sample in valid_samples})
        class_to_id = {name: index for index, name in enumerate(classes)}
        feature_by_id = features_by_id
        development = splits["train"] + splits["validation"]
        development_ids = [sample.sample_id for sample in development]
        development_x = np.stack([feature_by_id[sample_id] for sample_id in development_ids])
        development_y = np.asarray(
            [class_to_id[sample.label] for sample in development], dtype=np.int32
        )
        development_groups = np.asarray([sample.group_id for sample in development], dtype=str)
        cv_scores = _select_candidate(
            development_x,
            development_y,
            development_groups,
            classes,
            args.cv_folds,
            args.seed,
        )
        selected = max(
            cv_scores,
            key=lambda name: (
                cv_scores[name]["macro_f1_mean"],
                cv_scores[name]["accuracy_mean"],
            ),
        )
        feature_name, estimator = _candidate_models()[selected]
        estimator.fit(_feature_set(development_x, feature_name), development_y)

        test_samples = splits["test"]
        test_x = np.stack([feature_by_id[sample.sample_id] for sample in test_samples])
        test_y = np.asarray([class_to_id[sample.label] for sample in test_samples], dtype=np.int32)
        test_predictions = estimator.predict(_feature_set(test_x, feature_name))
        matrix = confusion_matrix(test_y, test_predictions, labels=np.arange(len(classes)))
        precision, recall, f1, support = precision_recall_fscore_support(
            test_y,
            test_predictions,
            labels=np.arange(len(classes)),
            zero_division=0,
        )
        report = classification_report(
            test_y,
            test_predictions,
            labels=np.arange(len(classes)),
            target_names=classes,
            output_dict=True,
            zero_division=0,
        )
        test_by_source = {
            source: sum(sample.source == source for sample in test_samples)
            for source in ("gtzan", "saraga")
        }
        previous_gtzan_test_overlap = 0
        previous_manifest = PROJECT_ROOT / "artifacts" / "yamnet_panns_combo_v1" / "split_manifest.json"
        if previous_manifest.is_file():
            previous_payload = json.loads(previous_manifest.read_text(encoding="utf-8"))
            previous_ids = {
                row.get("track_id")
                for row in previous_payload.get("splits", {}).get("test", [])
                if row.get("track_id")
            }
            current_ids = {
                sample.audio_hash for sample in test_samples if sample.source == "gtzan"
            }
            previous_gtzan_test_overlap = len(previous_ids & current_ids)
        summary = {
            "model_type": "yamnet_panns_logistic",
            "experiment": "GTZAN + Saraga Hindustani 11-label hybrid",
            "selected_candidate": selected,
            "selected_features": feature_name,
            "candidate_selection": (
                f"{args.cv_folds}-fold GroupKFold on train+validation only; "
                "Saraga release groups kept intact."
            ),
            "candidate_scores": cv_scores,
            "test_accuracy": float(accuracy_score(test_y, test_predictions)),
            "macro_precision": float(precision.mean()),
            "macro_recall": float(recall.mean()),
            "macro_f1": float(f1.mean()),
            "classes": classes,
            "class_hierarchy": {
                "Western classical": "Classical",
                "Hindustani classical": "Classical",
            },
            "test_track_count": len(test_y),
            "test_tracks_by_source": test_by_source,
            "gtzan_tracks_overlapping_previous_hybrid_test": previous_gtzan_test_overlap,
            "classification_report": report,
            "confusion_matrix": matrix.tolist(),
            "class_support": {
                name: int(support[index]) for index, name in enumerate(classes)
            },
            "track_counts_by_split_and_source": {
                split: {
                    source: sum(sample.source == source for sample in rows)
                    for source in ("gtzan", "saraga")
                }
                for split, rows in splits.items()
            },
            "split_integrity": {
                "track_level": "No track ID occurs in more than one split.",
                "saraga_grouping": "All performances in one top-level Saraga release stay in one split.",
            },
            "saraga_archive_source": "https://zenodo.org/records/4301737",
            "saraga_license_note": (
                "Saraga is published under CC BY-NC 4.0. Raw audio and this experiment's "
                "trained model artifacts remain local/ignored; review the dataset terms "
                "before redistributing derived artifacts or using them commercially."
            ),
            "evaluation_limitations": [
                "This is a controlled internal holdout, not an external benchmark.",
                "All Hindustani examples come from Saraga and all Western classical examples come from GTZAN, so class label and dataset source are confounded; the scores may partly reflect recording/source characteristics.",
                f"{previous_gtzan_test_overlap} of the {test_by_source['gtzan']} GTZAN test tracks overlap the previously recorded hybrid test set; this is not a fully blind final estimate.",
                "Saraga represents Hindustani performances and does not represent all Indian classical traditions.",
                "One centered 30-second excerpt represents each track for feature extraction.",
                "Model scores are not calibrated probabilities.",
            ],
            "exact_duplicate_handling": duplicate_stats,
            "feature_errors": len(feature_errors),
            "archive_index_errors": len(archive_errors),
            "spoken_items_excluded": len(excluded_speech),
            "seed": args.seed,
        }

        joblib.dump(estimator, args.output / "best_model.joblib")
        model_config = {
            "model_type": "yamnet_panns_logistic",
            "classes": classes,
            "class_hierarchy": summary["class_hierarchy"],
            "feature_order": ["yamnet_mean_std_max_3072", "panns_cnn14_16k_embedding_2048"],
            "feature_dimension": HYBRID_FEATURE_SIZE,
            "sample_rate": PANNs_SAMPLE_RATE,
            "audio_window_seconds": CLIP_SECONDS,
            "audio_hop_seconds": CLIP_SECONDS / 2,
            "yamnet_model_path": "artifacts/pretrained/yamnet",
            "panns_checkpoint_path": "artifacts/pretrained/panns/Cnn14_16k_mAP_0.438.pth",
            "panns_checkpoint_md5": PANNs_CHECKPOINT_MD5,
            "panns_architecture": "Cnn14_16k",
            "classifier": f"StandardScaler + LogisticRegression selected by group-CV ({selected})",
            "training_sources": ["GTZAN", "Saraga Hindustani"],
            "feature_extraction": (
                "Frozen YAMNet and PANNs embeddings from one centered 30-second mono 16 kHz window per track"
            ),
            "format_version": 1,
        }
        (args.output / "preprocessing.json").write_text(
            json.dumps(model_config, indent=2), encoding="utf-8"
        )
        (args.output / "evaluation.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        _plot_confusion_matrix(matrix, classes, args.output / "confusion_matrix.png")
        printable = {
            "selected_candidate": selected,
            "test_accuracy": summary["test_accuracy"],
            "macro_precision": summary["macro_precision"],
            "macro_recall": summary["macro_recall"],
            "macro_f1": summary["macro_f1"],
            "test_track_count": summary["test_track_count"],
            "test_tracks_by_source": test_by_source,
            "gtzan_tracks_overlapping_previous_hybrid_test": previous_gtzan_test_overlap,
            "hindustani_classical_test": report.get("Hindustani classical"),
            "western_classical_test": report.get("Western classical"),
        }
        print(json.dumps(printable, indent=2), flush=True)
        return summary
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gtzan-data", type=Path, default=Path("data/raw/Data/genres_original")
    )
    parser.add_argument(
        "--saraga-archive",
        type=Path,
        default=Path("data/raw/saraga_hindustani/saraga1.5_hindustani.zip"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/musify_saraga_hybrid_v1")
    )
    parser.add_argument(
        "--yamnet-model", type=Path, default=Path("artifacts/pretrained/yamnet")
    )
    parser.add_argument(
        "--panns-checkpoint",
        type=Path,
        default=Path("artifacts/pretrained/panns/Cnn14_16k_mAP_0.438.pth"),
    )
    parser.add_argument("--seed", type=int, default=31415)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    args = parser.parse_args()
    try:
        train(args)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
