"""Train Musify's from-scratch CNN with GTZAN and mapped FMA-Small tracks.

GTZAN drives validation and primary held-out evaluation. FMA-Small contributes
only its Hip-Hop, Pop, and Rock tracks; its held-out split is reported separately.
"""

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import io
import json
import os
import random
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from zipfile import BadZipFile, ZipFile

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from .audio import load_audio, mel_spectrogram
from .config import AudioConfig
from .data import discover_tracks, split_tracks
from .model import build_model


@dataclass(frozen=True)
class SourceTrack:
    """One labeled recording, kept together across all dataset splits."""

    path: Path
    label: str
    track_id: str
    source: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_fma_small_tracks(metadata_archive: Path, audio_root: Path) -> list[SourceTrack]:
    """Read FMA metadata and map only labels that match Musify's GTZAN classes."""
    label_map = {
        "hip-hop": "hiphop",
        "hip hop": "hiphop",
        "hiphop": "hiphop",
        "pop": "pop",
        "rock": "rock",
    }
    audio_root = audio_root.expanduser().resolve()
    if not audio_root.is_dir():
        raise FileNotFoundError(f"FMA-Small audio folder does not exist: {audio_root}")
    if not metadata_archive.is_file():
        raise FileNotFoundError(f"FMA metadata archive does not exist: {metadata_archive}")

    tracks: list[SourceTrack] = []
    missing = 0
    metadata_rows = 0
    try:
        with ZipFile(metadata_archive) as archive:
            csv_name = next(
                name for name in archive.namelist()
                if name.endswith("/tracks.csv")
            )
            with archive.open(csv_name) as binary_stream:
                reader = csv.reader(io.TextIOWrapper(binary_stream, encoding="utf-8"))
                groups = next(reader)
                columns = next(reader)
                next(reader)  # Units row.
                subset_index = next(
                    index for index, pair in enumerate(zip(groups, columns))
                    if pair == ("set", "subset")
                )
                genre_index = next(
                    index for index, pair in enumerate(zip(groups, columns))
                    if pair == ("track", "genre_top")
                )

                for row in reader:
                    if not row or len(row) <= max(subset_index, genre_index):
                        continue
                    if row[subset_index].strip().lower() != "small":
                        continue
                    metadata_rows += 1
                    label = label_map.get(row[genre_index].strip().lower())
                    if label is None:
                        continue
                    try:
                        numeric_id = int(row[0])
                    except (TypeError, ValueError):
                        continue
                    path = audio_root / f"{numeric_id // 1000:03d}" / f"{numeric_id:06d}.mp3"
                    if not path.is_file():
                        missing += 1
                        continue
                    tracks.append(
                        SourceTrack(
                            path=path,
                            label=label,
                            track_id=_sha256(path),
                            source="fma_small",
                        )
                    )
    except (BadZipFile, StopIteration, KeyError, OSError) as exc:
        raise ValueError(f"Could not read FMA metadata from {metadata_archive}: {exc}") from exc

    print(
        f"FMA-Small metadata rows: {metadata_rows}; "
        f"mapped audio found: {len(tracks)}; missing mapped audio: {missing}",
        flush=True,
    )
    if not tracks:
        raise ValueError(
            "No FMA-Small Hip-Hop, Pop, or Rock audio was found. "
            "Check that the archive was extracted and --fma-audio points at its root."
        )
    return tracks


def filter_decodable_fma_tracks(
    tracks: list[SourceTrack], *, minimum_seconds: float = 8.0
) -> tuple[list[SourceTrack], list[tuple[SourceTrack, str]]]:
    """Remove malformed/truncated MP3s before they can abort a training epoch."""

    def inspect(track: SourceTrack) -> str | None:
        try:
            info = sf.info(str(track.path))
            if info.duration < minimum_seconds:
                return f"audio shorter than {minimum_seconds:g} seconds ({info.duration:.3f}s)"
            audio, _ = sf.read(str(track.path), dtype="float32", always_2d=True)
            if audio.size == 0 or not np.isfinite(audio).all():
                return "audio is empty or contains non-finite samples"
        except Exception as exc:
            return f"audio decode failed ({type(exc).__name__}: {exc})"
        return None

    reasons: dict[int, str] = {}
    worker_count = min(8, max(1, os.cpu_count() or 4))
    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        futures = {
            pool.submit(inspect, track): index for index, track in enumerate(tracks)
        }
        for checked, future in enumerate(as_completed(futures), start=1):
            index = futures[future]
            reason = future.result()
            if reason is not None:
                reasons[index] = reason
            if checked % 500 == 0 or checked == len(tracks):
                print(f"Audio preflight: {checked}/{len(tracks)} FMA tracks checked", flush=True)

    usable = [track for index, track in enumerate(tracks) if index not in reasons]
    excluded = [(tracks[index], reason) for index, reason in sorted(reasons.items())]
    print(
        f"FMA audio preflight: usable={len(usable)}, excluded={len(excluded)}",
        flush=True,
    )
    return usable, excluded


def _deduplicate_sources(
    gtzan_tracks: list[SourceTrack], fma_tracks: list[SourceTrack]
) -> tuple[list[SourceTrack], list[SourceTrack], int]:
    """Remove exact byte duplicates across sources and conflicting-label groups."""
    first_by_hash: dict[str, SourceTrack] = {}
    conflicting: set[str] = set()
    duplicates = 0
    for track in (*gtzan_tracks, *fma_tracks):
        previous = first_by_hash.get(track.track_id)
        if previous is None:
            first_by_hash[track.track_id] = track
            continue
        duplicates += 1
        if previous.label != track.label:
            conflicting.add(track.track_id)

    if conflicting:
        warnings.warn(
            f"Excluding {len(conflicting)} exact-audio group(s) with conflicting labels.",
            RuntimeWarning,
        )
    kept = [
        track for digest, track in first_by_hash.items()
        if digest not in conflicting
    ]
    return (
        [track for track in kept if track.source == "gtzan"],
        [track for track in kept if track.source == "fma_small"],
        duplicates,
    )


class MultiSourceSequence(tf.keras.utils.Sequence):
    """Create one randomly cropped mel example per source track and epoch."""

    def __init__(
        self,
        tracks: list[SourceTrack],
        class_to_id: dict[str, int],
        config: AudioConfig,
        batch_size: int,
        *,
        training: bool,
        seed: int,
        sample_weights: np.ndarray | None = None,
    ):
        super().__init__()
        if sample_weights is not None and len(sample_weights) != len(tracks):
            raise ValueError("One training weight is required per track")
        self.tracks = tracks
        self.class_to_id = class_to_id
        self.config = config
        self.batch_size = batch_size
        self.training = training
        self.seed = seed
        self.sample_weights = sample_weights
        self.epoch = 0
        self.rng = np.random.default_rng(seed)
        self.order = np.arange(len(tracks))

    def __len__(self) -> int:
        return int(np.ceil(len(self.tracks) / self.batch_size))

    def on_epoch_end(self) -> None:
        self.epoch += 1
        self.rng = np.random.default_rng(self.seed + self.epoch)
        if self.training:
            self.rng.shuffle(self.order)

    def _spec_augment(self, features: np.ndarray) -> np.ndarray:
        features = features.copy()
        mel_bins, frames, _ = features.shape
        frequency_width = int(self.rng.integers(0, min(9, mel_bins + 1)))
        if frequency_width:
            start = int(self.rng.integers(0, mel_bins - frequency_width + 1))
            features[start:start + frequency_width, :, :] = 0.0
        time_width = int(self.rng.integers(0, min(21, frames + 1)))
        if time_width:
            start = int(self.rng.integers(0, frames - time_width + 1))
            features[:, start:start + time_width, :] = 0.0
        return features

    def __getitem__(self, index: int):
        indices = self.order[index * self.batch_size:(index + 1) * self.batch_size]
        xs, labels, weights = [], [], []
        for track_index in indices:
            track = self.tracks[int(track_index)]
            try:
                audio = load_audio(str(track.path), self.config)
                features = mel_spectrogram(
                    audio, self.config, training=self.training, rng=self.rng
                )
            except (ValueError, OSError) as exc:
                raise ValueError(f"Could not prepare {track.source} track {track.path}: {exc}") from exc
            if self.training:
                features = self._spec_augment(features)
            xs.append(features)
            labels.append(self.class_to_id[track.label])
            if self.sample_weights is not None:
                weights.append(float(self.sample_weights[int(track_index)]))

        x_batch = np.stack(xs).astype(np.float32, copy=False)
        y_batch = np.asarray(labels, dtype=np.int32)
        if self.sample_weights is None:
            return x_batch, y_batch
        return x_batch, y_batch, np.asarray(weights, dtype=np.float32)


def balanced_source_weights(
    tracks: list[SourceTrack], classes: list[str]
) -> np.ndarray:
    """Balance class mass, then divide shared-class mass evenly by data source."""
    group_counts = Counter((track.source, track.label) for track in tracks)
    source_counts_per_class = Counter(label for _, label in group_counts)
    weights = np.asarray(
        [
            1.0 / (
                len(classes)
                * source_counts_per_class[track.label]
                * group_counts[(track.source, track.label)]
            )
            for track in tracks
        ],
        dtype=np.float64,
    )
    weights *= len(weights) / weights.sum()
    return weights.astype(np.float32)


def _write_manifest(
    path: Path,
    seed: int,
    gtzan_root: Path,
    fma_root: Path,
    gtzan_splits: dict[str, list[SourceTrack]],
    fma_splits: dict[str, list[SourceTrack]],
) -> None:
    def dataset_payload(root: Path, splits: dict[str, list[SourceTrack]]) -> dict:
        resolved_root = root.expanduser().resolve()
        return {
            "paths_relative_to": "this dataset's root folder",
            "splits": {
                name: [
                    {
                        "path": track.path.relative_to(resolved_root).as_posix(),
                        "label": track.label,
                        "source_track_sha256": track.track_id,
                    }
                    for track in rows
                ]
                for name, rows in splits.items()
            },
        }

    payload = {
        "seed": seed,
        "split_policy": "track-level stratified split before audio cropping",
        "datasets": {
            "gtzan": dataset_payload(gtzan_root, gtzan_splits),
            "fma_small_mapped": dataset_payload(fma_root, fma_splits),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _evaluate(
    model: tf.keras.Model,
    tracks: list[SourceTrack],
    classes: list[str],
    class_to_id: dict[str, int],
    config: AudioConfig,
    batch_size: int,
    seed: int,
    *,
    labels_to_score: list[str] | None = None,
) -> dict:
    sequence = MultiSourceSequence(
        tracks, class_to_id, config, batch_size, training=False, seed=seed
    )
    probabilities = model.predict(sequence, verbose=1)
    y_true = np.asarray([class_to_id[track.label] for track in tracks], dtype=np.int32)
    y_pred = probabilities.argmax(axis=1)
    ids = list(range(len(classes))) if labels_to_score is None else [
        class_to_id[name] for name in labels_to_score
    ]
    target_names = classes if labels_to_score is None else labels_to_score
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=ids, zero_division=0
    )
    return {
        "test_accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "classes_scored": target_names,
        "track_count": len(y_true),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=ids,
            target_names=target_names,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=ids).tolist(),
    }


def _plot_confusion_matrix(matrix: np.ndarray, classes: list[str], output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(max(8, len(classes) * 0.8), max(7, len(classes) * 0.7)))
    image = axis.imshow(matrix, cmap="magma")
    axis.set(
        xticks=range(len(classes)),
        yticks=range(len(classes)),
        xticklabels=classes,
        yticklabels=classes,
        xlabel="Predicted genre",
        ylabel="True genre",
        title="GTZAN held-out test confusion matrix",
    )
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(len(classes)):
        for column in range(len(classes)):
            axis.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                color="white" if matrix[row, column] < threshold else "black",
                fontsize=8,
            )
    figure.colorbar(image, ax=axis, label="Track count")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def train(args) -> dict:
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.keras.utils.set_random_seed(args.seed)
    config = AudioConfig()

    gtzan_original = discover_tracks(args.gtzan_data)
    gtzan_tracks = [
        SourceTrack(track.path, track.label, track.track_id, "gtzan")
        for track in gtzan_original
    ]
    fma_tracks = discover_fma_small_tracks(args.fma_metadata, args.fma_audio)
    fma_tracks, excluded_fma_tracks = filter_decodable_fma_tracks(fma_tracks)
    gtzan_tracks, fma_tracks, duplicate_count = _deduplicate_sources(
        gtzan_tracks, fma_tracks
    )

    classes = sorted({track.label for track in gtzan_tracks})
    fma_labels = {track.label for track in fma_tracks}
    if not fma_labels.issubset(set(classes)):
        raise ValueError(f"Mapped FMA labels are not present in GTZAN: {sorted(fma_labels - set(classes))}")
    if fma_labels != {"hiphop", "pop", "rock"}:
        raise ValueError(
            "Expected FMA-Small to provide Hip-Hop, Pop, and Rock; "
            f"found {sorted(fma_labels)}."
        )

    gtzan_splits = split_tracks(
        gtzan_tracks, args.seed, args.test_fraction, args.validation_fraction
    )
    fma_splits = split_tracks(
        fma_tracks, args.seed, args.test_fraction, args.validation_fraction
    )
    train_tracks = gtzan_splits["train"] + fma_splits["train"]
    validation_tracks = gtzan_splits["validation"]
    test_tracks = gtzan_splits["test"]
    fma_test_tracks = fma_splits["test"]

    for label in classes:
        if sum(track.label == label for track in gtzan_splits["train"]) < 3:
            raise ValueError(f"Too few GTZAN train tracks for class {label}")

    args.output.mkdir(parents=True, exist_ok=False)
    excluded_payload = [
        {
            "path": track.path.relative_to(args.fma_audio.expanduser().resolve()).as_posix(),
            "label": track.label,
            "source_track_sha256": track.track_id,
            "reason": reason,
        }
        for track, reason in excluded_fma_tracks
    ]
    (args.output / "excluded_fma_tracks.json").write_text(
        json.dumps(excluded_payload, indent=2), encoding="utf-8"
    )
    _write_manifest(
        args.output / "split_manifest.json",
        args.seed,
        args.gtzan_data,
        args.fma_audio,
        gtzan_splits,
        fma_splits,
    )
    config.save(args.output / "preprocessing.json", classes)
    config_path = args.output / "preprocessing.json"
    config_payload = json.loads(config_path.read_text(encoding="utf-8"))
    config_payload.update(
        {
            "model_type": "musify_multiscale_cnn",
            "architecture": "Musify Multi-Scale Residual AudioNet v1",
            "training_sources": ["GTZAN", "FMA-Small"],
            "fma_label_mapping": {
                "Hip-Hop": "hiphop",
                "Pop": "pop",
                "Rock": "rock",
                "excluded_fma_genres": [
                    "Folk", "Experimental", "International", "Electronic", "Instrumental"
                ],
            },
            "sample_weight_policy": "equal class mass; shared-class mass split evenly by source",
            "format_version": 2,
        }
    )
    config_path.write_text(json.dumps(config_payload, indent=2), encoding="utf-8")
    class_to_id = {name: index for index, name in enumerate(classes)}
    sample_weights = balanced_source_weights(train_tracks, classes)
    model = build_model(config, len(classes), architecture="multiscale")
    model.summary()

    checkpoint = args.output / "best_model.keras"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint, monitor="val_accuracy", mode="max", save_best_only=True
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=args.patience,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6
        ),
        tf.keras.callbacks.CSVLogger(args.output / "training_history.csv"),
    ]
    history = model.fit(
        MultiSourceSequence(
            train_tracks,
            class_to_id,
            config,
            args.batch_size,
            training=True,
            seed=args.seed,
            sample_weights=sample_weights,
        ),
        validation_data=MultiSourceSequence(
            validation_tracks,
            class_to_id,
            config,
            args.batch_size,
            training=False,
            seed=args.seed,
        ),
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    model = tf.keras.models.load_model(checkpoint)
    gtzan_metrics = _evaluate(
        model, test_tracks, classes, class_to_id, config, args.batch_size, args.seed
    )
    fma_metrics = _evaluate(
        model,
        fma_test_tracks,
        classes,
        class_to_id,
        config,
        args.batch_size,
        args.seed,
        labels_to_score=["hiphop", "pop", "rock"],
    )
    matrix = np.asarray(gtzan_metrics["confusion_matrix"], dtype=np.int64)
    _plot_confusion_matrix(matrix, classes, args.output / "confusion_matrix.png")

    best_epoch = int(np.argmax(history.history["val_accuracy"]))
    report = {
        "model_architecture": config_payload["architecture"],
        "primary_test_dataset": "GTZAN",
        "test_accuracy": gtzan_metrics["test_accuracy"],
        "macro_precision": gtzan_metrics["macro_precision"],
        "macro_recall": gtzan_metrics["macro_recall"],
        "macro_f1": gtzan_metrics["macro_f1"],
        "classes": classes,
        "test_track_count": gtzan_metrics["track_count"],
        "classification_report": gtzan_metrics["classification_report"],
        "confusion_matrix": matrix.tolist(),
        "fma_small_external_test": fma_metrics,
        "track_counts": {
            "gtzan": {name: len(rows) for name, rows in gtzan_splits.items()},
            "fma_small_mapped": {name: len(rows) for name, rows in fma_splits.items()},
        },
        "exact_duplicate_tracks_removed": duplicate_count,
        "unreadable_or_truncated_fma_tracks_excluded": len(excluded_fma_tracks),
        "sample_weight_policy": config_payload["sample_weight_policy"],
        "checkpoint_selection": "GTZAN validation accuracy only",
        "checkpoint_epoch": best_epoch + 1,
        "checkpoint_val_accuracy": float(history.history["val_accuracy"][best_epoch]),
        "checkpoint_val_loss": float(history.history["val_loss"][best_epoch]),
        "best_gtzan_validation_accuracy": float(max(history.history["val_accuracy"])),
    }
    (args.output / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "gtzan_test_accuracy": report["test_accuracy"],
                "gtzan_macro_f1": report["macro_f1"],
                "fma_small_external_test_accuracy": fma_metrics["test_accuracy"],
                "fma_small_external_test_macro_f1": fma_metrics["macro_f1"],
                "checkpoint_epoch": report["checkpoint_epoch"],
                "test_track_count": report["test_track_count"],
            },
            indent=2,
        )
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gtzan-data", type=Path, default=Path("data/raw/Data/genres_original")
    )
    parser.add_argument("--fma-audio", type=Path, default=Path("data/raw/fma_small"))
    parser.add_argument(
        "--fma-metadata",
        type=Path,
        default=Path.home() / "Downloads" / "fma_metadata.zip",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/musify_fma_multisource_cnn_v1")
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    args = parser.parse_args()

    if args.epochs < 1 or args.patience < 1 or args.batch_size < 1:
        parser.error("--epochs, --patience, and --batch-size must be positive")
    try:
        train(args)
    except FileExistsError as exc:
        parser.error(f"Output already exists; choose another --output path: {exc.filename}")


if __name__ == "__main__":
    main()
