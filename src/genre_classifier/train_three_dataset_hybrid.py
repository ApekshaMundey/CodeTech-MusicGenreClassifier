"""Train a three-source YAMNet/PANNs genre classifier from saved embeddings.

This experiment combines the existing track-level GTZAN and FMA-Small splits
with the release-grouped Saraga Hindustani split. The feature extractors remain
frozen; only the StandardScaler + logistic-regression classifier is fitted.
The saved feature caches and source manifests must be present locally.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURE_DIMENSION = 5120
YAMNET_DIMENSION = 3072
FEATURE_SETS = ("hybrid", "yamnet", "panns")
SOURCES = ("gtzan", "fma_small", "saraga")
DEFAULT_C_VALUES = (0.001, 0.01, 0.1, 1.0)


@dataclass(frozen=True)
class Track:
    source: str
    split: str
    label: str
    track_id: str
    group_id: str


def _read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Required split manifest is missing: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read {path}: {exc}") from exc


def _load_manifest_tracks(gtzan_saraga_manifest: Path, fma_manifest: Path) -> list[Track]:
    """Read GTZAN/Saraga and FMA source split assignments without changing them."""
    combined_payload = _read_json(gtzan_saraga_manifest)
    fma_payload = _read_json(fma_manifest)
    tracks: list[Track] = []

    for split, rows in combined_payload.get("splits", {}).items():
        if split not in {"train", "validation", "test"}:
            continue
        for row in rows:
            source = row.get("source")
            if source not in {"gtzan", "saraga"}:
                continue
            track_id = row.get("sample_id")
            label = row.get("label")
            group_id = row.get("group_id", track_id)
            if not all(isinstance(value, str) and value for value in (track_id, label, group_id)):
                raise ValueError(f"Malformed {source} row in {gtzan_saraga_manifest}: {row}")
            tracks.append(Track(source, split, label, track_id, group_id))

    fma_splits = fma_payload.get("datasets", {}).get("fma_small_mapped", {}).get("splits", {})
    for split, rows in fma_splits.items():
        if split not in {"train", "validation", "test"}:
            continue
        for row in rows:
            track_id = row.get("source_track_sha256")
            label = row.get("label")
            if not isinstance(track_id, str) or not isinstance(label, str):
                raise ValueError(f"Malformed FMA row in {fma_manifest}: {row}")
            tracks.append(Track("fma_small", split, label, track_id, f"fma_small:{track_id}"))

    for source in SOURCES:
        source_tracks = [track for track in tracks if track.source == source]
        if not source_tracks:
            raise ValueError(f"No split records for source {source} in the manifests")
        for split in ("train", "validation", "test"):
            if not any(track.split == split for track in source_tracks):
                raise ValueError(f"Source {source} has no {split} tracks")

    seen: dict[tuple[str, str], Track] = {}
    for track in tracks:
        key = (track.source, track.track_id)
        if key in seen:
            raise ValueError(f"Duplicate {track.source} track in split manifests: {track.track_id}")
        seen[key] = track

    # Exact byte hashes are shared by the FMA and GTZAN manifests. Saraga IDs
    # include a source prefix, so compare the digest portion across sources.
    by_digest: dict[str, list[Track]] = defaultdict(list)
    for track in tracks:
        digest = track.track_id.split(":", 1)[-1]
        by_digest[digest].append(track)
    cross_source_duplicates = [
        rows for rows in by_digest.values()
        if len({row.source for row in rows}) > 1
    ]
    if cross_source_duplicates:
        example = [(row.source, row.track_id, row.split) for row in cross_source_duplicates[0]]
        raise ValueError(f"Exact audio occurs in multiple sources; rebuild deduplicated manifests: {example}")

    group_splits: dict[tuple[str, str], set[str]] = defaultdict(set)
    for track in tracks:
        group_splits[(track.source, track.group_id)].add(track.split)
    leaked_groups = [key for key, split_names in group_splits.items() if len(split_names) > 1]
    if leaked_groups:
        raise ValueError(f"Track/release groups cross splits, e.g. {leaked_groups[0]}")
    return tracks


def _load_feature_cache(path: Path, expected_dimension: int) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Required feature cache is missing: {path}")
    try:
        with np.load(path, allow_pickle=False) as cache:
            track_ids = cache["track_ids"].astype(str).tolist()
            features = np.asarray(cache["features"], dtype=np.float32)
            version = int(cache["format_version"].item())
    except (EOFError, KeyError, OSError, ValueError) as exc:
        raise ValueError(f"Could not read feature cache {path}: {exc}") from exc
    if version != 1 or features.shape != (len(track_ids), expected_dimension):
        raise ValueError(
            f"Unexpected cache version/shape in {path}: version={version}, "
            f"features={features.shape}, IDs={len(track_ids)}"
        )
    if len(track_ids) != len(set(track_ids)) or not np.isfinite(features).all():
        raise ValueError(f"Feature cache has duplicate IDs or invalid values: {path}")
    return dict(zip(track_ids, features, strict=True))


def _load_features(args, tracks: list[Track]) -> dict[str, np.ndarray]:
    """Join cached vectors by track hash and fail closed on missing or extra IDs."""
    gtzan_saraga = _load_feature_cache(args.gtzan_saraga_features, FEATURE_DIMENSION)
    fma_dev = _load_feature_cache(args.fma_dev_features, FEATURE_DIMENSION)
    fma_test = _load_feature_cache(args.fma_test_features, FEATURE_DIMENSION)
    fma_by_id = fma_dev | fma_test
    if set(fma_dev) & set(fma_test):
        raise ValueError("FMA development and test feature caches overlap")

    expected_gtzan_saraga = {
        track.track_id for track in tracks if track.source in {"gtzan", "saraga"}
    }
    expected_fma = {track.track_id for track in tracks if track.source == "fma_small"}
    if set(gtzan_saraga) != expected_gtzan_saraga:
        missing = sorted(expected_gtzan_saraga - set(gtzan_saraga))[:3]
        extra = sorted(set(gtzan_saraga) - expected_gtzan_saraga)[:3]
        raise ValueError(f"GTZAN/Saraga cache IDs do not match manifest (missing={missing}, extra={extra})")
    if set(fma_by_id) != expected_fma:
        missing = sorted(expected_fma - set(fma_by_id))[:3]
        extra = sorted(set(fma_by_id) - expected_fma)[:3]
        raise ValueError(f"FMA cache IDs do not match manifest (missing={missing}, extra={extra})")
    return gtzan_saraga | fma_by_id


def _feature_view(matrix: np.ndarray, name: str) -> np.ndarray:
    if name == "hybrid":
        return matrix
    if name == "yamnet":
        return matrix[:, :YAMNET_DIMENSION]
    if name == "panns":
        return matrix[:, YAMNET_DIMENSION:]
    raise ValueError(f"Unknown feature set: {name}")


def _balanced_source_weights(tracks: list[Track], classes: list[str]) -> np.ndarray:
    """Give each class equal weight and split it evenly across sources present."""
    counts = Counter((track.source, track.label) for track in tracks)
    sources_per_class = {
        label: sum(1 for source in SOURCES if counts[(source, label)])
        for label in classes
    }
    if any(sources_per_class[label] == 0 for label in classes):
        raise ValueError("A class has no training examples")
    weights = np.asarray(
        [
            1.0 / (
                len(classes)
                * sources_per_class[track.label]
                * counts[(track.source, track.label)]
            )
            for track in tracks
        ],
        dtype=np.float64,
    )
    weights *= len(weights) / weights.sum()
    return weights.astype(np.float32)


def _fit_classifier(
    features: np.ndarray,
    labels: np.ndarray,
    sample_weights: np.ndarray,
    c_value: float,
) -> Pipeline:
    scaler = StandardScaler()
    scaler.fit(features, sample_weight=sample_weights)
    scaled = scaler.transform(features)
    classifier = LogisticRegression(C=c_value, max_iter=3_000, solver="lbfgs")
    classifier.fit(scaled, labels, sample_weight=sample_weights)
    return Pipeline(
        [
            ("standardscaler", scaler),
            ("logisticregression", classifier),
        ]
    )


def _evaluate_source(
    tracks: list[Track],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: list[str],
) -> dict:
    active_labels = sorted({track.label for track in tracks})
    class_to_id = {label: index for index, label in enumerate(classes)}
    active_ids = [class_to_id[label] for label in active_labels]
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=active_ids, zero_division=0
    )
    return {
        "track_count": len(tracks),
        "classes_scored": active_labels,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "predictions_outside_source_labels": int(np.sum(~np.isin(y_pred, active_ids))),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=active_ids,
            target_names=active_labels,
            output_dict=True,
            zero_division=0,
        ),
    }


def _score_predictions(
    tracks: list[Track],
    matrix: np.ndarray,
    labels: np.ndarray,
    classes: list[str],
    feature_name: str,
    c_value: float,
) -> tuple[dict, dict[str, np.ndarray]]:
    probabilities = matrix
    predictions = probabilities.argmax(axis=1)
    reports: dict[str, dict] = {}
    source_predictions: dict[str, np.ndarray] = {}
    for source in SOURCES:
        indices = np.asarray([index for index, track in enumerate(tracks) if track.source == source])
        if indices.size == 0:
            continue
        source_tracks = [tracks[int(index)] for index in indices]
        source_y = labels[indices]
        source_pred = predictions[indices]
        source_predictions[source] = source_pred
        reports[source] = _evaluate_source(source_tracks, source_y, source_pred, classes)
    mean_source_f1 = float(np.mean([report["macro_f1"] for report in reports.values()]))
    mean_source_accuracy = float(np.mean([report["accuracy"] for report in reports.values()]))
    return (
        {
            "feature_set": feature_name,
            "C": float(c_value),
            "source_validation": reports,
            "mean_source_macro_f1": mean_source_f1,
            "mean_source_accuracy": mean_source_accuracy,
        },
        source_predictions,
    )


def _split_arrays(
    tracks: list[Track],
    features_by_id: dict[str, np.ndarray],
    classes: list[str],
    split: str,
) -> tuple[list[Track], np.ndarray, np.ndarray]:
    rows = [track for track in tracks if track.split == split]
    if not rows:
        raise ValueError(f"No tracks for split {split}")
    class_to_id = {label: index for index, label in enumerate(classes)}
    matrix = np.stack([features_by_id[track.track_id] for track in rows]).astype(np.float32)
    labels = np.asarray([class_to_id[track.label] for track in rows], dtype=np.int32)
    return rows, matrix, labels


def _plot_confusion_matrix(
    matrix: np.ndarray,
    classes: list[str],
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(max(10, len(classes) * 0.9), max(8, len(classes) * 0.75)))
    image = axis.imshow(matrix, cmap="magma")
    axis.set(
        xticks=range(len(classes)),
        yticks=range(len(classes)),
        xticklabels=classes,
        yticklabels=classes,
        xlabel="Predicted genre",
        ylabel="True genre",
        title="Three-source held-out test confusion matrix",
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
                fontsize=7,
            )
    figure.colorbar(image, ax=axis, label="Track count")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def train(args: argparse.Namespace) -> dict:
    if args.output.exists():
        raise FileExistsError(
            f"Output already exists; choose another --output path: {args.output}"
        )
    tracks = _load_manifest_tracks(args.gtzan_saraga_manifest, args.fma_manifest)
    features_by_id = _load_features(args, tracks)
    classes = sorted({track.label for track in tracks})
    train_tracks, train_matrix, train_labels = _split_arrays(
        tracks, features_by_id, classes, "train"
    )
    validation_tracks, validation_matrix, validation_labels = _split_arrays(
        tracks, features_by_id, classes, "validation"
    )
    test_tracks, test_matrix, test_labels = _split_arrays(
        tracks, features_by_id, classes, "test"
    )
    train_weights = _balanced_source_weights(train_tracks, classes)

    candidate_scores: list[dict] = []
    for feature_name in FEATURE_SETS:
        x_train = _feature_view(train_matrix, feature_name)
        x_validation = _feature_view(validation_matrix, feature_name)
        for c_value in args.c_values:
            candidate = _fit_classifier(
                x_train, train_labels, train_weights, float(c_value)
            )
            probabilities = candidate.predict_proba(x_validation)
            score, _ = _score_predictions(
                validation_tracks,
                probabilities,
                validation_labels,
                classes,
                feature_name,
                float(c_value),
            )
            candidate_scores.append(score)
            print(
                f"Validation {feature_name} C={c_value:g}: "
                f"source-macro-F1={score['mean_source_macro_f1']:.4f}, "
                f"source-accuracy={score['mean_source_accuracy']:.4f}",
                flush=True,
            )

    selected = max(
        candidate_scores,
        key=lambda row: (row["mean_source_macro_f1"], row["mean_source_accuracy"]),
    )
    selected_features = selected["feature_set"]
    selected_c = selected["C"]
    print(
        f"Selected {selected_features} features, C={selected_c:g} "
        "using equal-weight mean of the three validation-source macro-F1 scores.",
        flush=True,
    )

    development_tracks = train_tracks + validation_tracks
    development_matrix = np.concatenate((train_matrix, validation_matrix), axis=0)
    development_labels = np.concatenate((train_labels, validation_labels), axis=0)
    development_weights = _balanced_source_weights(development_tracks, classes)
    final_model = _fit_classifier(
        _feature_view(development_matrix, selected_features),
        development_labels,
        development_weights,
        selected_c,
    )

    test_probabilities = final_model.predict_proba(_feature_view(test_matrix, selected_features))
    test_predictions = test_probabilities.argmax(axis=1)
    source_metrics: dict[str, dict] = {}
    for source in SOURCES:
        indices = np.asarray([index for index, track in enumerate(test_tracks) if track.source == source])
        if indices.size:
            source_metrics[source] = _evaluate_source(
                [test_tracks[int(index)] for index in indices],
                test_labels[indices],
                test_predictions[indices],
                classes,
            )

    all_ids = list(range(len(classes)))
    precision, recall, f1, _ = precision_recall_fscore_support(
        test_labels, test_predictions, labels=all_ids, zero_division=0
    )
    matrix = confusion_matrix(test_labels, test_predictions, labels=all_ids)
    report = {
        "experiment": "GTZAN + FMA-Small + Saraga Hindustani 11-label hybrid",
        "model_type": "yamnet_panns_logistic",
        "selected_candidate": selected_features,
        "selected_C": selected_c,
        "candidate_selection": (
            "Grid search on fixed validation tracks only; mean of source-specific macro-F1 "
            "for GTZAN, FMA-Small, and Saraga, with each source weighted equally."
        ),
        "candidate_scores": candidate_scores,
        "test_accuracy_pooled": float(accuracy_score(test_labels, test_predictions)),
        "test_macro_precision_pooled": float(precision.mean()),
        "test_macro_recall_pooled": float(recall.mean()),
        "test_macro_f1_pooled": float(f1.mean()),
        "test_accuracy_equal_source_mean": float(
            np.mean([item["accuracy"] for item in source_metrics.values()])
        ),
        "test_macro_f1_equal_source_mean": float(
            np.mean([item["macro_f1"] for item in source_metrics.values()])
        ),
        "classes": classes,
        "class_hierarchy": {
            "Western classical": "Classical",
            "Hindustani classical": "Classical",
        },
        "test_track_count": len(test_tracks),
        "test_tracks_by_source": dict(Counter(track.source for track in test_tracks)),
        "source_metrics": source_metrics,
        "confusion_matrix_labels": classes,
        "confusion_matrix": matrix.tolist(),
        "training_tracks_by_source": dict(Counter(track.source for track in development_tracks)),
        "training_tracks_by_source_and_label": {
            source: dict(Counter(track.label for track in development_tracks if track.source == source))
            for source in SOURCES
        },
        "sample_weight_policy": (
            "Equal total weight per class, then equal class-mass shares among sources that "
            "provide that class; weights are applied to scaler and logistic regression."
        ),
        "feature_extractors": "Frozen YAMNet and PANNs CNN14 track embeddings; classifier fitted locally.",
        "test_set_caveat": (
            "These are saved held-out splits reused from prior experiments, not a fresh blind benchmark. "
            "Saraga is the only source for Hindustani classical and GTZAN is the only source for "
            "Western classical, so source and those labels are confounded."
        ),
        "confidence_note": "Logistic-regression scores are not calibrated probabilities.",
        "split_manifests": {
            "gtzan_and_saraga": str(args.gtzan_saraga_manifest),
            "fma_small": str(args.fma_manifest),
        },
    }

    args.output.mkdir(parents=True, exist_ok=False)
    joblib.dump(final_model, args.output / "best_model.joblib")
    (args.output / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (args.output / "split_manifest.json").write_text(
        json.dumps(
            {
                "split_policy": "GTZAN track-level and Saraga release-group manifests retained; FMA-Small original track splits retained.",
                "tracks": [asdict(track) for track in tracks],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    config = {
        "model_type": "yamnet_panns_logistic",
        "model_variant": "gtzan_fma_saraga_three_source",
        "classes": classes,
        "class_hierarchy": report["class_hierarchy"],
        "feature_order": ["yamnet_mean_std_max_3072", "panns_cnn14_16k_embedding_2048"],
        "feature_dimension": FEATURE_DIMENSION,
        "selected_features": selected_features,
        "sample_rate": 16000,
        "audio_window_seconds": 30.0,
        "audio_hop_seconds": 15.0,
        "yamnet_model_path": "artifacts/pretrained/yamnet",
        "panns_checkpoint_path": "artifacts/pretrained/panns/Cnn14_16k_mAP_0.438.pth",
        "panns_architecture": "Cnn14_16k",
        "classifier": f"StandardScaler(sample_weighted) + LogisticRegression(C={selected_c}, sample_weighted)",
        "training_sources": ["GTZAN", "FMA-Small", "Saraga Hindustani"],
        "training_protocol": report["candidate_selection"],
        "format_version": 1,
    }
    (args.output / "preprocessing.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    _plot_confusion_matrix(matrix, classes, args.output / "confusion_matrix.png")

    print(
        json.dumps(
            {
                "selected_candidate": selected_features,
                "selected_C": selected_c,
                "test_accuracy_pooled": report["test_accuracy_pooled"],
                "test_accuracy_equal_source_mean": report["test_accuracy_equal_source_mean"],
                "test_macro_f1_equal_source_mean": report["test_macro_f1_equal_source_mean"],
                "source_metrics": {
                    source: {
                        "accuracy": metrics["accuracy"],
                        "macro_f1": metrics["macro_f1"],
                        "track_count": metrics["track_count"],
                    }
                    for source, metrics in source_metrics.items()
                },
                "test_tracks_by_source": report["test_tracks_by_source"],
            },
            indent=2,
        ),
        flush=True,
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gtzan-saraga-manifest",
        type=Path,
        default=Path("artifacts/musify_saraga_hybrid_v1/split_manifest.json"),
    )
    parser.add_argument(
        "--fma-manifest",
        type=Path,
        default=Path("artifacts/yamnet_panns_multisource_v1/split_manifest.json"),
    )
    parser.add_argument(
        "--gtzan-saraga-features",
        type=Path,
        default=Path("artifacts/musify_saraga_hybrid_v1/hybrid_features.npz"),
    )
    parser.add_argument(
        "--fma-dev-features",
        type=Path,
        default=Path("artifacts/pretrained/fma_small_hybrid_dev_features_v1.npz"),
    )
    parser.add_argument(
        "--fma-test-features",
        type=Path,
        default=Path("artifacts/pretrained/fma_small_hybrid_test_features_v1.npz"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/musify_three_dataset_hybrid_v1")
    )
    parser.add_argument("--c-values", type=float, nargs="+", default=list(DEFAULT_C_VALUES))
    args = parser.parse_args()
    if not args.c_values or any(value <= 0 for value in args.c_values):
        parser.error("--c-values must contain positive values")
    try:
        train(args)
    except (FileNotFoundError, ValueError, RuntimeError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
