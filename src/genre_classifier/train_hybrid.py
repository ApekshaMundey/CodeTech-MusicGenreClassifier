"""Train and compare leakage-safe YAMNet/PANNs music-genre transfer models."""

import argparse
import json
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .data import discover_tracks, save_manifest, split_tracks
from .panns import (
    PANNs_CHECKPOINT_MD5,
    PANNs_EMBEDDING_SIZE,
    PANNs_SAMPLE_RATE,
    extract_panns_embedding,
    load_panns,
)
from .yamnet import (
    YAMNET_FEATURE_SIZE,
    extract_track_embedding_from_audio,
    load_yamnet,
    load_yamnet_audio,
)

FEATURE_CACHE_VERSION = 1
HYBRID_FEATURE_SIZE = YAMNET_FEATURE_SIZE + PANNs_EMBEDDING_SIZE


def load_or_extract_features(tracks, yamnet, panns, cache_path: Path) -> np.ndarray:
    """Cache full-track embeddings from frozen pretrained models by track hash."""
    track_ids = [track.track_id for track in tracks]
    if len(set(track_ids)) != len(track_ids):
        raise ValueError("Duplicate track IDs reached hybrid feature extraction")

    if cache_path.is_file():
        try:
            with np.load(cache_path, allow_pickle=False) as cache:
                cached_ids = cache["track_ids"].astype(str).tolist()
                features = np.asarray(cache["features"], dtype=np.float32)
                version = int(cache["format_version"].item())
            expected_shape = (len(tracks), HYBRID_FEATURE_SIZE)
            if (
                version == FEATURE_CACHE_VERSION
                and cached_ids == track_ids
                and features.shape == expected_shape
                and np.isfinite(features).all()
            ):
                print(f"Reusing hybrid feature cache: {cache_path}")
                return features
            warnings.warn("Hybrid feature cache does not match these tracks; recomputing it.")
        except (EOFError, KeyError, OSError, ValueError) as exc:
            warnings.warn(f"Could not read hybrid feature cache ({exc}); recomputing it.")

    rows = []
    for index, track in enumerate(tracks, start=1):
        audio = load_yamnet_audio(track.path)
        yamnet_features = extract_track_embedding_from_audio(yamnet, audio)
        panns_features = extract_panns_embedding(panns, audio)
        rows.append(np.concatenate((yamnet_features, panns_features)))
        if index % 50 == 0 or index == len(tracks):
            print(f"Hybrid tracks processed: {index}/{len(tracks)}", flush=True)

    features = np.stack(rows).astype(np.float32, copy=False)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = cache_path.with_name(cache_path.stem + ".tmp.npz")
    np.savez_compressed(
        temporary_path,
        track_ids=np.asarray(track_ids, dtype="U64"),
        features=features,
        format_version=np.asarray(FEATURE_CACHE_VERSION, dtype=np.int32),
    )
    temporary_path.replace(cache_path)
    return features


def candidate_models() -> dict[str, tuple[str, object]]:
    """Return frozen-feature classifiers considered for the final model."""
    return {
        "yamnet_linear_svm": (
            "yamnet",
            make_pipeline(
                StandardScaler(),
                SVC(C=0.01, kernel="linear", class_weight="balanced"),
            ),
        ),
        "panns_logistic": (
            "panns",
            make_pipeline(
                StandardScaler(),
                LogisticRegression(C=0.1, max_iter=3_000, class_weight="balanced"),
            ),
        ),
        "yamnet_logistic": (
            "yamnet",
            make_pipeline(
                StandardScaler(),
                LogisticRegression(C=0.01, max_iter=3_000, class_weight="balanced"),
            ),
        ),
        "yamnet_panns_logistic": (
            "hybrid",
            make_pipeline(
                StandardScaler(),
                LogisticRegression(C=0.01, max_iter=3_000, class_weight="balanced"),
            ),
        ),
    }


def score_candidates(features, labels, folds: int, seed: int) -> dict:
    """Compare candidates with stratified CV using train+validation tracks only."""
    if folds < 2:
        raise ValueError("Cross-validation requires at least two folds")
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    scores = {}
    for name, (_, estimator) in candidate_models().items():
        accuracies = []
        macro_f1s = []
        for train_indices, validation_indices in splitter.split(features["hybrid"], labels):
            fit_features = features["hybrid"][train_indices]
            validation_features = features["hybrid"][validation_indices]
            if name == "yamnet_linear_svm" or name == "yamnet_logistic":
                fit_features = features["yamnet"][train_indices]
                validation_features = features["yamnet"][validation_indices]
            elif name == "panns_logistic":
                fit_features = features["panns"][train_indices]
                validation_features = features["panns"][validation_indices]
            model = clone(estimator)
            model.fit(fit_features, labels[train_indices])
            predictions = model.predict(validation_features)
            accuracies.append(accuracy_score(labels[validation_indices], predictions))
            macro_f1s.append(
                f1_score(labels[validation_indices], predictions, average="macro")
            )
        scores[name] = {
            "accuracy_mean": float(np.mean(accuracies)),
            "accuracy_std": float(np.std(accuracies)),
            "macro_f1_mean": float(np.mean(macro_f1s)),
            "macro_f1_std": float(np.std(macro_f1s)),
        }
    return scores


def plot_confusion_matrix(matrix: np.ndarray, classes: list[str], output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(9, 8))
    image = axis.imshow(matrix, cmap="magma")
    axis.set(
        xticks=range(len(classes)),
        yticks=range(len(classes)),
        xticklabels=classes,
        yticklabels=classes,
        xlabel="Predicted genre",
        ylabel="True genre",
        title="Hybrid model held-out test confusion matrix",
    )
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    figure.colorbar(image, ax=axis, label="Track count")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def train(args) -> dict:
    tf.keras.utils.set_random_seed(args.seed)
    tracks = discover_tracks(args.data)
    splits = split_tracks(tracks, args.seed, args.test_fraction, args.validation_fraction)
    args.output.mkdir(parents=True, exist_ok=True)
    save_manifest(splits, args.output / "split_manifest.json", args.seed)

    ordered_tracks = splits["train"] + splits["validation"] + splits["test"]
    panns = load_panns(args.panns_checkpoint)
    yamnet = load_yamnet(args.yamnet_model)
    all_features = load_or_extract_features(
        ordered_tracks, yamnet, panns, args.output / "hybrid_features.npz"
    )
    feature_by_id = {track.track_id: all_features[i] for i, track in enumerate(ordered_tracks)}
    class_names = sorted({track.label for track in tracks})
    class_to_id = {name: i for i, name in enumerate(class_names)}

    def extract_split(name: str) -> tuple[dict[str, np.ndarray], np.ndarray]:
        rows = splits[name]
        ids = [track.track_id for track in rows]
        hybrid = np.stack([feature_by_id[track_id] for track_id in ids])
        yamnet_features = hybrid[:, :YAMNET_FEATURE_SIZE]
        panns_features = hybrid[:, YAMNET_FEATURE_SIZE:]
        features = {
            "hybrid": hybrid,
            "yamnet": yamnet_features,
            "panns": panns_features,
        }
        labels = np.asarray([class_to_id[track.label] for track in rows], dtype=np.int32)
        return features, labels

    train_features, train_labels = extract_split("train")
    validation_features, validation_labels = extract_split("validation")
    test_features, test_labels = extract_split("test")
    development_features = {
        name: np.concatenate((train_features[name], validation_features[name]))
        for name in train_features
    }
    development_labels = np.concatenate((train_labels, validation_labels))
    cv_scores = score_candidates(development_features, development_labels, args.cv_folds, args.seed)
    selected_name = max(
        cv_scores,
        key=lambda name: (cv_scores[name]["macro_f1_mean"], cv_scores[name]["accuracy_mean"]),
    )
    selected_features, estimator = candidate_models()[selected_name]
    estimator.fit(development_features[selected_features], development_labels)
    predictions = estimator.predict(test_features[selected_features])
    matrix = confusion_matrix(test_labels, predictions, labels=range(len(class_names)))
    precision, recall, f1, _ = precision_recall_fscore_support(
        test_labels, predictions, labels=range(len(class_names)), zero_division=0
    )

    metrics = {
        "selected_candidate": selected_name,
        "selected_feature_set": selected_features,
        "candidate_selection": f"{args.cv_folds}-fold stratified CV on train+validation tracks only",
        "candidate_scores": cv_scores,
        "test_accuracy": float(accuracy_score(test_labels, predictions)),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "classes": class_names,
        "test_track_count": len(test_labels),
        "classification_report": classification_report(
            test_labels,
            predictions,
            labels=range(len(class_names)),
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": matrix.tolist(),
    }
    (args.output / "evaluation.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    joblib.dump(estimator, args.output / "best_model.joblib")
    config = {
        "model_type": "yamnet_panns_logistic",
        "classes": class_names,
        "feature_order": ["yamnet_mean_std_max_3072", "panns_cnn14_16k_embedding_2048"],
        "feature_dimension": HYBRID_FEATURE_SIZE,
        "sample_rate": PANNs_SAMPLE_RATE,
        "yamnet_model_path": str(args.yamnet_model),
        "panns_checkpoint_path": str(args.panns_checkpoint),
        "panns_checkpoint_md5": PANNs_CHECKPOINT_MD5,
        "panns_architecture": "Cnn14_16k",
        "classifier": "StandardScaler + LogisticRegression(C=0.01, class_weight=balanced)",
        "format_version": 1,
    }
    (args.output / "preprocessing.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    plot_confusion_matrix(matrix, class_names, args.output / "confusion_matrix.png")
    summary = {key: metrics[key] for key in (
        "selected_candidate", "test_accuracy", "macro_f1", "test_track_count"
    )}
    print(json.dumps(summary, indent=2))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/raw/genres_original"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/yamnet_panns_combo_v1"))
    parser.add_argument("--yamnet-model", type=Path, default=Path("artifacts/pretrained/yamnet"))
    parser.add_argument(
        "--panns-checkpoint",
        type=Path,
        default=Path("artifacts/pretrained/panns") / "Cnn14_16k_mAP_0.438.pth",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    args = parser.parse_args()
    if args.cv_folds < 2:
        parser.error("--cv-folds must be at least 2")
    train(args)


if __name__ == "__main__":
    main()
