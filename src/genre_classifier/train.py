"""Train Musify's from-scratch multi-scale CNN with track-level data splits."""

import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from .audio import load_audio, mel_spectrogram
from .config import AudioConfig
from .data import discover_tracks, save_manifest, split_tracks
from .model import build_model


class TrackSequence(tf.keras.utils.Sequence):
    """Load one crop per track each epoch and augment only the training examples."""

    def __init__(
        self, tracks, class_to_id, config, batch_size, training, seed,
        architecture,
    ):
        super().__init__()
        self.tracks = tracks
        self.class_to_id = class_to_id
        self.config = config
        self.batch_size = batch_size
        self.training = training
        self.seed = seed
        self.architecture = architecture
        self.epoch = 0
        self.rng = np.random.default_rng(seed)
        self.order = np.arange(len(tracks))

    def __len__(self):
        return int(np.ceil(len(self.tracks) / self.batch_size))

    def on_epoch_end(self):
        self.epoch += 1
        self.rng = np.random.default_rng(self.seed + self.epoch)
        if self.training:
            self.rng.shuffle(self.order)

    def _spec_augment(self, features):
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

    def __getitem__(self, index):
        indices = self.order[index * self.batch_size:(index + 1) * self.batch_size]
        xs, labels = [], []
        for track_index in indices:
            track = self.tracks[int(track_index)]
            audio = load_audio(str(track.path), self.config)
            features = mel_spectrogram(
                audio, self.config, training=self.training, rng=self.rng
            )
            if self.training:
                features = self._spec_augment(features)
            xs.append(features)
            labels.append(self.class_to_id[track.label])

        x_batch = np.stack(xs).astype(np.float32, copy=False)
        y_ids = np.asarray(labels, dtype=np.int32)
        # Use one true track label per input. On this small corpus, MixUp kept
        # the from-scratch network near chance; random track crops and light
        # SpecAugment provide the training variation instead.
        return x_batch, y_ids


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.keras.utils.set_random_seed(args.seed)

    config = AudioConfig()
    tracks = discover_tracks(args.data)
    classes = sorted({track.label for track in tracks})
    for label in classes:
        count = sum(track.label == label for track in tracks)
        if count < 3:
            raise ValueError(
                f"Need at least 3 distinct tracks per class for stratified splits; "
                f"{label}: {count}"
            )

    splits = split_tracks(
        tracks, args.seed, args.test_fraction, args.validation_fraction
    )
    args.output.mkdir(parents=True, exist_ok=True)
    save_manifest(splits, args.output / "split_manifest.json", args.seed)
    config.save(args.output / "preprocessing.json", classes)
    config_payload = json.loads(
        (args.output / "preprocessing.json").read_text(encoding="utf-8")
    )
    model_types = {
        "multiscale": ("musify_multiscale_cnn", "Musify Multi-Scale Residual AudioNet v1"),
        "crnn": ("crnn", "Convolutional Recurrent Baseline"),
        "attention_crnn": (
            "musify_attention_crnn",
            "Musify Residual Attention CRNN v1",
        ),
    }
    config_payload["model_type"], config_payload["architecture"] = model_types[
        args.architecture
    ]
    (args.output / "preprocessing.json").write_text(
        json.dumps(config_payload, indent=2), encoding="utf-8"
    )

    class_to_id = {name: index for index, name in enumerate(classes)}
    model = build_model(config, len(classes), architecture=args.architecture)
    model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            args.output / "best_model.keras",
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=args.patience,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_accuracy",
            mode="max",
            factor=0.5,
            patience=4,
            min_lr=1e-6,
        ),
        tf.keras.callbacks.CSVLogger(args.output / "training_history.csv"),
    ]

    history = model.fit(
        TrackSequence(
            splits["train"], class_to_id, config, args.batch_size,
            True, args.seed, args.architecture,
        ),
        validation_data=TrackSequence(
            splits["validation"], class_to_id, config, args.batch_size,
            False, args.seed, args.architecture,
        ),
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    model = tf.keras.models.load_model(args.output / "best_model.keras")
    test_sequence = TrackSequence(
        splits["test"], class_to_id, config, args.batch_size,
        False, args.seed, args.architecture,
    )
    probabilities = model.predict(test_sequence, verbose=1)
    y_true = np.asarray([class_to_id[track.label] for track in splits["test"]])
    y_pred = probabilities.argmax(axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(classes)), zero_division=0
    )
    validation_accuracies = history.history["val_accuracy"]
    validation_losses = history.history["val_loss"]
    best_epoch = int(np.argmax(validation_accuracies))
    minimum_loss_epoch = int(np.argmin(validation_losses))
    matrix = confusion_matrix(y_true, y_pred, labels=range(len(classes)))
    metrics = {
        "model_architecture": config_payload["architecture"],
        "test_accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "classes": classes,
        "classification_report": classification_report(
            y_true, y_pred, labels=range(len(classes)), target_names=classes,
            output_dict=True, zero_division=0,
        ),
        "confusion_matrix": matrix.tolist(),
        "test_track_count": len(y_true),
        "checkpoint_selection_metric": "val_accuracy",
        "checkpoint_epoch": best_epoch + 1,
        "checkpoint_val_accuracy": float(validation_accuracies[best_epoch]),
        "checkpoint_val_loss": float(validation_losses[best_epoch]),
        "minimum_validation_loss": float(validation_losses[minimum_loss_epoch]),
        "minimum_validation_loss_epoch": minimum_loss_epoch + 1,
    }
    (args.output / "evaluation.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    figure, axis = plt.subplots(
        figsize=(max(8, len(classes) * 0.8), max(7, len(classes) * 0.7))
    )
    image = axis.imshow(matrix, cmap="magma")
    axis.set(
        xticks=range(len(classes)),
        yticks=range(len(classes)),
        xticklabels=classes,
        yticklabels=classes,
        xlabel="Predicted genre",
        ylabel="True genre",
        title="Held-out test confusion matrix",
    )
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(len(classes)):
        for column in range(len(classes)):
            axis.text(
                column, row, str(matrix[row, column]), ha="center", va="center",
                color="white" if matrix[row, column] < threshold else "black",
                fontsize=8,
            )
    figure.colorbar(image, ax=axis, label="Track count")
    figure.tight_layout()
    figure.savefig(args.output / "confusion_matrix.png", dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(json.dumps({
        key: value for key, value in metrics.items()
        if key not in ("classification_report", "confusion_matrix")
    }, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/raw/Data/genres_original"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/musify_multiscale_v1"))
    parser.add_argument(
        "--architecture",
        choices=("multiscale", "crnn", "attention_crnn"),
        default="multiscale",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    args = parser.parse_args()
    if args.epochs < 1 or args.patience < 1 or args.batch_size < 1:
        parser.error("--epochs, --patience, and --batch-size must be positive")
    train(args)


if __name__ == "__main__":
    main()
