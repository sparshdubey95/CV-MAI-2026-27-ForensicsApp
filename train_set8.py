#!/usr/bin/env python
"""Train Set 8 CIFAR-10 models or extract features from a saved passport."""

from __future__ import annotations

import argparse
import inspect
import json
import os
from pathlib import Path
import time
from typing import Any

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import cifar10

from forensics_app.ml.cv_models import BUILDERS, CLASS_NAMES, FEATURE_LAYER_NAMES
from forensics_app.tools.cnn_core import (
    DEFAULT_OUTPUT_DIR,
    feature_layer_name,
    load_model_from_passport,
    model_type_from_passport,
    prepare_model_inputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=BUILDERS, default="custom_cnn")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--n-train", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--output-dir", type=Path, default=Path("set8_outputs"))
    parser.add_argument("--quick", action="store_true", help="Train 2 epochs and project 500 test samples.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-umap", action="store_true", help="Skip UMAP if it is installed.")
    parser.add_argument(
        "--extract-features",
        action="store_true",
        help="Extract projections from a saved model without retraining it.",
    )
    parser.add_argument("--passport", type=Path, help="Passport to use with --extract-features.")
    parser.add_argument("--feature-samples", type=int, default=2000)
    return parser.parse_args()


def _stratified_indices(labels: np.ndarray, count: int, seed: int) -> np.ndarray:
    if count < len(CLASS_NAMES) or count > len(labels):
        raise ValueError(f"Sample count must be between {len(CLASS_NAMES)} and {len(labels)}.")
    if count == len(labels):
        return np.arange(len(labels))
    selected, _ = train_test_split(
        np.arange(len(labels)),
        train_size=count,
        stratify=labels,
        random_state=seed,
    )
    return np.asarray(selected)


def _history_values(history: tf.keras.callbacks.History) -> dict[str, list[float]]:
    return {
        key: [float(value) for value in history.history.get(key, [])]
        for key in ("loss", "val_loss", "accuracy", "val_accuracy")
    }


def _compute_tsne(features: np.ndarray, seed: int) -> np.ndarray:
    options: dict[str, Any] = {
        "n_components": 2,
        "random_state": seed,
        "init": "pca",
        "learning_rate": "auto",
        "perplexity": min(30.0, max(1.0, (len(features) - 1) / 3)),
    }
    iteration_argument = "max_iter" if "max_iter" in inspect.signature(TSNE).parameters else "n_iter"
    options[iteration_argument] = 1000
    return TSNE(**options).fit_transform(features).astype(np.float32)


def _compute_umap(features: np.ndarray, seed: int, skip: bool) -> np.ndarray:
    if skip:
        return np.empty((0, 2), dtype=np.float32)
    try:
        import umap
    except ImportError:
        print("umap-learn is not installed; saving an empty umap_2d array.")
        return np.empty((0, 2), dtype=np.float32)
    return umap.UMAP(n_components=2, random_state=seed).fit_transform(features).astype(np.float32)


def _stratified_feature_subset(
    images: np.ndarray,
    labels: np.ndarray,
    sample_count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = _stratified_indices(labels, min(sample_count, len(labels)), seed)
    return images[indices], labels[indices], indices


def _save_feature_archive(
    model: tf.keras.Model,
    model_type: str,
    passport: dict[str, Any],
    images: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
    sample_count: int,
    batch_size: int,
    seed: int,
    skip_umap: bool,
) -> Path:
    selected_images, selected_labels, indices = _stratified_feature_subset(
        images, labels, sample_count, seed
    )
    layer_name = feature_layer_name(model_type, passport)
    feature_model = tf.keras.Model(model.input, model.get_layer(layer_name).output)
    model_inputs = prepare_model_inputs(selected_images, model_type)
    features = feature_model.predict(model_inputs, batch_size=batch_size, verbose=0).astype(np.float32)
    tsne_2d = _compute_tsne(features, seed)
    umap_2d = _compute_umap(features, seed, skip_umap)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        features=features,
        labels=selected_labels,
        indices=indices,
        class_names=np.asarray(passport.get("class_names", CLASS_NAMES)),
        tsne_2d=tsne_2d,
        umap_2d=umap_2d,
        sample_images=selected_images.astype(np.uint8),
    )
    print(f"Saved {len(features)} {features.shape[1]}-D features to {output_path}")
    return output_path


def _record_feature_subset_metrics(
    model: tf.keras.Model,
    passport_path: Path | str,
    passport: dict[str, Any],
    feature_path: Path | str,
) -> None:
    if passport.get("test_metrics"):
        return
    with np.load(feature_path, allow_pickle=False) as archive:
        feature_vectors = tf.convert_to_tensor(archive["features"])
        test_labels = tf.convert_to_tensor(archive["labels"], dtype=tf.int32)
    probabilities = model.get_layer("predictions")(feature_vectors, training=False)
    test_loss = tf.keras.losses.sparse_categorical_crossentropy(test_labels, probabilities)
    test_accuracy = tf.reduce_mean(
        tf.cast(tf.argmax(probabilities, axis=1, output_type=tf.int32) == test_labels, tf.float32)
    )
    passport["test_metrics"] = {
        "accuracy": float(test_accuracy.numpy()),
        "loss": float(tf.reduce_mean(test_loss).numpy()),
        "evaluated_samples": int(len(test_labels)),
        "evaluation_source": "Stratified CIFAR-10 test subset",
    }
    Path(passport_path).write_text(json.dumps(passport, indent=2), encoding="utf-8")


def extract_saved_model_features(
    passport_path: Path | str,
    sample_count: int = 2000,
    batch_size: int = 128,
    seed: int = 42,
    skip_umap: bool = False,
    output_dir: Path | None = None,
) -> Path:
    """Extract stratified CIFAR-10 test features from saved weights, without training."""
    if sample_count < len(CLASS_NAMES) or batch_size < 1:
        raise ValueError("Feature sample count must be at least 10 and batch size must be positive.")
    model, passport = load_model_from_passport(passport_path)
    model_type = model_type_from_passport(passport)
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    del x_train, y_train
    model_name = passport.get("model_name")
    if not isinstance(model_name, str):
        raise ValueError("Passport must contain model_name to name the feature archive.")
    destination = (output_dir or Path(passport_path).parent) / f"{model_name}_features.npz"
    _save_feature_archive(
        model,
        model_type,
        passport,
        x_test,
        y_test.reshape(-1),
        destination,
        sample_count,
        batch_size,
        seed,
        skip_umap,
    )
    _record_feature_subset_metrics(model, passport_path, passport, destination)
    return destination


def train(args: argparse.Namespace) -> tuple[Path, Path, Path] | Path:
    if args.extract_features:
        passport_path = args.passport or args.output_dir / f"{args.model}_passport.json"
        return extract_saved_model_features(
            passport_path,
            sample_count=args.feature_samples,
            batch_size=args.batch_size,
            seed=args.seed,
            skip_umap=args.no_umap,
            output_dir=args.output_dir,
        )
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("--epochs and --batch-size must be positive.")

    epochs = 2 if args.quick else args.epochs
    train_count = 1000 if args.quick else args.n_train
    feature_count = 500 if args.quick else args.feature_samples
    if not 20 <= train_count <= 50000:
        raise ValueError("--n-train must be between 20 and 50000.")

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (x_train_full, y_train_full), (x_test, y_test) = cifar10.load_data()
    y_train_full, y_test = y_train_full.reshape(-1), y_test.reshape(-1)

    train_indices = _stratified_indices(y_train_full, train_count, args.seed)
    x_train = x_train_full[train_indices]
    y_train = y_train_full[train_indices]
    x_train, x_val, y_train, y_val = train_test_split(
        x_train,
        y_train,
        test_size=max(10, round(len(x_train) * 0.1)),
        stratify=y_train,
        random_state=args.seed,
    )

    model_type = args.model
    model = BUILDERS[model_type]()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    started = time.monotonic()
    history = model.fit(
        prepare_model_inputs(x_train, model_type),
        y_train,
        validation_data=(prepare_model_inputs(x_val, model_type), y_val),
        epochs=epochs,
        batch_size=args.batch_size,
        verbose=2,
    )
    train_duration = time.monotonic() - started
    test_loss, test_accuracy = model.evaluate(
        prepare_model_inputs(x_test, model_type), y_test, verbose=0
    )

    model_name = model_type
    weights_path = args.output_dir / f"{model_name}.weights.h5"
    passport_path = args.output_dir / f"{model_name}_passport.json"
    features_path = args.output_dir / f"{model_name}_features.npz"
    model.save_weights(weights_path)
    feature_name = FEATURE_LAYER_NAMES[model_type]
    feature_dimension = int(model.get_layer(feature_name).output.shape[-1])
    passport = {
        "model_name": model_name,
        "architecture_builder": model_type,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "framework": f"TensorFlow {tf.__version__}",
        "weights_file": weights_path.name,
        "dataset": {
            "name": "CIFAR-10",
            "n_train": int(len(x_train)),
            "n_val": int(len(x_val)),
            "n_test": int(len(x_test)),
            "num_classes": len(CLASS_NAMES),
            "input_shape": [32, 32, 3],
        },
        "sample_size": {
            "train": int(len(x_train)),
            "validation": int(len(x_val)),
            "test": int(len(x_test)),
            "total": int(len(x_train) + len(x_val) + len(x_test)),
        },
        "class_names": CLASS_NAMES,
        "input_shape": [32, 32, 3],
        "model_architecture": {
            "name": model.name,
            "total_params": int(model.count_params()),
            "feature_layer": feature_name,
            "feature_dim": feature_dimension,
        },
        "preprocessing": {
            "input_color": "RGB",
            "input_shape": [32, 32],
            "caller_input": (
                "Convert raw RGB pixels from [0,255] to [0,1]; model maps [0,1] to [-1,1]."
                if model_type == "inceptionv3_transfer"
                else "Pass raw RGB pixels in [0,255]; the model rescales internally to [0,1]."
            ),
        },
        "training": {
            "epochs": epochs,
            "batch_size": args.batch_size,
            "learning_rate": 1e-3,
            "optimizer": "Adam",
            "loss": "sparse_categorical_crossentropy",
            "duration_seconds": float(train_duration),
        },
        "history": _history_values(history),
        "test_metrics": {"accuracy": float(test_accuracy), "loss": float(test_loss)},
    }
    passport_path.write_text(json.dumps(passport, indent=2), encoding="utf-8")
    _save_feature_archive(
        model,
        model_type,
        passport,
        x_test,
        y_test,
        features_path,
        feature_count,
        args.batch_size,
        args.seed,
        args.no_umap,
    )
    print(f"Test accuracy: {test_accuracy:.4f}; artifacts saved to {args.output_dir.resolve()}")
    return weights_path, passport_path, features_path


def main() -> int:
    train(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
