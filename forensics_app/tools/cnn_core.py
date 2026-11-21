"""Model passport, inference-input, and feature archive helpers for Set 8."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
import warnings

import numpy as np
from PIL import Image

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore", category=UserWarning)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "set8_outputs"


def discover_passports(output_dir: Path = DEFAULT_OUTPUT_DIR) -> list[Path]:
    """Find every passport in the output directory, independent of its prefix."""
    return sorted(output_dir.glob("*_passport.json")) if output_dir.is_dir() else []


def model_type_from_passport(passport: dict[str, Any]) -> str:
    model_type = passport.get("architecture_builder") or passport.get("model_type")
    if not isinstance(model_type, str):
        raise ValueError("Passport must specify architecture_builder or model_type.")
    return model_type


def feature_layer_name(model_type: str, passport: dict[str, Any] | None = None) -> str:
    from forensics_app.ml.cv_models import FEATURE_LAYER_NAMES

    layer_name = (passport or {}).get("model_architecture", {}).get("feature_layer")
    if layer_name:
        return str(layer_name)
    try:
        return FEATURE_LAYER_NAMES[model_type]
    except KeyError as error:
        raise ValueError(f"Unknown model type in passport: {model_type}") from error


def load_model_from_passport(passport_path: Path | str) -> tuple[Any, dict[str, Any]]:
    """Build the passport's architecture and load the weights beside the passport."""
    passport_file = Path(passport_path)
    with passport_file.open(encoding="utf-8") as file:
        passport = json.load(file)

    model_type = model_type_from_passport(passport)
    model_name = passport.get("model_name")
    weights_name = passport.get("weights_file")
    if not isinstance(weights_name, str):
        if not isinstance(model_name, str):
            raise ValueError(f"Passport has no weights_file or model_name: {passport_file}")
        weights_name = f"{model_name}.weights.h5"

    from forensics_app.ml.cv_models import BUILDERS

    try:
        builder = BUILDERS[model_type]
    except KeyError as error:
        raise ValueError(f"Unknown architecture builder in passport: {model_type}") from error

    model = builder(weights=None) if model_type == "inceptionv3_transfer" else builder()
    weights_path = passport_file.parent / weights_name
    if not weights_path.is_file():
        raise FileNotFoundError(f"Model weights not found: {weights_path}")
    model.load_weights(weights_path)

    layer_name = feature_layer_name(model_type, passport)
    feature_layer = model.get_layer(layer_name)
    expected_dimension = passport.get("model_architecture", {}).get("feature_dim")
    actual_dimension = int(feature_layer.output.shape[-1])
    if expected_dimension is not None and int(expected_dimension) != actual_dimension:
        raise ValueError(
            f"Passport feature dimension {expected_dimension} does not match "
            f"{layer_name!r} output dimension {actual_dimension}."
        )
    return model, passport


def load_features_npz(features_path: Path | str) -> dict[str, np.ndarray]:
    """Load and validate precomputed features and both projection coordinates."""
    with np.load(features_path, allow_pickle=False) as archive:
        required = {
            "features", "labels", "indices", "class_names",
            "tsne_2d", "umap_2d", "sample_images",
        }
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"Feature archive is missing: {', '.join(sorted(missing))}")
        data = {key: archive[key] for key in required}

    feature_count = len(data["features"])
    if data["features"].ndim != 2 or not data["features"].shape[1]:
        raise ValueError("Features must be a non-empty 2D array.")
    if data["labels"].reshape(-1).shape != (feature_count,):
        raise ValueError("Labels must have one entry per feature vector.")
    if data["indices"].reshape(-1).shape != (feature_count,):
        raise ValueError("Indices must have one entry per feature vector.")
    if data["tsne_2d"].shape != (feature_count, 2):
        raise ValueError("t-SNE coordinates must have shape (samples, 2).")
    if data["umap_2d"].size and data["umap_2d"].shape != (feature_count, 2):
        raise ValueError("UMAP coordinates must have shape (samples, 2) or be empty.")
    if len(data["sample_images"]) != feature_count:
        raise ValueError("sample_images must have one image per feature vector.")
    return data


def prepare_model_inputs(images: np.ndarray, model_type: str) -> np.ndarray:
    """Convert RGB pixels to the input range expected by the selected model."""
    inputs = np.asarray(images, dtype=np.float32)
    if model_type not in {"custom_cnn", "vgg_net", "inceptionv3_transfer"}:
        raise ValueError(f"Unknown model type: {model_type}")
    if inputs.ndim != 4 or inputs.shape[-1] != 3:
        raise ValueError("Model inputs must be a batch of RGB images.")
    if model_type == "vgg_net" and inputs.shape[1:3] != (64, 64):
        import tensorflow as tf

        tf.get_logger().setLevel("ERROR")
        inputs = tf.image.resize(inputs, (64, 64), method="bilinear").numpy()
    return inputs / 255.0


def prepare_image(
    image: Image.Image,
    model_type: str = "custom_cnn",
    size: tuple[int, int] | None = None,
) -> np.ndarray:
    """Convert a PIL image to a model-ready batched RGB input."""
    if model_type == "vgg_net":
        rgb_image = np.asarray(image.convert("RGB"), dtype=np.float32)[None, ...]
        return prepare_model_inputs(rgb_image, model_type)
    if size is None:
        size = (32, 32)
    resized = image.convert("RGB").resize(size, Image.Resampling.BILINEAR)
    return prepare_model_inputs(np.asarray(resized)[None, ...], model_type)


def prepare_model_dataset(
    images: np.ndarray,
    labels: np.ndarray,
    model_type: str,
    batch_size: int,
    *,
    shuffle: bool = False,
    seed: int = 42,
) -> Any:
    """Create a batched dataset with the same scaling/resizing as inference."""
    if model_type not in {"custom_cnn", "vgg_net", "inceptionv3_transfer"}:
        raise ValueError(f"Unknown model type: {model_type}")
    if batch_size < 1:
        raise ValueError("Batch size must be positive.")
    import tensorflow as tf

    tf.get_logger().setLevel("ERROR")
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    if shuffle:
        dataset = dataset.shuffle(len(images), seed=seed, reshuffle_each_iteration=True)

    def preprocess(image: Any, label: Any) -> tuple[Any, Any]:
        image = tf.cast(image, tf.float32)
        if model_type == "vgg_net":
            image = tf.image.resize(image, (64, 64), method="bilinear")
        return image / 255.0, label

    return (
        dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )


def extract_feature_embedding(
    model: Any,
    image: Image.Image,
    model_type: str,
    passport: dict[str, Any] | None = None,
) -> np.ndarray:
    """Extract the model's architecture-specific penultimate embedding."""
    import tensorflow as tf

    tf.get_logger().setLevel("ERROR")
    layer_name = feature_layer_name(model_type, passport)
    feature_model = tf.keras.Model(model.input, model.get_layer(layer_name).output)
    return np.asarray(
        feature_model.predict(prepare_image(image, model_type), verbose=0)[0]
    )
