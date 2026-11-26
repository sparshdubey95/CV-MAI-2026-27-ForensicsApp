"""Core implementation for Set 8: Image Classification using CNN.

Includes:
  - 8.1 Model architectures (Shallow CNN, Deeper VGG, InceptionV3 transfer), training,
        saving weights, and generating model passport.
  - 8.2 Loading pretrained model and passport, displaying passport, and running
        image classification inference with verified preprocessing.
  - 8.3 Feature extraction from CIFAR-10 test set, feature saving/loading, and 2D
        projections via t-SNE and UMAP.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
import hashlib
import io
import json
import os
import pickle
from pathlib import Path
import shutil
import zipfile
from typing import Any, Callable

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless & Tkinter integration
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, normalize as l2_normalize

# ── Dependency Guards ─────────────────────────────────────────────────────────

TENSORFLOW_AVAILABLE = False
UMAP_AVAILABLE = False
UMAP_IMPORT_ERROR: str | None = None

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models, optimizers
    TENSORFLOW_AVAILABLE = True
except ImportError:
    tf = None
    keras = None
    layers = None
    models = None
    optimizers = None

try:
    import umap
    UMAP_AVAILABLE = True
except Exception as exc:
    umap = None
    UMAP_IMPORT_ERROR = str(exc)

try:
    from sklearn.manifold import TSNE
except ImportError:
    TSNE = None


# ── Constants ─────────────────────────────────────────────────────────────────

CIFAR10_CLASSES: tuple[str, ...] = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

CIFAR10_CLASS_COLORS: tuple[str, ...] = (
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
)

CIFAR10_REQUIRED_FILES: tuple[str, ...] = (
    "data_batch_1",
    "data_batch_2",
    "data_batch_3",
    "data_batch_4",
    "data_batch_5",
    "test_batch",
    "batches.meta",
)

BUNDLED_WEIGHTS_NAME = "cifar10_cnn_model.weights.h5"
BUNDLED_PASSPORT_NAME = "cifar10_cnn_passport.json"
INCEPTION_SPATIAL_SIZE = 128
SUPPORTED_NORMALIZATIONS = frozenset(
    {
        "div_255",
        "inception_v3",
        "div_255_then_inception_v3",
    }
)

DEFAULT_PREPROCESSING: dict[str, Any] = {
    "target_size": [32, 32, 3],
    "normalization": "div_255",
    "color_mode": "RGB",
    "dtype": "float32",
    "range": "[0.0, 1.0]",
    "description": "Resize image to 32x32 RGB, convert to float32, divide by 255.0",
}

TRANSFER_PREPROCESSING: dict[str, Any] = {
    "target_size": [32, 32, 3],
    "model_input_size": [INCEPTION_SPATIAL_SIZE, INCEPTION_SPATIAL_SIZE, 3],
    "normalization": "div_255_then_inception_v3",
    "color_mode": "RGB",
    "dtype": "float32",
    "range": "[0.0, 1.0] at model input; InceptionV3 preprocess_input applied inside the network",
    "description": (
        "Resize photograph to 32x32 RGB, convert to float32, divide by 255.0. "
        "The InceptionV3 model then bilinearly resizes to 128x128 and applies "
        "keras.applications.inception_v3.preprocess_input."
    ),
}


class Cifar10DataError(FileNotFoundError):
    """Raised when the official local CIFAR-10 Python batches cannot be loaded."""


class TrainingCancelled(RuntimeError):
    """Raised when the user cancels CNN training."""


class PassportCompatibilityError(ValueError):
    """Raised when weights and passport files do not describe the same model."""


# ── Active Session State ──────────────────────────────────────────────────────

_active_cnn_model: Any = None
_active_cnn_passport: dict[str, Any] | None = None
_active_cnn_passport_path: str | None = None
_active_cnn_weights_path: str | None = None
_active_features_bundle: dict[str, Any] | None = None
_last_cifar10_info: dict[str, Any] | None = None


def get_active_cnn_model() -> Any:
    return _active_cnn_model


def set_active_cnn_model(model: Any) -> None:
    global _active_cnn_model
    _active_cnn_model = model


def get_active_cnn_passport() -> dict[str, Any] | None:
    return _active_cnn_passport


def set_active_cnn_passport(passport: dict[str, Any] | None, path: str | Path | None = None) -> None:
    global _active_cnn_passport, _active_cnn_passport_path
    _active_cnn_passport = passport
    _active_cnn_passport_path = str(path) if path else None


def get_active_cnn_passport_path() -> str | None:
    return _active_cnn_passport_path


def get_active_cnn_weights_path() -> str | None:
    return _active_cnn_weights_path


def set_active_cnn_weights_path(path: str | None) -> None:
    global _active_cnn_weights_path
    _active_cnn_weights_path = path


def get_active_features_bundle() -> dict[str, Any] | None:
    return _active_features_bundle


def set_active_features_bundle(bundle: dict[str, Any] | None) -> None:
    global _active_features_bundle
    _active_features_bundle = bundle


def get_last_cifar10_info() -> dict[str, Any] | None:
    return _last_cifar10_info


# ── Path Discovery Helpers ────────────────────────────────────────────────────

def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_cifar10_data_dir() -> Path:
    """Return the default directory for CIFAR-10 models, passports, and features."""
    pkg_dir = Path(__file__).resolve().parent.parent / "data" / "cifar10"
    pkg_dir.mkdir(parents=True, exist_ok=True)
    return pkg_dir


def get_default_user_weights_path() -> Path:
    """Default save location that does not overwrite the bundled demonstration model."""
    runs = get_cifar10_data_dir() / "user_runs"
    runs.mkdir(parents=True, exist_ok=True)
    return runs / "cifar10_cnn.weights.h5"


def get_default_user_passport_path() -> Path:
    return get_cifar10_data_dir() / "user_runs" / "cifar10_cnn_passport.json"


def is_bundled_model_path(path: str | Path | None) -> bool:
    if not path:
        return False
    name = Path(path).name.lower()
    return name in {BUNDLED_WEIGHTS_NAME.lower(), BUNDLED_PASSPORT_NAME.lower()}


def find_cifar10_batches_dir(explicit: str | Path | None = None) -> Path:
    """Resolve the official CIFAR-10 Python batch directory (project-relative)."""
    if explicit:
        selected = Path(explicit).expanduser()
        validate_cifar10_batches_dir(selected)
        return selected.resolve()

    candidates: list[Path] = []
    env_dir = os.environ.get("FORENSICS_CIFAR10_DIR") or os.environ.get("CIFAR10_DIR")
    if env_dir:
        candidates.append(Path(env_dir).expanduser())

    root = get_project_root()
    pkg = Path(__file__).resolve().parent.parent
    candidates.extend(
        [
            root / "data" / "cifar-10-batches-py",
            pkg / "data" / "cifar-10-batches-py",
            Path.cwd() / "data" / "cifar-10-batches-py",
            Path.cwd() / "cifar-10-batches-py",
        ]
    )

    seen: set[Path] = set()
    for cand in candidates:
        resolved = cand.resolve() if cand.exists() else cand
        if resolved in seen:
            continue
        seen.add(resolved)
        if _cifar10_dir_complete(cand):
            return cand.resolve()

    searched = "\n".join(f"  - {c}" for c in candidates)
    raise Cifar10DataError(
        "Official CIFAR-10 Python batches were not found.\n"
        "Place the extracted archive in a project-relative folder named "
        "data/cifar-10-batches-py/ (with data_batch_1..5, test_batch, and batches.meta).\n"
        f"Searched:\n{searched}"
    )


def _cifar10_dir_complete(directory: Path) -> bool:
    if not directory.is_dir():
        return False
    return all((directory / name).is_file() for name in CIFAR10_REQUIRED_FILES)


def validate_cifar10_batches_dir(directory: Path) -> None:
    if not directory.is_dir():
        raise Cifar10DataError(
            f"CIFAR-10 dataset directory does not exist: {directory}\n"
            "Extract the official CIFAR-10 Python version into data/cifar-10-batches-py/."
        )
    missing = [name for name in CIFAR10_REQUIRED_FILES if not (directory / name).is_file()]
    if missing:
        raise Cifar10DataError(
            f"CIFAR-10 directory {directory} is missing required files: {', '.join(missing)}.\n"
            "Expected data_batch_1 through data_batch_5, test_batch, and batches.meta. "
            "Synthetic or downloaded substitutes are not used."
        )


def find_default_cnn_weights() -> str | None:
    """Search candidate paths for default pretrained CNN weights."""
    root = get_project_root()
    candidates = [
        get_cifar10_data_dir() / BUNDLED_WEIGHTS_NAME,
        get_cifar10_data_dir() / "cifar10_cnn_model.h5",
        root / "cifar10_models" / BUNDLED_WEIGHTS_NAME,
        root / BUNDLED_WEIGHTS_NAME,
        Path.cwd() / BUNDLED_WEIGHTS_NAME,
    ]
    for cand in candidates:
        if cand.is_file():
            return str(cand)
    return str(candidates[0])


def find_default_cnn_passport() -> str | None:
    """Search candidate paths for default CNN model passport."""
    root = get_project_root()
    candidates = [
        get_cifar10_data_dir() / BUNDLED_PASSPORT_NAME,
        root / "cifar10_models" / BUNDLED_PASSPORT_NAME,
        root / BUNDLED_PASSPORT_NAME,
        Path.cwd() / BUNDLED_PASSPORT_NAME,
    ]
    for cand in candidates:
        if cand.is_file():
            return str(cand)
    return str(candidates[0])


def find_default_features_path() -> str | None:
    """Search candidate paths for default extracted features."""
    root = get_project_root()
    candidates = [
        get_cifar10_data_dir() / "cifar10_features_test500.npz",
        root / "cifar10_models" / "cifar10_features_test500.npz",
        root / "cifar10_features_test500.npz",
        Path.cwd() / "cifar10_features_test500.npz",
    ]
    for cand in candidates:
        if cand.is_file():
            return str(cand)
    return str(candidates[0])


def find_sample_cifar_images() -> list[str]:
    """Find bundled sample CIFAR-10 test images if available."""
    samples_dir = get_cifar10_data_dir() / "samples"
    if samples_dir.is_dir():
        return [str(p) for p in sorted(samples_dir.glob("*.png"))]
    return []


def backup_bundled_cnn_artifacts() -> list[Path]:
    """Copy existing bundled weights/passport aside before replacing them."""
    data_dir = get_cifar10_data_dir()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backed: list[Path] = []
    for name in (BUNDLED_WEIGHTS_NAME, BUNDLED_PASSPORT_NAME):
        src = data_dir / name
        if src.is_file():
            dest = data_dir / f"{src.name}.bak_{stamp}"
            shutil.copy2(src, dest)
            backed.append(dest)
    return backed


# ── Model Passport Dataclass ──────────────────────────────────────────────────

@dataclass
class ModelPassport:
    """Model Passport capturing dataset, class names, preprocessing, architecture, and history."""

    model_name: str
    dataset: str
    sample_size: int
    class_names: list[str]
    preprocessing_instructions: dict[str, Any]
    architecture: dict[str, Any]
    training_history: dict[str, list[float]]
    metrics: dict[str, float]
    created_at: str
    weights_path: str
    dataset_source: str = ""
    dataset_provenance: str = ""
    dataset_directory: str = ""
    split_method: str = "stratified_shuffle_90_10"
    train_samples: int = 0
    val_samples: int = 0
    test_samples: int = 0
    random_seed: int = 42
    optimizer: str = "sgd"
    learning_rate: float = 0.01
    batch_size: int = 32
    epochs: int = 0
    test_metrics: dict[str, Any] = field(default_factory=dict)
    classification_report: dict[str, Any] = field(default_factory=dict)
    confusion_matrix: list[list[int]] = field(default_factory=list)
    weights_filename: str = ""
    pretrained_base_weights: str | None = None
    weights_sha256: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelPassport":
        return cls(
            model_name=data.get("model_name", "cifar10_model"),
            dataset=data.get("dataset", "CIFAR-10"),
            sample_size=int(data.get("sample_size", 0)),
            class_names=list(data.get("class_names", CIFAR10_CLASSES)),
            preprocessing_instructions=dict(data.get("preprocessing_instructions", DEFAULT_PREPROCESSING)),
            architecture=dict(data.get("architecture", {})),
            training_history=dict(data.get("training_history", {})),
            metrics=dict(data.get("metrics", {})),
            created_at=data.get("created_at", datetime.now().isoformat()),
            weights_path=data.get("weights_path", ""),
            dataset_source=str(data.get("dataset_source", "")),
            dataset_provenance=str(data.get("dataset_provenance", "")),
            dataset_directory=str(data.get("dataset_directory", data.get("dataset_source", ""))),
            split_method=str(data.get("split_method", "stratified_shuffle_90_10")),
            train_samples=int(data.get("train_samples", 0)),
            val_samples=int(data.get("val_samples", 0)),
            test_samples=int(data.get("test_samples", 0)),
            random_seed=int(data.get("random_seed", 42)),
            optimizer=str(data.get("optimizer", "sgd")),
            learning_rate=float(data.get("learning_rate", 0.01)),
            batch_size=int(data.get("batch_size", 32)),
            epochs=int(data.get("epochs", 0)),
            test_metrics=dict(data.get("test_metrics", {})),
            classification_report=dict(data.get("classification_report") or {}),
            confusion_matrix=list(data.get("confusion_matrix") or []),
            weights_filename=str(data.get("weights_filename", Path(str(data.get("weights_path", ""))).name)),
            pretrained_base_weights=data.get("pretrained_base_weights"),
            weights_sha256=str(data.get("weights_sha256", "")),
        )


def save_passport(passport_data: ModelPassport | dict[str, Any], file_path: str | Path) -> str:
    """Save model passport to JSON file."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = passport_data.to_dict() if isinstance(passport_data, ModelPassport) else passport_data
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return str(path)


def load_passport(file_path: str | Path) -> dict[str, Any]:
    """Load model passport from JSON file."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Passport file not found: {path}")
    try:
        with open(path, "r", encoding="utf-8") as f:
            value = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read model passport JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Model passport {path} must contain a JSON object.")
    return value


def resolve_architecture_type(architecture: str | dict[str, Any] | None) -> str:
    if isinstance(architecture, dict):
        raw = str(architecture.get("type", "shallow"))
    else:
        raw = str(architecture or "shallow")
    key = raw.lower().strip()
    if key in ("deeper", "vgg", "deeper_vgg", "vgg-style"):
        return "vgg"
    if key in ("transfer", "inception", "inceptionv3", "inception_v3", "mobilenet", "mobilenetv2"):
        return "transfer"
    return "shallow"


# ── Model Architectures (8.1) ─────────────────────────────────────────────────

def build_shallow_cnn(
    input_shape: tuple[int, int, int] = (32, 32, 3),
    num_classes: int = 10,
) -> Any:
    """Build Shallow CNN: Conv32 -> MaxPool -> Flatten -> Dense(100) -> Dense(10)."""
    if not TENSORFLOW_AVAILABLE:
        raise RuntimeError("TensorFlow is required to build CNN models.")

    model = models.Sequential(
        [
            layers.Input(shape=input_shape, name="input_layer"),
            layers.Conv2D(
                32,
                (3, 3),
                activation="relu",
                kernel_initializer="he_uniform",
                name="conv1",
            ),
            layers.MaxPooling2D((2, 2), name="pool1"),
            layers.Flatten(name="flatten"),
            layers.Dense(
                100,
                activation="relu",
                kernel_initializer="he_uniform",
                name="dense_feature",
            ),
            layers.Dense(num_classes, activation="softmax", name="predictions"),
        ],
        name="shallow_cifar10_cnn",
    )
    return model


def build_deeper_vgg_cnn(
    input_shape: tuple[int, int, int] = (32, 32, 3),
    num_classes: int = 10,
) -> Any:
    """Build Deeper VGG-style CNN for CIFAR-10."""
    if not TENSORFLOW_AVAILABLE:
        raise RuntimeError("TensorFlow is required to build CNN models.")

    model = models.Sequential(
        [
            layers.Input(shape=input_shape, name="input_layer"),
            layers.Conv2D(
                32,
                (3, 3),
                activation="relu",
                kernel_initializer="he_uniform",
                padding="same",
                name="conv1_1",
            ),
            layers.Conv2D(
                32,
                (3, 3),
                activation="relu",
                kernel_initializer="he_uniform",
                padding="same",
                name="conv1_2",
            ),
            layers.MaxPooling2D((2, 2), name="pool1"),
            layers.Conv2D(
                64,
                (3, 3),
                activation="relu",
                kernel_initializer="he_uniform",
                padding="same",
                name="conv2_1",
            ),
            layers.Conv2D(
                64,
                (3, 3),
                activation="relu",
                kernel_initializer="he_uniform",
                padding="same",
                name="conv2_2",
            ),
            layers.MaxPooling2D((2, 2), name="pool2"),
            layers.Flatten(name="flatten"),
            layers.Dense(
                128,
                activation="relu",
                kernel_initializer="he_uniform",
                name="dense_feature",
            ),
            layers.Dense(num_classes, activation="softmax", name="predictions"),
        ],
        name="deeper_vgg_cifar10_cnn",
    )
    return model


def _inception_preprocess(x: Any) -> Any:
    """Map [0, 1] RGB tensors to InceptionV3's expected input range."""
    return tf.keras.applications.inception_v3.preprocess_input(x * 255.0)


def build_transfer_learning_model(
    base_name: str = "InceptionV3",
    input_shape: tuple[int, int, int] = (32, 32, 3),
    num_classes: int = 10,
    weights: str | None = "imagenet",
    restore_only: bool = False,
) -> Any:
    """Build frozen-base InceptionV3 transfer model with ImageNet weights.

    CIFAR-10 images stay 32x32 at the model input. They are resized to 128x128
    (InceptionV3 minimum is 75x75) and preprocessed inside the graph.
    """
    if not TENSORFLOW_AVAILABLE:
        raise RuntimeError("TensorFlow is required to build CNN models.")

    requested = (base_name or "InceptionV3").strip()
    if requested.lower() not in ("inceptionv3", "inception_v3", "inception"):
        raise ValueError(
            f"Unsupported transfer base '{requested}'. This assignment requires InceptionV3, "
            "not MobileNetV2 or a randomly initialised substitute."
        )

    if weights != "imagenet" and not (restore_only and weights is None):
        raise RuntimeError(
            "InceptionV3 transfer learning requires genuine pretrained ImageNet weights "
            "(weights='imagenet'). Random initialisation is not reported as transfer learning."
        )

    try:
        base_model = tf.keras.applications.InceptionV3(
            include_top=False,
            weights=weights,
            input_shape=(INCEPTION_SPATIAL_SIZE, INCEPTION_SPATIAL_SIZE, 3),
        )
    except Exception as exc:
        raise RuntimeError(
            "Could not load genuine ImageNet pretrained InceptionV3 weights. "
            "Transfer learning was not started with random or unverified weights. "
            "Ensure Keras can access its cache or download the official ImageNet checkpoint, "
            f"then retry. Original error: {exc}"
        ) from exc

    base_model.trainable = False
    for layer in base_model.layers:
        layer.trainable = False

    inputs = layers.Input(shape=input_shape, name="input_layer")
    x = layers.Resizing(
        INCEPTION_SPATIAL_SIZE,
        INCEPTION_SPATIAL_SIZE,
        interpolation="bilinear",
        name="inception_resize",
    )(inputs)
    x = layers.Lambda(_inception_preprocess, name="inception_preprocess")(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.Dense(1024, activation="relu", name="dense_feature")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    model = models.Model(inputs=inputs, outputs=outputs, name="transfer_inceptionv3_cifar10")
    return model


def build_cnn_architecture(
    architecture: str,
    num_classes: int = 10,
    restore_only: bool = False,
) -> Any:
    arch_type = resolve_architecture_type(architecture)
    if arch_type == "vgg":
        return build_deeper_vgg_cnn(input_shape=(32, 32, 3), num_classes=num_classes)
    if arch_type == "transfer":
        if restore_only:
            # The saved, fingerprint-verified checkpoint replaces every weight
            # before the restored model is returned to callers.
            return build_transfer_learning_model(
                input_shape=(32, 32, 3), num_classes=num_classes, weights=None, restore_only=True
            )
        return build_transfer_learning_model(input_shape=(32, 32, 3), num_classes=num_classes)
    return build_shallow_cnn(input_shape=(32, 32, 3), num_classes=num_classes)


def architecture_display_name(arch_type: str) -> str:
    if arch_type == "vgg":
        return "cifar10_vgg_cnn"
    if arch_type == "transfer":
        return "cifar10_transfer_inceptionv3"
    return "cifar10_shallow_cnn"


def preprocessing_for_architecture(arch_type: str) -> dict[str, Any]:
    if arch_type == "transfer":
        return dict(TRANSFER_PREPROCESSING)
    return dict(DEFAULT_PREPROCESSING)


# ── Official CIFAR-10 loading (local pickle only) ─────────────────────────────

def _unpickle(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return pickle.load(handle, encoding="bytes")


def _batch_get(raw: dict[str, Any], key: str) -> Any:
    if key in raw:
        return raw[key]
    bkey = key.encode("ascii")
    if bkey in raw:
        return raw[bkey]
    raise KeyError(f"CIFAR-10 pickle {key!r} not found in {list(raw)[:12]}")


def _decode_images_and_labels(raw: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    data = np.asarray(_batch_get(raw, "data"))
    if data.ndim != 2 or data.shape[1] != 3072:
        raise Cifar10DataError(
            f"Unexpected CIFAR-10 data shape {data.shape}; expected (N, 3072) RGB planes."
        )
    images = data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    if "labels" in raw or b"labels" in raw:
        labels = np.asarray(_batch_get(raw, "labels"), dtype=np.int32)
    else:
        labels = np.asarray(_batch_get(raw, "fine_labels"), dtype=np.int32)
    if labels.ndim != 1 or len(labels) != len(images):
        raise Cifar10DataError(
            f"CIFAR-10 labels have shape {labels.shape}; expected one label per image ({len(images)})."
        )
    if np.any((labels < 0) | (labels > 9)):
        raise Cifar10DataError("CIFAR-10 labels must be integers in the range 0 through 9.")
    if images.dtype != np.uint8:
        images = np.clip(images, 0, 255).astype(np.uint8)
    return images, labels.reshape(-1)


def _decode_class_names(meta: dict[str, Any]) -> list[str]:
    names = _batch_get(meta, "label_names")
    decoded: list[str] = []
    for name in names:
        if isinstance(name, bytes):
            decoded.append(name.decode("utf-8", errors="replace"))
        else:
            decoded.append(str(name))
    if len(decoded) != 10:
        raise Cifar10DataError(
            f"batches.meta listed {len(decoded)} class names; CIFAR-10 must have 10."
        )
    return decoded


@dataclass
class Cifar10Dataset:
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    class_names: list[str]
    source_path: str
    provenance: str


def load_cifar10_dataset(
    dataset_dir: str | Path | None = None,
    sample_limit: int | None = None,
    seed: int = 42,
    normalize: bool = True,
) -> Cifar10Dataset:
    """Load the official local CIFAR-10 Python batches. Never synthesizes or downloads."""
    global _last_cifar10_info

    batches_dir = find_cifar10_batches_dir(dataset_dir)
    validate_cifar10_batches_dir(batches_dir)

    train_images: list[np.ndarray] = []
    train_labels: list[np.ndarray] = []
    for idx in range(1, 6):
        raw = _unpickle(batches_dir / f"data_batch_{idx}")
        images, labels = _decode_images_and_labels(raw)
        train_images.append(images)
        train_labels.append(labels)

    x_train = np.concatenate(train_images, axis=0)
    y_train = np.concatenate(train_labels, axis=0)
    x_test, y_test = _decode_images_and_labels(_unpickle(batches_dir / "test_batch"))
    class_names = _decode_class_names(_unpickle(batches_dir / "batches.meta"))

    if x_train.shape[1:] != (32, 32, 3) or x_test.shape[1:] != (32, 32, 3):
        raise Cifar10DataError(
            f"CIFAR-10 images must be 32x32 RGB; got train {x_train.shape}, test {x_test.shape}."
        )
    if len(np.unique(y_train)) != 10 or len(np.unique(y_test)) != 10:
        raise Cifar10DataError("CIFAR-10 batches did not contain all 10 class labels.")

    if sample_limit is not None:
        x_train, y_train = stratified_subsample(x_train, y_train, sample_limit, seed=seed)

    provenance = (
        "Official CIFAR-10 Python version loaded from local pickle batches "
        f"({batches_dir})."
    )
    if normalize:
        x_train = x_train.astype(np.float32) / 255.0
        x_test = x_test.astype(np.float32) / 255.0
    else:
        x_train = x_train.astype(np.uint8)
        x_test = x_test.astype(np.uint8)

    dataset = Cifar10Dataset(
        x_train=x_train,
        y_train=y_train.astype(np.int32),
        x_test=x_test,
        y_test=y_test.astype(np.int32),
        class_names=class_names,
        source_path=str(batches_dir),
        provenance=provenance,
    )
    _last_cifar10_info = {
        "source_path": dataset.source_path,
        "provenance": dataset.provenance,
        "class_names": list(dataset.class_names),
        "n_train": int(len(dataset.x_train)),
        "n_test": int(len(dataset.x_test)),
        "image_shape": list(dataset.x_train.shape[1:]),
    }
    return dataset


def load_cifar10_data(
    sample_limit: int | None = None,
    dataset_dir: str | Path | None = None,
    seed: int = 42,
    **_ignored: Any,
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Load official CIFAR-10 as float32 [0, 1] arrays. Extra kwargs are ignored for API compatibility."""
    if _ignored:
        unsupported = [k for k in _ignored if k in {"allow_download", "use_synthetic_fallback"}]
        if any(_ignored.get(k) for k in unsupported):
            raise Cifar10DataError(
                "Automatic CIFAR-10 downloading and synthetic fallbacks are disabled. "
                "Provide the official local Python batches instead."
            )
    data = load_cifar10_dataset(dataset_dir=dataset_dir, sample_limit=sample_limit, seed=seed, normalize=True)
    return (data.x_train, data.y_train), (data.x_test, data.y_test)


def stratified_subsample(
    x: np.ndarray,
    y: np.ndarray,
    sample_limit: int,
    seed: int = 42,
    n_classes: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw a shuffled, class-balanced subset from x/y without touching the test set."""
    y = np.asarray(y).reshape(-1)
    if sample_limit <= 0:
        raise ValueError("sample_limit must be a positive integer.")
    if sample_limit >= len(x):
        rng = np.random.RandomState(seed)
        order = rng.permutation(len(x))
        return x[order], y[order]

    if sample_limit < n_classes * 2:
        raise ValueError(
            f"Need at least {n_classes * 2} samples for a stratified 90/10 split "
            f"that keeps all {n_classes} classes; got sample_limit={sample_limit}."
        )

    rng = np.random.RandomState(seed)
    per = sample_limit // n_classes
    remainder = sample_limit % n_classes
    chosen: list[int] = []
    for cls in range(n_classes):
        cls_idx = np.where(y == cls)[0]
        if len(cls_idx) == 0:
            raise ValueError(f"Class {cls} is missing from the training batches.")
        rng.shuffle(cls_idx)
        take = per + (1 if cls < remainder else 0)
        if take > len(cls_idx):
            raise ValueError(
                f"Class {cls} has only {len(cls_idx)} training images; cannot take {take}."
            )
        chosen.extend(cls_idx[:take].tolist())
    order = np.array(chosen, dtype=np.int64)
    rng.shuffle(order)
    return x[order], y[order]


def stratified_train_val_split(
    x: np.ndarray,
    y: np.ndarray,
    val_fraction: float = 0.1,
    seed: int = 42,
    n_classes: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reproducible shuffled stratified 90/10 split. The official test set is not used."""
    y = np.asarray(y).reshape(-1)
    if not 0.0 < val_fraction < 0.5:
        raise ValueError("validation_split must be between 0 and 0.5 (default 0.1).")

    unique, counts = np.unique(y, return_counts=True)
    if unique.size < n_classes:
        missing = sorted(set(range(n_classes)) - set(unique.tolist()))
        raise ValueError(f"Training subset is missing classes {missing}; all 10 CIFAR-10 classes are required.")
    if counts.min() < 2:
        raise ValueError(
            "Each class needs at least 2 samples for a stratified train/validation split. "
            f"Smallest class count is {int(counts.min())}."
        )

    rng = np.random.RandomState(seed)
    class_indices = [np.where(y == cls)[0].copy() for cls in range(n_classes)]
    if any(len(indices) == 0 for indices in class_indices):
        missing = [cls for cls, indices in enumerate(class_indices) if len(indices) == 0]
        raise ValueError(f"Training subset is missing classes {missing}; all 10 CIFAR-10 classes are required.")
    target_val_count = int(round(len(y) * val_fraction))
    ideal_val_counts = np.asarray([len(indices) * val_fraction for indices in class_indices])
    val_counts = np.floor(ideal_val_counts).astype(int)
    val_counts = np.maximum(val_counts, 1)
    val_counts = np.minimum(val_counts, np.asarray([len(indices) - 1 for indices in class_indices]))
    # Allocate rounding leftovers by largest fractional remainder so the total
    # validation set matches the requested fraction exactly (when feasible).
    tie_break = rng.random_sample(n_classes)
    while int(val_counts.sum()) < target_val_count:
        eligible = [i for i, indices in enumerate(class_indices) if val_counts[i] < len(indices) - 1]
        if not eligible:
            break
        idx = max(eligible, key=lambda i: (ideal_val_counts[i] - np.floor(ideal_val_counts[i]), tie_break[i]))
        val_counts[idx] += 1
        tie_break[idx] = rng.random_sample()
    while int(val_counts.sum()) > target_val_count:
        eligible = [i for i in range(n_classes) if val_counts[i] > 1]
        if not eligible:
            break
        idx = min(eligible, key=lambda i: (ideal_val_counts[i] - np.floor(ideal_val_counts[i]), tie_break[i]))
        val_counts[idx] -= 1
        tie_break[idx] = rng.random_sample()

    train_idx: list[int] = []
    val_idx: list[int] = []
    for cls, cls_idx in enumerate(class_indices):
        rng.shuffle(cls_idx)
        n_val = int(val_counts[cls])
        val_idx.extend(cls_idx[:n_val].tolist())
        train_idx.extend(cls_idx[n_val:].tolist())

    train_order = np.array(train_idx, dtype=np.int64)
    val_order = np.array(val_idx, dtype=np.int64)
    rng.shuffle(train_order)
    rng.shuffle(val_order)
    x_train, y_train = x[train_order], y[train_order]
    x_val, y_val = x[val_order], y[val_order]
    if len(np.unique(y_train)) != n_classes or len(np.unique(y_val)) != n_classes:
        raise ValueError(
            "Stratified split failed to keep all 10 classes in both the training and "
            "validation sets. Increase the sample count."
        )
    return x_train, y_train, x_val, y_val


def validate_training_hparams(
    sample_size: int,
    epochs: int,
    batch_size: int,
    validation_split: float = 0.1,
) -> None:
    if sample_size < 20:
        raise ValueError("sample_size must be at least 20 so every CIFAR-10 class can appear in train and validation.")
    if epochs < 1:
        raise ValueError("epochs must be a positive integer.")
    if batch_size < 1:
        raise ValueError("batch_size must be a positive integer.")
    if not 0.0 < validation_split < 0.5:
        raise ValueError("validation_split must be in (0, 0.5).")
    min_train = int(round(sample_size * (1.0 - validation_split)))
    if batch_size > min_train:
        raise ValueError(
            f"batch_size ({batch_size}) is larger than the expected training split "
            f"({min_train} images). Reduce batch size or increase sample_size."
        )


# ── Image Preprocessing Pipeline (8.2) ────────────────────────────────────────

def preprocess_image_for_cifar(
    image: Image.Image | np.ndarray,
    target_size: tuple[int, int] = (32, 32),
    normalization: str = "div_255",
) -> np.ndarray:
    """Preprocess an image to a (1, H, W, 3) float32 batch using passport rules."""
    norm = (normalization or "div_255").lower()
    if norm not in SUPPORTED_NORMALIZATIONS:
        raise ValueError(
            f"Unsupported preprocessing normalization {normalization!r}. "
            f"Supported values: {sorted(SUPPORTED_NORMALIZATIONS)}."
        )

    if isinstance(image, np.ndarray):
        if image.ndim == 2:
            pil_img = Image.fromarray(image).convert("RGB")
        elif image.ndim == 3:
            arr = image
            if arr.shape[2] == 4:
                arr = arr[:, :, :3]
            if np.issubdtype(arr.dtype, np.floating):
                if arr.max() <= 1.5:
                    arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
                else:
                    arr = np.clip(arr, 0, 255).astype(np.uint8)
            pil_img = Image.fromarray(arr).convert("RGB")
        else:
            raise ValueError(f"Unsupported image array shape: {image.shape}")
    else:
        pil_img = image.convert("RGB")

    resized = pil_img.resize(target_size, Image.Resampling.BILINEAR)
    arr = np.array(resized, dtype=np.float32)
    if norm in {"div_255", "div_255_then_inception_v3"}:
        arr = arr / 255.0
    elif norm == "inception_v3":
        if TENSORFLOW_AVAILABLE:
            arr = tf.keras.applications.inception_v3.preprocess_input(arr)
        else:
            arr = (arr / 127.5) - 1.0
    if arr.ndim == 3:
        arr = np.expand_dims(arr, axis=0)
    return arr.astype(np.float32)


def spatial_size_from_passport(passport: dict[str, Any] | None) -> tuple[int, int]:
    spec = (passport or {}).get("preprocessing_instructions", {}) if passport else {}
    ts = spec.get("target_size", [32, 32, 3])
    if isinstance(ts, (list, tuple)) and len(ts) >= 2:
        return int(ts[0]), int(ts[1])
    return 32, 32


def normalization_from_passport(passport: dict[str, Any] | None) -> str:
    spec = (passport or {}).get("preprocessing_instructions", {}) if passport else {}
    return str(spec.get("normalization", "div_255"))


# ── Training callbacks ────────────────────────────────────────────────────────

def _make_train_callback(
    progress_callback: Callable[[str], None] | None,
    cancel_event: Any | None,
) -> Any:
    if not TENSORFLOW_AVAILABLE:
        return None

    class _ProgressCancelCallback(keras.callbacks.Callback):
        def on_epoch_begin(self, epoch, logs=None):
            if progress_callback:
                total = int(self.params.get("epochs") or 0)
                progress_callback(f"Starting epoch {epoch + 1}/{total or '?'}...")
            self._abort_if_cancelled()

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            acc = logs.get("accuracy")
            val_acc = logs.get("val_accuracy")
            parts = [f"Finished epoch {epoch + 1}"]
            if acc is not None:
                parts.append(f"acc={float(acc):.3f}")
            if val_acc is not None:
                parts.append(f"val_acc={float(val_acc):.3f}")
            if progress_callback:
                progress_callback(" | ".join(parts))
            self._abort_if_cancelled()

        def on_train_batch_end(self, batch, logs=None):
            if batch % 20 == 0:
                self._abort_if_cancelled()

        def _abort_if_cancelled(self) -> None:
            if cancel_event is not None and getattr(cancel_event, "is_set", lambda: False)():
                self.model.stop_training = True
                raise TrainingCancelled("CNN training was cancelled.")

    return _ProgressCancelCallback()


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    return obj


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ── Training & Passport Generation (8.1) ──────────────────────────────────────

def train_cnn_model(
    architecture: str = "shallow",
    sample_size: int = 1000,
    epochs: int = 5,
    batch_size: int = 32,
    optimizer_name: str = "sgd",
    learning_rate: float | None = None,
    validation_split: float = 0.1,
    weights_path: str | Path | None = None,
    passport_path: str | Path | None = None,
    x_train_custom: np.ndarray | None = None,
    y_train_custom: np.ndarray | None = None,
    dataset_dir: str | Path | None = None,
    random_seed: int = 42,
    progress_callback: Callable[[str], None] | None = None,
    cancel_event: Any | None = None,
    max_test_eval: int | None = None,
) -> tuple[Any, dict[str, Any], Path, Path]:
    """Train a Keras CNN on official CIFAR-10, evaluate the held-out test set, and save a passport."""
    if not TENSORFLOW_AVAILABLE:
        raise RuntimeError("TensorFlow is required for CNN training.")

    validate_training_hparams(sample_size, epochs, batch_size, validation_split)
    arch_type = resolve_architecture_type(architecture)
    if learning_rate is None:
        learning_rate = 0.001 if optimizer_name.lower() == "adam" else 0.01
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be a finite positive number.")
    tf.keras.utils.set_random_seed(int(random_seed))

    class_names = list(CIFAR10_CLASSES)
    dataset_source = ""
    dataset_provenance = ""

    if progress_callback:
        progress_callback("Loading official local CIFAR-10 batches...")

    if x_train_custom is not None or y_train_custom is not None:
        raise Cifar10DataError(
            "Custom training arrays are not accepted by the CIFAR-10 training pipeline. "
            "Load the official local CIFAR-10 Python batches instead."
        )

    data = load_cifar10_dataset(
        dataset_dir=dataset_dir,
        sample_limit=sample_size,
        seed=random_seed,
        normalize=True,
    )
    x_pool, y_pool = data.x_train, data.y_train
    x_test, y_test = data.x_test, data.y_test
    class_names = list(data.class_names)
    dataset_source = data.source_path
    dataset_provenance = data.provenance

    x_train, y_train, x_val, y_val = stratified_train_val_split(
        x_pool,
        y_pool,
        val_fraction=validation_split,
        seed=random_seed,
    )

    y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes=10)
    y_val_cat = tf.keras.utils.to_categorical(y_val, num_classes=10)

    if progress_callback:
        progress_callback(f"Building {arch_type} CNN ({len(x_train)} train / {len(x_val)} val)...")

    model = build_cnn_architecture(arch_type, num_classes=10)
    model_name = architecture_display_name(arch_type)

    if optimizer_name.lower() == "adam":
        opt = optimizers.Adam(learning_rate=learning_rate)
    else:
        opt = optimizers.SGD(learning_rate=learning_rate)

    model.compile(
        loss="categorical_crossentropy",
        optimizer=opt,
        metrics=["accuracy"],
    )

    callbacks = [
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-5,
            verbose=0,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=6,
            restore_best_weights=True,
            verbose=0,
        ),
    ]
    cb = _make_train_callback(progress_callback, cancel_event)
    if cb is not None:
        callbacks.append(cb)

    history = model.fit(
        x_train,
        y_train_cat,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_val, y_val_cat),
        shuffle=True,
        verbose=0 if progress_callback is not None else 1,
        callbacks=callbacks,
    )

    history_dict: dict[str, list[float]] = {}
    for key, values in history.history.items():
        history_dict[key] = [float(val) for val in values]

    # EarlyStopping restores the best validation weights. Report train/validation
    # metrics from that same epoch so the passport describes the saved model.
    best_epoch_idx = (
        int(np.argmin(history_dict["val_loss"])) if history_dict.get("val_loss") else len(history_dict["loss"]) - 1
    )
    final_loss = float(history_dict["loss"][best_epoch_idx])
    final_acc = float(history_dict["accuracy"][best_epoch_idx])
    val_loss = float(history_dict["val_loss"][best_epoch_idx]) if "val_loss" in history_dict else final_loss
    val_acc = float(history_dict["val_accuracy"][best_epoch_idx]) if "val_accuracy" in history_dict else final_acc

    if progress_callback:
        progress_callback("Evaluating on the official CIFAR-10 test set...")

    # The official test set is held out in full. Keep max_test_eval in the
    # signature for compatibility, but never let it truncate evaluation.
    x_eval, y_eval = x_test, y_test

    y_eval_cat = tf.keras.utils.to_categorical(y_eval, num_classes=10)
    test_loss, test_acc = model.evaluate(x_eval, y_eval_cat, verbose=0)
    y_prob = model.predict(x_eval, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)
    report = classification_report(
        y_eval,
        y_pred,
        labels=list(range(10)),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_eval, y_pred, labels=list(range(10)))

    data_dir = get_cifar10_data_dir()
    if weights_path is None:
        save_weights_path = get_default_user_weights_path()
    else:
        save_weights_path = Path(weights_path)
    save_weights_path.parent.mkdir(parents=True, exist_ok=True)

    if passport_path is None:
        save_passport_path = get_default_user_passport_path()
    else:
        save_passport_path = Path(passport_path)
    save_passport_path.parent.mkdir(parents=True, exist_ok=True)

    if progress_callback:
        progress_callback(f"Saving weights to {save_weights_path}...")

    model.save_weights(str(save_weights_path))

    pretrained_flag: str | None = "imagenet" if arch_type == "transfer" else None
    passport = ModelPassport(
        model_name=model_name,
        dataset="CIFAR-10",
        sample_size=int(len(x_pool)),
        class_names=list(class_names),
        preprocessing_instructions=preprocessing_for_architecture(arch_type),
        architecture={
            "type": arch_type,
            "base_name": "InceptionV3" if arch_type == "transfer" else None,
            "input_shape": [32, 32, 3],
            "total_parameters": int(model.count_params()),
            "layers": [layer.name for layer in model.layers],
            "base_frozen": True if arch_type == "transfer" else None,
        },
        training_history=history_dict,
        metrics={
            "final_loss": final_loss,
            "final_accuracy": final_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "best_epoch": best_epoch_idx + 1,
            "test_loss": float(test_loss),
            "test_accuracy": float(test_acc),
        },
        created_at=datetime.now().isoformat(),
        weights_path=str(save_weights_path),
        dataset_source=dataset_source,
        dataset_provenance=dataset_provenance,
        dataset_directory=dataset_source,
        split_method=f"shuffled stratified train/validation split ({1-validation_split:.0%}/{validation_split:.0%})",
        train_samples=int(len(x_train)),
        val_samples=int(len(x_val)),
        test_samples=int(len(x_eval)),
        random_seed=int(random_seed),
        optimizer=optimizer_name.lower(),
        learning_rate=float(learning_rate),
        batch_size=int(batch_size),
        epochs=int(epochs),
        test_metrics={
            "loss": float(test_loss),
            "accuracy": float(test_acc),
            "evaluated_on": "official CIFAR-10 test_batch (full 10,000 images)",
        },
        classification_report=_json_safe(report),
        confusion_matrix=_json_safe(cm.tolist()),
        weights_filename=save_weights_path.name,
        pretrained_base_weights=pretrained_flag,
        weights_sha256=_sha256_file(save_weights_path),
    )
    save_passport(passport, save_passport_path)

    set_active_cnn_model(model)
    set_active_cnn_passport(passport.to_dict(), save_passport_path)
    set_active_cnn_weights_path(str(save_weights_path))
    return model, passport.to_dict(), save_weights_path, save_passport_path


def render_training_curves(
    history: dict[str, list[float]],
    title: str = "CIFAR-10 CNN Training History",
) -> Image.Image:
    """Render stacked subplots of cross-entropy loss and classification accuracy."""
    fig, (ax_loss, ax_acc) = plt.subplots(2, 1, figsize=(8, 7), dpi=100)
    fig.patch.set_facecolor("#ffffff")

    epochs_range = range(1, len(history.get("loss", [])) + 1)

    ax_loss.set_title("Cross Entropy Loss", fontsize=11, fontweight="bold")
    if "loss" in history:
        ax_loss.plot(epochs_range, history["loss"], color="#0055aa", label="Train Loss", linewidth=2)
    if "val_loss" in history:
        ax_loss.plot(epochs_range, history["val_loss"], color="#d62728", linestyle="--", label="Val Loss", linewidth=2)
    ax_loss.set_ylabel("Loss")
    ax_loss.grid(True, linestyle=":", alpha=0.6)
    ax_loss.legend(loc="upper right")

    ax_acc.set_title("Classification Accuracy", fontsize=11, fontweight="bold")
    if "accuracy" in history:
        ax_acc.plot(epochs_range, history["accuracy"], color="#2ca02c", label="Train Accuracy", linewidth=2)
    if "val_accuracy" in history:
        ax_acc.plot(epochs_range, history["val_accuracy"], color="#ff7f0e", linestyle="--", label="Val Accuracy", linewidth=2)
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.grid(True, linestyle=":", alpha=0.6)
    ax_acc.legend(loc="lower right")

    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


def render_confusion_matrix_image(
    cm: np.ndarray | list[list[int]],
    class_names: list[str] | None = None,
    title: str = "CIFAR-10 Test Confusion Matrix",
    accuracy: float | None = None,
) -> Image.Image:
    matrix = np.asarray(cm, dtype=np.int32)
    names = class_names or list(CIFAR10_CLASSES)
    fig, ax = plt.subplots(figsize=(8.2, 6.6), dpi=100)
    fig.patch.set_facecolor("#ffffff")
    im = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(np.arange(len(names)))
    ax.set_yticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=40, ha="right", fontsize=8)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    thresh = matrix.max() / 2.0 if matrix.size and matrix.max() > 0 else 1.0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j,
                i,
                str(int(matrix[i, j])),
                ha="center",
                va="center",
                color="white" if matrix[i, j] > thresh else "black",
                fontsize=7,
            )
    full_title = title if accuracy is None else f"{title}\nTest accuracy = {accuracy * 100.0:.1f}%"
    ax.set_title(full_title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


def render_training_report(
    history: dict[str, list[float]],
    confusion: np.ndarray | list[list[int]] | None,
    class_names: list[str] | None = None,
    title: str = "CIFAR-10 CNN Training & Test Evaluation",
    test_accuracy: float | None = None,
) -> Image.Image:
    curves = render_training_curves(history, title=title)
    if confusion is None:
        return curves
    cm_img = render_confusion_matrix_image(
        confusion,
        class_names=class_names,
        accuracy=test_accuracy,
    )
    canvas = Image.new(
        "RGB",
        (max(curves.width, cm_img.width), curves.height + cm_img.height + 16),
        color=(255, 255, 255),
    )
    canvas.paste(curves.convert("RGB"), (0, 0))
    canvas.paste(cm_img.convert("RGB"), (0, curves.height + 16))
    return canvas


# ── Pretrained Model Loading & Inference (8.2) ────────────────────────────────

def _passport_weights_name(passport: dict[str, Any]) -> str:
    name = str(passport.get("weights_filename") or "").strip()
    if name:
        return Path(name).name
    wp = str(passport.get("weights_path") or "").strip()
    return Path(wp).name if wp else ""


def validate_weights_passport_pair(weights_path: Path, passport: dict[str, Any]) -> None:
    if not passport:
        raise PassportCompatibilityError(
            "A model passport JSON is required so the correct architecture and preprocessing can be restored."
        )
    expected = _passport_weights_name(passport)
    actual = weights_path.name
    if expected and expected != actual:
        raise PassportCompatibilityError(
            f"Weights file {actual!r} does not match passport weights filename {expected!r}. "
            "Select the passport that was saved with these weights."
        )
    expected_hash = str(passport.get("weights_sha256") or "").strip().lower()
    if expected_hash:
        actual_hash = _sha256_file(weights_path)
        if actual_hash != expected_hash:
            raise PassportCompatibilityError(
                f"Weights file {actual!r} does not match the SHA-256 fingerprint in this passport. "
                "Select the passport saved with these exact weights."
            )
    classes = passport.get("class_names") or []
    if list(classes) != list(CIFAR10_CLASSES):
        raise PassportCompatibilityError("Passport class_names must contain the 10 CIFAR-10 classes in canonical order.")
    arch = passport.get("architecture") or {}
    raw_type = str(arch.get("type", "")).strip().lower()
    if raw_type not in {"shallow", "vgg", "deeper", "deeper_vgg", "vgg-style", "transfer", "inception", "inceptionv3", "inception_v3"}:
        raise PassportCompatibilityError(f"Passport architecture type {raw_type!r} is not supported.")
    arch_type = resolve_architecture_type(arch)
    if arch.get("input_shape") not in (None, [32, 32, 3], (32, 32, 3)):
        raise PassportCompatibilityError("Passport input dimensions are incompatible; expected 32x32 RGB.")
    if arch_type == "transfer":
        base = str(arch.get("base_name") or "InceptionV3")
        if base.lower().replace("_", "") not in {"inceptionv3", "inception"}:
            raise PassportCompatibilityError(
                f"Passport transfer base {base!r} is not InceptionV3. Refusing to load an incompatible network."
            )
        if str(passport.get("pretrained_base_weights") or "").lower() != "imagenet":
            raise PassportCompatibilityError("Transfer passport does not confirm genuine ImageNet pretrained weights.")
    norm = normalization_from_passport(passport)
    if norm not in SUPPORTED_NORMALIZATIONS:
        raise PassportCompatibilityError(
            f"Passport preprocessing {norm!r} is not supported. Refusing to run inference with an unverified pipeline."
        )
    expected_norm = "div_255_then_inception_v3" if arch_type == "transfer" else "div_255"
    if norm != expected_norm:
        raise PassportCompatibilityError(
            f"Passport preprocessing {norm!r} is incompatible with architecture {arch_type!r}; "
            f"expected {expected_norm!r}."
        )


def load_pretrained_cnn(
    weights_path: str | Path,
    passport_path: str | Path | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Load model weights and passport information for Set 8.2."""
    if not TENSORFLOW_AVAILABLE:
        raise RuntimeError("TensorFlow is required to load CNN models.")

    w_path = Path(weights_path)
    if not w_path.is_file():
        raise FileNotFoundError(f"Model weights file not found: {w_path}")

    p_dict: dict[str, Any] = {}
    resolved_passport: Path | None = None
    if passport_path:
        p_file = Path(passport_path)
        if not p_file.is_file():
            raise FileNotFoundError(f"Passport file not found: {p_file}")
        resolved_passport = p_file
        p_dict = load_passport(p_file)
    else:
        candidates = [
            w_path.with_name(w_path.name.replace(".weights.h5", "_passport.json")),
            w_path.with_name(w_path.stem + "_passport.json"),
            w_path.with_suffix(".json"),
            w_path.parent / BUNDLED_PASSPORT_NAME,
        ]
        for cand in candidates:
            if cand.is_file():
                resolved_passport = cand
                p_dict = load_passport(cand)
                break
        if not p_dict:
            raise FileNotFoundError(
                "No model passport JSON was provided or found next to the weights file. "
                "Inference cannot verify architecture or preprocessing without a passport."
            )

    validate_weights_passport_pair(w_path, p_dict)

    try:
        arch_type = resolve_architecture_type(p_dict.get("architecture"))
        model = build_cnn_architecture(arch_type, num_classes=10, restore_only=True)
        model.load_weights(str(w_path))
    except PassportCompatibilityError:
        raise
    except Exception as exc:
        raise PassportCompatibilityError(
            f"Failed to load weights {w_path} with passport {resolved_passport}: {exc}. "
            "The files may be mismatched or corrupted."
        ) from exc

    set_active_cnn_model(model)
    set_active_cnn_passport(p_dict, resolved_passport)
    set_active_cnn_weights_path(str(w_path))
    return model, p_dict


def run_cnn_inference(
    model: Any,
    image: Image.Image | np.ndarray,
    passport: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Classify one image with the passport's preprocessing pipeline."""
    if passport is None:
        raise ValueError("A model passport is required so inference uses the training preprocessing.")
    norm = normalization_from_passport(passport)
    if norm not in SUPPORTED_NORMALIZATIONS:
        raise ValueError(
            f"Unsupported passport preprocessing {norm!r}. "
            "Refusing to invent an alternative normalization."
        )
    target_size = spatial_size_from_passport(passport)
    input_tensor = preprocess_image_for_cifar(image, target_size=target_size, normalization=norm)
    predictions = model.predict(input_tensor, verbose=0)[0]
    if predictions.shape != (10,) or not np.isfinite(predictions).all():
        raise PassportCompatibilityError(
            f"CNN produced invalid output shape/values {predictions.shape}; expected 10 finite class probabilities."
        )
    if np.any(predictions < 0.0) or np.any(predictions > 1.0) or not np.isclose(float(predictions.sum()), 1.0, atol=1e-4):
        raise PassportCompatibilityError("CNN output is not a normalized 10-class probability distribution.")

    class_names = list(passport.get("class_names") or CIFAR10_CLASSES)
    top_idx = int(np.argmax(predictions))
    confidence = float(predictions[top_idx])
    predicted_class = class_names[top_idx] if top_idx < len(class_names) else f"class_{top_idx}"

    probabilities: dict[str, float] = {}
    for i, p in enumerate(predictions):
        cname = class_names[i] if i < len(class_names) else f"class_{i}"
        probabilities[cname] = float(p)

    return {
        "predicted_class": predicted_class,
        "predicted_index": top_idx,
        "confidence": confidence,
        "probabilities": probabilities,
        "passport": passport,
        "note": (
            "CIFAR-10 models are trained on 32x32 images. Arbitrary photographs are resized "
            "to that resolution and may be classified poorly."
        ),
    }


def render_classification_visualization(
    image: Image.Image,
    predicted_class: str,
    confidence: float,
    probabilities: dict[str, float],
    passport: dict[str, Any] | None = None,
) -> Image.Image:
    """Render a 2-panel figure showing the input image and class probability distribution."""
    fig = plt.figure(figsize=(9, 4.8), dpi=100)
    fig.patch.set_facecolor("#ffffff")
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.4], wspace=0.3)

    ax_img = fig.add_subplot(gs[0, 0])
    ax_bar = fig.add_subplot(gs[0, 1])

    ax_img.imshow(image)
    ax_img.set_title(
        f"Prediction: {predicted_class.upper()}\nConfidence: {confidence * 100.0:.1f}%",
        fontsize=12,
        fontweight="bold",
        color="#0055aa",
        pad=8,
    )
    ax_img.axis("off")

    classes = list(probabilities.keys())
    scores = [probabilities[c] for c in classes]
    y_pos = np.arange(len(classes))
    bar_colors = ["#2ca02c" if c == predicted_class else "#4a90e2" for c in classes]
    bars = ax_bar.barh(y_pos, [s * 100.0 for s in scores], color=bar_colors, height=0.65, edgecolor="none")
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(classes, fontsize=9, fontweight="medium")
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel("Probability (%)", fontsize=10, fontweight="bold")
    ax_bar.set_xlim(0, 105)
    ax_bar.grid(axis="x", linestyle=":", alpha=0.6)

    for bar in bars:
        width = bar.get_width()
        if width > 0.5:
            ax_bar.text(
                width + 1.5,
                bar.get_y() + bar.get_height() / 2,
                f"{width:.1f}%",
                va="center",
                fontsize=8,
                fontweight="bold",
                color="#333333",
            )

    passport_info = ""
    if passport:
        mname = passport.get("model_name", "CNN")
        samples = passport.get("sample_size", "N/A")
        acc = passport.get("metrics", {}).get("test_accuracy")
        if acc is None:
            acc = passport.get("metrics", {}).get("final_accuracy")
        acc_str = f" | Acc: {acc * 100:.1f}%" if acc is not None else ""
        passport_info = f"Model: {mname} | Trained on {samples} samples{acc_str}"

    fig.suptitle(
        f"CIFAR-10 Classification Inference\n{passport_info}" if passport_info else "CIFAR-10 Classification Inference",
        fontsize=12,
        fontweight="bold",
        y=0.98,
    )
    plt.subplots_adjust(top=0.88, bottom=0.12, left=0.08, right=0.95, wspace=0.32)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


# ── Feature Extraction & Projections (8.3) ────────────────────────────────────

def _feature_extractor(model: Any, layer_name: str | None = None) -> tuple[Any, str]:
    target_layer = None
    if layer_name:
        target_layer = model.get_layer(layer_name)
    else:
        for layer in reversed(model.layers[:-1]):
            if layer.__class__.__name__ in {"Dense", "Flatten", "GlobalAveragePooling2D"}:
                target_layer = layer
                break
        if target_layer is None:
            target_layer = model.layers[-2]
    if not model.layers or target_layer.name == model.layers[-1].name:
        raise ValueError("Feature extraction must use an intermediate layer before the final classification output.")
    extractor = models.Model(inputs=model.inputs, outputs=target_layer.output, name="cnn_feature_extractor")
    return extractor, target_layer.name


def extract_cifar10_features(
    model: Any | None = None,
    samples_per_class: int = 50,
    layer_name: str | None = None,
    save_path: str | Path | None = None,
    x_test_custom: np.ndarray | None = None,
    y_test_custom: np.ndarray | None = None,
    weights_path: str | Path | None = None,
    passport_path: str | Path | None = None,
    dataset_dir: str | Path | None = None,
    require_trained_weights: bool = True,
    random_seed: int = 42,
) -> dict[str, Any]:
    """Extract feature vectors from a trained CNN on the official CIFAR-10 test set."""
    if not TENSORFLOW_AVAILABLE:
        raise RuntimeError("TensorFlow is required to extract features.")
    del require_trained_weights  # Kept for API compatibility; extraction always requires saved trained weights.
    if x_test_custom is not None or y_test_custom is not None:
        raise ValueError("Feature extraction uses the official CIFAR-10 test set; custom arrays are not accepted.")
    if samples_per_class < 1:
        raise ValueError("samples_per_class must be a positive integer.")

    selected_weights = weights_path or get_active_cnn_weights_path()
    selected_passport = passport_path or get_active_cnn_passport_path()
    if selected_passport is None and selected_weights is None:
        active_passport = get_active_cnn_passport() or {}
        candidate = active_passport.get("weights_path")
        if candidate:
            selected_weights = str(candidate)
    if not selected_weights:
        raise RuntimeError(
            "Feature extraction requires saved trained weights and their passport. "
            "Train a CNN or select a matching weights/passport pair; random model weights are never used."
        )
    if not Path(selected_weights).is_file():
        raise RuntimeError(
            f"Feature extraction requires an existing saved trained weights file; not found: {selected_weights}. "
            "Train a CNN or select a matching weights/passport pair."
        )
    if selected_passport is None:
        active_passport = get_active_cnn_passport() or {}
        active_weights = get_active_cnn_weights_path()
        if active_passport and active_weights and Path(active_weights).resolve() == Path(selected_weights).resolve():
            # The passport still goes through the same pair validation as an explicitly selected file.
            selected_passport = active_passport.get("passport_path")
        if not selected_passport:
            candidates = [
                Path(selected_weights).with_name(Path(selected_weights).name.replace(".weights.h5", "_passport.json")),
                Path(selected_weights).with_name(Path(selected_weights).stem + "_passport.json"),
                Path(selected_weights).with_suffix(".json"),
            ]
            selected_passport = next((str(p) for p in candidates if p.is_file()), None)
    if not selected_passport:
        raise RuntimeError("Feature extraction requires the passport saved with the selected weights.")

    # Always reconstruct from the saved pair. A caller-supplied in-memory model
    # alone is not evidence that its weights were trained or match a passport.
    model, passport = load_pretrained_cnn(selected_weights, selected_passport)
    used_weights = str(Path(selected_weights).resolve())
    data = load_cifar10_dataset(dataset_dir=dataset_dir, sample_limit=None, seed=42, normalize=True)
    x_test, y_test = data.x_test, data.y_test
    source = data.source_path
    class_names = list(data.class_names)
    provenance = data.provenance

    rng = np.random.RandomState(int(random_seed))
    selected_indices: list[int] = []
    for cls in range(10):
        cls_idxs = np.where(y_test == cls)[0]
        count = min(int(samples_per_class), len(cls_idxs))
        selected_indices.extend(rng.choice(cls_idxs, size=count, replace=False).tolist())
    selected_indices_arr = np.array(selected_indices, dtype=np.int64)
    rng.shuffle(selected_indices_arr)
    x_strat = x_test[selected_indices_arr]
    y_strat = y_test[selected_indices_arr]
    if not np.array_equal(y_test[selected_indices_arr], y_strat):
        raise RuntimeError("Feature-label alignment check failed for the official CIFAR-10 test set.")

    extractor, used_layer = _feature_extractor(model, layer_name=layer_name)

    batch_size = 32
    features_list = []
    for i in range(0, len(x_strat), batch_size):
        batch = x_strat[i : i + batch_size]
        feats = extractor.predict(batch, verbose=0)
        if feats.ndim > 2:
            feats = feats.reshape(len(batch), -1)
        features_list.append(feats)
    features = np.concatenate(features_list, axis=0)
    if features.ndim != 2 or len(features) != len(y_strat):
        raise RuntimeError("Feature extractor returned a shape mismatch; expected one vector per labeled image.")
    valid = np.isfinite(features).all(axis=1) & (np.linalg.norm(np.nan_to_num(features), axis=1) > 1e-12)
    features = features[valid]
    y_strat = y_strat[valid]
    selected_indices_arr = selected_indices_arr[valid]
    if len(features) == 0 or set(np.unique(y_strat).tolist()) != set(range(10)):
        raise RuntimeError("Feature extraction left fewer than one valid vector for one or more CIFAR-10 classes.")

    model_name = str((passport or {}).get("model_name") or getattr(model, "name", "cnn"))
    weights_hash = _sha256_file(used_weights)
    model_test_accuracy = (passport or {}).get("metrics", {}).get("test_accuracy")
    bundle: dict[str, Any] = {
        "features": features,
        "labels": y_strat,
        "class_names": np.array(class_names),
        "indices": selected_indices_arr,
        "extractor_layer": used_layer,
        "feature_dim": int(features.shape[1]),
        "num_samples": int(features.shape[0]),
        "dataset_source": source,
        "dataset_provenance": provenance,
        "model_name": model_name,
        "weights_path": used_weights or "",
        "weights_sha256": weights_hash,
        "architecture": str(((passport or {}).get("architecture") or {}).get("type", "unknown")),
        "sampling_method": "seeded balanced random sample from official CIFAR-10 test set",
        "random_seed": int(random_seed),
        "model_test_accuracy": float(model_test_accuracy) if model_test_accuracy is not None else None,
    }

    if save_path:
        out_path = Path(save_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            str(out_path),
            features=features,
            labels=y_strat,
            class_names=np.array(class_names),
            indices=selected_indices_arr,
            feature_dim=np.array([features.shape[1]]),
            dataset_source=np.array(source),
            dataset_provenance=np.array(provenance),
            model_name=np.array(model_name),
            weights_path=np.array(used_weights or ""),
            weights_sha256=np.array(weights_hash),
            architecture=np.array(bundle["architecture"]),
            sampling_method=np.array(bundle["sampling_method"]),
            random_seed=np.array([random_seed], dtype=np.int32),
            model_test_accuracy=np.array(
                [float(model_test_accuracy) if model_test_accuracy is not None else -1.0], dtype=np.float32
            ),
            extractor_layer=np.array(used_layer),
        )
        bundle["saved_path"] = str(out_path)

    set_active_features_bundle(bundle)
    return bundle


def _npz_text(data: Any, key: str, default: str = "") -> str:
    if key not in data.files:
        return default
    value = data[key]
    if np.ndim(value) == 0:
        return str(value)
    if getattr(value, "size", 0) == 1:
        return str(value.reshape(-1)[0])
    return str(value)


def _passport_metrics_for_weights(weights_path: str | Path, expected_hash: str = "") -> dict[str, Any]:
    """Find a sibling passport that identifies these exact weights, if present."""
    path = Path(weights_path)
    if not path.is_file():
        return {}
    actual_hash = _sha256_file(path)
    for candidate in path.parent.glob("*.json"):
        try:
            passport = load_passport(candidate)
        except (OSError, ValueError):
            continue
        if _passport_weights_name(passport) != path.name:
            continue
        passport_hash = str(passport.get("weights_sha256") or "").strip().lower()
        if expected_hash and actual_hash != expected_hash.lower():
            continue
        if passport_hash and passport_hash != actual_hash:
            continue
        return dict(passport.get("metrics") or {})
    return {}


def load_extracted_features(file_path: str | Path) -> dict[str, Any]:
    """Load extracted feature vectors from a saved .npz file."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Feature file not found: {path}")

    try:
        data = np.load(str(path), allow_pickle=False)
        required = {
            "features", "labels", "class_names", "indices", "dataset_source",
            "model_name", "weights_path", "extractor_layer",
        }
        missing = sorted(required.difference(data.files))
        if missing:
            raise ValueError(f"Feature file is missing required metadata: {', '.join(missing)}.")
        features = np.asarray(data["features"])
        labels = np.asarray(data["labels"])
        class_names = [str(c) for c in data["class_names"]]
        indices = np.asarray(data["indices"])
        if features.ndim != 2 or not np.issubdtype(features.dtype, np.number) or not np.isfinite(features).all():
            raise ValueError("Feature matrix must be a finite numeric 2D array.")
        if labels.ndim != 1 or len(labels) != len(features) or not np.issubdtype(labels.dtype, np.integer):
            raise ValueError("Feature labels must be a 1D integer array with one label per feature vector.")
        if np.any((labels < 0) | (labels > 9)):
            raise ValueError("Feature labels must be CIFAR-10 class indices from 0 through 9.")
        if len(class_names) != 10 or class_names != list(CIFAR10_CLASSES):
            raise ValueError("Feature file class_names must list canonical CIFAR-10 classes in order.")
        if indices.ndim != 1 or len(indices) != len(features):
            raise ValueError("Feature indices must be a 1D array with one entry per feature vector.")
        if not _npz_text(data, "dataset_source").strip() or not _npz_text(data, "model_name").strip():
            raise ValueError("Feature file must identify its dataset source and model.")
        if not _npz_text(data, "weights_path").strip() or not _npz_text(data, "extractor_layer").strip():
            raise ValueError("Feature file must identify its weights and extraction layer.")
        recorded_hash = _npz_text(data, "weights_sha256").strip().lower()
        recorded_weights = Path(_npz_text(data, "weights_path"))
        if recorded_hash and recorded_weights.is_file() and _sha256_file(recorded_weights) != recorded_hash:
            raise ValueError("Feature vectors do not match the recorded weights fingerprint; re-extract from current weights.")
        if "feature_dim" in data and int(np.asarray(data["feature_dim"]).reshape(-1)[0]) != features.shape[1]:
            raise ValueError("Feature file feature_dim metadata does not match the feature matrix.")
    except (OSError, KeyError, ValueError, EOFError, zipfile.BadZipFile) as exc:
        raise ValueError(f"Invalid CNN feature file {path}: {exc}") from exc

    bundle = {
        "features": features,
        "labels": labels,
        "class_names": class_names,
        "indices": indices,
        "feature_dim": int(features.shape[1]),
        "num_samples": int(features.shape[0]),
        "saved_path": str(path),
        "dataset_source": _npz_text(data, "dataset_source"),
        "dataset_provenance": _npz_text(data, "dataset_provenance"),
        "model_name": _npz_text(data, "model_name"),
        "weights_path": _npz_text(data, "weights_path"),
        "weights_sha256": _npz_text(data, "weights_sha256"),
        "architecture": _npz_text(data, "architecture", "unknown"),
        "sampling_method": _npz_text(data, "sampling_method", "legacy archive"),
        "random_seed": int(np.asarray(data["random_seed"]).reshape(-1)[0]) if "random_seed" in data else None,
        "extractor_layer": _npz_text(data, "extractor_layer", "unknown"),
    }
    if "model_test_accuracy" in data:
        saved_accuracy = float(np.asarray(data["model_test_accuracy"]).reshape(-1)[0])
        if 0.0 <= saved_accuracy <= 1.0:
            bundle["model_test_accuracy"] = saved_accuracy
    if "model_test_accuracy" not in bundle:
        metrics = _passport_metrics_for_weights(bundle["weights_path"], bundle["weights_sha256"])
        accuracy = metrics.get("test_accuracy")
        if accuracy is not None and np.isfinite(float(accuracy)) and 0.0 <= float(accuracy) <= 1.0:
            bundle["model_test_accuracy"] = float(accuracy)

    if "tsne_2d" in data:
        coords = np.asarray(data["tsne_2d"])
        if coords.shape != (len(features), 2) or not np.isfinite(coords).all():
            raise ValueError("Invalid t-SNE coordinates in feature file; expected finite shape (N, 2).")
        bundle["tsne_2d"] = coords
    if "umap_2d" in data:
        coords = np.asarray(data["umap_2d"])
        if coords.shape != (len(features), 2) or not np.isfinite(coords).all():
            raise ValueError("Invalid UMAP coordinates in feature file; expected finite shape (N, 2).")
        bundle["umap_2d"] = coords

    set_active_features_bundle(bundle)
    return bundle


def prepare_features_for_projection(
    features: np.ndarray,
    labels: np.ndarray | None = None,
    n_components: int | None = None,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """Clean, balance and standardize CNN features, with optional PCA reduction.

    The returned mask maps retained rows back to the supplied arrays in their
    original order. When labels are supplied, only class-balanced rows are
    retained, using the same deterministic sampling for both projection methods.
    """
    raw = np.asarray(features)
    if raw.ndim != 2 or raw.shape[1] == 0 or not np.issubdtype(raw.dtype, np.number):
        raise ValueError("CNN features must be a numeric 2D matrix.")
    valid = np.isfinite(raw).all(axis=1) & (np.linalg.norm(np.nan_to_num(raw), axis=1) > 1e-12)
    if labels is not None:
        labels = np.asarray(labels)
        if labels.ndim != 1 or len(labels) != len(raw):
            raise ValueError("Feature labels must be a 1D array aligned one-to-one with feature rows.")
        if not np.issubdtype(labels.dtype, np.integer) or np.any((labels < 0) | (labels > 9)):
            raise ValueError("Feature labels must be CIFAR-10 class indices from 0 through 9.")

    keep = np.zeros(len(raw), dtype=bool)
    if labels is None:
        keep = valid.copy()
        selected = np.flatnonzero(keep)
        retained_labels = None
    else:
        counts = [int(np.sum(valid & (labels == cls))) for cls in range(10)]
        per_class = min(counts)
        if per_class == 0:
            raise ValueError(f"After removing invalid vectors, CIFAR-10 class counts are {counts}; all classes are required.")
        rng = np.random.RandomState(random_state)
        selected: list[int] = []
        for cls in range(10):
            indices = np.flatnonzero(valid & (labels == cls))
            selected.extend(rng.choice(indices, size=per_class, replace=False).tolist())
        selected = np.asarray(selected, dtype=np.int64)
        selected = np.sort(selected)
        keep[selected] = True
        retained_labels = labels[selected]

    clean = np.asarray(raw[selected], dtype=np.float32)
    if len(clean) < 3:
        raise ValueError("At least 3 valid feature vectors are required for 2D projection.")
    standardized = StandardScaler().fit_transform(clean)
    if n_components is not None and int(n_components) < standardized.shape[1]:
        n_pca = min(max(1, int(n_components)), len(clean) - 1, standardized.shape[1])
        pca = PCA(n_components=n_pca, random_state=random_state)
        pca_scores = pca.fit_transform(standardized)
        nonzero_components = pca.explained_variance_ > 1e-12
        if not np.any(nonzero_components):
            raise ValueError("CNN feature matrix has no measurable variance after standardization.")
        reduced = pca_scores[:, nonzero_components] / np.sqrt(pca.explained_variance_[nonzero_components])
        prepared = l2_normalize(reduced).astype(np.float32)
    else:
        prepared = standardized.astype(np.float32)
    if not np.isfinite(prepared).all():
        raise ValueError("Feature standardization produced non-finite values; check for degenerate CNN features.")
    return prepared, retained_labels, keep


def compute_tsne_projection(
    features: np.ndarray,
    perplexity: float = 40.0,
    n_iter: int = 1000,
    random_state: int = 42,
    prepared: bool = False,
) -> np.ndarray:
    """Project feature vectors into 2D space using t-SNE."""
    if TSNE is None:
        raise RuntimeError("scikit-learn TSNE is required for t-SNE projection.")
    features = np.asarray(features)
    if features.ndim != 2 or not np.issubdtype(features.dtype, np.number):
        raise ValueError("t-SNE input must be a numeric 2D feature matrix.")
    if not prepared:
        if not np.isfinite(features).all() or np.any(np.linalg.norm(np.nan_to_num(features), axis=1) <= 1e-12):
            raise ValueError("Invalid rows must be filtered with aligned labels using prepare_features_for_projection first.")
        features, _, _ = prepare_features_for_projection(features, random_state=random_state)
    if features.ndim != 2 or not np.isfinite(features).all():
        raise ValueError("t-SNE input must be a finite 2D feature matrix.")
    n_samples = len(features)
    if n_samples < 3:
        raise ValueError("t-SNE needs at least 3 feature vectors to produce a 2D projection.")
    if perplexity <= 0:
        raise ValueError("t-SNE perplexity must be positive.")
    safe_perplexity = min(float(perplexity), float(n_samples - 1) - 1e-3)
    kwargs: dict[str, Any] = {
        "n_components": 2,
        "perplexity": safe_perplexity,
        "random_state": random_state,
        "init": "pca",
        "learning_rate": "auto",
    }
    try:
        tsne = TSNE(max_iter=n_iter, **kwargs)
    except TypeError:
        tsne = TSNE(n_iter=n_iter, **kwargs)
    result = tsne.fit_transform(features)
    if result.shape != (n_samples, 2) or not np.isfinite(result).all():
        raise RuntimeError("t-SNE did not produce a finite (N, 2) projection.")
    return result


def compute_umap_projection(
    features: np.ndarray,
    n_neighbors: int = 30,
    min_dist: float = 0.1,
    random_state: int = 42,
    metric: str = "euclidean",
    prepared: bool = False,
) -> np.ndarray:
    """Project feature vectors into 2D space using UMAP."""
    if not UMAP_AVAILABLE:
        detail = f" ({UMAP_IMPORT_ERROR})" if UMAP_IMPORT_ERROR else ""
        raise RuntimeError(
            "UMAP is unavailable because the umap-learn package is not installed or failed to import"
            f"{detail}. Install umap-learn to enable UMAP, or run t-SNE only."
        )

    features = np.asarray(features)
    if features.ndim != 2 or not np.issubdtype(features.dtype, np.number):
        raise ValueError("UMAP input must be a numeric 2D feature matrix.")
    if not prepared:
        if not np.isfinite(features).all() or np.any(np.linalg.norm(np.nan_to_num(features), axis=1) <= 1e-12):
            raise ValueError("Invalid rows must be filtered with aligned labels using prepare_features_for_projection first.")
        features, _, _ = prepare_features_for_projection(features, random_state=random_state)
    if features.ndim != 2 or not np.isfinite(features).all():
        raise ValueError("UMAP input must be a finite 2D feature matrix.")
    n_samples = len(features)
    if n_samples < 3:
        raise ValueError("UMAP needs at least 3 feature vectors to produce a 2D projection.")
    if n_neighbors < 2:
        raise ValueError("UMAP n_neighbors must be at least 2.")
    if min_dist < 0.0:
        raise ValueError("UMAP min_dist cannot be negative.")
    safe_neighbors = min(n_neighbors, n_samples - 1)
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=safe_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )
    result = reducer.fit_transform(features)
    if result.shape != (n_samples, 2) or not np.isfinite(result).all():
        raise RuntimeError("UMAP did not produce a finite (N, 2) projection.")
    return result


def umap_unavailable_message() -> str:
    if UMAP_AVAILABLE:
        return ""
    detail = f" Import error: {UMAP_IMPORT_ERROR}" if UMAP_IMPORT_ERROR else ""
    return (
        "UMAP projection was skipped because umap-learn is not available."
        f"{detail} t-SNE can still be used."
    )


def render_feature_projections(
    tsne_2d: np.ndarray | None,
    umap_2d: np.ndarray | None,
    labels: np.ndarray,
    class_names: list[str] | None = None,
    title: str = "CIFAR-10 Feature Space 2D Projections",
) -> Image.Image:
    """Render 2D feature projection plots with 10-class colour coding."""
    classes = class_names or list(CIFAR10_CLASSES)
    num_panels = 2 if (tsne_2d is not None and umap_2d is not None) else 1
    if tsne_2d is None and umap_2d is None:
        raise ValueError("At least one of t-SNE or UMAP coordinates is required.")

    fig, axes = plt.subplots(1, num_panels, figsize=(7.2 * num_panels, 6.0), dpi=100)
    fig.patch.set_facecolor("#ffffff")
    if num_panels == 1:
        axes = [axes]

    panel_idx = 0
    if tsne_2d is not None:
        ax = axes[panel_idx]
        for c_idx in range(len(classes)):
            mask = labels == c_idx
            if np.any(mask):
                color = CIFAR10_CLASS_COLORS[c_idx % len(CIFAR10_CLASS_COLORS)]
                ax.scatter(
                    tsne_2d[mask, 0],
                    tsne_2d[mask, 1],
                    label=classes[c_idx],
                    color=color,
                    alpha=0.8,
                    s=20,
                    edgecolors="white",
                    linewidths=0.25,
                )
        ax.set_title("t-SNE Projection", fontsize=12, fontweight="bold", pad=8)
        ax.set_xlabel("t-SNE 1", fontsize=10, fontweight="bold")
        ax.set_ylabel("t-SNE 2", fontsize=10, fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(loc="best", fontsize=8, framealpha=0.8)
        panel_idx += 1

    if umap_2d is not None and panel_idx < len(axes):
        ax = axes[panel_idx]
        for c_idx in range(len(classes)):
            mask = labels == c_idx
            if np.any(mask):
                color = CIFAR10_CLASS_COLORS[c_idx % len(CIFAR10_CLASS_COLORS)]
                ax.scatter(
                    umap_2d[mask, 0],
                    umap_2d[mask, 1],
                    label=classes[c_idx],
                    color=color,
                    alpha=0.8,
                    s=20,
                    edgecolors="white",
                    linewidths=0.25,
                )
        ax.set_title("UMAP Projection", fontsize=12, fontweight="bold", pad=8)
        ax.set_xlabel("UMAP 1", fontsize=10, fontweight="bold")
        ax.set_ylabel("UMAP 2", fontsize=10, fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(loc="best", fontsize=8, framealpha=0.8)

    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)
