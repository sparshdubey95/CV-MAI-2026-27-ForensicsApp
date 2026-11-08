"""Core Eigenfaces and PCA mathematical pipeline (Set 6.2 & Set 6.3).

This module implements:
  - Mathematical Eigenfaces workflow:
      1. Calculate mean face μ = (1/N) ∑ x_i
      2. Centre images: x_i - μ
      3. Compute first K principal components via PCA/SVD (U_K)
      4. Project training images: w_i = U_K^T (x_i - μ)
      5. Store coefficients with corresponding identity labels
  - Complete Train/Test separation using LFW:
      - subset="train" for training
      - subset="test" for testing
      - Automatic export of test images to 'LFW_Test_Images' with labels.csv & manifest.json
  - Full model serialization as .npy (NumPy binary format) and .npz
  - Visualization of the first 12 eigenfaces indexed 0..11
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
import re
import threading
from typing import Any, Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.utils import Bunch

SUPPORTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

# Active session model cache
_ACTIVE_MODEL: dict[str, Any] | None = None


def get_active_model() -> dict[str, Any] | None:
    """Retrieve the currently active Eigenface model from session memory."""
    global _ACTIVE_MODEL
    return _ACTIVE_MODEL


def set_active_model(model: dict[str, Any] | None) -> None:
    """Cache the active Eigenface model in session memory."""
    global _ACTIVE_MODEL
    _ACTIVE_MODEL = model


def extract_person_name_from_filename(stem: str) -> str:
    """Extract a clean person name from a filename stem.

    Examples:
        'George_W_Bush_0001' -> 'George W Bush'
        'Colin_Powell_04'    -> 'Colin Powell'
    """
    cleaned = re.sub(r"[_\-\d]+$", "", stem).strip("_ -")
    if not cleaned:
        cleaned = stem
    cleaned = cleaned.replace("_", " ").strip()
    return cleaned.title() if cleaned else stem


# ---------------------------------------------------------------------------
# Train / Test Separation and LFW Dataset Management
# ---------------------------------------------------------------------------

def export_test_dataset(
    images: np.ndarray,
    labels: np.ndarray | list[str],
    export_dir: str | Path = "LFW_Test_Images",
    target_names: list[str] | np.ndarray | None = None,
) -> tuple[Path, Path]:
    """Export test images organized by identity and generate manifest files.

    Parameters
    ----------
    images : np.ndarray
        Array of test images of shape (N, H, W) or (N, D).
    labels : np.ndarray | list[str]
        Identity labels or target indices corresponding to each image.
    export_dir : str | Path
        Directory to export test images into (default 'LFW_Test_Images').
    target_names : list[str] | np.ndarray | None
        Optional mapping from integer targets to string identities.

    Returns
    -------
    csv_path : Path
        Path to generated labels.csv.
    json_path : Path
        Path to generated manifest.json.
    """
    out_dir = Path(export_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_records: list[dict[str, str]] = []
    manifest_map: dict[str, str] = {}

    for idx, raw_img in enumerate(images):
        raw_label = labels[idx]
        if target_names is not None and isinstance(raw_label, (int, np.integer)):
            identity = str(target_names[raw_label])
        else:
            identity = str(raw_label)

        safe_name = re.sub(r"[^\w\s-]", "", identity).strip().replace(" ", "_")
        if not safe_name:
            safe_name = f"Identity_{idx}"

        person_dir = out_dir / safe_name
        person_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{safe_name}_{idx:04d}.png"
        file_path = person_dir / filename
        rel_path = file_path.relative_to(out_dir).as_posix()

        # Format image to uint8 [0, 255]
        img_arr = np.array(raw_img, dtype=np.float32)
        if img_arr.ndim == 1:
            # Reshape if flattened and square-ish
            side = int(np.round(np.sqrt(img_arr.size)))
            if side * side == img_arr.size:
                img_arr = img_arr.reshape((side, side))

        if img_arr.max() <= 1.0 and img_arr.min() >= 0.0:
            uint8_img = np.clip(img_arr * 255.0, 0, 255).astype(np.uint8)
        else:
            uint8_img = np.clip(img_arr, 0, 255).astype(np.uint8)

        Image.fromarray(uint8_img).save(file_path)

        manifest_records.append({
            "filename": filename,
            "identity": identity,
            "relative_path": rel_path,
        })
        manifest_map[filename] = identity
        manifest_map[rel_path] = identity

    # 1. Write labels.csv
    csv_path = out_dir / "labels.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "identity", "relative_path"])
        writer.writeheader()
        writer.writerows(manifest_records)

    # 2. Write manifest.json
    json_path = out_dir / "manifest.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(manifest_map, f, indent=2)

    return csv_path, json_path


def fetch_lfw_people(
    subset: str | None = None,
    min_faces_per_person: int = 20,
    color: bool = False,
    resize: float | None = 0.5,
    download_if_missing: bool = True,
    test_size: float = 0.25,
    random_state: int = 42,
    auto_export_test: bool = True,
    export_dir: str | Path = "LFW_Test_Images",
    **kwargs: Any,
) -> Bunch:
    """Fetch LFW dataset with strict train/test separation support.

    Wraps `sklearn.datasets.fetch_lfw_people` to support `subset='train'`
    and `subset='test'`. Training PCA on training images only guarantees
    zero test data leakage. Automatically exports test images to `LFW_Test_Images`
    with full ground truth manifest.

    Parameters
    ----------
    subset : str | None
        'train', 'test', or None/'all'.
    min_faces_per_person : int
        Filter for individuals with at least this many face images.
    color : bool
        Whether to retain RGB color channels.
    resize : float | None
        Resize ratio applied by sklearn fetcher.
    download_if_missing : bool
        Download LFW if missing locally.
    test_size : float
        Proportion of dataset reserved for testing (default 0.25).
    random_state : int
        Random seed for deterministic train/test splitting.
    auto_export_test : bool
        If True, exports test images to `export_dir` when test split is formed.
    export_dir : str | Path
        Directory where test images are exported (default 'LFW_Test_Images').
    """
    try:
        from sklearn.datasets import fetch_lfw_people as _sk_fetch_lfw
    except ImportError as err:
        raise ImportError(f"scikit-learn is required to load LFW dataset: {err}") from err

    # sklearn fetch_lfw_people does not accept 'subset', filter kwargs
    sk_kwargs = {
        "min_faces_per_person": max(min_faces_per_person, 0),
        "color": color,
        "resize": resize,
        "download_if_missing": download_if_missing,
    }
    for k in ("data_home", "funneled", "slice_"):
        if k in kwargs:
            sk_kwargs[k] = kwargs[k]

    raw_lfw = _sk_fetch_lfw(**sk_kwargs)
    n_samples = len(raw_lfw.images)
    if n_samples == 0:
        raise ValueError("LFW dataset returned 0 face images.")

    if subset is None or subset.lower() in ("all", "none"):
        return raw_lfw

    subset_norm = subset.lower().strip()
    if subset_norm not in ("train", "test"):
        raise ValueError(f"Unknown subset: {subset}. Must be 'train', 'test', or 'all'.")

    indices = np.arange(n_samples)
    try:
        train_idx, test_idx = train_test_split(
            indices,
            test_size=test_size,
            random_state=random_state,
            stratify=raw_lfw.target,
        )
    except ValueError:
        train_idx, test_idx = train_test_split(
            indices,
            test_size=test_size,
            random_state=random_state,
        )

    # Automatically export test images if requested
    if auto_export_test:
        try:
            export_test_dataset(
                images=raw_lfw.images[test_idx],
                labels=raw_lfw.target[test_idx],
                export_dir=export_dir,
                target_names=raw_lfw.target_names,
            )
        except Exception as export_err:
            # Non-fatal warning
            pass

    selected_idx = train_idx if subset_norm == "train" else test_idx

    bunch = Bunch(
        data=raw_lfw.data[selected_idx],
        images=raw_lfw.images[selected_idx],
        target=raw_lfw.target[selected_idx],
        target_names=raw_lfw.target_names,
        DESCR=f"LFW Face Dataset (Subset: {subset_norm}, {len(selected_idx)} samples)",
    )
    return bunch


def load_face_dataset(
    source: str = "folder",
    folder_path: str | Path | None = None,
    target_size: tuple[int, int] = (64, 64),
    min_faces_per_person: int = 1,
    subset: str = "train",
    export_test_images: bool = True,
    export_dir: str | Path = "LFW_Test_Images",
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    """Load face images and return preprocessed matrices and labels.

    Parameters
    ----------
    source : str
        'folder' for local directory or 'lfw' for fetch_lfw_people.
    folder_path : str | Path | None
        Path to local folder (required if source == 'folder').
    target_size : tuple[int, int]
        (width, height) for image normalization.
    min_faces_per_person : int
        Minimum number of images per person for LFW dataset.
    subset : str
        'train' or 'test' (used for LFW).
    export_test_images : bool
        Whether to export test images when loading LFW.
    export_dir : str | Path
        Directory for test images export.

    Returns
    -------
    X : np.ndarray
        (N, D) float32 matrix of flattened face vectors.
    labels : np.ndarray
        (N,) array of person identity names.
    unique_names : list[str]
        List of unique person names.
    thumbnails : np.ndarray
        (N, H, W) uint8 array of preprocessed face images.
    """
    target_w, target_h = target_size
    flattened_faces: list[np.ndarray] = []
    face_labels: list[str] = []
    face_thumbnails: list[np.ndarray] = []

    if source.lower() == "lfw":
        lfw = fetch_lfw_people(
            subset=subset,
            min_faces_per_person=max(min_faces_per_person, 1),
            color=False,
            resize=None,
            download_if_missing=True,
            auto_export_test=export_test_images,
            export_dir=export_dir,
        )

        for i, raw_face in enumerate(lfw.images):
            if raw_face.max() <= 1.0:
                uint8_face = np.clip(raw_face * 255.0, 0, 255).astype(np.uint8)
            else:
                uint8_face = np.clip(raw_face, 0, 255).astype(np.uint8)

            pil_face = Image.fromarray(uint8_face).resize((target_w, target_h), Image.Resampling.LANCZOS)
            arr = np.array(pil_face, dtype=np.uint8)
            person_name = str(lfw.target_names[lfw.target[i]])

            face_thumbnails.append(arr)
            flattened_faces.append(arr.astype(np.float32).reshape(-1))
            face_labels.append(person_name)

    elif source.lower() == "folder":
        if folder_path is None:
            raise ValueError("A folder path must be provided when loading from 'folder'.")
        root = Path(folder_path)
        if not root.is_dir():
            raise FileNotFoundError(f"Folder not found: {root}")

        subdirs = [p for p in root.iterdir() if p.is_dir()]
        has_subdirs = any(
            any(f.suffix.lower() in SUPPORTED_IMAGE_EXTS for f in d.iterdir() if f.is_file())
            for d in subdirs
        )

        image_entries: list[tuple[Path, str]] = []
        if has_subdirs:
            for subdir in sorted(subdirs):
                person_name = subdir.name.replace("_", " ").title()
                for img_path in sorted(subdir.iterdir()):
                    if img_path.is_file() and img_path.suffix.lower() in SUPPORTED_IMAGE_EXTS:
                        image_entries.append((img_path, person_name))
        else:
            for img_path in sorted(root.iterdir()):
                if img_path.is_file() and img_path.suffix.lower() in SUPPORTED_IMAGE_EXTS:
                    person_name = extract_person_name_from_filename(img_path.stem)
                    image_entries.append((img_path, person_name))

        if not image_entries:
            raise ValueError(f"No supported image files found in folder: {root}")

        for img_path, person_name in image_entries:
            try:
                with Image.open(img_path) as opened:
                    gray = opened.convert("L")
                    resized = gray.resize((target_w, target_h), Image.Resampling.LANCZOS)
                    arr = np.array(resized, dtype=np.uint8)
                    face_thumbnails.append(arr)
                    flattened_faces.append(arr.astype(np.float32).reshape(-1))
                    face_labels.append(person_name)
            except Exception:
                continue

        if not flattened_faces:
            raise ValueError(f"Failed to read any valid face images from: {root}")
    else:
        raise ValueError(f"Unknown dataset source: {source}")

    X = np.stack(flattened_faces, axis=0)
    labels = np.array(face_labels, dtype=object)
    unique_names = sorted(list(set(face_labels)))
    thumbnails = np.stack(face_thumbnails, axis=0)

    return X, labels, unique_names, thumbnails


# ---------------------------------------------------------------------------
# Mathematical Eigenfaces Pipeline (PCA / SVD)
# ---------------------------------------------------------------------------

def compute_mean_face(X: np.ndarray) -> np.ndarray:
    """Calculate mean face: μ = (1/N) ∑ x_i.

    Parameters
    ----------
    X : np.ndarray
        (N, D) float32 matrix of face vectors.

    Returns
    -------
    mean_face : np.ndarray
        (D,) float32 mean face vector.
    """
    if len(X) == 0:
        raise ValueError("Cannot compute mean face of empty array.")
    return np.mean(X, axis=0).astype(np.float32)


def center_faces(X: np.ndarray, mean_face: np.ndarray) -> np.ndarray:
    """Center face images: x_i - μ.

    Parameters
    ----------
    X : np.ndarray
        (N, D) float32 matrix.
    mean_face : np.ndarray
        (D,) float32 vector.

    Returns
    -------
    X_centered : np.ndarray
        (N, D) float32 zero-mean matrix.
    """
    return (X.astype(np.float32) - mean_face.astype(np.float32)).astype(np.float32)


def compute_principal_components(
    X_centered: np.ndarray,
    n_components: int = 12,
) -> tuple[np.ndarray, PCA]:
    """Calculate first K principal components (Eigenfaces) using PCA/SVD.

    Parameters
    ----------
    X_centered : np.ndarray
        (N, D) centered face vectors.
    n_components : int
        Number of principal components K to retain (default 12).

    Returns
    -------
    eigenfaces : np.ndarray
        (K, D) float32 matrix of orthonormal principal component vectors (U_K).
    pca : PCA
        Fitted scikit-learn PCA instance.
    """
    n_samples, n_features = X_centered.shape
    if n_samples < 2:
        raise ValueError("At least 2 training images are required to compute PCA.")

    k = max(1, min(int(n_components), n_samples - 1, n_features))
    # Note: Fitting on centered data gives identical components to fitting on uncentered
    pca = PCA(n_components=k, svd_solver="randomized", whiten=False)
    pca.fit(X_centered)
    eigenfaces = pca.components_.astype(np.float32)
    return eigenfaces, pca


def project_faces(
    X_centered: np.ndarray,
    eigenfaces: np.ndarray,
) -> np.ndarray:
    """Project face images onto eigenfaces: w_i = U_K (x_i - μ).

    Parameters
    ----------
    X_centered : np.ndarray
        (N, D) or (D,) centered face vectors.
    eigenfaces : np.ndarray
        (K, D) orthonormal eigenface matrix U_K.

    Returns
    -------
    coefficients : np.ndarray
        (N, K) or (K,) float32 projection coefficients.
    """
    if X_centered.ndim == 1:
        # (K, D) @ (D,) -> (K,)
        return np.dot(eigenfaces, X_centered).astype(np.float32)
    # (N, D) @ (D, K) -> (N, K)
    return np.dot(X_centered, eigenfaces.T).astype(np.float32)


def train_eigenfaces(
    X: np.ndarray,
    n_components: int = 12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, PCA]:
    """Execute complete Eigenfaces mathematical training workflow.

    Workflow:
      1. Calculate mean face μ
      2. Centre training images: x_i - μ
      3. Calculate first K principal components using PCA/SVD
      4. Project training images: w_i = U_K^T (x_i - μ)
      5. Return mean, components, coefficients and fitted PCA

    Parameters
    ----------
    X : np.ndarray
        (N, D) float32 matrix of flattened face vectors.
    n_components : int
        Number of principal components K (default 12).

    Returns
    -------
    mean_face : np.ndarray
        (D,) float32 mean face vector.
    eigenfaces : np.ndarray
        (K, D) float32 matrix of principal components U_K.
    coefficients : np.ndarray
        (N, K) float32 matrix of projected training weights.
    pca : PCA
        Fitted PCA instance.
    """
    n_samples, n_features = X.shape
    if n_samples < 2:
        raise ValueError("At least 2 training images are required to compute PCA.")

    k = max(1, min(int(n_components), n_samples - 1, n_features))

    # 1. Calculate mean face
    mean_face = compute_mean_face(X)

    # 2. Centre images
    X_centered = center_faces(X, mean_face)

    # 3. Calculate principal components
    pca = PCA(n_components=k, svd_solver="randomized", whiten=False)
    # Fit PCA
    pca.fit(X)
    eigenfaces = pca.components_.astype(np.float32)
    mean_face = pca.mean_.astype(np.float32)

    # 4. Project training images: w_i = U_K (x_i - μ)
    coefficients = project_faces(center_faces(X, mean_face), eigenfaces)

    return mean_face, eigenfaces, coefficients, pca


# ---------------------------------------------------------------------------
# Model Serialization (.npy and .npz)
# ---------------------------------------------------------------------------

def save_eigenface_model(
    filepath: str | Path,
    mean_face: np.ndarray,
    eigenfaces: np.ndarray,
    coefficients: np.ndarray,
    labels: np.ndarray | list[str],
    image_shape: tuple[int, int],
    target_names: list[str] | np.ndarray | None = None,
    thumbnails: np.ndarray | None = None,
    explained_variance_ratio: np.ndarray | None = None,
    k: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save the trained Eigenface model through file dialog as .npy or .npz.

    Includes:
      - mean face μ
      - components U_K (eigenfaces)
      - training coefficients (weights)
      - labels
      - target names
      - training images (thumbnails)
      - dimensions (H, W)
      - K
      - preprocessing metadata
    """
    save_path = Path(filepath)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if target_names is None:
        target_names = sorted(list(set(str(l) for l in labels)))

    k_val = k if k is not None else len(eigenfaces)

    meta = {
        "k": int(k_val),
        "image_shape": tuple(int(x) for x in image_shape),
        "n_samples": len(labels),
        "n_classes": len(target_names),
        "format": "eigenfaces_model_v2",
    }
    if metadata:
        meta.update(metadata)

    model_dict: dict[str, Any] = {
        "mean_face": mean_face.astype(np.float32),
        "components": eigenfaces.astype(np.float32),
        "eigenfaces": eigenfaces.astype(np.float32),
        "training_coefficients": coefficients.astype(np.float32),
        "coefficients": coefficients.astype(np.float32),
        "labels": np.array([str(x) for x in labels], dtype=str),
        "target_names": np.array([str(x) for x in target_names], dtype=str),
        "dimensions": np.array(image_shape, dtype=np.int32),
        "image_shape": np.array(image_shape, dtype=np.int32),
        "k": int(k_val),
        "metadata": meta,
    }

    if thumbnails is not None:
        model_dict["training_images"] = thumbnails.astype(np.uint8)
        model_dict["thumbnails"] = thumbnails.astype(np.uint8)

    if explained_variance_ratio is not None:
        model_dict["explained_variance_ratio"] = explained_variance_ratio.astype(np.float32)

    if save_path.suffix.lower() == ".npz":
        np.savez_compressed(save_path, **model_dict)
    else:
        # Default save as .npy
        np.save(save_path, model_dict, allow_pickle=True)


def load_eigenface_model(filepath: str | Path) -> dict[str, Any]:
    """Load an Eigenface model from .npy or .npz file."""
    load_path = Path(filepath)
    if not load_path.is_file():
        raise FileNotFoundError(f"Eigenface model file not found: {load_path}")

    if load_path.suffix.lower() == ".npz":
        with np.load(load_path, allow_pickle=True) as npz:
            raw_dict = {k: npz[k] for k in npz.files}
    else:
        loaded = np.load(load_path, allow_pickle=True)
        if isinstance(loaded, np.ndarray) and loaded.shape == ():
            raw_dict = loaded.item()
        elif isinstance(loaded, dict):
            raw_dict = loaded
        else:
            raw_dict = loaded.item()

    # Reconcile key aliases
    mean_face = raw_dict.get("mean_face")
    eigenfaces = raw_dict.get("eigenfaces")
    if eigenfaces is None:
        eigenfaces = raw_dict.get("components")

    coefficients = raw_dict.get("coefficients")
    if coefficients is None:
        coefficients = raw_dict.get("training_coefficients")

    labels = raw_dict.get("labels")
    if labels is not None:
        labels = np.array([str(x) for x in labels], dtype=str)

    image_shape_raw = raw_dict.get("image_shape")
    if image_shape_raw is None:
        image_shape_raw = raw_dict.get("dimensions")
    if image_shape_raw is not None:
        image_shape = tuple(int(x) for x in np.array(image_shape_raw).reshape(-1)[:2])
    else:
        image_shape = (64, 64)

    thumbnails = raw_dict.get("thumbnails")
    if thumbnails is None:
        thumbnails = raw_dict.get("training_images")

    target_names = raw_dict.get("target_names")
    if target_names is None and labels is not None:
        target_names = np.array(sorted(list(set(labels))), dtype=str)

    if mean_face is None or eigenfaces is None or coefficients is None or labels is None:
        raise KeyError("Model file is missing mandatory keys (mean_face, eigenfaces, coefficients, labels).")

    model = {
        "mean_face": np.array(mean_face, dtype=np.float32),
        "eigenfaces": np.array(eigenfaces, dtype=np.float32),
        "components": np.array(eigenfaces, dtype=np.float32),
        "coefficients": np.array(coefficients, dtype=np.float32),
        "training_coefficients": np.array(coefficients, dtype=np.float32),
        "labels": labels,
        "target_names": target_names,
        "image_shape": image_shape,
        "dimensions": image_shape,
        "thumbnails": thumbnails,
        "training_images": thumbnails,
        "explained_variance_ratio": raw_dict.get("explained_variance_ratio"),
        "k": int(raw_dict.get("k", len(eigenfaces))),
        "metadata": raw_dict.get("metadata", {}),
        "filepath": str(load_path),
    }
    return model


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def render_eigenfaces_montage(
    mean_face: np.ndarray,
    eigenfaces: np.ndarray,
    image_shape: tuple[int, int],
    max_eigenfaces: int = 12,
    explained_variance_ratio: np.ndarray | None = None,
    include_mean: bool = True,
) -> Image.Image:
    """Visualise Eigenfaces (first 12, indexed 0..11) matching Set 6.2 slide.

    Shows eigenfaces labeled 'eigenface 0', 'eigenface 1', ..., 'eigenface 11'
    in a clean 3x4 grid (or with Mean face panel on top/side).

    Parameters
    ----------
    mean_face : np.ndarray
        (D,) float32 mean face vector.
    eigenfaces : np.ndarray
        (K, D) float32 matrix.
    image_shape : tuple[int, int]
        (H, W) dimensions.
    max_eigenfaces : int
        Number of eigenfaces to display (default 12).
    explained_variance_ratio : np.ndarray | None
        Optional explained variance ratios.
    include_mean : bool
        Whether to include mean face as the initial tile.
    """
    target_h, target_w = image_shape
    k = min(len(eigenfaces), max_eigenfaces)

    if include_mean:
        total_panels = 1 + k
        cols = 4
        rows = int(np.ceil(total_panels / cols))
    else:
        total_panels = k
        cols = 4 if k >= 4 else k
        rows = int(np.ceil(total_panels / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.8), dpi=100)
    fig.patch.set_facecolor("#f8f9fa")
    flat_axes = np.array(axes).reshape(-1)

    offset = 0
    if include_mean:
        # Panel 0: Mean Face
        ax_mean = flat_axes[0]
        mean_img = mean_face.reshape((target_h, target_w))
        ax_mean.imshow(mean_img, cmap="gray")
        ax_mean.set_title("Mean Face (μ)", fontsize=11, fontweight="bold", pad=6)
        ax_mean.axis("off")
        offset = 1

    # Panels: eigenfaces indexed 0..11
    for i in range(k):
        ax = flat_axes[i + offset]
        ef = eigenfaces[i].reshape((target_h, target_w))
        ef_norm = ef - ef.min()
        if ef_norm.max() > 0:
            ef_norm = ef_norm / ef_norm.max()

        ax.imshow(ef_norm, cmap="bone")
        if explained_variance_ratio is not None and i < len(explained_variance_ratio):
            var_pct = explained_variance_ratio[i] * 100.0
            ax.set_title(f"eigenface {i}\n({var_pct:.1f}% var)", fontsize=9, pad=4)
        else:
            ax.set_title(f"eigenface {i}", fontsize=10, pad=4)
        ax.axis("off")

    for j in range(total_panels, len(flat_axes)):
        flat_axes[j].axis("off")

    plt.suptitle("PCA Eigenfaces (Set 6.2)", fontsize=13, fontweight="bold", y=0.99)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)
