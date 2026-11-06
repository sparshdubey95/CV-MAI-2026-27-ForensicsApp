"""Eigenface model builder using PCA (Set 6.2).

This module implements:
  - Face dataset loading from local folders or sklearn.datasets.fetch_lfw_people.
  - Image-to-person name correspondence tracking.
  - Consistent grayscale conversion and resizing.
  - Principal Component Analysis (PCA) to compute:
      - Mean face
      - Eigenfaces (principal eigenvectors)
      - Training image PCA coefficients (projections)
  - Model serialization and loading via NumPy (.npz).
  - Visualization of the mean face and the top K eigenfaces.
"""

from __future__ import annotations

import io
from pathlib import Path
import re
import tkinter as tk
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
from .dialogs import (
    ask_eigenface_builder_parameters,
    ask_image_folder,
    ask_save_npz_file,
    show_error,
)


SUPPORTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

# Active model cache stored in memory during an application session
_ACTIVE_MODEL: dict[str, Any] | None = None


def get_active_model() -> dict[str, Any] | None:
    """Retrieve the currently loaded or trained Eigenface model from memory."""
    global _ACTIVE_MODEL
    return _ACTIVE_MODEL


def set_active_model(model: dict[str, Any] | None) -> None:
    """Cache the active Eigenface model in memory."""
    global _ACTIVE_MODEL
    _ACTIVE_MODEL = model


def extract_person_name_from_filename(stem: str) -> str:
    """Extract a human-readable person name from a filename stem.

    Examples
    --------
    - 'George_W_Bush_0001' -> 'George W Bush'
    - 'Colin_Powell_04'    -> 'Colin Powell'
    - 'person1_05'         -> 'Person 1'
    """
    cleaned = re.sub(r"[_\-\d]+$", "", stem).strip("_ -")
    if not cleaned:
        cleaned = stem
    cleaned = cleaned.replace("_", " ").strip()
    return cleaned.title() if cleaned else stem


def load_face_dataset(
    source: str = "folder",
    folder_path: str | Path | None = None,
    target_size: tuple[int, int] = (64, 64),
    min_faces_per_person: int = 1,
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    """Load face images and return preprocessed matrices and labels.

    Parameters
    ----------
    source : str
        'folder' for a local directory or 'lfw' for fetch_lfw_people.
    folder_path : str | Path | None
        Path to local folder (required if source == 'folder').
    target_size : tuple[int, int]
        (width, height) for image resizing.
    min_faces_per_person : int
        Minimum number of images per person for LFW dataset.

    Returns
    -------
    X : np.ndarray
        (N, D) float32 matrix of flattened face vectors.
    labels : np.ndarray
        (N,) string array of person names.
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
        try:
            from sklearn.datasets import fetch_lfw_people
        except ImportError as err:
            raise ImportError(f"scikit-learn is required to load LFW dataset: {err}") from err

        lfw = fetch_lfw_people(
            min_faces_per_person=max(min_faces_per_person, 1),
            color=False,
            resize=None,
            download_if_missing=True,
        )
        if len(lfw.images) == 0:
            raise ValueError("LFW dataset returned 0 face images.")

        for i, raw_face in enumerate(lfw.images):
            # Normalize raw float face to uint8 [0, 255]
            if raw_face.max() <= 1.0:
                uint8_face = np.clip(raw_face * 255.0, 0, 255).astype(np.uint8)
            else:
                uint8_face = np.clip(raw_face, 0, 255).astype(np.uint8)

            pil_face = Image.fromarray(uint8_face).resize((target_w, target_h), Image.Resampling.LANCZOS)
            arr = np.array(pil_face, dtype=np.uint8)
            person_name = lfw.target_names[lfw.target[i]]

            face_thumbnails.append(arr)
            flattened_faces.append(arr.astype(np.float32).reshape(-1))
            face_labels.append(str(person_name))

    elif source.lower() == "folder":
        if folder_path is None:
            raise ValueError("A folder path must be provided when loading from 'folder'.")
        root = Path(folder_path)
        if not root.is_dir():
            raise FileNotFoundError(f"Folder not found: {root}")

        # Check for subdirectories (each subdirectory representing a person)
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
            # Flat folder
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
            except Exception as err:
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


def train_eigenfaces(
    X: np.ndarray,
    n_components: int = 50,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, PCA]:
    """Compute mean face, eigenfaces, and projection coefficients using PCA.

    Parameters
    ----------
    X : np.ndarray
        (N, D) float32 matrix of face vectors.
    n_components : int
        Maximum number of principal components to keep.

    Returns
    -------
    mean_face : np.ndarray
        (D,) float32 vector representing the average face.
    eigenfaces : np.ndarray
        (K, D) float32 matrix where each row is an eigenface.
    coefficients : np.ndarray
        (N, K) float32 matrix of training images projected into PCA space.
    pca : PCA
        Fitted scikit-learn PCA instance.
    """
    n_samples, n_features = X.shape
    if n_samples < 2:
        raise ValueError("At least 2 training images are required to compute PCA.")

    # Number of components cannot exceed min(n_samples - 1, n_features)
    k = max(1, min(int(n_components), n_samples - 1, n_features))

    pca = PCA(n_components=k, svd_solver="randomized", whiten=False)
    coefficients = pca.fit_transform(X).astype(np.float32)
    mean_face = pca.mean_.astype(np.float32)
    eigenfaces = pca.components_.astype(np.float32)

    return mean_face, eigenfaces, coefficients, pca


def save_eigenface_model(
    filepath: str | Path,
    mean_face: np.ndarray,
    eigenfaces: np.ndarray,
    coefficients: np.ndarray,
    labels: np.ndarray,
    image_shape: tuple[int, int],
    thumbnails: np.ndarray | None = None,
    explained_variance_ratio: np.ndarray | None = None,
) -> None:
    """Save the trained PCA model to a compressed NumPy archive (.npz)."""
    save_path = Path(filepath)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    data: dict[str, Any] = {
        "mean_face": mean_face.astype(np.float32),
        "eigenfaces": eigenfaces.astype(np.float32),
        "coefficients": coefficients.astype(np.float32),
        "labels": np.array(labels, dtype=str),
        "image_shape": np.array(image_shape, dtype=np.int32),
    }
    if thumbnails is not None:
        data["thumbnails"] = thumbnails.astype(np.uint8)
    if explained_variance_ratio is not None:
        data["explained_variance_ratio"] = explained_variance_ratio.astype(np.float32)

    np.savez_compressed(save_path, **data)


def load_eigenface_model(filepath: str | Path) -> dict[str, Any]:
    """Load a trained PCA eigenface model from a NumPy archive (.npz)."""
    load_path = Path(filepath)
    if not load_path.is_file():
        raise FileNotFoundError(f"Eigenface model file not found: {load_path}")

    with np.load(load_path, allow_pickle=False) as npz:
        required_keys = {"mean_face", "eigenfaces", "coefficients", "labels", "image_shape"}
        missing = required_keys - set(npz.files)
        if missing:
            raise KeyError(f"Invalid model archive, missing keys: {missing}")

        model = {
            "mean_face": npz["mean_face"],
            "eigenfaces": npz["eigenfaces"],
            "coefficients": npz["coefficients"],
            "labels": npz["labels"],
            "image_shape": tuple(int(x) for x in npz["image_shape"]),
            "thumbnails": npz["thumbnails"] if "thumbnails" in npz.files else None,
            "explained_variance_ratio": (
                npz["explained_variance_ratio"] if "explained_variance_ratio" in npz.files else None
            ),
            "filepath": str(load_path),
        }

    return model


def render_eigenfaces_montage(
    mean_face: np.ndarray,
    eigenfaces: np.ndarray,
    image_shape: tuple[int, int],
    max_eigenfaces: int = 12,
    explained_variance_ratio: np.ndarray | None = None,
) -> Image.Image:
    """Render a visualization of the mean face and the top K eigenfaces."""
    target_h, target_w = image_shape
    k = min(len(eigenfaces), max_eigenfaces)

    # 1 panel for mean face + k panels for eigenfaces
    total_panels = 1 + k
    cols = 4 if total_panels >= 4 else total_panels
    rows = int(np.ceil(total_panels / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.6, rows * 2.8), dpi=100)
    fig.patch.set_facecolor("#f8f9fa")

    flat_axes = np.array(axes).reshape(-1)

    # 1. Render Mean Face
    mean_img = mean_face.reshape((target_h, target_w))
    ax_mean = flat_axes[0]
    ax_mean.imshow(mean_img, cmap="gray")
    ax_mean.set_title("Mean Face", fontsize=11, fontweight="bold", pad=6)
    ax_mean.axis("off")

    # 2. Render Top K Eigenfaces
    for i in range(k):
        ax = flat_axes[i + 1]
        ef = eigenfaces[i].reshape((target_h, target_w))
        # Normalize eigenface for optimal visual contrast
        ef_norm = ef - ef.min()
        if ef_norm.max() > 0:
            ef_norm = ef_norm / ef_norm.max()

        ax.imshow(ef_norm, cmap="bone")
        if explained_variance_ratio is not None and i < len(explained_variance_ratio):
            var_pct = explained_variance_ratio[i] * 100.0
            ax.set_title(f"Eigenface {i + 1}\n({var_pct:.1f}% var)", fontsize=9, pad=4)
        else:
            ax.set_title(f"Eigenface {i + 1}", fontsize=9, pad=4)
        ax.axis("off")

    # Turn off unused subplot axes
    for j in range(total_panels, len(flat_axes)):
        flat_axes[j].axis("off")

    plt.suptitle("PCA Eigenfaces Visualization", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


class EigenfaceBuilderTool(ForensicsTool):
    """Set 6.2: Build and save PCA Eigenfaces from a folder or LFW dataset."""

    tool_id = "eigenface_builder"
    title = "Eigenface builder"
    category = "Set 6: Face Detection & Recognition"
    description = "Compute PCA mean face, eigenfaces and coefficients from face dataset and save model."
    requires_image = False

    def run(self, parent: tk.Misc | None, document: ImageDocument) -> ToolResult | None:
        # Prompt for parameters (source, components K, dimensions)
        params = ask_eigenface_builder_parameters(
            parent,
            initial_source="folder",
            initial_k=50,
            initial_size="64, 64",
        )
        if params is None:
            return None

        source, k_req, (target_w, target_h) = params
        folder_path: str | None = None

        if source == "folder":
            folder_path = ask_image_folder(parent, title="Select folder of face images")
            if not folder_path:
                return None

        # Load face dataset
        try:
            X, labels, unique_names, thumbnails = load_face_dataset(
                source=source,
                folder_path=folder_path,
                target_size=(target_w, target_h),
                min_faces_per_person=1,
            )
        except Exception as err:
            if parent is not None:
                show_error(parent, "Dataset Load Error", str(err))
            return None

        # Train PCA
        try:
            mean_face, eigenfaces, coefficients, pca = train_eigenfaces(X, n_components=k_req)
        except Exception as err:
            if parent is not None:
                show_error(parent, "PCA Training Error", str(err))
            return None

        # Ask destination to save model .npz file
        save_path = ask_save_npz_file(
            parent,
            title="Save trained Eigenface model (.npz)",
            initialfile="eigenface_model.npz",
        )
        if save_path:
            try:
                save_eigenface_model(
                    save_path,
                    mean_face=mean_face,
                    eigenfaces=eigenfaces,
                    coefficients=coefficients,
                    labels=labels,
                    image_shape=(target_h, target_w),
                    thumbnails=thumbnails,
                    explained_variance_ratio=pca.explained_variance_ratio_,
                )
            except Exception as err:
                if parent is not None:
                    show_error(parent, "Save Error", f"Could not save model: {err}")

        # Update session active model
        model_dict = {
            "mean_face": mean_face,
            "eigenfaces": eigenfaces,
            "coefficients": coefficients,
            "labels": labels,
            "image_shape": (target_h, target_w),
            "thumbnails": thumbnails,
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "filepath": str(save_path) if save_path else "In-memory session",
        }
        set_active_model(model_dict)

        # Render Eigenface visualization
        montage = render_eigenfaces_montage(
            mean_face=mean_face,
            eigenfaces=eigenfaces,
            image_shape=(target_h, target_w),
            max_eigenfaces=11,
            explained_variance_ratio=pca.explained_variance_ratio_,
        )

        total_variance = float(np.sum(pca.explained_variance_ratio_)) * 100.0
        details = {
            "Operation": "PCA Eigenface Training",
            "Dataset source": f"{source.upper()} ({Path(folder_path).name if folder_path else 'sklearn'})",
            "Total training images": len(X),
            "Unique identities": len(unique_names),
            "Image dimensions (WxH)": f"{target_w} × {target_h}",
            "Components (K)": len(eigenfaces),
            "Total explained variance": f"{total_variance:.2f}%",
            "Saved model": str(Path(save_path).name) if save_path else "In-memory active session",
        }

        message = (
            f"Eigenface model built successfully: {len(X)} images, "
            f"{len(unique_names)} persons, K={len(eigenfaces)} ({total_variance:.1f}% variance)."
        )

        return ToolResult(
            image=montage,
            message=message,
            details=details,
        )
