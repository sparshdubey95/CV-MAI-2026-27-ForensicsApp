"""Face recognition using saved Eigenface PCA model (Set 6.3).

This module implements:
  - Loading pre-trained Eigenface models (.npz) without retraining PCA.
  - Applying identical preprocessing (grayscale conversion, resizing to model shape).
  - Projecting the test image into PCA space to compute its coefficients.
  - Comparing test coefficients against training coefficients using Euclidean distance:
        d = ||c_test - c_train_i||_2
  - Visualizing the test image alongside the closest 2–3 training face matches with
    their names and distance scores.
"""

from __future__ import annotations

import io
from pathlib import Path
import tkinter as tk
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
from .dialogs import ask_choice, ask_image_file, ask_npz_file, show_error
from .eigenface_builder import get_active_model, load_eigenface_model, set_active_model


def preprocess_face_image(
    image: Image.Image | np.ndarray,
    target_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Preprocess a test image identically to the training pipeline.

    Parameters
    ----------
    image : Image.Image | np.ndarray
        Input image.
    target_size : tuple[int, int]
        (width, height) expected by the trained model.

    Returns
    -------
    x_test : np.ndarray
        (D,) float32 flattened vector.
    thumbnail : np.ndarray
        (H, W) uint8 grayscale image.
    """
    target_w, target_h = target_size

    if isinstance(image, Image.Image):
        pil_img = image.convert("L")
    elif isinstance(image, np.ndarray):
        if image.ndim == 2:
            pil_img = Image.fromarray(image.astype(np.uint8))
        elif image.ndim == 3:
            # RGB / RGBA
            pil_img = Image.fromarray(image.astype(np.uint8)).convert("L")
        else:
            raise ValueError(f"Invalid image array dimension: {image.ndim}")
    else:
        raise TypeError(f"Expected PIL Image or np.ndarray, got {type(image)}")

    resized = pil_img.resize((target_w, target_h), Image.Resampling.LANCZOS)
    thumbnail = np.array(resized, dtype=np.uint8)
    x_test = thumbnail.astype(np.float32).reshape(-1)

    return x_test, thumbnail


def project_test_face(
    x_test: np.ndarray,
    mean_face: np.ndarray,
    eigenfaces: np.ndarray,
) -> np.ndarray:
    """Project centered test face vector onto eigenfaces to compute PCA coefficients.

    c_test = (x_test - mean_face) · eigenfaces^T
    """
    centered = x_test.astype(np.float32) - mean_face.astype(np.float32)
    # eigenfaces is (K, D), centered is (D,) -> c_test is (K,)
    c_test = np.dot(eigenfaces, centered)
    return c_test.astype(np.float32)


def recognize_face(
    test_image: Image.Image | np.ndarray,
    model: dict[str, Any],
    top_k: int = 3,
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray]:
    """Identify the closest matching training faces for a given test image.

    Parameters
    ----------
    test_image : Image.Image | np.ndarray
        Query face image.
    model : dict[str, Any]
        Loaded Eigenface model containing mean_face, eigenfaces, coefficients, labels, etc.
    top_k : int
        Number of closest matches to return (default 3).

    Returns
    -------
    top_matches : list[dict[str, Any]]
        List of dictionaries for the top matching candidates.
    c_test : np.ndarray
        (K,) PCA projection coefficients for the query image.
    test_thumbnail : np.ndarray
        (H, W) preprocessed grayscale query face.
    """
    target_h, target_w = model["image_shape"]
    mean_face = model["mean_face"]
    eigenfaces = model["eigenfaces"]
    coefficients = model["coefficients"]  # (N, K)
    labels = model["labels"]              # (N,)
    thumbnails = model.get("thumbnails")  # (N, H, W) or None

    x_test, test_thumbnail = preprocess_face_image(test_image, (target_w, target_h))
    c_test = project_test_face(x_test, mean_face, eigenfaces)

    # Compute Euclidean distance between c_test and all training coefficients
    # d_i = ||c_test - c_train_i||_2
    diffs = coefficients - c_test.reshape(1, -1)
    distances = np.linalg.norm(diffs, axis=1)

    # Rank closest candidates
    n_candidates = len(distances)
    k_retrieve = min(max(1, top_k), n_candidates)
    sorted_indices = np.argsort(distances)[:k_retrieve]

    top_matches: list[dict[str, Any]] = []
    for rank, idx in enumerate(sorted_indices):
        name = str(labels[idx])
        dist = float(distances[idx])
        thumb = thumbnails[idx] if thumbnails is not None and idx < len(thumbnails) else None
        top_matches.append({
            "rank": rank + 1,
            "index": int(idx),
            "name": name,
            "distance": dist,
            "thumbnail": thumb,
        })

    return top_matches, c_test, test_thumbnail


def render_recognition_result(
    test_thumbnail: np.ndarray,
    top_matches: list[dict[str, Any]],
    query_title: str = "Query Face",
) -> Image.Image:
    """Render a side-by-side figure showing the test face and top matches with distance scores."""
    n_matches = len(top_matches)
    total_panels = 1 + n_matches

    fig, axes = plt.subplots(1, total_panels, figsize=(total_panels * 2.8, 3.4), dpi=100)
    fig.patch.set_facecolor("#f8f9fa")

    if total_panels == 1:
        axes = [axes]

    # Panel 0: Query Face
    ax_query = axes[0]
    ax_query.imshow(test_thumbnail, cmap="gray")
    ax_query.set_title(f"{query_title}\n(Input)", fontsize=11, fontweight="bold", pad=8)
    ax_query.axis("off")
    # Draw blue/cyan outline
    for spine in ax_query.spines.values():
        spine.set_edgecolor("#0066cc")
        spine.set_linewidth(3)
        spine.set_visible(True)

    # Panels 1..N: Top Matches
    for rank, match in enumerate(top_matches):
        ax = axes[rank + 1]
        thumb = match["thumbnail"]
        if thumb is not None:
            ax.imshow(thumb, cmap="gray")
        else:
            # Placeholder if thumbnails were not saved
            ax.text(
                0.5,
                0.5,
                match["name"],
                ha="center",
                va="center",
                fontsize=10,
                wrap=True,
            )
            ax.set_facecolor("#e0e0e0")

        # Top 1 match highlighted with green outline
        border_color = "#28a745" if rank == 0 else "#6c757d"
        for spine in ax.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(3 if rank == 0 else 1.5)
            spine.set_visible(True)

        ax.set_title(
            f"Rank #{rank + 1}: {match['name']}\nDist: {match['distance']:.2f}",
            fontsize=10,
            fontweight="bold" if rank == 0 else "normal",
            pad=8,
        )
        ax.axis("off")

    plt.suptitle("Face Recognition Results (Eigenface PCA Matching)", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


class FaceRecognitionTool(ForensicsTool):
    """Set 6.3: Recognize face using saved Eigenface PCA model."""

    tool_id = "face_recognition"
    title = "Face recognition (Eigenfaces)"
    category = "Set 6: Face Detection & Recognition"
    description = "Match a test face to the closest training faces using PCA projection and Euclidean distance."
    requires_image = False

    def run(self, parent: tk.Misc | None, document: ImageDocument) -> ToolResult | None:
        # 1. Acquire Model
        active_model = get_active_model()
        model: dict[str, Any] | None = None

        if active_model is not None and parent is not None:
            choice = ask_choice(
                parent,
                "Select Model Source",
                "An active Eigenface model is available in memory. Choose model source:",
                [
                    ("active", "Use active model currently in memory"),
                    ("file", "Browse and load a saved model (.npz) file..."),
                ],
                initial="active",
            )
            if choice is None:
                return None
            if choice == "active":
                model = active_model
            else:
                npz_path = ask_npz_file(parent, title="Select saved Eigenface model (.npz)")
                if not npz_path:
                    return None
                try:
                    model = load_eigenface_model(npz_path)
                    set_active_model(model)
                except Exception as err:
                    show_error(parent, "Model Load Error", f"Failed to load model: {err}")
                    return None
        elif active_model is not None:
            # Headless fallback
            model = active_model
        else:
            # No active model; prompt user to select one
            npz_path = ask_npz_file(parent, title="Select saved Eigenface model (.npz)")
            if not npz_path:
                if parent is not None:
                    show_error(
                        parent,
                        "Model Required",
                        "Please select a trained Eigenface model (.npz) or use the Eigenface Builder first.",
                    )
                return None
            try:
                model = load_eigenface_model(npz_path)
                set_active_model(model)
            except Exception as err:
                if parent is not None:
                    show_error(parent, "Model Load Error", f"Failed to load model: {err}")
                return None

        # 2. Acquire Test Image
        test_image: Image.Image | None = None
        source_name = "Query Image"

        if document.is_loaded and document.current is not None:
            if parent is not None:
                choice = ask_choice(
                    parent,
                    "Select Test Image",
                    "Choose test image source for recognition:",
                    [
                        ("current", "Use currently opened image in ForensicsApp"),
                        ("browse", "Browse for a test face image file..."),
                    ],
                    initial="current",
                )
                if choice is None:
                    return None
                if choice == "current":
                    test_image = document.current
                    if document.path:
                        source_name = document.path.name
                else:
                    path = ask_image_file(parent, title="Select test face image")
                    if not path:
                        return None
                    try:
                        with Image.open(path) as opened:
                            test_image = opened.copy()
                        source_name = Path(path).name
                    except Exception as err:
                        show_error(parent, "Open Error", f"Could not open image: {err}")
                        return None
            else:
                test_image = document.current
                if document.path:
                    source_name = document.path.name
        else:
            path = ask_image_file(parent, title="Select test face image")
            if not path:
                return None
            try:
                with Image.open(path) as opened:
                    test_image = opened.copy()
                source_name = Path(path).name
            except Exception as err:
                if parent is not None:
                    show_error(parent, "Open Error", f"Could not open image: {err}")
                return None

        # 3. Perform Face Recognition
        try:
            top_matches, c_test, test_thumbnail = recognize_face(
                test_image=test_image,
                model=model,
                top_k=3,
            )
        except Exception as err:
            if parent is not None:
                show_error(parent, "Recognition Error", str(err))
            return None

        # 4. Render Result Montage
        rendered_montage = render_recognition_result(
            test_thumbnail=test_thumbnail,
            top_matches=top_matches,
            query_title=source_name,
        )

        best_match = top_matches[0]
        details: dict[str, Any] = {
            "Operation": "Eigenface PCA Recognition",
            "Query image": source_name,
            "Predicted identity": best_match["name"],
            "Rank 1 Match": f"{best_match['name']} (Distance: {best_match['distance']:.3f})",
        }
        for m in top_matches[1:]:
            details[f"Rank {m['rank']} Match"] = f"{m['name']} (Distance: {m['distance']:.3f})"

        details["Model K components"] = len(model["eigenfaces"])
        details["Training images compared"] = len(model["coefficients"])
        if "filepath" in model:
            details["Model source"] = Path(model["filepath"]).name

        message = (
            f"Recognition complete: Closest identity is '{best_match['name']}' "
            f"(Euclidean distance: {best_match['distance']:.2f})."
        )

        return ToolResult(
            image=rendered_montage,
            message=message,
            details=details,
        )
