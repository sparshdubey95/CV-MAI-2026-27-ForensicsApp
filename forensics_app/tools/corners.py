"""FAST and Harris corner detection (Set 5.1)."""

from __future__ import annotations

import io
import tkinter as tk

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage import color, feature, util

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
from .dialogs import ask_choice, ask_float, ask_int, show_error


def _grayscale(image: Image.Image | np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 3:
        array = color.rgb2gray(array[..., :3])
    return util.img_as_float(array)


def detect_corners(
    image: Image.Image | np.ndarray,
    algorithm: str = "FAST",
    min_distance: int = 5,
    threshold_rel: float = 0.1,
) -> np.ndarray:
    """Return peak corner coordinates as (row, column) pairs."""
    algorithm_name = algorithm.upper()
    gray = _grayscale(image)
    if min_distance < 1:
        raise ValueError("min_distance must be at least 1.")
    if not 0 <= threshold_rel <= 1:
        raise ValueError("threshold_rel must be between 0 and 1.")
    if algorithm_name == "FAST":
        response = feature.corner_fast(gray)
    elif algorithm_name == "HARRIS":
        response = feature.corner_harris(gray)
    else:
        raise ValueError("algorithm must be 'FAST' or 'Harris'.")
    return feature.corner_peaks(
        response,
        min_distance=min_distance,
        threshold_rel=threshold_rel,
    )


def render_corners(
    image: Image.Image | np.ndarray,
    corners: np.ndarray,
    algorithm: str,
) -> Image.Image:
    """Render grayscale image with detected corners shown as hollow red circles."""
    gray = _grayscale(image)
    fig, ax = plt.subplots(figsize=(7, 6), dpi=100)
    ax.imshow(gray, cmap="gray")
    if corners.size:
        ax.scatter(
            corners[:, 1],
            corners[:, 0],
            facecolors="none",
            edgecolors="r",
        )
    ax.set_title(f"{algorithm} corners")
    ax.axis("off")
    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    with Image.open(buffer) as output:
        return output.convert("RGB")


class CornerDetectionTool(ForensicsTool):
    """Set 5.1: Detect FAST or Harris corners."""

    tool_id = "corner_detection"
    title = "Corner detection"
    category = "Set 5: Features & Matching"
    description = "Detect FAST or Harris corners and highlight them on the image."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None
        algorithm = ask_choice(
            parent, "Corner Detection", "Select corner detector:",
            [("FAST", "FAST"), ("Harris", "Harris")], initial="FAST",
        )
        if algorithm is None:
            return None
        min_distance = ask_int(
            parent, "Corner Spacing", "Minimum distance between corners:",
            initial=5, min_val=1,
        )
        if min_distance is None:
            return None
        threshold_rel = ask_float(
            parent, "Corner Threshold", "Relative corner threshold (0 to 1):",
            initial=0.1, min_val=0.0, max_val=1.0,
        )
        if threshold_rel is None:
            return None
        try:
            corners = detect_corners(document.current, algorithm, min_distance, threshold_rel)
            output = render_corners(document.current, corners, algorithm)
        except ValueError as error:
            show_error(parent, "Corner Detection Error", str(error))
            return None
        return ToolResult(
            image=output,
            message=f"Detected {len(corners)} {algorithm} corners.",
            details={
                "Algorithm": algorithm,
                "Corner count": len(corners),
                "Minimum distance": min_distance,
                "Relative threshold": threshold_rel,
            },
        )
