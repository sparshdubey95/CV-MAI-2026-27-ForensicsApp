"""Canny edge detection using skimage.feature.canny with configurable parameters."""

from __future__ import annotations

import tkinter as tk
import numpy as np
import skimage.feature
from PIL import Image

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
from .dialogs import ask_float


def to_grayscale_float(image: Image.Image) -> np.ndarray:
    """Convert PIL image to a 2D float64 array in [0.0, 1.0]."""
    return np.asarray(image.convert("L")).astype(np.float64) / 255.0


def apply_canny(
    image: Image.Image,
    sigma: float = 1.0,
    low_threshold: float | None = None,
    high_threshold: float | None = None,
) -> np.ndarray:
    """Run Canny edge detection and return a boolean edge mask.

    How Canny works (the three steps):
      1. Gaussian blur: smooth the image using sigma to reduce noise.
      2. Gradient magnitude: find how sharply intensity changes at each pixel.
      3. Hysteresis thresholding: keep only strong edges (> high_threshold)
         and weak edges that connect to strong ones (> low_threshold).
    """
    if sigma <= 0:
        raise ValueError(f"Sigma must be positive, got {sigma}.")
    if low_threshold is not None and high_threshold is not None:
        if low_threshold >= high_threshold:
            raise ValueError("low_threshold must be less than high_threshold.")

    gray = to_grayscale_float(image)
    edges = skimage.feature.canny(
        gray,
        sigma=sigma,
        low_threshold=low_threshold,
        high_threshold=high_threshold,
    )
    return edges  # dtype bool, True where an edge was detected


def edges_to_image(edge_mask: np.ndarray) -> Image.Image:
    """Convert a boolean edge mask to a black-and-white PIL Image (edges = white)."""
    uint8 = (edge_mask * 255).astype(np.uint8)
    return Image.fromarray(uint8, mode="L")


class CannyEdgeTool(ForensicsTool):
    tool_id = "canny_edges"
    title = "Canny edge detection"
    category = "Set 3"
    description = "Detect edges with skimage.feature.canny. Configurable sigma and thresholds."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None

        # --- configurable sigma ---
        # (higher sigma = fewer details, smoother edges).
        sigma = ask_float(
            parent,
            "Canny Edge Detection",
            "Sigma for Gaussian smoothing (higher = less noise, fewer edges):",
            initial=1.0,
            min_val=0.1,
            max_val=10.0,
        )
        if sigma is None:
            return None

        # --- optional low threshold ---
        low = ask_float(
            parent,
            "Canny – Low Threshold",
            "Low hysteresis threshold (0–1). Leave 0 to use skimage default:",
            initial=0.0,
            min_val=0.0,
            max_val=1.0,
        )
        if low is None:
            return None
        low_threshold = low if low > 0 else None

        # --- optional high threshold ---
        high = ask_float(
            parent,
            "Canny – High Threshold",
            "High hysteresis threshold (0–1). Leave 0 to use skimage default:",
            initial=0.0,
            min_val=0.0,
            max_val=1.0,
        )
        if high is None:
            return None
        high_threshold = high if high > 0 else None

        try:
            edge_mask = apply_canny(
                document.current,
                sigma=sigma,
                low_threshold=low_threshold,
                high_threshold=high_threshold,
            )
        except ValueError as error:
            from .dialogs import show_error
            show_error(parent, "Canny Edge Detection", str(error))
            return None

        output = edges_to_image(edge_mask)
        edge_count = int(np.sum(edge_mask))
        total = edge_mask.size

        return ToolResult(
            image=output,
            message=f"Canny edges detected (sigma={sigma:g}). White pixels are edges.",
            details={
                "Operation": "Canny edge detection",
                "Sigma": sigma,
                "Low threshold": low_threshold if low_threshold is not None else "auto",
                "High threshold": high_threshold if high_threshold is not None else "auto",
                "Edge pixels": f"{edge_count} / {total}",
            },
        )
