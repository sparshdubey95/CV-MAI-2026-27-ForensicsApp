"""Filters from skimage.filters: Gaussian, Median, Sobel, and Prewitt."""

from __future__ import annotations

import tkinter as tk
import numpy as np
import skimage.filters
import skimage.morphology
from PIL import Image

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
from .dialogs import ask_choice, ask_float, ask_int, show_error


def to_float_array(image: Image.Image) -> tuple[np.ndarray, bool]:
    """Convert PIL image to float numpy array in [0.0, 1.0]; return array and is_rgb flag."""
    arr = np.asarray(image).astype(np.float64) / 255.0
    is_rgb = arr.ndim == 3 and arr.shape[2] >= 3
    if is_rgb and arr.shape[2] > 3:
        arr = arr[:, :, :3]  # drop alpha if present
    return arr, is_rgb


def to_grayscale_float(image: Image.Image) -> np.ndarray:
    """Convert PIL image to 2D float grayscale array in [0.0, 1.0].
    because edge-detection algorithms work best on single-channel intensity data"""
    gray_image = image.convert("L")
    return np.asarray(gray_image).astype(np.float64) / 255.0


def from_float_array(arr: np.ndarray) -> Image.Image:
    """Convert float numpy array in [0.0, 1.0] to uint8 PIL Image."""
    clipped = np.clip(arr, 0.0, 1.0)
    uint8_arr = (clipped * 255.0).round().astype(np.uint8)
    return Image.fromarray(uint8_arr)


def apply_gaussian(image: Image.Image, sigma: float = 2.0) -> Image.Image:
    """Apply skimage.filters.gaussian with configurable sigma."""
    if sigma <= 0:
        raise ValueError(f"Sigma must be positive, got {sigma}.")
    arr, is_rgb = to_float_array(image)
    filtered = skimage.filters.gaussian(arr, sigma=sigma, channel_axis=-1 if is_rgb else None)
    return from_float_array(filtered)


# Median filters are fantastic for removing "salt-and-pepper" noise (random noisy specs on an image)
# without completely blurring out sharp edges.
def apply_median(image: Image.Image, radius: int = 2) -> Image.Image:
    """Apply skimage.filters.median with configurable disk footprint radius."""
    if radius < 1:
        raise ValueError(f"Radius must be at least 1, got {radius}.")
    footprint = skimage.morphology.disk(radius)
    arr, is_rgb = to_float_array(image)

    if is_rgb:
        # Apply median channel by channel
        channels = [
            skimage.filters.median(arr[:, :, c], footprint=footprint)
            for c in range(arr.shape[2])
        ]
        filtered = np.stack(channels, axis=-1)
    else:
        filtered = skimage.filters.median(arr, footprint=footprint)

    return from_float_array(filtered)

# Sobel & prewitt are classic edge detectors. They calculate gradients (changes in pixel brightness)
# to find outlines and borders in an image, converting the result to a clean normalized range between 0 and 1.
def apply_sobel(image: Image.Image) -> Image.Image:
    """Apply skimage.filters.sobel edge detector."""
    gray = to_grayscale_float(image)
    edges = skimage.filters.sobel(gray)
    # Normalize edges to [0, 1] if needed
    max_val = np.max(edges)
    if max_val > 0:
        edges = edges / max_val
    return from_float_array(edges)


def apply_prewitt(image: Image.Image) -> Image.Image:
    """Apply skimage.filters.prewitt edge detector."""
    gray = to_grayscale_float(image)
    edges = skimage.filters.prewitt(gray)
    max_val = np.max(edges)
    if max_val > 0:
        edges = edges / max_val
    return from_float_array(edges)


SKIMAGE_FILTER_CHOICES = [
    ("gaussian", "Gaussian filter (configurable sigma)"),
    ("median", "Median filter (configurable radius)"),
    ("sobel", "Sobel edge detector"),
    ("prewitt", "Prewitt edge detector"),
]


class SkimageFiltersTool(ForensicsTool):
    tool_id = "skimage_filters"
    title = "Scikit-image filters"
    category = "Set 3"
    description = "Apply Gaussian, Median, Sobel, or Prewitt filters from skimage.filters."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None

        choice = ask_choice(
            parent,
            "Scikit-Image Filters",
            "Choose a filter from skimage.filters:",
            SKIMAGE_FILTER_CHOICES,
            initial="gaussian",
        )
        if choice is None:
            return None

        details: dict[str, object] = {
            "Operation": "skimage.filters",
            "Filter": choice,
        }

        if choice == "gaussian":
            sigma = ask_float(
                parent,
                "Gaussian Filter",
                "Sigma (standard deviation):",
                initial=2.0,
                min_val=0.1,
                max_val=50.0,
            )
            if sigma is None:
                return None
            output = apply_gaussian(document.current, sigma=sigma)
            details["Sigma"] = sigma
            message = f"Applied skimage.filters.gaussian (sigma={sigma:g})."

        elif choice == "median":
            radius = ask_int(
                parent,
                "Median Filter",
                "Disk footprint radius (pixels):",
                initial=2,
                min_val=1,
                max_val=30,
            )
            if radius is None:
                return None
            output = apply_median(document.current, radius=radius)
            details["Radius"] = radius
            message = f"Applied skimage.filters.median (disk radius={radius})."

        elif choice == "sobel":
            output = apply_sobel(document.current)
            message = "Applied skimage.filters.sobel edge detector."

        elif choice == "prewitt":
            output = apply_prewitt(document.current)
            message = "Applied skimage.filters.prewitt edge detector."

        else:
            show_error(parent, "Filters", f"Unknown filter choice {choice!r}")
            return None

        return ToolResult(
            image=output,
            message=message,
            details=details,
        )
