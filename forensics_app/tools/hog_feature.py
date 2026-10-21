"""HOG feature extraction and multi-orientation visualization (Set 4.3).

This module implements:
  - Extraction of Histogram of Oriented Gradients (HOG) descriptors.
  - Multi-orientation side-by-side comparison figure directly matching Slide 4:
      [Original image | HoG orientation=2 | HoG orientation=4 | HoG orientation=9]
  - Configurable orientation bins, pixels_per_cell (e.g. 6x6 or 8x8),
    and cells_per_block (e.g. 3x3 or 2x2).
"""

from __future__ import annotations

import io
import tkinter as tk
from typing import Any

import matplotlib
matplotlib.use("Agg")  # Non-interactive off-screen plotting
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import skimage.exposure as exposure
import skimage.feature as feature

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
from .dialogs import show_error


# ── Algorithmic Functions ───────────────────────────────────────────────────

def compute_hog(
    image: Image.Image | np.ndarray,
    orientations: int = 9,
    pixels_per_cell: tuple[int, int] = (8, 8),
    cells_per_block: tuple[int, int] = (2, 2),
) -> tuple[np.ndarray, Image.Image]:
    """Extract a HOG descriptor and visualization image using skimage.feature.hog.

    What HOG (Histogram of Oriented Gradients) does in plain English:
      1. Gradients: It calculates edge directions and strengths across small blocks.
      2. Cells & Histograms: Groups pixels into cells (e.g., 6x6 or 8x8) and bins
         gradient directions into discrete angle bins (e.g., 2, 4, or 9 bins).
      3. Normalization: Groups cells into larger blocks (e.g., 2x2 or 3x3) and normalizes
         the energy to make detection invariant to illumination and shadows.

    Parameters
    ----------
    image : PIL.Image.Image or np.ndarray
        Input image (RGB or grayscale).
    orientations : int
        Number of gradient orientation bins (e.g., 2, 4, 8, 9, 16).
    pixels_per_cell : tuple[int, int]
        Size of each cell in pixels (e.g., (6, 6) or (8, 8)).
    cells_per_block : tuple[int, int]
        Number of cells in each normalization block (e.g., (3, 3) or (2, 2)).

    Returns
    -------
    tuple[np.ndarray, PIL.Image.Image]
        (1D feature descriptor vector, HOG visualization image as 8-bit PIL Image)
    """
    if orientations < 1:
        raise ValueError(f"orientations must be positive, got {orientations}")
    if pixels_per_cell[0] < 1 or pixels_per_cell[1] < 1:
        raise ValueError(f"pixels_per_cell values must be positive, got {pixels_per_cell}")
    if cells_per_block[0] < 1 or cells_per_block[1] < 1:
        raise ValueError(f"cells_per_block values must be positive, got {cells_per_block}")

    # Check minimum image dimensions required to form at least one block
    min_h = pixels_per_cell[0] * cells_per_block[0]
    min_w = pixels_per_cell[1] * cells_per_block[1]

    if isinstance(image, Image.Image):
        arr = np.asarray(image)
        w, h = image.size
    else:
        arr = image
        h, w = arr.shape[:2]

    if h < min_h or w < min_w:
        raise ValueError(
            f"Image dimensions ({w}x{h}) are too small for block size ({min_w}x{min_h})."
        )

    # Use channel_axis=-1 for 3D color images (RGB), or None for 2D grayscale arrays
    channel_axis = -1 if arr.ndim == 3 and arr.shape[2] in (3, 4) else None
    if channel_axis is not None and arr.shape[2] == 4:
        # Drop alpha channel for HOG computation if present
        arr = arr[:, :, :3]

    fd, hog_image = feature.hog(
        arr,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        visualize=True,
        channel_axis=channel_axis,
    )

    # Rescale intensity to stretch contrast so gradient lines appear bright and crisp
    if hog_image.max() > 0:
        hog_rescaled = exposure.rescale_intensity(hog_image, in_range="image")
    else:
        hog_rescaled = hog_image

    hog_pil = Image.fromarray((hog_rescaled * 255.0).astype(np.uint8), mode="L")
    return fd, hog_pil


def compute_hog_comparison_figure(
    image: Image.Image | np.ndarray,
    orientations_list: list[int] = [2, 4, 9],
    pixels_per_cell: tuple[int, int] = (6, 6),
    cells_per_block: tuple[int, int] = (3, 3),
) -> tuple[Image.Image, dict[str, Any]]:
    """Render the 4-panel figure directly matching Slide 4 of the PDF.

    Slide 4 shows:
      Panel 1: 'Original image' (Color image with axes ticks)
      Panel 2: 'HoG image with orientation=2' (with axes ticks)
      Panel 3: 'HoG image with orientation=4' (with axes ticks)
      Panel 4: 'HoG image with orientation=9' (with axes ticks)

    Configurable parameters (Slide 4):
      - orientations (e.g. 2, 4, 9)
      - pixels_per_cell (default (6, 6))
      - cells_per_block (default (3, 3))
    """
    if isinstance(image, Image.Image):
        arr = np.asarray(image)
    else:
        arr = image

    channel_axis = -1 if arr.ndim == 3 and arr.shape[2] in (3, 4) else None
    if channel_axis is not None and arr.shape[2] == 4:
        arr = arr[:, :, :3]

    num_panels = 1 + len(orientations_list)
    fig, axes = plt.subplots(1, num_panels, figsize=(4.5 * num_panels, 4.5), dpi=100)

    # Panel 0: Original image
    if arr.ndim == 2:
        axes[0].imshow(arr, cmap="gray")
    else:
        axes[0].imshow(arr)
    axes[0].set_title("Original image", fontsize=12)

    descriptors: dict[str, int] = {}

    # Panels 1..N: HOG visualization for each orientation in orientations_list
    for idx, ori in enumerate(orientations_list, start=1):
        fd, hog_img = feature.hog(
            arr,
            orientations=ori,
            pixels_per_cell=pixels_per_cell,
            cells_per_block=cells_per_block,
            visualize=True,
            channel_axis=channel_axis,
        )
        descriptors[f"Descriptor length (ori={ori})"] = len(fd)

        # Rescale intensity for crisp contrast
        if hog_img.max() > 0:
            hog_disp = exposure.rescale_intensity(hog_img, in_range="image")
        else:
            hog_disp = hog_img

        axes[idx].imshow(hog_disp, cmap="gray")
        axes[idx].set_title(f"HoG image with orientation={ori}", fontsize=12)

    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    plt.close(fig)
    buf.seek(0)
    output_image = Image.open(buf).convert("RGB")

    details = {
        "Operation": "HOG Feature Extraction",
        "Visualization": f"Multi-orientation ({', '.join(str(o) for o in orientations_list)})",
        "Pixels per cell": f"{pixels_per_cell[0]} × {pixels_per_cell[1]}",
        "Cells per block": f"{cells_per_block[0]} × {cells_per_block[1]}",
        "Orientations": orientations_list[-1],
        "Descriptor length": descriptors.get(f"Descriptor length (ori={orientations_list[-1]})", 0),
        **descriptors,
    }
    return output_image, details


# ── Tool Class ───────────────────────────────────────────────────────────────

class HOGFeatureTool(ForensicsTool):
    """Set 4.3: Render the fixed multi-orientation comparison from Slide 4."""

    tool_id = "hog_feature"
    title = "HOG feature extraction"
    category = "Set 4: Feature & HOG Detection"
    description = "Extract HOG features and display the 4-panel comparison from Slide 4."

    def run(self, parent: tk.Misc | None, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None

        try:
            output_image, details = compute_hog_comparison_figure(
                document.current,
                orientations_list=[2, 4, 9],
                pixels_per_cell=(6, 6),
                cells_per_block=(3, 3),
            )
        except ValueError as err:
            show_error(parent, "HOG Error", str(err))
            return None

        return ToolResult(
            image=output_image,
            message="Compared HOG visualizations for 2, 4, and 9 orientations.",
            details=details,
        )
