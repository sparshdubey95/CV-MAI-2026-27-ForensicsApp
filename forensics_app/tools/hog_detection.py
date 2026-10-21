"""Person / object detection using HOG and sliding window (Set 4.4).

This module implements:
  - Sliding-window object detection using HOG feature descriptors.
  - Compares sliding window patches against a target template using:
      1. Euclidean Distance: d_E = ||f_tmpl - f_patch||_2  (lower is better)
      2. Dot Product: s = f_tmpl · f_patch               (higher is better)
  - Visualizes detection using the 3-panel comparison from Slide 5:
      [Original image | Euclidean distance | Dot product]
    with red bounding boxes marking the best matches in each panel.
"""

from __future__ import annotations

import io
from pathlib import Path
import tkinter as tk
from typing import Any

import matplotlib
matplotlib.use("Agg")  # Non-interactive off-screen plotting
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image
import skimage.feature as feature

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
from .dialogs import ask_image_file, show_error
from .template_matching import draw_bounding_box


# ── Helper: Locate course assignment person template ────────────────────────

def find_default_person_template(
    document_path: str | None = None,
) -> Image.Image | None:
    """Try to find the official person template (person_template.bmp) from Slide 5."""
    candidate_paths: list[Path] = []

    # 1. Look in the same folder as the currently opened image
    if document_path:
        folder = Path(document_path).parent
        candidate_paths.extend((folder / "person_template.bmp", folder.parent / "person_template.bmp"))

    for path in candidate_paths:
        if path.is_file():
            try:
                with Image.open(path) as img:
                    return img.copy()
            except OSError:
                continue

    return None


def acquire_person_template(
    parent: tk.Misc | None,
    image: Image.Image,
    document_path: str | None = None,
) -> Image.Image | None:
    """Use a nearby assignment template, or prompt for one with a crop fallback."""
    default_tmpl = find_default_person_template(document_path)

    if default_tmpl is not None:
        tw, th = default_tmpl.size
        iw, ih = image.size
        if tw <= iw and th <= ih:
            return default_tmpl

    if parent is not None:
        path = ask_image_file(parent, title="Select person template image file")
        if path:
            try:
                with Image.open(path) as selected:
                    return selected.copy()
            except OSError as error:
                show_error(parent, "Template Error", f"Could not open template image: {error}")
                return None

    w, h = image.size
    crop_w = min(w, max(8, w // 3))
    crop_h = min(h, max(8, h // 2))
    left = (w - crop_w) // 2
    top = (h - crop_h) // 2
    return image.crop((left, top, left + crop_w, top + crop_h))


# ── Algorithmic Functions ───────────────────────────────────────────────────

def detect_person_hog(
    image: Image.Image,
    template: Image.Image,
    step_size: int = 16,
    orientations: int = 16,
    pixels_per_cell: tuple[int, int] = (8, 8),
    cells_per_block: tuple[int, int] = (2, 2),
    method: str = "side_by_side",
    as_figure: bool = False,
) -> tuple[Image.Image, dict[str, Any]]:
    """Locate template within search image using sliding window HOG matching (Slide 5).

    Plain English Explanation of the Algorithm:
      1. First, we compute the HOG descriptor for the template (target person).
      2. Next, we slide a window of the same width and height across the search image,
         moving by `step_size` pixels horizontally and vertically.
      3. For every window position (i, j):
         - Extract HOG descriptor of that local window patch.
         - Compute Euclidean distance: ||f_template - f_patch||_2  (lower means more similar).
         - Compute Dot product similarity: f_template · f_patch  (higher means more similar).
      4. Find the best position for both metrics:
         - Best Euclidean match: index of minimum distance.
         - Best Dot Product match: index of maximum similarity.
      5. Draw red bounding boxes at the detected positions to display results.

    Parameters
    ----------
    image : PIL.Image.Image
        The search image containing pedestrians.
    template : PIL.Image.Image
        The reference person template patch.
    step_size : int
        Step stride in pixels for the sliding window (e.g., 4, 8, 16).
    orientations : int
        Number of gradient orientation bins (Slide 5 uses 16).
    pixels_per_cell : tuple[int, int]
        Cell size in pixels (Slide 5 uses (8, 8)).
    cells_per_block : tuple[int, int]
        Block size in cells (Slide 5 uses (2, 2)).
    method : str
        'side_by_side', 'both', 'euclidean', or 'dot_product'.
    as_figure : bool
        If True, renders the 3-panel matplotlib figure directly matching Slide 5:
        [Original image | Euclidean distance | Dot product] with red boxes.
        If False, returns raw RGB image/montage for programmatic tests.

    Returns
    -------
    tuple[PIL.Image.Image, dict[str, Any]]
        (result_image, details_dictionary)
    """
    img_rgb = np.asarray(image.convert("RGB"))
    tmpl_rgb = np.asarray(template.convert("RGB"))

    # Height (x) and Width (y) matching Slide 5 nomenclature:
    # size_person_x = img.shape[0] (rows / height), size_person_y = img.shape[1] (cols / width)
    size_person_x, size_person_y = img_rgb.shape[:2]
    size_window_x, size_window_y = tmpl_rgb.shape[:2]

    # Block size validation
    min_h = pixels_per_cell[0] * cells_per_block[0]
    min_w = pixels_per_cell[1] * cells_per_block[1]
    if size_window_x < min_h or size_window_y < min_w:
        raise ValueError(
            f"Template size ({size_window_y}x{size_window_x}) is smaller than block size ({min_w}x{min_h})."
        )
    if size_window_x > size_person_x or size_window_y > size_person_y:
        raise ValueError(
            f"Template size ({size_window_y}x{size_window_x}) exceeds search image ({size_person_y}x{size_person_x})."
        )

    # 1. Compute template HOG descriptor (Slide 5: channel_axis=-1 for color images)
    tmpl_fd = feature.hog(
        tmpl_rgb,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        visualize=False,
        channel_axis=-1,
    )

    # 2. Sliding window search across the search image (Slide 5 loop)
    euclidean_differences: list[float] = []
    dot_prod_differences: list[float] = []
    coord: list[list[int]] = []

    # Slide across rows (i) and columns (j) with step_size stride
    for i in range(0, max(1, size_person_x - size_window_x + 1), step_size):
        for j in range(0, max(1, size_person_y - size_window_y + 1), step_size):
            coord.append([i, j])
            person_slice = img_rgb[i : i + size_window_x, j : j + size_window_y]

            # Compute HOG descriptor for current sliding window
            img_fd = feature.hog(
                person_slice,
                orientations=orientations,
                pixels_per_cell=pixels_per_cell,
                cells_per_block=cells_per_block,
                visualize=False,
                channel_axis=-1,
            )

            # Slide 5 distance and similarity calculations:
            euclidean_differences.append(float(np.linalg.norm(tmpl_fd - img_fd)))
            dot_prod_differences.append(float(np.dot(img_fd, tmpl_fd)))

    if not coord:
        raise ValueError("No search windows could be generated.")

    # 3. Find optimal detection locations (Slide 5)
    min_euclid_location = int(np.argmin(euclidean_differences))
    max_dot_prod_location = int(np.argmax(dot_prod_differences))

    best_euclid_coord = coord[min_euclid_location]      # [row_y, col_x]
    best_dot_prod_coord = coord[max_dot_prod_location]  # [row_y, col_x]

    min_euclid_val = euclidean_differences[min_euclid_location]
    max_dot_prod_val = dot_prod_differences[max_dot_prod_location]

    # 4. Generate the output visualization
    if as_figure:
        # Render the exact 3-panel figure from Slide 5:
        # [Original image | Euclidean distance | Dot product] with ax.set_axis_off()
        fig, ax = plt.subplots(1, 3, figsize=(14, 5), dpi=100)

        # Panel 0: Original image
        ax[0].imshow(img_rgb)
        ax[0].set_title("Original image", fontsize=12)
        ax[0].set_axis_off()

        # Panel 1: Euclidean distance detection with red rectangle
        ax[1].imshow(img_rgb)
        rect_euclid = Rectangle(
            (best_euclid_coord[1], best_euclid_coord[0]),
            size_window_y,
            size_window_x,
            facecolor="none",
            linewidth=2,
            edgecolor="red",
        )
        ax[1].add_patch(rect_euclid)
        ax[1].set_axis_off()
        ax[1].set_title("Euclidean distance", fontsize=12)

        # Panel 2: Dot product detection with red rectangle
        ax[2].imshow(img_rgb)
        rect_dot = Rectangle(
            (best_dot_prod_coord[1], best_dot_prod_coord[0]),
            size_window_y,
            size_window_x,
            facecolor="none",
            linewidth=2,
            edgecolor="red",
        )
        ax[2].add_patch(rect_dot)
        ax[2].set_axis_off()
        ax[2].set_title("Dot product", fontsize=12)

        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
        plt.close(fig)
        buf.seek(0)
        output_image = Image.open(buf).convert("RGB")

    else:
        # Programmatic / unit test output format
        base_rgb = img_rgb.copy()
        if method == "side_by_side":
            euclid_img = draw_bounding_box(
                base_rgb, best_euclid_coord[0], best_euclid_coord[1], size_window_x, size_window_y, color_rgb=(255, 0, 0), thickness=2
            )
            dot_img = draw_bounding_box(
                base_rgb, best_dot_prod_coord[0], best_dot_prod_coord[1], size_window_x, size_window_y, color_rgb=(255, 0, 0), thickness=2
            )
            montage_arr = np.concatenate([base_rgb, euclid_img, dot_img], axis=1)
            output_image = Image.fromarray(montage_arr)

        elif method == "euclidean":
            out_arr = draw_bounding_box(
                base_rgb, best_euclid_coord[0], best_euclid_coord[1], size_window_x, size_window_y, color_rgb=(255, 0, 0), thickness=2
            )
            output_image = Image.fromarray(out_arr)

        elif method == "dot_product":
            out_arr = draw_bounding_box(
                base_rgb, best_dot_prod_coord[0], best_dot_prod_coord[1], size_window_x, size_window_y, color_rgb=(255, 0, 0), thickness=2
            )
            output_image = Image.fromarray(out_arr)

        else:  # 'both'
            out_arr = draw_bounding_box(
                base_rgb, best_euclid_coord[0], best_euclid_coord[1], size_window_x, size_window_y, color_rgb=(255, 0, 0), thickness=2
            )
            out_arr = draw_bounding_box(
                out_arr, best_dot_prod_coord[0], best_dot_prod_coord[1], size_window_x, size_window_y, color_rgb=(0, 255, 0), thickness=2
            )
            output_image = Image.fromarray(out_arr)

    details = {
        "Operation": "HOG Person Detection",
        "Window size (WxH)": f"{size_window_y} × {size_window_x}",
        "Windows evaluated": len(coord),
        "Step size": f"{step_size} px",
        "Best Euclidean location": f"y={best_euclid_coord[0]}, x={best_euclid_coord[1]}",
        "Min Euclidean distance": f"{min_euclid_val:.4f}",
        "Best Dot Product location": f"y={best_dot_prod_coord[0]}, x={best_dot_prod_coord[1]}",
        "Max Dot Product similarity": f"{max_dot_prod_val:.4f}",
        "Visualization": "Slide 5 figure" if as_figure else method,
    }
    return output_image, details


# ── Tool Class ───────────────────────────────────────────────────────────────

class HOGPersonDetectionTool(ForensicsTool):
    """Set 4.4: Sliding-window person detection using HOG Euclidean distance and dot product."""

    tool_id = "hog_person_detection"
    title = "HOG person detection"
    category = "Set 4: Feature & HOG Detection"
    description = "Locate person using HOG sliding-window matching and display the 3-panel figure from Slide 5."

    def run(self, parent: tk.Misc | None, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None

        # 1. Ask for person template (smart default to person_template.bmp from Slide 5)
        doc_path_str = str(document.path) if document.path else None
        template = acquire_person_template(parent, document.current, document_path=doc_path_str)
        if template is None:
            return None

        try:
            output_image, details = detect_person_hog(
                document.current,
                template,
                step_size=16,
                orientations=16,          # Slide 5 standard
                pixels_per_cell=(8, 8),   # Slide 5 standard
                cells_per_block=(2, 2),   # Slide 5 standard
                method="side_by_side",
                as_figure=True,
            )
        except ValueError as err:
            if parent is not None:
                show_error(parent, "Detection Error", str(err))
            return None

        return ToolResult(
            image=output_image,
            message=(
                f"Detection complete ({details['Windows evaluated']} windows). "
                f"Euclid: ({details['Best Euclidean location']}), "
                f"DotProd: ({details['Best Dot Product location']})."
            ),
            details=details,
        )
