"""Tools for normalized cross-correlation and template matching (Set 4.1 & 4.2).

This module implements:
  - Set 4.1: Normalized cross-correlation (NCC) using skimage.feature.match_template.
             Displays a 4-panel figure: Original image, Template, NCC map, and Thresholded image.
  - Set 4.2: Template matching with configurable rotation angle and detection threshold,
             displaying a 2-panel figure: Rotated template and Thresholded image (dimmed background
             with red detections).
"""

from __future__ import annotations

import io
import tkinter as tk
from typing import Any

import matplotlib
matplotlib.use("Agg")  # Off-screen backend for rendering plots
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import skimage.color as color
import skimage.feature as feature
import skimage.transform as transform

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
from .dialogs import ask_float, ask_image_file, show_error


# ── Template Selection ──────────────────────────────────────────────────────

def select_template(parent: tk.Misc | None) -> Image.Image | None:
    """Prompt the user to select a template image using a simple file picker.
    Handles cancellation and invalid image files safely.
    """
    file_path = ask_image_file(parent, title="Select Template Image")
    if not file_path:
        return None  # Cancelled

    try:
        with Image.open(file_path) as img:
            return img.copy()
    except Exception as err:
        if parent is not None:
            show_error(parent, "Template Error", f"Could not open template image: {err}")
        return None


# ── Core Image Helpers ──────────────────────────────────────────────────────
def to_gray_float(image: Image.Image | np.ndarray) -> np.ndarray:
    """Convert an image to a 2D float64 grayscale array in [0.0, 1.0]."""
    if isinstance(image, np.ndarray):
        arr = image
    else:
        arr = np.asarray(image.convert("RGB"))

    if arr.ndim == 3:
        return color.rgb2gray(arr)
    if arr.dtype == np.uint8:
        return arr.astype(np.float64) / 255.0
    return arr.astype(np.float64)


def compute_ncc_map(
    image: Image.Image | np.ndarray,
    template: Image.Image | np.ndarray,
    pad_input: bool = True,
) -> np.ndarray:
    """Compute normalized cross-correlation (NCC) map using skimage.feature.match_template.

    Parameters
    ----------
    image : PIL.Image.Image or np.ndarray
        Search image.
    template : PIL.Image.Image or np.ndarray
        Template image to locate.
    pad_input : bool, optional
        If True, pads the image so the output matches image dimensions. Default is True.

    Returns
    -------
    np.ndarray
        2D array of correlation scores in [-1.0, 1.0].
    """
    gray_img = to_gray_float(image)
    gray_tmpl = to_gray_float(template)

    ih, iw = gray_img.shape[:2]
    th, tw = gray_tmpl.shape[:2]

    if th > ih or tw > iw:
        raise ValueError(
            f"Template size ({tw}x{th}) cannot exceed image size ({iw}x{ih})."
        )

    corr_map = feature.match_template(gray_img, gray_tmpl, pad_input=pad_input)
    corr_map = np.nan_to_num(corr_map, nan=0.0)
    return corr_map


def render_ncc_image(corr_map: np.ndarray) -> Image.Image:
    """Render correlation map as a normalized 8-bit grayscale image."""
    clean = np.nan_to_num(corr_map, nan=0.0)
    c_min = float(np.min(clean))
    c_max = float(np.max(clean))
    if c_max > c_min:
        norm = (clean - c_min) / (c_max - c_min)
    else:
        norm = np.zeros_like(clean)
    uint8 = np.clip(norm * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(uint8, mode="L")


def rotate_template(template: Image.Image | np.ndarray, angle: float) -> np.ndarray:
    """Rotate template by angle in degrees using skimage.transform.rotate."""
    gray_tmpl = to_gray_float(template)
    if angle % 360 == 0:
        return gray_tmpl
    return transform.rotate(gray_tmpl, angle, resize=False, mode="edge")


def draw_bounding_box(
    image_rgb: np.ndarray,
    top: int,
    left: int,
    height: int,
    width: int,
    color_rgb: tuple[int, int, int] = (255, 0, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Draw a rectangular bounding box on an RGB numpy array (H, W, 3)."""
    out = image_rgb.copy()
    H, W = out.shape[:2]

    r_min = max(0, top)
    r_max = min(H, top + height)
    c_min = max(0, left)
    c_max = min(W, left + width)

    if r_max > r_min and c_max > c_min:
        out[r_min : min(H, r_min + thickness), c_min:c_max] = color_rgb
        out[max(0, r_max - thickness) : r_max, c_min:c_max] = color_rgb
        out[r_min:r_max, c_min : min(W, c_min + thickness)] = color_rgb
        out[r_min:r_max, max(0, c_max - thickness) : c_max] = color_rgb

    return out


# ── Figure Renderers ────────────────────────────────────────────────────────

def render_ncc_figure(
    image: Image.Image | np.ndarray,
    template: Image.Image | np.ndarray,
    corr_map: np.ndarray,
    threshold: float = 0.8,
    layout: str = "4_panel",
) -> Image.Image:
    """Render 4-panel figure: Original image, Template, NCC map, and Thresholded image.

    Detections are shown in red on a background dimmed to 40%.
    If no positions pass the threshold, the global maximum is highlighted.
    """
    gray_img = to_gray_float(image)
    gray_tmpl = to_gray_float(template)
    clean_corr = np.nan_to_num(corr_map, nan=0.0)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), dpi=100)

    # Panel 1: Original image
    axes[0, 0].imshow(gray_img, cmap="gray")
    axes[0, 0].set_title("Original image")

    # Panel 2: Template
    axes[0, 1].imshow(gray_tmpl, cmap="gray")
    axes[0, 1].set_title("Template")

    # Panel 3: Normalized cross-correlation map
    axes[1, 0].imshow(clean_corr, cmap="gray")
    axes[1, 0].set_title("Normalized cross-corr image")

    # Panel 4: Thresholded image (background dimmed to 40% with red detections)
    rgb_base = color.gray2rgb(gray_img) * 0.4
    matches_mask = clean_corr >= threshold

    if np.any(matches_mask):
        rgb_base[matches_mask] = [1.0, 0.0, 0.0]
    else:
        # If no position passes the threshold, highlight the global maximum
        peak_y, peak_x = np.unravel_index(np.argmax(clean_corr), clean_corr.shape)
        rgb_base[peak_y, peak_x] = [1.0, 0.0, 0.0]

    axes[1, 1].imshow(rgb_base)
    axes[1, 1].set_title("Thresholded image")

    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def render_template_matching_figure(
    template_r: np.ndarray,
    thresholded_rgb: np.ndarray,
    angle: float = 0.0,
    threshold: float = 0.8,
) -> Image.Image:
    """Render 2-panel figure: Rotated template and Thresholded image."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=100)

    # Left: Rotated template with angle label
    axes[0].imshow(template_r, cmap="gray")
    axes[0].set_title("Template", fontsize=12)
    axes[0].set_xlabel(f"Configurable angle: {angle:.1f}°", fontsize=10)

    # Right: Thresholded image with threshold label
    if thresholded_rgb.dtype == np.uint8:
        display_img = thresholded_rgb.astype(np.float64) / 255.0
    else:
        display_img = thresholded_rgb

    axes[1].imshow(display_img)
    axes[1].set_title("Thresholded image", fontsize=12)
    axes[1].set_xlabel(f"Configurable threshold: {threshold:.2f}", fontsize=10)

    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


# ── Template Matching Computation ───────────────────────────────────────────

def match_template_configured(
    image: Image.Image,
    template: Image.Image,
    angle: float = 0.0,
    threshold: float = 0.8,
    style: str = "dimmed",
    as_figure: bool = False,
) -> tuple[Image.Image, dict[str, Any]]:
    """Perform template matching with rotation, thresholding, and dimmed background display."""
    gray_img = to_gray_float(image)
    gray_tmpl = to_gray_float(template)

    if angle % 360 != 0:
        gray_tmpl = rotate_template(gray_tmpl, angle)

    ih, iw = gray_img.shape[:2]
    th, tw = gray_tmpl.shape[:2]
    if th > ih or tw > iw:
        raise ValueError(
            f"Rotated template size ({tw}x{th}) exceeds search image ({iw}x{ih})."
        )

    corr_map = feature.match_template(gray_img, gray_tmpl, pad_input=True)
    corr_map = np.nan_to_num(corr_map, nan=0.0)

    peak_y, peak_x = np.unravel_index(np.argmax(corr_map), corr_map.shape)
    peak_val = float(corr_map[peak_y, peak_x])

    top = int(round(peak_y - th / 2.0))
    left = int(round(peak_x - tw / 2.0))

    matches_mask = corr_map >= threshold
    matches_count = int(np.sum(matches_mask))

    # Dimmed background (40%) with red detections
    out_float = color.gray2rgb(gray_img) * 0.4
    if matches_count > 0:
        out_float[matches_mask] = [1.0, 0.0, 0.0]
    else:
        out_float[peak_y, peak_x] = [1.0, 0.0, 0.0]

    result_uint8 = np.clip(out_float * 255.0, 0, 255).astype(np.uint8)

    details = {
        "Operation": "Template Matching",
        "Peak correlation": f"{peak_val:.4f}",
        "Peak location (y, x)": f"({peak_y}, {peak_x})",
        "Bounding box top-left": f"({top}, {left})",
        "Template size": f"{tw} × {th}",
        "Rotation angle": f"{angle:.1f}°",
        "Threshold": f"{threshold:.2f}",
        "Matches above threshold": matches_count,
        "Display style": "dimmed",
    }

    if as_figure:
        fig_img = render_template_matching_figure(
            template_r=gray_tmpl,
            thresholded_rgb=result_uint8,
            angle=angle,
            threshold=threshold,
        )
        return fig_img, details

    return Image.fromarray(result_uint8, mode="RGB"), details


# ── ForensicsTool Classes ───────────────────────────────────────────────────

class NormalizedCrossCorrTool(ForensicsTool):
    """Set 4.1: Compute normalized cross-correlation and render 4-panel figure."""

    tool_id = "normalized_cross_corr"
    title = "Normalized cross-correlation"
    category = "Set 4: Feature & HOG Detection"
    description = "Compute normalized cross-correlation map and display the 4-panel figure."

    def run(self, parent: tk.Misc | None, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None

        template = select_template(parent)
        if template is None:
            return None

        try:
            corr_map = compute_ncc_map(document.current, template, pad_input=True)
        except ValueError as err:
            if parent is not None:
                show_error(parent, "NCC Error", str(err))
            return None

        peak_y, peak_x = np.unravel_index(np.argmax(corr_map), corr_map.shape)
        peak_val = float(corr_map[peak_y, peak_x])

        output_image = render_ncc_figure(
            image=document.current,
            template=template,
            corr_map=corr_map,
            threshold=0.8,
        )

        return ToolResult(
            image=output_image,
            message=f"Normalized cross-correlation computed (peak: {peak_val:.4f} at y={peak_y}, x={peak_x}).",
            details={
                "Operation": "Normalized Cross-Correlation",
                "Peak correlation": f"{peak_val:.4f}",
                "Peak coordinate (y, x)": f"({peak_y}, {peak_x})",
                "Template dimensions": f"{template.width} × {template.height}",
                "Search image dimensions": f"{document.current.width} × {document.current.height}",
                "Threshold": "0.80",
            },
        )


class TemplateMatchingTool(ForensicsTool):
    """Set 4.2: Template matching with configurable angle and threshold."""

    tool_id = "template_matching_configs"
    title = "Template matching configs"
    category = "Set 4: Feature & HOG Detection"
    description = "Template matching with configurable rotation and thresholding."

    def run(self, parent: tk.Misc | None, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None

        template = select_template(parent)
        if template is None:
            return None

        angle = ask_float(
            parent,
            "Template Rotation",
            "Template rotation angle in degrees (-360 to 360):",
            initial=0.0,
            min_val=-360.0,
            max_val=360.0,
        )
        if angle is None:
            return None

        threshold = ask_float(
            parent,
            "Match Threshold",
            "Detection threshold (0.0 to 1.0):",
            initial=0.8,
            min_val=0.0,
            max_val=1.0,
        )
        if threshold is None:
            return None

        try:
            output_image, details = match_template_configured(
                document.current,
                template,
                angle=angle,
                threshold=threshold,
                as_figure=True,
            )
        except ValueError as err:
            if parent is not None:
                show_error(parent, "Template Matching Error", str(err))
            return None

        return ToolResult(
            image=output_image,
            message=f"Template matched ({details['Matches above threshold']} matches >= {threshold:.2f}, peak: {details['Peak correlation']}).",
            details=details,
        )
