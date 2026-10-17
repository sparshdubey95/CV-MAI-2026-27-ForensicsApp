"""Tools for normalized cross-correlation and template matching (Set 4.1 & 4.2).

This module implements:
  - Set 4.1: Normalized cross-correlation (NCC) matching using skimage.feature.match_template.
             Renders the multi-panel figure (Original image, Template, NCC map, Thresholded image)
             matching the lecture slides.
  - Set 4.2: Template matching with configurable rotation angle, detection threshold,
             and superimposition styles (dimmed background with red detection markers,
             red overlays, transparent overlays, and bounding boxes).
"""

from __future__ import annotations

import io
import os
import tkinter as tk
from typing import Any

import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend so plots render off-screen without popup windows
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import skimage.color as color
import skimage.feature as feature
import skimage.transform as transform

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
from .dialogs import ask_choice, ask_float, ask_image_file, show_error


# ── Helper: Locate course assignment templates ──────────────────────────────

def find_default_template(
    image: Image.Image,
    document_path: str | None = None,
) -> Image.Image | None:
    """Try to find the official assignment template for the current image.

    Why this helper exists:
    In the course assignment (Set 4.1 & 4.2), the target image is Albert Einstein
    and the template is his eye (15 rows by 14 columns). If the user opens
    'einstein.png', we want to automatically offer 'template.png' so the student
    does not have to manually guess or crop the coordinates.
    """
    candidate_paths: list[str] = []

    # 1. Look in the same directory as the currently opened image
    if document_path:
        folder = os.path.dirname(document_path)
        candidate_paths.append(os.path.join(folder, "template.png"))
        candidate_paths.append(os.path.join(folder, "template_br.png"))

    # 2. Look in common download folders for Set 4 materials
    downloads_dirs = [
        r"D:\Downloads\Set4_images\Set4_images",
        r"D:\Downloads\Set4_images",
        r"C:\Users\Sparsh\Downloads\Set4_images\Set4_images",
        r"C:\Users\Sparsh\Downloads\Set4_images",
        os.path.join(os.path.expanduser("~"), "Downloads", "Set4_images", "Set4_images"),
    ]
    for d in downloads_dirs:
        candidate_paths.append(os.path.join(d, "template.png"))
        candidate_paths.append(os.path.join(d, "template_br.png"))

    # Check if any candidate file exists on disk and can be opened
    for path in candidate_paths:
        if os.path.isfile(path):
            try:
                with Image.open(path) as img:
                    return img.copy()
            except Exception:
                pass

    # 3. If image size is (180, 232), this is the standard Einstein image!
    # The official template is located at row 94..109, col 51..65 (Einstein's right eye).
    w, h = image.size
    if w == 180 and h == 232:
        # Crop the exact 14x15 eye patch matching template.png
        return image.crop((51, 94, 65, 109))

    return None


def acquire_template(
    parent: tk.Misc | None,
    image: Image.Image,
    document_path: str | None = None,
    title: str = "Select Template",
) -> Image.Image | None:
    """Prompt the user to choose a template, with smart defaults for course images.

    Options provided:
      1. Default template (Einstein's eye if detected)
      2. Choose any template image file from disk
      3. Crop center region from current image
    """
    # Look for the default assignment template first
    default_tmpl = find_default_template(image, document_path)

    # In headless / unit testing mode, return immediately
    if parent is None:
        if default_tmpl is not None:
            tw, th = default_tmpl.size
            iw, ih = image.size
            if tw <= iw and th <= ih:
                return default_tmpl
        # Fallback to center crop
        w, h = image.size
        crop_w = max(8, min(w, w // 3))
        crop_h = max(8, min(h, h // 3))
        left = (w - crop_w) // 2
        top = (h - crop_h) // 2
        return image.crop((left, top, left + crop_w, top + crop_h))

    # Build the options list for the user dialog
    options: list[tuple[str, str]] = []
    if default_tmpl is not None:
        options.append(
            ("default", "Standard assignment template (template.png – Einstein's eye, as in PDF)")
        )
    options.append(("file", "Choose template image file from disk..."))
    options.append(("crop", "Crop center region from current image"))

    choice = ask_choice(
        parent,
        title,
        "Select how to provide the template for matching:",
        options,
        initial="default" if default_tmpl is not None else "crop",
    )
    if choice is None:
        return None  # User clicked Cancel

    if choice == "default" and default_tmpl is not None:
        return default_tmpl

    if choice == "file":
        initial_dir = os.path.dirname(document_path) if document_path else None
        path = ask_image_file(parent, title="Select template image file")
        if not path:
            return None
        try:
            return Image.open(path)
        except Exception as error:
            show_error(parent, "Template Error", f"Could not open template image: {error}")
            return None

    # 'crop': fallback to a central crop of the image
    w, h = image.size
    crop_w = max(8, min(w, w // 3))
    crop_h = max(8, min(h, h // 3))
    left = (w - crop_w) // 2
    top = (h - crop_h) // 2
    return image.crop((left, top, left + crop_w, top + crop_h))


# ── Core Algorithmic Functions ──────────────────────────────────────────────

def to_gray_float(image: Image.Image | np.ndarray) -> np.ndarray:
    """Convert an image (PIL Image or NumPy array) to a 2D float64 grayscale array in [0.0, 1.0].

    Easy English Explanation:
      Normalized cross-correlation needs pixel values as floating-point numbers
      between 0.0 (pure black) and 1.0 (pure white) in a 2D array (height x width).
    """
    if isinstance(image, np.ndarray):
        arr = image
    else:
        arr = np.asarray(image.convert("RGB"))

    if arr.ndim == 3:
        # RGB to grayscale conversion using standard luminance weights
        return color.rgb2gray(arr)
    if arr.dtype == np.uint8:
        # Scale 0..255 uint8 to 0.0..1.0 float
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
        The larger search image (e.g., Einstein portrait).
    template : PIL.Image.Image or np.ndarray
        The small template image to look for (e.g., Einstein's eye).
    pad_input : bool, optional
        If True, pads the image so the output has the EXACT same dimensions as `image`,
        and the maximum peak score aligns with the center of the template. Default is True.

    Returns
    -------
    np.ndarray
        A 2D array of correlation scores in the range [-1.0, 1.0].
        Values close to 1.0 indicate an almost perfect match.
    """
    gray_img = to_gray_float(image)
    gray_tmpl = to_gray_float(template)

    ih, iw = gray_img.shape[:2]
    th, tw = gray_tmpl.shape[:2]

    # The template must fit inside the search image
    if th > ih or tw > iw:
        raise ValueError(
            f"Template size ({tw}x{th}) cannot exceed image size ({iw}x{ih})."
        )

    # Perform normalized cross correlation via scikit-image
    corr_map = feature.match_template(gray_img, gray_tmpl, pad_input=pad_input)

    # In case of constant/flat patches, division by zero may produce NaNs; replace them with 0.0
    corr_map = np.nan_to_num(corr_map, nan=0.0)
    return corr_map


def render_ncc_image(corr_map: np.ndarray) -> Image.Image:
    """Render correlation map as a normalized 8-bit grayscale display image.

    Rescales values from [min, max] to [0, 255] for visual display.
    """
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
    """Rotate template by angle in degrees using skimage.transform.rotate.

    Parameters
    ----------
    template : PIL.Image.Image or np.ndarray
        The template patch to rotate.
    angle : float
        Rotation angle in degrees (e.g., 0, 15, 45, 90).

    Returns
    -------
    np.ndarray
        The rotated template array as float64 in [0.0, 1.0].
    """
    gray_tmpl = to_gray_float(template)
    if angle % 360 == 0:
        return gray_tmpl
    # Rotate with smooth bicubic interpolation and edge padding
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
    """Draw a rectangular bounding box directly on an RGB numpy array (H, W, 3)."""
    out = image_rgb.copy()
    H, W = out.shape[:2]

    r_min = max(0, top)
    r_max = min(H, top + height)
    c_min = max(0, left)
    c_max = min(W, left + width)

    if r_max > r_min and c_max > c_min:
        # Top and bottom horizontal borders
        out[r_min : min(H, r_min + thickness), c_min:c_max] = color_rgb
        out[max(0, r_max - thickness) : r_max, c_min:c_max] = color_rgb
        # Left and right vertical borders
        out[r_min:r_max, c_min : min(W, c_min + thickness)] = color_rgb
        out[r_min:r_max, max(0, c_max - thickness) : c_max] = color_rgb

    return out


# ── PDF Matplotlib Figure Renderers ─────────────────────────────────────────

def render_ncc_figure(
    image: Image.Image | np.ndarray,
    template: Image.Image | np.ndarray,
    corr_map: np.ndarray,
    threshold: float = 0.8,
    layout: str = "4_panel",
) -> Image.Image:
    """Render the multi-panel figure directly matching Slide 1 & Slide 2 of the PDF.

    Slide 2 shows:
      1. 'Original image' (Grayscale with axes ticks)
      2. 'Template' (Grayscale with axes ticks)
      3. 'Normalized cross-corr image' (Grayscale correlation surface with axes ticks)
      4. 'Thresholded image' (Dimmed image with bright red detection marker on the eye)

    Slide 2 code on the left shows:
      plt.subplots(1, 3, figsize=(15, 7)) -> 3-panel layout.
    Both layouts are fully supported.
    """
    gray_img = to_gray_float(image)
    gray_tmpl = to_gray_float(template)

    # Clean correlation map
    clean_corr = np.nan_to_num(corr_map, nan=0.0)

    if layout == "3_panel":
        # 1x3 subplots as written in the slide code
        fig, axes = plt.subplots(1, 3, figsize=(15, 6), dpi=100)

        axes[0].imshow(gray_img, cmap="gray")
        axes[0].set_title("Original image", fontsize=12)

        axes[1].imshow(gray_tmpl, cmap="gray")
        axes[1].set_title("Template", fontsize=12)

        axes[2].imshow(clean_corr, cmap="gray")
        axes[2].set_title("Normalized cross-correlation image", fontsize=12)

    else:
        # 2x2 grid (4 panels) as shown on the right side of Slide 2 and top row of Slide 1
        fig, axes = plt.subplots(2, 2, figsize=(10, 8), dpi=100)

        # Panel 1: Original image
        axes[0, 0].imshow(gray_img, cmap="gray")
        axes[0, 0].set_title("Original image", fontsize=11)

        # Panel 2: Template
        axes[0, 1].imshow(gray_tmpl, cmap="gray")
        axes[0, 1].set_title("Template", fontsize=11)

        # Panel 3: Normalized cross-correlation response surface
        axes[1, 0].imshow(clean_corr, cmap="gray")
        axes[1, 0].set_title("Normalized cross-corr image", fontsize=11)

        # Panel 4: Thresholded image (Slide 2 & 3: dimmed background + red detection marker)
        rgb_base = color.gray2rgb(gray_img) * 0.4  # Dim factor 0.4 from Slide 3
        matches_mask = clean_corr >= threshold

        if np.any(matches_mask):
            rgb_base[matches_mask] = [1.0, 0.0, 0.0]
        else:
            # If no pixel exceeded the threshold, highlight the global maximum peak
            py, px = np.unravel_index(np.argmax(clean_corr), clean_corr.shape)
            rgb_base[py, px] = [1.0, 0.0, 0.0]

        axes[1, 1].imshow(rgb_base)
        axes[1, 1].set_title("Thresholded image", fontsize=11)

    fig.tight_layout()

    # Save to memory buffer and convert to PIL Image
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
    """Render the 2-panel figure directly matching Slide 3 of the PDF.

    Slide 3 shows:
      - Left panel: 'Template' with subtitle 'Configurable angle'
      - Right panel: 'Thresholded image' with subtitle 'Configurable threshold'
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=100)

    # Left: Rotated template
    axes[0].imshow(template_r, cmap="gray")
    axes[0].set_title("Template", fontsize=12)
    axes[0].set_xlabel(f"Configurable angle: {angle:.1f}°", fontsize=10)

    # Right: Thresholded image with superimposition
    # If values are float in [0.0, 1.0], imshow handles them directly
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


# ── Template Matching with Configurations (Set 4.2) ─────────────────────────

def match_template_configured(
    image: Image.Image,
    template: Image.Image,
    angle: float = 0.0,
    threshold: float = 0.8,
    style: str = "dimmed",
    as_figure: bool = False,
) -> tuple[Image.Image, dict[str, Any]]:
    """Perform template matching with configurable rotation, thresholding, and superimposition.

    Parameters
    ----------
    image : PIL.Image.Image
        The search image.
    template : PIL.Image.Image
        The template to match.
    angle : float
        Rotation angle for template in degrees.
    threshold : float
        Correlation threshold in [0.0, 1.0] to flag detections.
    style : str
        Superimposition style from Slide 3:
          - 'dimmed': Technique #3 (dimmed background factor 0.4 + bright red detections [1, 0, 0]).
                      This is the exact visualization displayed in Slide 3!
          - 'red_on_gray': Technique #1 (red detections on grayscale image).
          - 'transparent': Technique #2 (transparent red overlay).
          - 'overlay' / 'bounding_box': Red detections plus bounding box around the best match.
          - 'heatmap': Correlation response map.
    as_figure : bool
        If True, returns the 2-panel figure ('Template' | 'Thresholded image') matching Slide 3.
        If False, returns only the processed image array (matches unit test expectations).

    Returns
    -------
    tuple[PIL.Image.Image, dict[str, Any]]
        (output_image, matching_details_dictionary)
    """
    gray_img = to_gray_float(image)
    gray_tmpl = to_gray_float(template)

    # 1. Rotate template by angle 'a' (Slide 3: template_r = transform.rotate(template, a))
    if angle % 360 != 0:
        gray_tmpl = rotate_template(gray_tmpl, angle)

    ih, iw = gray_img.shape[:2]
    th, tw = gray_tmpl.shape[:2]
    if th > ih or tw > iw:
        raise ValueError(
            f"Rotated template size ({tw}x{th}) exceeds search image ({iw}x{ih})."
        )

    # 2. Compute normalized cross-correlation
    corr_map = feature.match_template(gray_img, gray_tmpl, pad_input=True)
    corr_map = np.nan_to_num(corr_map, nan=0.0)

    # Find peak match location
    peak_y, peak_x = np.unravel_index(np.argmax(corr_map), corr_map.shape)
    peak_val = float(corr_map[peak_y, peak_x])

    # With pad_input=True, peak is at template center. Compute bounding box coordinates:
    top = int(round(peak_y - th / 2.0))
    left = int(round(peak_x - tw / 2.0))

    # Identify all locations meeting or exceeding the threshold
    matches_mask = corr_map >= threshold
    matches_count = int(np.sum(matches_mask))

    # Base grayscale image converted to RGB float in [0.0, 1.0]
    base_rgb_float = color.gray2rgb(gray_img)

    # 3. Apply Superimposition from Set 3 (as specified in Slide 3):
    if style == "dimmed":
        # Slide 3 Technique #3: Darken everything except matches
        # dimmed = rgb * 0.4
        # dimmed[matches] = [1, 0, 0]  (bright red)
        out_float = base_rgb_float * 0.4
        if matches_count > 0:
            out_float[matches_mask] = [1.0, 0.0, 0.0]
        else:
            # If no points pass threshold, highlight best peak
            out_float[peak_y, peak_x] = [1.0, 0.0, 0.0]
        result_uint8 = np.clip(out_float * 255.0, 0, 255).astype(np.uint8)

    elif style == "red_on_gray":
        # Slide 3 Technique #1: Red edges/matches on grayscale image
        out_float = base_rgb_float.copy()
        if matches_count > 0:
            out_float[matches_mask] = [1.0, 0.0, 0.0]
        else:
            out_float[peak_y, peak_x] = [1.0, 0.0, 0.0]
        result_uint8 = np.clip(out_float * 255.0, 0, 255).astype(np.uint8)

    elif style == "transparent":
        # Slide 3 Technique #2: Transparent red overlay
        # overlay = np.zeros_like(rgb); overlay[..., 0] = matches.astype(float)
        out_float = base_rgb_float.copy()
        alpha = 0.6
        if matches_count > 0:
            out_float[matches_mask] = (1.0 - alpha) * out_float[matches_mask] + alpha * np.array([1.0, 0.0, 0.0])
        else:
            out_float[peak_y, peak_x] = [1.0, 0.0, 0.0]
        result_uint8 = np.clip(out_float * 255.0, 0, 255).astype(np.uint8)

    elif style == "heatmap":
        heatmap = render_ncc_image(corr_map).convert("RGB")
        heatmap_arr = np.asarray(heatmap).copy()
        result_uint8 = draw_bounding_box(heatmap_arr, top, left, th, tw, color_rgb=(255, 0, 0), thickness=2)

    else:  # 'overlay' / 'bounding_box'
        out_float = base_rgb_float.copy()
        if matches_count > 0:
            out_float[matches_mask] = [1.0, 0.0, 0.0]
        result_uint8 = np.clip(out_float * 255.0, 0, 255).astype(np.uint8)
        result_uint8 = draw_bounding_box(result_uint8, top, left, th, tw, color_rgb=(255, 0, 0), thickness=2)

    details = {
        "Operation": "Template Matching",
        "Peak correlation": f"{peak_val:.4f}",
        "Peak location (y, x)": f"({peak_y}, {peak_x})",
        "Bounding box top-left": f"({top}, {left})",
        "Template size": f"{tw} × {th}",
        "Rotation angle": f"{angle:.1f}°",
        "Threshold": f"{threshold:.2f}",
        "Matches above threshold": matches_count,
        "Display style": style,
    }

    if as_figure:
        # Generate the 2-panel figure matching Slide 3
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
    """Set 4.1: Compute normalized cross-correlation and render comparison figure."""

    tool_id = "normalized_cross_corr"
    title = "Normalized cross-correlation"
    category = "Set 4: Feature & HOG Detection"
    description = "Compute normalized cross-correlation map and display the 4-panel figure from Slide 2."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None

        # 1. Ask for template (smart default to Einstein's eye from Slide 2)
        doc_path_str = str(document.path) if document.path else None
        template = acquire_template(
            parent,
            document.current,
            document_path=doc_path_str,
            title="Select Template for NCC",
        )
        if template is None:
            return None

        # 2. Offer visual presentation format
        layout = "4_panel"
        if parent is not None:
            layout_choice = ask_choice(
                parent,
                "NCC Visualization Layout",
                "Choose how to display the normalized cross-correlation result:",
                [
                    ("4_panel", "4-panel figure (Original | Template | NCC | Thresholded – Slide 1 & 2)"),
                    ("3_panel", "3-panel figure (Original | Template | NCC – Slide 2 code)"),
                    ("map_only", "Raw correlation map only"),
                ],
                initial="4_panel",
            )
            if layout_choice is None:
                return None
            layout = layout_choice

        # 3. Compute NCC
        try:
            corr_map = compute_ncc_map(document.current, template, pad_input=True)
        except ValueError as err:
            if parent is not None:
                show_error(parent, "NCC Error", str(err))
            return None

        peak_y, peak_x = np.unravel_index(np.argmax(corr_map), corr_map.shape)
        peak_val = float(corr_map[peak_y, peak_x])

        # 4. Generate final output image
        if layout == "map_only":
            output_image = render_ncc_image(corr_map)
        else:
            output_image = render_ncc_figure(
                image=document.current,
                template=template,
                corr_map=corr_map,
                threshold=0.8,
                layout=layout,
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
                "Visualization layout": layout,
            },
        )


class TemplateMatchingTool(ForensicsTool):
    """Set 4.2: Template matching with configurable angle, threshold, and superimposition."""

    tool_id = "template_matching_configs"
    title = "Template matching configs"
    category = "Set 4: Feature & HOG Detection"
    description = "Template matching with configurable rotation, thresholding, and superimposition (Slide 3)."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None

        # 1. Ask for template (smart default to Einstein's eye from Slide 3)
        doc_path_str = str(document.path) if document.path else None
        template = acquire_template(
            parent,
            document.current,
            document_path=doc_path_str,
            title="Select Template for Matching",
        )
        if template is None:
            return None

        # 2. Configurable angle (Slide 3: template_r = transform.rotate(template, a))
        angle = ask_float(
            parent,
            "Template Rotation (Configurable Angle)",
            "Template rotation angle in degrees (0–360):",
            initial=0.0,
            min_val=-360.0,
            max_val=360.0,
        )
        if angle is None:
            return None

        # 3. Configurable threshold (Slide 3: Configurable threshold)
        threshold = ask_float(
            parent,
            "Match Threshold (Configurable Threshold)",
            "Detection threshold (0.1–1.0):",
            initial=0.8,
            min_val=0.0,
            max_val=1.0,
        )
        if threshold is None:
            return None

        # 4. Superimposition style from Slide 3
        style = ask_choice(
            parent,
            "Superimposition from Set 3",
            "Choose superimposition style (Slide 3):",
            [
                ("dimmed", "3. Darken everything except matches (Slide 3: dimmed = rgb * 0.4)"),
                ("red_on_gray", "1. Red matches on grayscale image"),
                ("transparent", "2. Transparent red overlay"),
                ("overlay", "4. Bounding box overlay on detections"),
            ],
            initial="dimmed",
        )
        if style is None:
            return None

        # 5. Display presentation layout
        display_mode = "side_by_side"
        if parent is not None:
            disp_choice = ask_choice(
                parent,
                "Presentation Layout",
                "Choose presentation layout:",
                [
                    ("side_by_side", "2-panel comparison (Template | Thresholded image – as in Slide 3)"),
                    ("thresholded_only", "Thresholded image only"),
                ],
                initial="side_by_side",
            )
            if disp_choice is None:
                return None
            display_mode = disp_choice

        try:
            output_image, details = match_template_configured(
                document.current,
                template,
                angle=angle,
                threshold=threshold,
                style=style,
                as_figure=(display_mode == "side_by_side"),
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
