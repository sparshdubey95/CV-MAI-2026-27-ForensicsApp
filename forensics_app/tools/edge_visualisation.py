"""Edge visualization: superimpose Canny edges onto the original image."""

from __future__ import annotations

import tkinter as tk
import numpy as np
import skimage.feature
from PIL import Image

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
from .dialogs import ask_choice, ask_float
from .canny_edges import apply_canny


# ── Pure image-processing functions ──────────────────────────────────────────

def overlay_colored_edges(
    original: Image.Image,
    edge_mask: np.ndarray,
    edge_color: tuple[int, int, int] = (255, 0, 0),
) -> Image.Image:
    """Paint detected edges in a solid colour on top of the original image.

    Pixels where edge_mask is True are replaced by edge_color.
    All other pixels come from the original.
    """
    rgb = np.asarray(original.convert("RGB")).copy()
    rgb[edge_mask] = edge_color          # replace edge pixels with chosen colour
    return Image.fromarray(rgb)


def overlay_dimmed_background(
    original: Image.Image,
    edge_mask: np.ndarray,
    dim_factor: float = 0.3,
    edge_color: tuple[int, int, int] = (0, 255, 255),
) -> Image.Image:
    """Dim the background and highlight edges in a bright colour.

    This makes edges stand out clearly against a darkened version of the photo.
    """
    rgb = np.asarray(original.convert("RGB")).astype(np.float64)
    # Multiply non-edge pixels by dim_factor to darken the background
    rgb[~edge_mask] *= dim_factor
    # Paint edge pixels with a bright colour
    rgb[edge_mask] = edge_color
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))


def overlay_transparent_edges(
    original: Image.Image,
    edge_mask: np.ndarray,
    alpha: float = 0.7,
    edge_color: tuple[int, int, int] = (255, 0, 0),
) -> Image.Image:
    """Blend edge colour with the original using alpha blending.

    result = (1 - alpha) * original_pixel + alpha * edge_color
    This keeps some of the original image visible underneath the edge colour.
    """
    rgb = np.asarray(original.convert("RGB")).astype(np.float64)
    ec = np.array(edge_color, dtype=np.float64)
    # Only blend pixels where an edge was detected
    rgb[edge_mask] = (1.0 - alpha) * rgb[edge_mask] + alpha * ec
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))


# ── Tool class ────────────────────────────────────────────────────────────────

VISUALISATION_CHOICES = [
    ("colored", "Coloured edge overlay (red edges on original)"),
    ("dimmed", "Dimmed background with bright cyan edges"),
    ("transparent", "Transparent blend (partially visible original)"),
]


class EdgeVisualisationTool(ForensicsTool):
    tool_id = "edge_visualisation"
    title = "Edge visualisation"
    category = "Set 3"
    description = "Detect Canny edges and superimpose them on the original image."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None

        sigma = ask_float(
            parent,
            "Edge Visualisation – Canny",
            "Sigma for Gaussian smoothing:",
            initial=1.5,
            min_val=0.1,
            max_val=10.0,
        )
        if sigma is None:
            return None

        style = ask_choice(
            parent,
            "Edge Visualisation – Style",
            "Choose how to display edges on the image:",
            VISUALISATION_CHOICES,
            initial="colored",
        )
        if style is None:
            return None

        # Detect edges using Canny
        edge_mask = apply_canny(document.current, sigma=sigma)
        edge_count = int(np.sum(edge_mask))

        # Apply chosen visualisation
        if style == "colored":
            output = overlay_colored_edges(document.current, edge_mask, edge_color=(255, 0, 0))
        elif style == "dimmed":
            output = overlay_dimmed_background(document.current, edge_mask, dim_factor=0.3)
        elif style == "transparent":
            output = overlay_transparent_edges(document.current, edge_mask, alpha=0.7)
        else:
            output = overlay_colored_edges(document.current, edge_mask)

        return ToolResult(
            image=output,
            message=f"Edge overlay applied ({style} style, sigma={sigma:g}).",
            details={
                "Operation": "Edge visualisation",
                "Style": style,
                "Sigma": sigma,
                "Edge pixels": edge_count,
            },
        )
