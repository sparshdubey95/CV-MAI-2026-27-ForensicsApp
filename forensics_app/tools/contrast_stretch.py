"""Stretch intensities so a chosen percentile range fills 0–255."""

from __future__ import annotations

import tkinter as tk
from tkinter import simpledialog

from PIL import Image

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult


def intensity_at_percentile(histogram: list[int], percentile: float) -> int:
    """Map a percentile in [0, 100] to a gray value using the 256-bin histogram."""
    total = sum(histogram)
    if total == 0:
        return 0
    if percentile <= 0:
        for index, count in enumerate(histogram):
            if count:
                return index
        return 0
    if percentile >= 100:
        for index in range(len(histogram) - 1, -1, -1):
            if histogram[index]:
                return index
        return 255
    target = total * (percentile / 100.0)
    accumulated = 0
    for index, count in enumerate(histogram):
        accumulated += count
        if accumulated >= target:
            return index
    return 255


def stretch_band(band: Image.Image, low_percentile: float, high_percentile: float) -> tuple[Image.Image, int, int]:
    histogram = band.histogram()[:256]
    low = intensity_at_percentile(histogram, low_percentile)
    high = intensity_at_percentile(histogram, high_percentile)
    if high <= low:
        return band.copy(), low, high
    scale = 255.0 / (high - low)
    lookup = [max(0, min(255, int((value - low) * scale + 0.5))) for value in range(256)]
    return band.point(lookup), low, high


def stretch_contrast(
    image: Image.Image,
    low_percentile: float = 0.0,
    high_percentile: float = 100.0,
) -> tuple[Image.Image, dict[str, str]]:
    """Stretch each RGB band (or a grayscale image) independently."""
    if not 0 <= low_percentile < high_percentile <= 100:
        raise ValueError("Need 0 ≤ low < high ≤ 100")

    notes: dict[str, str] = {}
    if image.mode in {"L", "1"}:
        stretched, low, high = stretch_band(image.convert("L"), low_percentile, high_percentile)
        notes["L in"] = f"{low}–{high}"
        return stretched, notes

    red, green, blue = image.convert("RGB").split()
    parts = []
    for name, band in (("R", red), ("G", green), ("B", blue)):
        stretched, low, high = stretch_band(band, low_percentile, high_percentile)
        parts.append(stretched)
        notes[f"{name} in"] = f"{low}–{high}"
    return Image.merge("RGB", tuple(parts)), notes


class ContrastStretchTool(ForensicsTool):
    tool_id = "contrast_stretch"
    title = "Contrast stretching"
    category = "Set 2"
    description = "Remap a percentile intensity range to the full 0–255 display range."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        low = simpledialog.askfloat(
            "Contrast stretching",
            "Low percentile (0 = actual minimum):",
            parent=parent,
            minvalue=0.0,
            maxvalue=99.9,
            initialvalue=0.0,
        )
        if low is None:
            return None
        high = simpledialog.askfloat(
            "Contrast stretching",
            "High percentile (100 = actual maximum):",
            parent=parent,
            minvalue=low + 0.1,
            maxvalue=100.0,
            initialvalue=100.0,
        )
        if high is None:
            return None

        assert document.current is not None
        output, notes = stretch_contrast(document.current, low, high)
        details: dict[str, object] = {
            "Operation": "Contrast stretching",
            "Low percentile": low,
            "High percentile": high,
        }
        details.update(notes)
        return ToolResult(
            image=output,
            message=f"Stretched contrast using percentiles {low:g}–{high:g}.",
            details=details,
        )
