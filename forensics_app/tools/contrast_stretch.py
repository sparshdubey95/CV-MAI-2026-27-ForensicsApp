"""Stretch the minimum and maximum intensities to 0 and 255."""

from __future__ import annotations

import tkinter as tk

from PIL import Image

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult


def stretch_band(band: Image.Image) -> tuple[Image.Image, int, int]:
    """Stretch one image band using its minimum and maximum values."""

    # Find the darkest and brightest pixel in this band.
    minimum, maximum = band.getextrema()

    # If all pixels have the same value, there is nothing to stretch.
    if minimum == maximum:
        return band.copy(), minimum, maximum

    # Create a lookup table for all possible pixel values.
    lookup = []
    # We create a conversion table that tells the computer what every old pixel value should become.
    # Then we apply that table to the image.
    for value in range(256):
        # Move the minimum value to 0.
        new_value = value - minimum

        # Stretch the value to the range 0–255.This is the actual stretching.
        new_value = new_value * 255 / (maximum - minimum)

        # Convert the result to an integer.
        new_value = int(new_value)

        # If the answer goes outside the valid pixel range, bring it back.
        new_value = max(0, min(255, new_value))

        lookup.append(new_value)

    # Apply the new values to the image.
    stretched = band.point(lookup)

    return stretched, minimum, maximum


# stretch_band() works on one channel.
# stretch_contrast() works on the whole image.
def stretch_contrast(image: Image.Image,) -> tuple[Image.Image, dict[str, str]]:
    """Apply min–max contrast stretching."""

    notes: dict[str, str] = {}

    # If the image is grayscale, stretch it directly.
    if image.mode in {"L", "1"}:
        stretched, minimum, maximum = stretch_band(image)

        #the original minimum and maximum
        notes["L in"] = f"{minimum}–{maximum}"
        return stretched, notes

    # If the image is RGB, separate it into red, green and blue.
    red, green, blue = image.convert("RGB").split()

    stretched_bands = []

    for name, band in (
        # First iteration-- name = "R" , band = red
        ("R", red),
        # Second iteration -- name = "G", band = green
        ("G", green),
        # and so on
        ("B", blue),
    ):
        stretched, minimum, maximum = stretch_band(band)

        stretched_bands.append(stretched)
        notes[f"{name} in"] = f"{minimum}–{maximum}"

    # Put the three stretched channels back together.
    output = Image.merge("RGB", tuple(stretched_bands)) # Pillow's Image.merge() expects the channels as a tuple.

    return output, notes


class ContrastStretchTool(ForensicsTool):
    tool_id = "contrast_stretch"
    title = "Contrast stretching"
    category = "Set 2"
    description = "Stretch the minimum and maximum intensity to 0–255."

    def run(
        self,
        parent: tk.Misc,
        document: ImageDocument,
    ) -> ToolResult | None:

        assert document.current is not None

        # Apply min–max contrast stretching.
        output, notes = stretch_contrast(document.current)

        details: dict[str, object] = {
            "Operation": "Contrast stretching",
        }

        details.update(notes)

        return ToolResult(
            image=output,
            message="Stretched contrast using the minimum and maximum intensities.",
            details=details,
        )