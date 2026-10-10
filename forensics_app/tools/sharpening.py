"""Image sharpening via unsharp masking."""

from __future__ import annotations

import tkinter as tk
import numpy as np
import skimage.filters
from PIL import Image

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
from .dialogs import ask_float


def apply_unsharp_mask(
    image: Image.Image,
    radius: float = 1.0,
    amount: float = 1.0,
) -> Image.Image:
    """Sharpen the image using unsharp masking.

    Concept – unsharp masking in three steps:
      1. Create a blurred (unsharp) copy of the image using Gaussian blur.
      2. Compute the detail mask = original − blurred.
         This mask contains only the fine details and high-frequency edges.
      3. Add the detail mask back: sharpened = original + amount × mask.

    - 'radius' (sigma): controls blur spread (spatial scale of details).
    - 'amount': controls sharpening strength (scaling factor for the mask).
    """
    if radius <= 0:
        raise ValueError(f"Radius must be positive, got {radius}.")
    if amount <= 0:
        raise ValueError(f"Amount must be positive, got {amount}.")

    is_rgb = image.mode != "L"
    # Converts the image into a floating-point NumPy array scaled between 0.0 and 1.0
    if is_rgb:
        img_arr = np.asarray(image.convert("RGB")).astype(np.float64) / 255.0
        # channel_axis=-1 is a parameter that tells the filter function which dimension of your image array represents the color channels
        blurred = skimage.filters.gaussian(img_arr, sigma=radius, channel_axis=-1)
    else:
        img_arr = np.asarray(image).astype(np.float64) / 255.0
        blurred = skimage.filters.gaussian(img_arr, sigma=radius)

    mask = img_arr - blurred # Isolates the high-frequency edges.
    sharpened = img_arr + amount * mask # Adds the details back
    clipped = np.clip(sharpened, 0.0, 1.0)
    uint8_arr = (clipped * 255.0).round().astype(np.uint8)
    return Image.fromarray(uint8_arr)


class SharpeningTool(ForensicsTool):
    tool_id = "sharpening"
    title = "Image sharpening"
    category = "Set 3"
    description = "Sharpen image via unsharp masking with configurable radius and amount."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None

        radius = ask_float(
            parent,
            "Image Sharpening",
            "Blur radius / sigma (larger = broader edge sharpening):",
            initial=1.0,
            min_val=0.1,
            max_val=20.0,
        )
        if radius is None:
            return None

        amount = ask_float(
            parent,
            "Image Sharpening",
            "Sharpening amount / strength (e.g. 1.0 = standard, 2.0 = strong):",
            initial=1.0,
            min_val=0.1,
            max_val=10.0,
        )
        if amount is None:
            return None

        try:
            output = apply_unsharp_mask(document.current, radius=radius, amount=amount)
        except ValueError as error:
            from .dialogs import show_error
            show_error(parent, "Sharpening", str(error))
            return None

        return ToolResult(
            image=output,
            message=f"Image sharpened (radius={radius:g}, amount={amount:g}).",
            details={
                "Operation": "Unsharp masking",
                "Radius (sigma)": radius,
                "Amount": amount,
            },
        )
