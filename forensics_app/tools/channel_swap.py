"""Reorder the RGB channels (for example RGB → BGR)."""
# a pixel might be: (R, G, B) = (200, 100, 50)
# If we choose BGR, the new pixel becomes: (B, G, R) = (50, 100, 200)
# So the red and blue information changes position.
# User clicks "Channel swap"
#           ↓
# ChannelSwapTool.run()
#           ↓
# Show channel-order dialog
#           ↓
# User chooses BGR / GRB / etc.
#           ↓
# Get document.current
#           ↓
# Convert to RGB if needed
#           ↓
# Split image into R, G, B
#           ↓
# Store channels in a dictionary
#           ↓
# Read the selected order
#           ↓
# Rearrange the channels
#           ↓
# Merge them into a new RGB image
#           ↓
# Return ToolResult
#           ↓
# Main application applies the result


from __future__ import annotations

import tkinter as tk

from PIL import Image

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
from .dialogs import ask_choice

SWAP_CHOICES = [
    ("BGR", "BGR (swap red and blue)"),
    ("GRB", "GRB (swap red and green)"),
    ("RBG", "RBG (swap green and blue)"),
    ("BRG", "BRG (rotate channels: (blue moves to red, red moves to green, green moves to blue))"),
    ("GBR", "GBR (rotate channels: (green moves to red, blue moves to green, red moves to blue))"),
]


def to_rgb(image: Image.Image) -> Image.Image:
    if image.mode == "RGB":
        return image
    else:
        return image.convert("RGB")


def swap_channels(image: Image.Image, order: str) -> Image.Image:
    """Rebuild the image using the letters in ``order`` (a permutation of RGB)."""
    order = order.upper()
    if sorted(order) != ["B", "G", "R"]:
        raise ValueError(f"Order must be a permutation of RGB, got {order!r}") # !r shows '' around the order

    # Split the image into R, G and B channels.
    # Store them in a dictionary so each channel can be accessed by its letter.
    rgb_image = to_rgb(image)
    red, green, blue = rgb_image.split()

    bands = {
        "R": red,
        "G": green,
        "B": blue,
    }
    return Image.merge("RGB", (bands[order[0]], bands[order[1]], bands[order[2]]))# Create a new RGB image with the new order


class ChannelSwapTool(ForensicsTool):
    tool_id = "channel_swap"
    title = "Channel swap"
    category = "Set 2"
    description = "Permute RGB channels, e.g. swap red and blue to get BGR."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        order = ask_choice(
            parent,
            "Channel swap",
            "New channel order (letters are the source channels in R, G, B slots):",
            SWAP_CHOICES,
            initial="BGR",
        )
        if order is None:
            return None

        assert document.current is not None
        output = swap_channels(document.current, order)
        return ToolResult(
            image=output,
            message=f"Reordered channels to {order}.",
            details={
                "Operation": "Channel swap",
                "From": "RGB",
                "To": order,
            },
        )
