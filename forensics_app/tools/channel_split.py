"""Show one RGB channel, or all three side by side."""
#channel_split.py
#│
#├── CHANNEL_CHOICES       ← choices/data
#│
#├── to_rgb()              ← image-processing function
#│
#├── channel_plane()       ← image-processing function
#│
#├── split_all_planes()    ← image-processing function
#│
#└── ChannelSplitTool      ← connects everything to ForensicsApp
from __future__ import annotations

import tkinter as tk

from PIL import Image

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
from .dialogs import ask_choice
# Each tuple = (internal value, user-facing label)
CHANNEL_CHOICES = [
    ("R", "Red plane"),
    ("G", "Green plane"),
    ("B", "Blue plane"),
    ("all", "All three, side by side"),
]


def to_rgb(image: Image.Image) -> Image.Image: # Make sure the image is in RGB format.
    if image.mode == "RGB":
        return image
    else:
        return image.convert("RGB")


def channel_plane(image: Image.Image, channel: str) -> Image.Image:
    """Return one colour channel as a grayscale image."""
    red, green, blue = to_rgb(image).split()

    if channel == "R":
        return red
    if channel == "G":
        return green
    if channel == "B":
        return blue

    raise ValueError(f"Unknown channel: {channel}")


def split_all_planes(image: Image.Image, gap: int = 8) -> Image.Image: # This function is for the "All three, side by side" option.
    """Place the R, G, and B colour planes in a horizontal strip."""
    planes = []
    for name in "RGB": # Do something three times, once for R, once for G, and once for B.
        plane = channel_plane(image, name)
        planes.append(plane)
    # Gets the width and height of the first image in the planes list.
    width = planes[0].width
    height = planes[0].height

    # This creates a new empty image that will be used to put the three planes side by side.
    canvas = Image.new("L", (width * 3 + gap * 2, height), 0)
    canvas.paste(planes[0], (0, 0))
    canvas.paste(planes[1], (width + gap, 0))
    canvas.paste(planes[2], (2 * (width + gap), 0))
    return canvas


class ChannelSplitTool(ForensicsTool):
    tool_id = "channel_split"
    title = "Channel split"
    category = "Set 2"
    description = "Extract the red, green, or blue plane, or show all three."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        # ask_choice needs 5 pieces of information:
        # parent   → which window owns this dialog
        # title    → title of the dialog
        # prompt   → question shown to the user
        # options  → choices the user can select
        # initial  → which choice is selected initially
        choice = ask_choice(
            parent,
            "Channel split",
            "Which channel should be shown?",
            CHANNEL_CHOICES,
            initial="all",
        )
        if choice is None:
            return None

        assert document.current is not None
        source = document.current
        if choice == "all":
            output = split_all_planes(source)
            message = "Split the image into red, green, and blue planes."
        else:
            output = channel_plane(source, choice)
            message = f"Extracted the {choice} colour plane."
        return ToolResult(
            image=output,
            message=message,
            details={
                "Operation": "Channel split",
                "Channel": "R | G | B" if choice == "all" else choice,
                "Source mode": source.mode,
                "Output size": f"{output.width} × {output.height}",
            },
        )
