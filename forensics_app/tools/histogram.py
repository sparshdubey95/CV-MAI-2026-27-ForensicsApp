"""Draw RGB (or grayscale) histograms as an image shown in the preview."""

from __future__ import annotations

import tkinter as tk

from PIL import Image, ImageDraw

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult

PANEL_HEIGHT = 120
CHART_WIDTH = 640
LEFT = 44
RIGHT = 16
TOP = 22
BOTTOM = 18


def _band_stats(band: Image.Image) -> dict[str, float | int]:
    histogram = band.histogram()[:256]  # band.histogram() counts the pixels.
    total = band.width * band.height
    if total == 0:
        return {"min": 0, "max": 0, "mean": 0.0, "peak": 0}
    # Starting from intensity 0, what is the first intensity that actually appears in the image?
    low = next((index for index, count in enumerate(histogram) if count), 0)
    high = next((index for index in range(255, -1, -1) if histogram[index]), 0)
    # This calculates the total intensity.
    weighted = sum(index * count for index, count in enumerate(histogram))
    peak = max(range(256), key=lambda index: histogram[index]) # This asks which intensity has the most pixels?
    return {
        "min": low,
        "max": high,
        "mean": round(weighted / total, 2), # gives the average intensity.
        "peak": peak,
    }


# Decide whether we need one histogram or three histograms.
def histogram_panels(image: Image.Image) -> list[tuple[str, list[int], tuple[int, int, int]]]:
    # L means grayscale.
    # 1 means 1-bit black/white.
    if image.mode in {"L", "1"}:
        gray = image.convert("L") # makes sure it is grayscale.
        histogram = gray.histogram()[:256]
        colour = (50, 50, 55)
        # Returns a list containing one grayscale channel. Its name, histogram data, and the colour to use when drawing it.
        return [
            ("L", histogram, colour)
        ]
    # This makes sure the image is in RGB format. .split() separates the RGB image into three separate grayscale images.
    red, green, blue = image.convert("RGB").split()
    return [
        ("R", red.histogram()[:256], (200, 50, 50)),
        ("G", green.histogram()[:256], (40, 150, 60)),
        ("B", blue.histogram()[:256], (50, 90, 210)),
    ]


def render_histogram(image: Image.Image) -> Image.Image:
    """Paint one chart per band. This does not change pixel values of ``image``."""
    panels = histogram_panels(image) # Get the histogram data
    # The 12 gives some space between panels. The 36 gives space for the title at the top.
    height = 36 + len(panels) * (PANEL_HEIGHT + 12)
    canvas = Image.new("RGB", (CHART_WIDTH, height), (248, 248, 250)) # Creates a blank image for the chart with a white bg
    draw = ImageDraw.Draw(canvas) # Now draw can put things onto canvas.
    draw.text((LEFT, 8), "Intensity histogram (0 to 255)", fill=(30, 30, 32)) # Draws the title

    # The whole canvas is wider than the actual graph.
    # We leave some space on the: left, right
    # So: plot_width is the width available for the actual histogram.
    plot_width = CHART_WIDTH - LEFT - RIGHT
    # Each panel has a fixed height, but we leave space at the: top, bottom for labels and spacing.
    plot_height = PANEL_HEIGHT - TOP - BOTTOM

    # enumerate() gives us the index as well as the item.
    # The index is important because we need to know where vertically to place each chart.
    for index, (name, counts, colour) in enumerate(panels):
        origin_y = 32 + index * (PANEL_HEIGHT + 12) # This determines the vertical position of each histogram.
        x0, y0 = LEFT, origin_y + TOP # This is the top-left corner of the actual plotting area.
        x1, y1 = LEFT + plot_width, origin_y + TOP + plot_height # This is the bottom-right corner.

        # This draws the box around the histogram. outline is the border colour.
        draw.rectangle((x0, y0, x1, y1), outline=(180, 180, 185), fill=(255, 255, 255))
        # Draw the channel name
        draw.text((8, origin_y + TOP), name, fill=colour)
        # counts contains the number of pixels at every intensity.
        peak = max(counts) or 1 # Why or 1? It prevents a division-by-zero error.

        # Loop through every histogram bin
        # A histogram has 256 bins.
        # Each bin represents an intensity
        # bin_index tells us the intensity.
        # count tells us how many pixels have that intensity.
        for bin_index, count in enumerate(counts):
            if count == 0:
                continue # Ignore empty bins
            x = x0 + int(bin_index * plot_width / 255)
            bar_height = int(count / peak * (plot_height - 1))
            draw.line((x, y1, x, y1 - bar_height), fill=colour)
        draw.text((x0, y1 + 2), "0", fill=(90, 90, 95))
        draw.text((x1 - 24, y1 + 2), "255", fill=(90, 90, 95))
    return canvas


class HistogramTool(ForensicsTool):
    tool_id = "histogram"
    title = "Histogram visualization"
    category = "Set 2"
    description = "Plot intensity histograms. Use Undo to return to the photo."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult:
        assert document.current is not None
        source = document.current
        chart = render_histogram(source)
        details: dict[str, object] = {"Operation": "Histogram", "Source mode": source.mode}
        if source.mode in {"L", "1"}:
            stats = _band_stats(source.convert("L"))
            details.update(
                {
                    "Min": stats["min"],
                    "Max": stats["max"],
                    "Mean": stats["mean"],
                    "Peak bin": stats["peak"],
                }
            )
        else:
            red, green, blue = source.convert("RGB").split()
            for name, band in (("R", red), ("G", green), ("B", blue)):
                stats = _band_stats(band)
                details[f"{name} min–max"] = f"{stats['min']}–{stats['max']}"
                details[f"{name} mean"] = stats["mean"]
        return ToolResult(
            image=chart,
            message="Histogram chart shown. Undo to restore the working image.",
            details=details,
        )
