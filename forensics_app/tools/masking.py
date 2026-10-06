"""Build a threshold mask and optionally apply it to the colour image."""
# A mask is basically a map that says: Keep this pixel or remove this pixel.
from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk

from PIL import Image, ImageOps

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult


def to_rgb(image: Image.Image) -> Image.Image:
    if image.mode == "RGB":
        return image
    else:
        return image.convert("RGB")


# main image-processing function.
def threshold_mask(image: Image.Image, threshold: int, keep: str) -> Image.Image:
    """Return an L-mode mask: 255 for kept pixels, 0 for the rest.

    ``keep`` is ``"above"`` (intensity >= threshold) or ``"below"`` (<=).
    """

    # A grayscale pixel is represented by a value from 0 to 255. Reject thresholds outside this range.
    if not 0 <= threshold <= 255:
        raise ValueError("threshold must be between 0 and 255")
    gray = ImageOps.grayscale(image)
    if keep == "above":
        lookup = [255 if value >= threshold else 0 for value in range(256)]
        return gray.point(lookup)
    if keep == "below":
        lookup = [255 if value <= threshold else 0 for value in range(256)]
        return gray.point(lookup)
    raise ValueError(f"keep must be 'above' or 'below', got {keep!r}")


def apply_mask(image: Image.Image, mask: Image.Image) -> Image.Image:
    """Keep original colour where the mask is white; black elsewhere."""
    rgb = to_rgb(image)
    background = Image.new("RGB", rgb.size, (0, 0, 0))
    return Image.composite(rgb, background, mask.convert("L"))


def _ask_mask_options(parent: tk.Misc) -> tuple[int, str, str] | None:
    chosen: dict[str, object] = {"ok": False}
    window = tk.Toplevel(parent)
    window.title("Masking")
    window.transient(parent)
    window.resizable(False, False)

    frame = ttk.Frame(window, padding=12)
    frame.pack(fill="both", expand=True)

    ttk.Label(frame, text="Threshold (0–255) on the grayscale version of the image:").pack(
        anchor="w"
    )
    threshold_var = tk.StringVar(value="128")
    ttk.Entry(frame, textvariable=threshold_var, width=8).pack(anchor="w", pady=(4, 10))

    ttk.Label(frame, text="Keep pixels that are:").pack(anchor="w")
    keep_var = tk.StringVar(value="above")
    ttk.Radiobutton(frame, text="At or above the threshold (brighter)", variable=keep_var, value="above").pack(
        anchor="w"
    )
    ttk.Radiobutton(frame, text="At or below the threshold (darker)", variable=keep_var, value="below").pack(
        anchor="w"
    )

    ttk.Label(frame, text="Output:").pack(anchor="w", pady=(10, 0))
    output_var = tk.StringVar(value="masked")
    ttk.Radiobutton(frame, text="Masked colour image (background black)", variable=output_var, value="masked").pack(
        anchor="w"
    )
    ttk.Radiobutton(frame, text="Binary mask only (white = kept)", variable=output_var, value="mask").pack(
        anchor="w"
    )

    def confirm() -> None:
        raw = threshold_var.get().strip()
        try:
            value = int(raw)
        except ValueError:
            messagebox.showerror("Masking", "Threshold must be a whole number.", parent=window)
            return
        if not 0 <= value <= 255:
            messagebox.showerror("Masking", "Threshold must be between 0 and 255.", parent=window)
            return
        chosen["ok"] = True
        chosen["threshold"] = value
        chosen["keep"] = keep_var.get()
        chosen["output"] = output_var.get()
        window.destroy()

    def cancel() -> None:
        window.destroy()

    buttons = ttk.Frame(frame)
    buttons.pack(fill="x", pady=(12, 0))
    ttk.Button(buttons, text="OK", command=confirm).pack(side="right")
    ttk.Button(buttons, text="Cancel", command=cancel).pack(side="right", padx=(0, 6))
    window.protocol("WM_DELETE_WINDOW", cancel)
    window.grab_set()
    window.wait_window()
    if not chosen["ok"]:
        return None
    return int(chosen["threshold"]), str(chosen["keep"]), str(chosen["output"])


class MaskingTool(ForensicsTool):
    tool_id = "masking"
    title = "Masking"
    category = "Set 2"
    description = "Build a brightness threshold mask and optionally apply it."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        options = _ask_mask_options(parent)
        if options is None:
            return None
        threshold, keep, output_kind = options

        assert document.current is not None
        mask = threshold_mask(document.current, threshold, keep)
        if output_kind == "mask":
            result_image = mask
            message = "Computed a binary threshold mask."
        else:
            result_image = apply_mask(document.current, mask)
            message = "Applied the threshold mask to the image."
        kept_pixels = mask.histogram()[255]
        total = mask.width * mask.height
        return ToolResult(
            image=result_image,
            message=message,
            details={
                "Operation": "Masking",
                "Threshold": threshold,
                "Keep": "≥ threshold" if keep == "above" else "≤ threshold",
                "Output": "binary mask" if output_kind == "mask" else "masked image",
                "Kept pixels": f"{kept_pixels} / {total}",
            },
        )
