import tkinter as tk
from tkinter import simpledialog

from PIL import ImageFilter

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult


class BlurTool(ForensicsTool):
    tool_id = "gaussian_blur"       # unique, stable identifier
    title = "Gaussian blur"         # text shown in the sidebar
    category = "Filtering"
    description = "Blur the working image with a chosen radius."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        radius = simpledialog.askfloat(
            "Gaussian blur",
            "Radius (pixels):",
            parent=parent,
            minvalue=0.0,
            initialvalue=2.0,
        )
        if radius is None:
            return None

        assert document.current is not None
        output = document.current.filter(ImageFilter.GaussianBlur(radius))
        return ToolResult(
            image=output,
            message=f"Applied Gaussian blur (radius {radius:g}).",
            details={"Operation": "Gaussian blur", "Radius": radius},
        )