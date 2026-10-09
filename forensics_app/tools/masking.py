"""Binary masking and jacket texture replacement tools."""

from __future__ import annotations

from pathlib import Path
import tkinter as tk

from PIL import Image, ImageOps

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
from .dialogs import ask_choice, ask_image_file, show_error


def to_rgb(image: Image.Image) -> Image.Image:
    """Ensure the image is in RGB format."""
    if image.mode == "RGB":
        return image
    return image.convert("RGB")


def create_jacket_mask(jacket_image: Image.Image, threshold: int = 1) -> Image.Image:
    """Create an 8-bit binary mask (mode 'L') from the jacket image. """
    gray = ImageOps.grayscale(to_rgb(jacket_image))
    lookup = []
    for value in range(256):
        if value >= threshold:
            lookup.append(255)
        else:
            lookup.append(0)
    return gray.point(lookup)


def composite_jacket(base_image: Image.Image, jacket_image: Image.Image, mask: Image.Image) -> Image.Image:
    """Composite the jacket onto the base model image using a binary mask.

    Where mask is 255 (jacket area), pixels are taken from jacket_image.
    Where mask is 0 (outside jacket), pixels are preserved from base_image.
    """
    if jacket_image.size != base_image.size or mask.size != base_image.size:
        raise ValueError("Image dimensions must match for compositing.")
    # Image.composite(image1, image2, mask) , the order matters here img1 is the white pixel, img2 is
    return Image.composite(to_rgb(jacket_image), to_rgb(base_image), mask.convert("L"))


def apply_texture(base_image: Image.Image, texture_image: Image.Image, mask: Image.Image) -> Image.Image:
    """Composite a texture into the jacket region defined by the binary mask.

    Where mask is 255 (jacket area), pixels are replaced by texture_image.
    Where mask is 0 (outside jacket), pixels are preserved from base_image.
    """
    if texture_image.size != base_image.size or mask.size != base_image.size:
        raise ValueError("Image dimensions must match for compositing.")
    return Image.composite(to_rgb(texture_image), to_rgb(base_image), mask.convert("L"))


MASKING_CHOICES = [
    ("jacket", "Put jacket on model"),
    ("texture", "Apply texture to jacket"),
]


class MaskingTool(ForensicsTool):
    tool_id = "masking"
    title = "Masking"
    category = "Set 2"
    description = "Put jacket on model or apply textures using binary masking."

    def __init__(self) -> None:
        self._jacket_mask: Image.Image | None = None
        self._base_image: Image.Image | None = None
        self._jacket_result: Image.Image | None = None
        self._texture_results: list[Image.Image] = []

    def reset_state(self) -> None:
        """Reset internal jacket mask and operation state."""
        self._jacket_mask = None
        self._base_image = None
        self._jacket_result = None
        self._texture_results.clear()

    def is_jacket_applied(self, document: ImageDocument) -> bool:
        """Verify whether the jacket has been applied and document.current is in a valid state."""
        if self._jacket_mask is None or self._base_image is None or document.current is None:
            return False
        if document.current.size != self._jacket_mask.size:
            return False
        # Valid state if document.current matches the jacket result or a subsequent texture result
        if self._jacket_result is not None and document.current == self._jacket_result:
            return True
        return any(document.current == res for res in self._texture_results)

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        if document.current is None:
            show_error(parent, "Masking", "No base image is loaded. Open an image first.")
            return None

        choice = ask_choice(
            parent,
            title="Masking",
            prompt="Select a masking operation:",
            options=MASKING_CHOICES,
            initial="jacket",
        )
        if choice is None:
            return None

        if choice == "jacket":
            return self._run_put_jacket(parent, document)
        if choice == "texture":
            return self._run_apply_texture(parent, document)
        return None

    def _run_put_jacket(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None

        # 1. Ask the user to select the jacket image
        file_path = ask_image_file(parent, title="Select jacket image")
        if file_path is None:
            return None  # Harmless cancellation

        # 2. Validate that the image can be opened
        try:
            with Image.open(file_path) as loaded:
                jacket_image = loaded.convert("RGB")
        except Exception as error:
            show_error(parent, "Masking", f"Could not open jacket image: {error}")
            return None

        # 3. Check compatible dimensions
        if jacket_image.size != document.current.size:
            show_error(
                parent,
                "Masking",
                f"Jacket dimensions ({jacket_image.width} × {jacket_image.height}) "
                f"do not match the base image ({document.current.width} × {document.current.height}).",
            )
            return None

        # 4. Create binary mask from the jacket image
        mask = create_jacket_mask(jacket_image)

        # 5. Composite jacket onto model image
        result_image = composite_jacket(document.current, jacket_image, mask)

        # Update state for texture operations
        self._jacket_mask = mask.copy()
        self._base_image = document.current.copy()
        self._jacket_result = result_image.copy()
        self._texture_results.clear()

        # 6. Return result through ToolResult
        jacket_pixels = mask.histogram()[255]
        total_pixels = mask.width * mask.height
        return ToolResult(
            image=result_image,
            message="Put jacket on model successfully.",
            details={
                "Operation": "Put jacket on model",
                "Jacket file": Path(file_path).name,
                "Dimensions": f"{jacket_image.width} × {jacket_image.height}",
                "Jacket coverage": f"{jacket_pixels} / {total_pixels} pixels",
            },
        )


    def _run_apply_texture(
            self,
            parent: tk.Misc,
            document: ImageDocument,
    ) -> ToolResult | None:
        assert document.current is not None

        # 1. Verify that the jacket has already been applied successfully
        if not self.is_jacket_applied(document):
            show_error(
                parent,
                "Masking",
                "Please put the jacket on the model before applying a texture.",
            )
            return None

        # 2. Ask the user to select a texture image
        file_path = ask_image_file(parent, title="Select texture image")
        if file_path is None:
            return None

        # 3. Open the texture image
        try:
            with Image.open(file_path) as loaded:
                texture_image = loaded.convert("RGB")
        except Exception as error:
            show_error(
                parent,
                "Masking",
                f"Could not open texture image: {error}",
            )
            return None

        # 4. Resize the texture to match the base image
        original_size = texture_image.size
        target_size = self._base_image.size if self._base_image else document.current.size

        if texture_image.size != target_size:
            texture_image = texture_image.resize(
                target_size,
                Image.Resampling.LANCZOS,
            )

        # 5. Reuse the original jacket mask
        assert self._jacket_mask is not None
        assert self._base_image is not None

        # 6. Apply the texture only to the jacket area
        result_image = apply_texture(
            self._base_image,
            texture_image,
            self._jacket_mask,
        )

        # 7. Record this result for future texture operations
        self._texture_results.append(result_image.copy())

        # 8. Calculate jacket coverage
        texture_pixels = self._jacket_mask.histogram()[255]
        total_pixels = self._jacket_mask.width * self._jacket_mask.height

        return ToolResult(
            image=result_image,
            message="Applied texture to jacket successfully.",
            details={
                "Operation": "Apply texture to jacket",
                "Texture file": Path(file_path).name,
                "Original texture dimensions": (
                    f"{original_size[0]} × {original_size[1]}"
                ),
                "Final texture dimensions": (
                    f"{texture_image.width} × {texture_image.height}"
                ),
                "Textured area": f"{texture_pixels} / {total_pixels} pixels",
            },
        )



