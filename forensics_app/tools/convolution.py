"""Custom 1D and 2D convolution with arbitrary kernels using SciPy and NumPy."""

from __future__ import annotations

import tkinter as tk
import numpy as np # creates and manipulates kernel matrices and image arrays.
import scipy.ndimage # performs convolution.
import skimage.data # supplies a standard test image.
from PIL import Image

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
from .dialogs import ask_choice, ask_int, show_error


def load_camera_sample() -> Image.Image:
    """Load the standard scikit-image camera sample image as a grayscale PIL Image."""
    cam_array = skimage.data.camera()
    return Image.fromarray(cam_array, mode="L")


def create_1d_kernel(length: int, direction: str = "horizontal", kernel_type: str = "box") -> np.ndarray:
    """Create a 1D convolution kernel using NumPy.

    - direction: 'horizontal' -> shape (1, length); 'vertical' -> shape (length, 1)
    - kernel_type: 'box' (averaging) or 'gaussian' (smoothing)
    """
    if length < 2:
        raise ValueError(f"Kernel length must be at least 2, got {length}.")

    if kernel_type == "gaussian":
        sigma = max(0.5, length / 6.0)
        x = np.arange(length) - (length - 1) / 2.0
        k = np.exp(-0.5 * (x / sigma) ** 2)
        k = k / np.sum(k)
    else:  # box averaging filter
        k = np.ones(length, dtype=np.float64) / length

    if direction == "horizontal":
        return k.reshape(1, length)
    elif direction == "vertical":
        return k.reshape(length, 1)
    else:
        raise ValueError(f"Invalid direction {direction!r}. Choose 'horizontal' or 'vertical'.")


def create_2d_kernel(size: tuple[int, int] | int, kernel_type: str = "box") -> np.ndarray:
    """Create a 2D convolution kernel using NumPy.

    - size: (height, width) or scalar integer for square (size, size)
    - kernel_type: 'box', 'gaussian', or 'edges'
    """
    if isinstance(size, int):
        h, w = size, size
    else:
        h, w = size

    if h < 2 or w < 2:
        raise ValueError(f"Kernel dimensions must be at least 2x2, got {h}x{w}.")

    if kernel_type == "box":
        return np.ones((h, w), dtype=np.float64) / (h * w)
    elif kernel_type == "gaussian":
        ky = create_1d_kernel(h, direction="vertical", kernel_type="gaussian")
        kx = create_1d_kernel(w, direction="horizontal", kernel_type="gaussian")
        k2d = ky @ kx
        return k2d / np.sum(k2d)
    elif kernel_type == "edges":
        k = -np.ones((h, w), dtype=np.float64)
        cy, cx = h // 2, w // 2
        k[cy, cx] = float(h * w - 1)
        return k
    else:
        raise ValueError(f"Unknown kernel_type {kernel_type!r}")


def apply_convolution(image: Image.Image, kernel: np.ndarray) -> Image.Image:
    """Convolve an image with a custom kernel using scipy.ndimage.convolve."""
    h_img, w_img = image.height, image.width
    kh, kw = kernel.shape

    # Validate kernel size bounds (min 2, max image dimension)
    if kh < 1 or kw < 1:
        raise ValueError("Kernel dimensions must be >= 1.")
    if kh < 2 and kw < 2:
        raise ValueError("Kernel must have at least one dimension >= 2 (min 2x2 for 2D, or length >= 2 for 1D).")
    if kh > h_img or kw > w_img:
        raise ValueError(
            f"Kernel size ({kh}x{kw}) exceeds image dimensions ({h_img}x{w_img}). Maximum size is the image size."
        )

    # Convert image to float array in [0.0, 1.0]
    img_arr = np.asarray(image).astype(np.float64) / 255.0

    if img_arr.ndim == 2:
        convolved = scipy.ndimage.convolve(img_arr, kernel, mode="reflect")
    elif img_arr.ndim == 3:
        channels = [
            scipy.ndimage.convolve(img_arr[:, :, c], kernel, mode="reflect")
            for c in range(img_arr.shape[2])
        ]
        convolved = np.stack(channels, axis=-1)
    else:
        raise ValueError(f"Unsupported image shape: {img_arr.shape}")

    # For high-pass/edge kernels whose weights sum to <= 0, use absolute value
    if np.sum(kernel) <= 0:
        convolved = np.abs(convolved)

    convolved = np.clip(convolved, 0.0, 1.0)
    uint8_result = (convolved * 255.0).round().astype(np.uint8)
    return Image.fromarray(uint8_result)


CONV_CHOICES = [
    ("1d_h", "1D Horizontal kernel (1 × K)"),
    ("1d_v", "1D Vertical kernel (K × 1)"),
    ("2d_box", "2D Box blur kernel (K × K)"),
    ("2d_gauss", "2D Gaussian kernel (K × K)"),
    ("2d_edges", "2D Edge detection kernel (K × K)"),
]


class ConvolutionTool(ForensicsTool):
    tool_id = "convolution"
    title = "Custom convolution"
    category = "Set 3"
    description = "Convolve image with arbitrary 1D or 2D kernels using scipy.ndimage."
    requires_image = False

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        used_sample = False

        # If no image is currently loaded, or user chooses sample, load skimage camera sample
        if document.current is None:
            source_image = load_camera_sample()
            used_sample = True
        else:
            source_choice = ask_choice(
                parent,
                "Select image source",
                "Choose which image to convolve:",
                [
                    ("current", "Use currently open image"),
                    ("sample", "Load sample camera image (skimage.data.camera())"),
                ],
                initial="current",
            )
            if source_choice is None:
                return None
            if source_choice == "sample":
                source_image = load_camera_sample()
                used_sample = True
            else:
                source_image = document.current

        # 1. Ask user for kernel type
        choice = ask_choice(
            parent,
            "Convolution Kernel",
            "Choose kernel type:",
            CONV_CHOICES,
            initial="2d_box",
        )
        if choice is None:
            return None

        # Determine max allowed kernel size based on image bounds
        if choice == "1d_h":
            max_size = source_image.width
        elif choice == "1d_v":
            max_size = source_image.height
        else:
            max_size = min(source_image.width, source_image.height)

        # 2. Ask user for kernel size K (min 2, max image dimension)
        k = ask_int(
            parent,
            "Kernel Size",
            f"Enter kernel size K (min 2, max {max_size}):",
            initial=min(5, max_size),
            min_val=2,
            max_val=max_size,
        )
        if k is None:
            return None

        # Create the kernel
        if choice == "1d_h":
            kernel = create_1d_kernel(k, direction="horizontal", kernel_type="box")
        elif choice == "1d_v":
            kernel = create_1d_kernel(k, direction="vertical", kernel_type="box")
        elif choice == "2d_box":
            kernel = create_2d_kernel((k, k), kernel_type="box")
        elif choice == "2d_gauss":
            kernel = create_2d_kernel((k, k), kernel_type="gaussian")
        elif choice == "2d_edges":
            kernel = create_2d_kernel((k, k), kernel_type="edges")
        else:
            show_error(parent, "Convolution", f"Unknown kernel choice {choice!r}")
            return None

        # Apply convolution
        try:
            output = apply_convolution(source_image, kernel)
        except Exception as error:
            show_error(parent, "Convolution failed", str(error))
            return None

        return ToolResult(
            image=output,
            message=f"Convolved image with {choice} kernel ({kernel.shape[0]}×{kernel.shape[1]}).",
            details={
                "Operation": "Convolution",
                "Kernel type": choice,
                "Kernel shape": f"{kernel.shape[0]} × {kernel.shape[1]}",
                "Source": "skimage.data.camera()" if used_sample else "Current image",
                "Image size": f"{output.width} × {output.height}",
            },
        )
