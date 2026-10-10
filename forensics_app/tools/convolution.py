"""Custom 1D and 2D convolution with arbitrary kernels using SciPy and NumPy."""

from __future__ import annotations

import tkinter as tk
import numpy as np # creates and manipulates kernel matrices and image arrays.
from scipy.ndimage import convolve # performs convolution.
from PIL import Image

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
from .dialogs import ask_choice, ask_int, show_error


def create_1d_kernel(length: int, direction: str = "horizontal", kernel_type: str = "box") -> np.ndarray:
    """Create a 1D convolution kernel using NumPy.

    - direction: 'horizontal' -> shape (1, length); 'vertical' -> shape (length, 1)
    - kernel_type: 'box' (averaging) or 'gaussian' (smoothing)
    """
    if length < 2:
        raise ValueError(f"Kernel length must be at least 2, got {length}.")

    if kernel_type == "gaussian":
        sigma = max(0.5, length / 6.0)      # chooses sigma using a simple rule. It does not ask the user to choose the amount of blur.
        x = np.arange(length) - (length - 1) / 2.0 # finds the distance of each position from the centre.
        k = np.exp(-0.5 * (x / sigma) ** 2) # converts distances into weights.
        k = k / np.sum(k) # normalizes the weights, so that the uniform brightness remains unchanged.
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

    # isinstance is a built-in Python function that checks the data type of variable.
    if isinstance(size, int):
        h, w = size, size # If size is a single integer, this assigns that same number to both
    else:
        h, w = size # it unpacks those multiple values directly

    if h < 2 or w < 2:
        raise ValueError(f"Kernel dimensions must be at least 2x2, got {h}x{w}.")

    if kernel_type == "box":
        return np.ones((h, w), dtype=np.float64) / (h * w)
    elif kernel_type == "gaussian":
        ky = create_1d_kernel(h, direction="vertical", kernel_type="gaussian")
        kx = create_1d_kernel(w, direction="horizontal", kernel_type="gaussian")
        k2d = ky @ kx # multiplies kx and ky together using matrix multiplication (@)
        return k2d / np.sum(k2d)
    elif kernel_type == "edges":
        k = -np.ones((h, w), dtype=np.float64)
        cy, cx = h // 2, w // 2 # Finding the exact center, '//' is integer division (it drops any decimals)
        k[cy, cx] = float(h * w - 1) # Planting a positive number in the middle
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

    # Applying the Filter (Grayscale vs. Color)
    # Grayscale images (2D): If the image only has a height and width, it applies the convolution directly using mode="reflect".
    if img_arr.ndim == 2:
        convolved = convolve(img_arr, kernel, mode="reflect")
    # Color images (3D): Color images have a third dimension for RGB channels (Red, Green, Blue).
    # This code loops through each color channel separately, applies the kernel to Red, then Green, then Blue,
    # and glues them back together using np.stack.
    elif img_arr.ndim == 3:
        channels = [
            convolve(img_arr[:, :, c], kernel, mode="reflect")
            for c in range(img_arr.shape[2])
        ]
        convolved = np.stack(channels, axis=-1)
    else:
        raise ValueError(f"Unsupported image shape: {img_arr.shape}")

    # For high-pass/edge kernels whose weights sum to <= 0, use absolute value to avoid getting negative brightness
    if np.sum(kernel) <= 0:
        convolved = np.abs(convolved)
    # Ensures that no pixel value accidentally went below 0.0 or above 1.0 during the math.
    convolved = np.clip(convolved, 0.0, 1.0)
    # Scaling back up
    uint8_result = (convolved * 255.0).round().astype(np.uint8)
    # Turns the processed NumPy array back into a standard PIL Image object
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
        # Use only the image currently open in the application.
        if document.current is None:
            show_error(
                parent,
                "Convolution",
                "Please open an image before applying convolution.",
            )
            return None

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

        # Determine maximum allowed kernel size based on image dimensions.
        if choice == "1d_h":
            max_size = source_image.width
        elif choice == "1d_v":
            max_size = source_image.height
        else:
            max_size = min(source_image.width, source_image.height)

        # 2. Ask user for kernel size K.
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

        # Create the selected kernel.
        if choice == "1d_h":
            kernel = create_1d_kernel(
                k, direction="horizontal", kernel_type="box"
            )
        elif choice == "1d_v":
            kernel = create_1d_kernel(
                k, direction="vertical", kernel_type="box"
            )
        elif choice == "2d_box":
            kernel = create_2d_kernel((k, k), kernel_type="box")
        elif choice == "2d_gauss":
            kernel = create_2d_kernel((k, k), kernel_type="gaussian")
        elif choice == "2d_edges":
            kernel = create_2d_kernel((k, k), kernel_type="edges")
        else:
            show_error(
                parent,
                "Convolution",
                f"Unknown kernel choice {choice!r}",
            )
            return None

        # Apply convolution to the currently opened image.
        try:
            output = apply_convolution(source_image, kernel)
        except Exception as error:
            show_error(parent, "Convolution failed", str(error))
            return None

        # Return the processed image and operation details.
        return ToolResult(
            image=output,
            message=(
                f"Convolved image with {choice} kernel "
                f"({kernel.shape[0]}×{kernel.shape[1]})."
            ),
            details={
                "Operation": "Convolution",
                "Kernel type": choice,
                "Kernel shape": (
                    f"{kernel.shape[0]} × {kernel.shape[1]}"
                ),
                "Source": "Current image",
                "Image size": f"{output.width} × {output.height}",
            },
        )
