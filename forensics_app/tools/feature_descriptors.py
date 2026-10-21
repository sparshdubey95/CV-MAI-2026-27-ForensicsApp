"""ORB and SIFT feature descriptor extraction (Set 5.2)."""

from __future__ import annotations

import io
import tkinter as tk
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage import color, feature, util

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
from .dialogs import ask_choice, ask_float, ask_int, show_error


def to_grayscale(image: Image.Image | np.ndarray) -> np.ndarray:
    """Convert a PIL or NumPy image to grayscale float values in [0, 1]."""
    array = np.asarray(image)
    if array.ndim == 3:
        array = color.rgb2gray(array[..., :3])
    return util.img_as_float(array)


def extract_keypoints(
    image: Image.Image | np.ndarray,
    algorithm: str = "ORB",
    n_keypoints: int = 500,
    n_octaves: int = 8,
    n_scales: int = 3,
    sigma_min: float = 1.6,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract keypoint coordinates and descriptors using ORB or SIFT."""
    algorithm_name = algorithm.upper()
    gray = to_grayscale(image)
    if algorithm_name == "ORB":
        if n_keypoints < 1:
            raise ValueError("n_keypoints must be positive.")
        detector = feature.ORB(n_keypoints=n_keypoints)
    elif algorithm_name == "SIFT":
        if n_octaves < 1 or n_scales < 1 or sigma_min <= 0:
            raise ValueError("SIFT octaves, scales, and sigma_min must be positive.")
        detector = feature.SIFT(
            n_octaves=n_octaves,
            n_scales=n_scales,
            sigma_min=sigma_min,
        )
    else:
        raise ValueError("algorithm must be 'ORB' or 'SIFT'.")
    try:
        detector.detect_and_extract(gray)
    except RuntimeError as error:
        raise ValueError(f"{algorithm_name} could not find descriptors: {error}") from error
    return detector.keypoints, detector.descriptors


def render_keypoints(
    image: Image.Image | np.ndarray,
    keypoints: np.ndarray,
    algorithm: str,
) -> Image.Image:
    """Render grayscale image with keypoint locations as hollow red circles."""
    gray = to_grayscale(image)
    fig, ax = plt.subplots(figsize=(7, 6), dpi=100)
    ax.imshow(gray, cmap="gray")
    if keypoints.size:
        ax.scatter(
            keypoints[:, 1],
            keypoints[:, 0],
            facecolors="none",
            edgecolors="r",
        )
    ax.set_title(f"{algorithm} keypoints")
    ax.axis("off")
    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    with Image.open(buffer) as output:
        return output.convert("RGB")


class FeatureDescriptorTool(ForensicsTool):
    """Set 5.2: Extract and visualize ORB or SIFT descriptors."""

    tool_id = "feature_descriptors"
    title = "Feature descriptors"
    category = "Set 5: Features & Matching"
    description = "Extract ORB or SIFT keypoints and descriptors."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None
        algorithm = ask_choice(
            parent, "Feature Descriptors", "Select descriptor algorithm:",
            [("ORB", "ORB"), ("SIFT", "SIFT")], initial="ORB",
        )
        if algorithm is None:
            return None
        options: dict[str, Any] = {}
        if algorithm == "ORB":
            n_keypoints = ask_int(
                parent, "ORB Keypoints", "Maximum number of keypoints:",
                initial=500, min_val=1,
            )
            if n_keypoints is None:
                return None
            options["n_keypoints"] = n_keypoints
        else:
            n_octaves = ask_int(parent, "SIFT Octaves", "Number of octaves:", initial=8, min_val=1)
            if n_octaves is None:
                return None
            n_scales = ask_int(parent, "SIFT Scales", "Scales per octave:", initial=3, min_val=1)
            if n_scales is None:
                return None
            sigma_min = ask_float(
                parent, "SIFT Sigma", "Minimum Gaussian sigma:", initial=1.6, min_val=0.01
            )
            if sigma_min is None:
                return None
            options.update(n_octaves=n_octaves, n_scales=n_scales, sigma_min=sigma_min)
        try:
            keypoints, descriptors = extract_keypoints(
                document.current, algorithm=algorithm, **options
            )
            output = render_keypoints(document.current, keypoints, algorithm)
        except ValueError as error:
            show_error(parent, "Feature Descriptor Error", str(error))
            return None
        return ToolResult(
            image=output,
            message=f"Extracted {len(keypoints)} {algorithm} keypoints.",
            details={
                "Algorithm": algorithm,
                "Keypoints": len(keypoints),
                "Descriptor shape": str(descriptors.shape),
                **options,
            },
        )
