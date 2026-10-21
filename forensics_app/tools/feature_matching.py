"""Two-image ORB/SIFT descriptor matching (Set 5.3)."""

from __future__ import annotations

import io
import tkinter as tk
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage import feature

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
from .dialogs import ask_choice, ask_image_file, show_error
from .feature_descriptors import extract_keypoints, to_grayscale


def match_two_images(
    image1: Image.Image | np.ndarray,
    image2: Image.Image | np.ndarray,
    algorithm: str = "ORB",
    n_keypoints: int = 500,
    n_octaves: int = 8,
    n_scales: int = 3,
    sigma_min: float = 1.6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return matching descriptor index pairs and keypoints for each image."""
    algorithm_name = algorithm.upper()
    if algorithm_name not in ("ORB", "SIFT"):
        raise ValueError("algorithm must be 'ORB' or 'SIFT'.")
    options: dict[str, Any] = {
        "n_keypoints": n_keypoints,
        "n_octaves": n_octaves,
        "n_scales": n_scales,
        "sigma_min": sigma_min,
    }
    keypoints1, descriptors1 = extract_keypoints(image1, algorithm_name, **options)
    keypoints2, descriptors2 = extract_keypoints(image2, algorithm_name, **options)
    metric = "hamming" if algorithm_name == "ORB" else "euclidean"
    matches = feature.match_descriptors(descriptors1, descriptors2, metric=metric)
    return matches, keypoints1, keypoints2


def render_matches(
    image1: Image.Image | np.ndarray,
    keypoints1: np.ndarray,
    image2: Image.Image | np.ndarray,
    keypoints2: np.ndarray,
    matches: np.ndarray,
    title: str,
) -> Image.Image:
    """Render feature matches using plot_matches where available."""
    gray1 = to_grayscale(image1)
    gray2 = to_grayscale(image2)
    fig, ax = plt.subplots(figsize=(10, 5), dpi=100)
    plot_matches = getattr(feature, "plot_matches", None)
    if plot_matches is not None:
        plot_matches(gray1, keypoints1, gray2, keypoints2, matches, ax=ax)
    else:
        feature.plot_matched_features(
            gray1,
            gray2,
            keypoints0=keypoints1,
            keypoints1=keypoints2,
            matches=matches,
            ax=ax,
        )
    ax.set_title(title)
    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    with Image.open(buffer) as output:
        return output.convert("RGB")


class FeatureMatchingTool(ForensicsTool):
    """Set 5.3: Match ORB or SIFT descriptors between two images."""

    tool_id = "feature_matching"
    title = "Feature matching"
    category = "Set 5: Features & Matching"
    description = "Match feature descriptors between the current and a selected image."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None
        path = ask_image_file(parent, "Select second image")
        if not path:
            return None
        algorithm = ask_choice(
            parent, "Feature Matching", "Select descriptor algorithm:",
            [("ORB", "ORB"), ("SIFT", "SIFT")], initial="ORB",
        )
        if algorithm is None:
            return None
        try:
            with Image.open(path) as selected:
                image2 = selected.copy()
            matches, keypoints1, keypoints2 = match_two_images(
                document.current, image2, algorithm=algorithm
            )
            ref_name = document.path.name if document.path else "Current image"
            title = f"Reference: {ref_name} <---> {Path(path).name} (matches={len(matches)})"
            output = render_matches(
                document.current, keypoints1, image2, keypoints2, matches, title
            )
        except (OSError, ValueError) as error:
            show_error(parent, "Feature Matching Error", str(error))
            return None
        return ToolResult(
            image=output,
            message=f"Found {len(matches)} {algorithm} descriptor matches.",
            details={"Algorithm": algorithm, "Matches": len(matches), "Image": path},
        )
