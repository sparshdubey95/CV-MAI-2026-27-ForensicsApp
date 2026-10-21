"""Folder-based descriptor image lookup (Set 5.4)."""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from skimage import feature

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
from .dialogs import ask_choice, ask_image_folder, show_error
from .feature_descriptors import extract_keypoints
from .feature_matching import render_matches

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def lookup_closest_image(
    reference_image: Image.Image | np.ndarray,
    folder: str | Path,
    algorithm: str = "ORB",
    n_keypoints: int = 500,
    n_octaves: int = 8,
    n_scales: int = 3,
    sigma_min: float = 1.6,
) -> tuple[Path, np.ndarray, np.ndarray, np.ndarray]:
    """Find the candidate with the greatest descriptor-match count.

    Returns the winning path, reference keypoints, winning-image keypoints, and
    matching descriptor index pairs.
    """
    folder_path = Path(folder)
    candidates = sorted(
        path for path in folder_path.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not candidates:
        raise ValueError(f"No candidate image files found in {folder_path}.")
    algorithm_name = algorithm.upper()
    if algorithm_name not in ("ORB", "SIFT"):
        raise ValueError("algorithm must be 'ORB' or 'SIFT'.")
    options: dict[str, Any] = {
        "n_keypoints": n_keypoints,
        "n_octaves": n_octaves,
        "n_scales": n_scales,
        "sigma_min": sigma_min,
    }
    reference_keypoints, reference_descriptors = extract_keypoints(
        reference_image, algorithm_name, **options
    )
    metric = "hamming" if algorithm_name == "ORB" else "euclidean"

    best_path: Path | None = None
    best_keypoints: np.ndarray | None = None
    best_matches: np.ndarray | None = None
    for candidate in candidates:
        try:
            with Image.open(candidate) as opened:
                image = opened.copy()
        except OSError as error:
            raise ValueError(f"Could not open candidate image {candidate}: {error}") from error
        keypoints, descriptors = extract_keypoints(image, algorithm_name, **options)
        matches = feature.match_descriptors(
            reference_descriptors,
            descriptors,
            metric=metric,
        )
        if best_matches is None or len(matches) > len(best_matches):
            best_path = candidate
            best_keypoints = keypoints
            best_matches = matches

    assert best_path is not None
    assert best_keypoints is not None
    assert best_matches is not None
    return best_path, reference_keypoints, best_keypoints, best_matches


class ImageLookupTool(ForensicsTool):
    """Set 5.4: Locate the best-matching image in a folder."""

    tool_id = "image_lookup"
    title = "Folder image lookup"
    category = "Set 5: Features & Matching"
    description = "Find the folder image with the most matching ORB or SIFT descriptors."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None
        folder = ask_image_folder(parent, "Select folder containing candidate images")
        if not folder:
            return None
        algorithm = ask_choice(
            parent, "Image Lookup", "Select descriptor algorithm:",
            [("ORB", "ORB"), ("SIFT", "SIFT")], initial="ORB",
        )
        if algorithm is None:
            return None
        try:
            best_path, ref_keypoints, best_keypoints, matches = lookup_closest_image(
                document.current, folder, algorithm=algorithm
            )
            with Image.open(best_path) as opened:
                best_image = opened.copy()
            ref_name = document.path.name if document.path else "Current image"
            title = (
                f"Reference: {ref_name} <---> Best: {best_path.name} "
                f"(matches={len(matches)})"
            )
            output = render_matches(
                document.current,
                ref_keypoints,
                best_image,
                best_keypoints,
                matches,
                title,
            )
        except (OSError, ValueError) as error:
            show_error(parent, "Image Lookup Error", str(error))
            return None
        return ToolResult(
            image=output,
            message=f"Best match: {best_path.name} ({len(matches)} descriptor matches).",
            details={
                "Algorithm": algorithm,
                "Best image": str(best_path),
                "Matches": len(matches),
            },
        )
