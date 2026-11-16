"""Folder-based descriptor image lookup (Set 5.4)."""
# The Big Picture: What Does This Tool Do?
# Scans a Folder: It looks inside a user-selected folder and filters for valid image files (like .png, .jpg, etc.).
# Extracts Reference Features: It computes the feature descriptors for your main reference image once so it doesn’t have to recalculate it repeatedly.
# The Search Loop: It opens each candidate image in the folder one by one, extracts its features, and matches them against the reference image.
# Finds the Winner: It counts how many matches each image gets and keeps track of the highest score.
# Draws the Final Output: Once it finds the best match, it uses the drawing function from the previous script to show you the side-by-side comparison lines.
from __future__ import annotations

import tkinter as tk
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from skimage import feature, measure, transform

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
from .dialogs import ask_choice, ask_image_folder, show_error
from .feature_descriptors import extract_keypoints
from .feature_matching import render_matches

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

# What it does:
# Finds all valid images in the target folder using folder_path.iterdir(). If the folder is empty, it raises an error.
# Extracts descriptors from the reference image once (saving precious processing time).
# Loops through every single candidate image in a for loop.
# Compares the candidate's descriptors to the reference descriptors using feature.match_descriptors(...) (using Hamming for ORB or Euclidean for SIFT).
# Filters out false matches using RANSAC geometric verification (skimage.measure.ransac + AffineTransform).
# It uses a max-tracking variable (best_inliers_count). If a candidate has more inliers than the current record-holder, it replaces it as the new winner (best_path).
def lookup_closest_image(
    reference_image: Image.Image | np.ndarray,
    folder: str | Path,
    algorithm: str = "ORB",
    n_keypoints: int = 500,
    n_octaves: int = 8,
    n_scales: int = 3,
    sigma_min: float = 1.6,
    min_samples: int = 4,
    residual_threshold: float = 2.0,
) -> tuple[Path, np.ndarray, np.ndarray, np.ndarray]:
    """Find the candidate with the greatest inlier match count using RANSAC.

    Returns the winning path, reference keypoints, winning-image keypoints, and
    filtered inlier descriptor index pairs.
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
    best_inlier_matches: np.ndarray | None = None
    best_inlier_count: int = -1

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

        inlier_matches = np.empty((0, 2), dtype=int)
        if len(matches) >= min_samples:
            # Matches coordinates: reference keypoints are src, candidate keypoints are dst.
            # Convert (row, col) to (x, y) coordinates via [:, ::-1]
            src_coords = reference_keypoints[matches[:, 0]][:, ::-1]
            dst_coords = keypoints[matches[:, 1]][:, ::-1]

            try:
                model, inliers = measure.ransac(
                    (src_coords, dst_coords),
                    transform.AffineTransform,
                    min_samples=min_samples,
                    residual_threshold=residual_threshold,
                    max_trials=500,
                )
                if inliers is not None and np.any(inliers):
                    inlier_matches = matches[inliers]
            except Exception:
                inlier_matches = np.empty((0, 2), dtype=int)

        inlier_count = len(inlier_matches)
        if inlier_count > best_inlier_count:
            best_path = candidate
            best_keypoints = keypoints
            best_inlier_matches = inlier_matches
            best_inlier_count = inlier_count

    if best_path is None:
        best_path = candidates[0]
        best_keypoints = np.empty((0, 2), dtype=float)
        best_inlier_matches = np.empty((0, 2), dtype=int)

    return best_path, reference_keypoints, best_keypoints, best_inlier_matches


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
            # Reuses render_matches (imported from the previous file) to draw the lines between the reference image and the winning folder image.
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
