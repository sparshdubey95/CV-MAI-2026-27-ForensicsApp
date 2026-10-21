"""Headless tests for Set 5 feature detection, descriptors, and matching."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.transform import rotate

from forensics_app.tools import build_tool_registry
from forensics_app.tools.corners import detect_corners
from forensics_app.tools.feature_descriptors import extract_keypoints
from forensics_app.tools.feature_matching import match_two_images, render_matches
from forensics_app.tools.image_lookup import lookup_closest_image


def make_pattern(size: int = 128, rgb: bool = False) -> np.ndarray:
    """Create a repeatable geometric image with many distinct corners."""
    image = np.zeros((size, size), dtype=np.uint8)
    shapes = [(12, 12, 25), (50, 10, 30), (20, 70, 40), (75, 65, 45)]
    for top, left, side in shapes:
        image[top : top + side, left : left + side] = 220
        image[top + 4 : top + side - 4, left + 4 : left + side - 4] = 25
    image[::8, :] = 120
    image[:, ::11] = 180
    if rgb:
        return np.stack((image, np.roll(image, 3, axis=0), np.roll(image, 5, axis=1)), axis=-1)
    return image


def rotated_pattern() -> np.ndarray:
    rotated = rotate(make_pattern(), 8, preserve_range=True)
    return np.clip(rotated, 0, 255).astype(np.uint8)


class CornerDetectionTests(unittest.TestCase):
    def test_fast_detects_corners(self) -> None:
        corners = detect_corners(make_pattern(rgb=True), "FAST", min_distance=2)
        self.assertEqual(corners.ndim, 2)
        self.assertEqual(corners.shape[1], 2)
        self.assertGreater(len(corners), 0)

    def test_harris_detects_corners(self) -> None:
        corners = detect_corners(make_pattern(), "Harris", min_distance=2)
        self.assertEqual(corners.ndim, 2)
        self.assertEqual(corners.shape[1], 2)
        self.assertGreater(len(corners), 0)


class FeatureDescriptorTests(unittest.TestCase):
    def test_orb_returns_keypoints_and_2d_descriptors(self) -> None:
        keypoints, descriptors = extract_keypoints(make_pattern(rgb=True), "ORB")
        self.assertGreater(len(keypoints), 0)
        self.assertGreater(descriptors.shape[0], 0)
        self.assertEqual(descriptors.ndim, 2)

    def test_sift_returns_keypoints_and_2d_descriptors(self) -> None:
        keypoints, descriptors = extract_keypoints(
            make_pattern(), "SIFT", n_octaves=6, n_scales=3, sigma_min=1.6
        )
        self.assertGreater(len(keypoints), 0)
        self.assertGreater(descriptors.shape[0], 0)
        self.assertEqual(descriptors.ndim, 2)


class FeatureMatchingTests(unittest.TestCase):
    def test_matches_pattern_to_rotated_copy(self) -> None:
        matches, keypoints1, keypoints2 = match_two_images(
            make_pattern(), rotated_pattern(), algorithm="ORB"
        )
        self.assertGreater(len(keypoints1), 0)
        self.assertGreater(len(keypoints2), 0)
        self.assertGreater(len(matches), 0)

    def test_sift_matching_and_visualization(self) -> None:
        first = make_pattern()
        second = rotated_pattern()
        matches, keypoints1, keypoints2 = match_two_images(
            first, second, algorithm="SIFT"
        )
        self.assertGreater(len(matches), 0)
        rendered = render_matches(
            first, keypoints1, second, keypoints2, matches, "SIFT matches"
        )
        self.assertEqual(rendered.mode, "RGB")
        self.assertGreater(rendered.width, 100)


class ImageLookupTests(unittest.TestCase):
    def test_lookup_selects_candidate_with_maximum_matches(self) -> None:
        reference = make_pattern()
        with tempfile.TemporaryDirectory() as temporary_directory:
            folder = Path(temporary_directory)
            best_candidate = folder / "rotated.png"
            Image.fromarray(rotated_pattern()).save(best_candidate)
            distractor = np.zeros((128, 128), dtype=np.uint8)
            distractor[20:30, 20:30] = 200
            Image.fromarray(distractor).save(folder / "unrelated.png")

            path, reference_keypoints, candidate_keypoints, matches = lookup_closest_image(
                reference, folder, algorithm="ORB"
            )

            self.assertEqual(path, best_candidate)
            self.assertGreater(len(reference_keypoints), 0)
            self.assertGreater(len(candidate_keypoints), 0)
            self.assertGreater(len(matches), 0)


class Set5RegistrationTests(unittest.TestCase):
    def test_tools_registered_in_set5_category(self) -> None:
        registry = build_tool_registry()
        categories = dict(registry.categories())
        expected = {
            "corner_detection",
            "feature_descriptors",
            "feature_matching",
            "image_lookup",
        }
        set5_tools = categories["Set 5: Features & Matching"]
        self.assertEqual({tool.tool_id for tool in set5_tools}, expected)


if __name__ == "__main__":
    unittest.main()
