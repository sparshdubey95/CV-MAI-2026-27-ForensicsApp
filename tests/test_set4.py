"""Comprehensive tests for Set 4: Feature Detection & Matching, and UI scrolling."""

from __future__ import annotations

import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools import build_tool_registry
from forensics_app.tools.template_matching import (
    NormalizedCrossCorrTool,
    TemplateMatchingTool,
    compute_ncc_map,
    render_ncc_image,
    render_ncc_figure,
    render_template_matching_figure,
    rotate_template,
    match_template_configured,
    draw_bounding_box,
    select_template,
)
from forensics_app.tools.hog_feature import (
    HOGFeatureTool,
    compute_hog,
    compute_hog_comparison_figure,
)
from forensics_app.tools.hog_detection import (
    HOGPersonDetectionTool,
    detect_person_hog,
)


def make_pattern_image(width: int = 64, height: int = 64) -> Image.Image:
    """Create a deterministic synthetic image with distinct patterns for matching."""
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    # Background gradient
    for y in range(height):
        for x in range(width):
            arr[y, x] = [x * 3 % 256, y * 3 % 256, (x + y) * 2 % 256]
    # Distinct target patch at (y=20..36, x=24..44)
    arr[20:36, 24:44] = [255, 200, 50]
    # Add a cross pattern inside the target patch
    arr[28, 24:44] = [10, 20, 200]
    arr[20:36, 34] = [10, 20, 200]
    return Image.fromarray(arr, mode="RGB")


class NormalizedCrossCorrTests(unittest.TestCase):

    def setUp(self) -> None:
        self.image = make_pattern_image(64, 64)
        # Exact target patch from image
        self.template = self.image.crop((24, 20, 44, 36))

    def test_ncc_map_shape_and_range(self) -> None:
        corr_map = compute_ncc_map(self.image, self.template, pad_input=True)
        self.assertEqual(corr_map.shape, (64, 64))
        self.assertFalse(np.isnan(corr_map).any())
        self.assertLessEqual(np.max(corr_map), 1.0001)
        self.assertGreaterEqual(np.min(corr_map), -1.0001)

    def test_ncc_finds_exact_template_location(self) -> None:
        corr_map = compute_ncc_map(self.image, self.template, pad_input=True)
        peak_y, peak_x = np.unravel_index(np.argmax(corr_map), corr_map.shape)
        # With pad_input=True, peak is at the center of the template:
        # y center = (20 + 36) / 2 = 28, x center = (24 + 44) / 2 = 34
        self.assertAlmostEqual(corr_map[peak_y, peak_x], 1.0, places=2)
        self.assertEqual((peak_y, peak_x), (28, 34))

    def test_ncc_template_larger_than_image_raises(self) -> None:
        large_template = Image.new("RGB", (100, 100), "white")
        with self.assertRaises(ValueError):
            compute_ncc_map(self.image, large_template)

    def test_ncc_zero_variance_handled(self) -> None:
        flat_img = Image.new("RGB", (32, 32), "gray")
        flat_tmpl = Image.new("RGB", (8, 8), "gray")
        corr_map = compute_ncc_map(flat_img, flat_tmpl, pad_input=True)
        self.assertFalse(np.isnan(corr_map).any())
        self.assertEqual(corr_map.shape, (32, 32))

    def test_render_ncc_image(self) -> None:
        corr_map = compute_ncc_map(self.image, self.template, pad_input=True)
        rendered = render_ncc_image(corr_map)
        self.assertEqual(rendered.mode, "L")
        self.assertEqual(rendered.size, (64, 64))

    def test_render_ncc_figure_has_three_panels(self) -> None:
        corr_map = compute_ncc_map(self.image, self.template, pad_input=True)
        import matplotlib.pyplot as plt
        with patch(
            "forensics_app.tools.template_matching.plt.subplots",
            wraps=plt.subplots,
        ) as subplots:
            fig_img = render_ncc_figure(self.image, self.template, corr_map)
            self.assertEqual(subplots.call_args.args[:2], (1, 3))
        self.assertIsInstance(fig_img, Image.Image)
        self.assertEqual(fig_img.mode, "RGB")
        self.assertGreater(fig_img.width, 100)

    def test_render_ncc_figure_keeps_single_row_layout(self) -> None:
        corr_map = compute_ncc_map(self.image, self.template, pad_input=True)
        fig_img = render_ncc_figure(self.image, self.template, corr_map)
        self.assertIsInstance(fig_img, Image.Image)
        self.assertEqual(fig_img.mode, "RGB")
        self.assertGreater(fig_img.width, 100)

    def test_normalized_cross_corr_tool_run_headless(self) -> None:
        doc = ImageDocument()
        doc.current = self.image
        tool = NormalizedCrossCorrTool()
        with patch("forensics_app.tools.template_matching.select_template", return_value=self.template):
            result = tool.run(None, doc)
            self.assertIsNotNone(result)
            self.assertIsNotNone(result.image)
            self.assertIn("Peak correlation", result.details)
            self.assertIn("Peak coordinate (y, x)", result.details)

    def test_normalized_cross_corr_tool_cancel(self) -> None:
        doc = ImageDocument()
        doc.current = self.image
        tool = NormalizedCrossCorrTool()
        with patch("forensics_app.tools.template_matching.select_template", return_value=None):
            result = tool.run(None, doc)
            self.assertIsNone(result)


class TemplateMatchingConfigTests(unittest.TestCase):

    def setUp(self) -> None:
        self.image = make_pattern_image(64, 64)
        self.template = self.image.crop((24, 20, 44, 36))

    def test_rotate_template_identity(self) -> None:
        arr = np.asarray(self.template.convert("L")).astype(float) / 255.0
        rotated = rotate_template(arr, 0.0)
        np.testing.assert_array_almost_equal(arr, rotated)

    def test_rotate_template_angle(self) -> None:
        rotated = rotate_template(self.template, 90.0)
        self.assertIsInstance(rotated, np.ndarray)
        # 90 degree rotation swaps dimensions (approx)
        self.assertGreater(rotated.shape[0], 0)

    def test_match_template_configured_dimmed(self) -> None:
        out_img, details = match_template_configured(
            self.image,
            self.template,
            angle=0.0,
            threshold=0.8,
            style="dimmed",
        )
        self.assertEqual(out_img.size, (64, 64))
        self.assertEqual(out_img.mode, "RGB")
        self.assertEqual(details["Display style"], "dimmed")
        self.assertGreater(details["Matches above threshold"], 0)
        self.assertIn("Bounding box top-left", details)

    def test_match_template_configured_with_rotation(self) -> None:
        out_img, details = match_template_configured(
            self.image,
            self.template,
            angle=45.0,
            threshold=0.8,
        )
        self.assertEqual(out_img.size, (64, 64))
        self.assertEqual(details["Rotation angle"], "45.0°")

    def test_match_template_configured_as_figure(self) -> None:
        fig_img, details = match_template_configured(
            self.image,
            self.template,
            angle=0.0,
            threshold=0.8,
            style="dimmed",
            as_figure=True,
        )
        self.assertIsInstance(fig_img, Image.Image)
        self.assertEqual(fig_img.mode, "RGB")
        self.assertGreater(fig_img.width, 100)
        self.assertGreater(fig_img.height, 100)
        self.assertEqual(details["Display style"], "dimmed")

    def test_draw_bounding_box_within_bounds(self) -> None:
        arr = np.zeros((20, 20, 3), dtype=np.uint8)
        boxed = draw_bounding_box(arr, 2, 3, 10, 8, color_rgb=(255, 0, 0), thickness=1)
        # Check top border
        self.assertEqual(list(boxed[2, 3]), [255, 0, 0])
        self.assertEqual(list(boxed[2, 10]), [255, 0, 0])
        # Check interior is untouched
        self.assertEqual(list(boxed[5, 5]), [0, 0, 0])

    def test_render_template_matching_figure(self) -> None:
        tmpl_arr = np.asarray(self.template.convert("L")).astype(float) / 255.0
        thresh_arr = np.asarray(self.image.convert("RGB"))
        fig_img = render_template_matching_figure(tmpl_arr, thresh_arr, angle=15.0, threshold=0.7)
        self.assertIsInstance(fig_img, Image.Image)
        self.assertEqual(fig_img.mode, "RGB")
        self.assertGreater(fig_img.width, 100)

    def test_template_matching_tool_run_headless(self) -> None:
        doc = ImageDocument()
        doc.current = self.image
        tool = TemplateMatchingTool()
        with patch("forensics_app.tools.template_matching.select_template", return_value=self.template):
            result = tool.run(None, doc)
            self.assertIsNotNone(result)
            self.assertIsNotNone(result.image)
            self.assertIn("Peak correlation", result.details)
            self.assertEqual(result.details["Rotation angle"], "0.0°")
            self.assertEqual(result.details["Threshold"], "0.80")

    def test_template_matching_tool_cancel(self) -> None:
        doc = ImageDocument()
        doc.current = self.image
        tool = TemplateMatchingTool()
        with patch("forensics_app.tools.template_matching.select_template", return_value=None):
            result = tool.run(None, doc)
            self.assertIsNone(result)

    def test_template_matching_tool_cancel_parameters(self) -> None:
        doc = ImageDocument()
        doc.current = self.image
        tool = TemplateMatchingTool()
        with patch("forensics_app.tools.template_matching.select_template", return_value=self.template), \
             patch("forensics_app.tools.template_matching.ask_template_matching_parameters", return_value=None):
            result = tool.run(None, doc)
            self.assertIsNone(result)

    def test_select_template_cancel_and_error(self) -> None:
        # User cancels file picker
        with patch("forensics_app.tools.template_matching.ask_image_file", return_value=None):
            self.assertIsNone(select_template(None))

        # Invalid file selected
        with patch("forensics_app.tools.template_matching.ask_image_file", return_value="non_existent_file.xyz"):
            self.assertIsNone(select_template(None))


class HOGFeatureTests(unittest.TestCase):

    def setUp(self) -> None:
        self.image = make_pattern_image(64, 64)

    def test_compute_hog_orientations(self) -> None:
        for orientations in [2, 4, 8, 9, 16]:
            fd, hog_img = compute_hog(
                self.image,
                orientations=orientations,
                pixels_per_cell=(8, 8),
                cells_per_block=(2, 2),
            )
            self.assertGreater(len(fd), 0)
            self.assertEqual(hog_img.size, (64, 64))
            self.assertEqual(hog_img.mode, "L")
            # Verify feature descriptor scales with orientations
            expected_blocks_y = (64 // 8) - 2 + 1  # 7
            expected_blocks_x = (64 // 8) - 2 + 1  # 7
            expected_len = expected_blocks_y * expected_blocks_x * 2 * 2 * orientations
            self.assertEqual(len(fd), expected_len)

    def test_compute_hog_custom_cells_and_blocks(self) -> None:
        fd, hog_img = compute_hog(
            self.image,
            orientations=9,
            pixels_per_cell=(6, 6),
            cells_per_block=(3, 3),
        )
        self.assertGreater(len(fd), 0)
        self.assertEqual(hog_img.size, (64, 64))

    def test_compute_hog_invalid_params_raise(self) -> None:
        with self.assertRaises(ValueError):
            compute_hog(self.image, orientations=0)
        with self.assertRaises(ValueError):
            compute_hog(self.image, pixels_per_cell=(0, 8))
        with self.assertRaises(ValueError):
            compute_hog(self.image, cells_per_block=(8, 0))

    def test_compute_hog_too_small_image_raises(self) -> None:
        small_img = Image.new("L", (8, 8), "gray")
        with self.assertRaises(ValueError):
            compute_hog(small_img, pixels_per_cell=(8, 8), cells_per_block=(2, 2))

    def test_compute_hog_comparison_figure(self) -> None:
        fig_img, details = compute_hog_comparison_figure(
            self.image,
            orientations_list=[2, 4, 9],
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
        )
        self.assertIsInstance(fig_img, Image.Image)
        self.assertEqual(fig_img.mode, "RGB")
        self.assertGreater(fig_img.width, 100)
        self.assertIn("Descriptor length (ori=2)", details)
        self.assertIn("Descriptor length (ori=4)", details)
        self.assertIn("Descriptor length (ori=9)", details)

    def test_hog_feature_tool_run_headless(self) -> None:
        doc = ImageDocument()
        doc.current = self.image
        tool = HOGFeatureTool()
        result = tool.run(None, doc)
        self.assertIsNotNone(result)
        self.assertIsNotNone(result.image)
        self.assertIn("Descriptor length", result.details)
        self.assertEqual(result.details["Orientations"], 9)
        self.assertEqual(result.details["Pixels per cell"], "6 × 6")
        self.assertEqual(result.details["Cells per block"], "3 × 3")

    def test_hog_feature_tool_configured_params(self) -> None:
        doc = ImageDocument()
        doc.current = self.image
        tool = HOGFeatureTool()
        with patch(
            "forensics_app.tools.hog_feature.ask_hog_parameters",
            return_value=([4, 8], (8, 8), (2, 2)),
        ):
            result = tool.run(None, doc)
            self.assertIsNotNone(result)
            self.assertIsNotNone(result.image)
            self.assertEqual(result.details["Orientations"], 8)
            self.assertEqual(result.details["Pixels per cell"], "8 × 8")
            self.assertEqual(result.details["Cells per block"], "2 × 2")
            self.assertIn("Descriptor length (ori=4)", result.details)
            self.assertIn("Descriptor length (ori=8)", result.details)
            self.assertNotIn("Descriptor length (ori=9)", result.details)

    def test_hog_feature_tool_cancel(self) -> None:
        doc = ImageDocument()
        doc.current = self.image
        tool = HOGFeatureTool()
        with patch("forensics_app.tools.hog_feature.ask_hog_parameters", return_value=None):
            result = tool.run(None, doc)
            self.assertIsNone(result)

    def test_ask_hog_parameters_headless_defaults_and_parsing(self) -> None:
        from forensics_app.tools.dialogs import ask_hog_parameters

        # Headless default
        params = ask_hog_parameters(None)
        self.assertEqual(params, ([2, 4, 9], (6, 6), (3, 3)))

        # Custom values parsing
        custom = ask_hog_parameters(
            None,
            initial_orientations="4, 8, 12",
            initial_pixels_per_cell="8, 8",
            initial_cells_per_block="2, 2",
        )
        self.assertEqual(custom, ([4, 8, 12], (8, 8), (2, 2)))

        # Single integer dimension e.g. "8" or "8x8"
        custom2 = ask_hog_parameters(
            None,
            initial_orientations="6",
            initial_pixels_per_cell="8",
            initial_cells_per_block="2x2",
        )
        self.assertEqual(custom2, ([6], (8, 8), (2, 2)))


class HOGDetectionTests(unittest.TestCase):

    def setUp(self) -> None:
        # Create image with textured pattern
        rng = np.random.default_rng(123)
        arr = rng.integers(50, 220, size=(64, 64), dtype=np.uint8)
        # Distinct target object at (row=16..48, col=24..40)
        arr[16:48, 24:40] = rng.integers(0, 50, size=(32, 16), dtype=np.uint8)
        self.image = Image.fromarray(arr, mode="L")
        self.template = self.image.crop((24, 16, 40, 48))  # width=16, height=32

    def test_detect_person_hog_finds_correct_location(self) -> None:
        out_img, details = detect_person_hog(
            self.image,
            self.template,
            step_size=8,
            orientations=8,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            method="both",
        )
        self.assertEqual(out_img.size, (64, 64))
        # Best location should be exact target location (y=16, x=24)
        self.assertEqual(details["Best Euclidean location"], "y=16, x=24")
        self.assertAlmostEqual(float(details["Min Euclidean distance"]), 0.0, places=2)
        self.assertEqual(details["Best Dot Product location"], "y=16, x=24")

    def test_detect_person_hog_side_by_side_montage(self) -> None:
        out_img, details = detect_person_hog(
            self.image,
            self.template,
            step_size=8,
            orientations=8,
            method="side_by_side",
        )
        # 3 panels side by side -> width = 3 * 64 = 192
        self.assertEqual(out_img.size, (192, 64))
        self.assertEqual(details["Visualization"], "side_by_side")

    def test_detect_person_hog_as_figure(self) -> None:
        out_img, details = detect_person_hog(
            self.image,
            self.template,
            step_size=8,
            orientations=8,
            as_figure=True,
        )
        self.assertIsInstance(out_img, Image.Image)
        self.assertEqual(out_img.mode, "RGB")
        self.assertGreater(out_img.width, 100)
        self.assertEqual(details["Visualization"], "Slide 5 figure")

    def test_detect_person_hog_methods(self) -> None:
        for method in ["euclidean", "dot_product"]:
            out_img, details = detect_person_hog(
                self.image,
                self.template,
                step_size=8,
                method=method,
            )
            self.assertEqual(out_img.size, (64, 64))
            self.assertEqual(details["Visualization"], method)

    def test_detect_person_hog_template_exceeds_image_raises(self) -> None:
        big_tmpl = Image.new("L", (80, 80), "white")
        with self.assertRaises(ValueError):
            detect_person_hog(self.image, big_tmpl)

    def test_hog_person_detection_tool_run_headless(self) -> None:
        doc = ImageDocument()
        doc.current = self.image
        tool = HOGPersonDetectionTool()
        result = tool.run(None, doc)
        self.assertIsNotNone(result)
        self.assertIsNotNone(result.image)
        self.assertIn("Best Euclidean location", result.details)
        self.assertIn("Best Dot Product location", result.details)
        self.assertEqual(result.details["Step size"], "16 px")
        self.assertEqual(result.details["Visualization"], "Slide 5 figure")

class RegistryAndUITests(unittest.TestCase):

    def test_all_set4_tools_registered(self) -> None:
        registry = build_tool_registry()
        tool_ids = [t.tool_id for t in registry.all()]
        expected_ids = [
            "normalized_cross_corr",
            "template_matching_configs",
            "hog_feature",
            "hog_person_detection",
        ]
        for tid in expected_ids:
            self.assertIn(tid, tool_ids)

    def test_tool_ids_are_unique(self) -> None:
        registry = build_tool_registry()
        ids = [t.tool_id for t in registry.all()]
        self.assertEqual(len(ids), len(set(ids)))

    def test_set4_category_name(self) -> None:
        registry = build_tool_registry()
        categories = dict(registry.categories())
        self.assertIn("Set 4: Feature & HOG Detection", categories)
        self.assertEqual(len(categories["Set 4: Feature & HOG Detection"]), 4)

    def test_ui_scrollable_sidebar(self) -> None:
        import tkinter as tk
        from forensics_app.ui.main_window import MainWindow

        root = tk.Tk()
        try:
            reg = build_tool_registry()
            app = MainWindow(root, reg)
            root.update()

            # Find Canvas inside MainWindow widget tree
            canvas_widget = None

            def find_canvas(widget: tk.Misc) -> tk.Canvas | None:
                for child in widget.winfo_children():
                    if isinstance(child, tk.Canvas):
                        return child
                    res = find_canvas(child)
                    if res is not None:
                        return res
                return None

            canvas_widget = find_canvas(root)
            self.assertIsNotNone(canvas_widget, "Scrollable sidebar canvas should exist in MainWindow")

            # Verify scrolling commands execute and change canvas position
            initial_top = canvas_widget.yview()[0]
            self.assertEqual(initial_top, 0.0)

            canvas_widget.yview_scroll(5, "units")
            scrolled_top = canvas_widget.yview()[0]
            self.assertGreater(scrolled_top, 0.0)

            canvas_widget.yview_moveto(0.0)
            self.assertEqual(canvas_widget.yview()[0], 0.0)
        finally:
            root.destroy()


if __name__ == "__main__":
    unittest.main()
