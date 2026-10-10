"""Focused tests for Set 3 image-processing functions.

These tests cover the pure image-processing logic (no Tkinter required).
Each test creates synthetic images in memory so no files need to be present.
"""

import unittest
import numpy as np
from PIL import Image

# ── helpers shared by several tests ──────────────────────────────────────────

def make_gray(width: int = 64, height: int = 64, value: int = 128) -> Image.Image:
    """Create a uniform grayscale PIL image."""
    return Image.new("L", (width, height), value)


def make_rgb(width: int = 64, height: int = 64, color: tuple = (100, 150, 200)) -> Image.Image:
    """Create a uniform RGB PIL image."""
    return Image.new("RGB", (width, height), color)


def make_gradient_gray(width: int = 64, height: int = 64) -> Image.Image:
    """Create a simple horizontal gradient grayscale image for edge testing."""
    arr = np.zeros((height, width), dtype=np.uint8)
    arr[:, width // 2 :] = 200
    arr[:, : width // 2] = 50
    return Image.fromarray(arr, mode="L")


# ── 1. Convolution ────────────────────────────────────────────────────────────

from forensics_app.tools.convolution import (
    apply_convolution,
    create_1d_kernel,
    create_2d_kernel,
)


class KernelCreationTests(unittest.TestCase):

    def test_1d_horizontal_box_shape_and_sum(self):
        k = create_1d_kernel(5, direction="horizontal", kernel_type="box")
        self.assertEqual(k.shape, (1, 5))
        self.assertAlmostEqual(np.sum(k), 1.0)

    def test_1d_vertical_box_shape_and_sum(self):
        k = create_1d_kernel(4, direction="vertical", kernel_type="box")
        self.assertEqual(k.shape, (4, 1))
        self.assertAlmostEqual(np.sum(k), 1.0)

    def test_1d_gaussian_sums_to_one(self):
        k = create_1d_kernel(7, direction="horizontal", kernel_type="gaussian")
        self.assertAlmostEqual(np.sum(k), 1.0, places=5)

    def test_2d_box_shape_and_sum(self):
        k = create_2d_kernel(4, kernel_type="box")
        self.assertEqual(k.shape, (4, 4))
        self.assertAlmostEqual(np.sum(k), 1.0)

    def test_2d_gaussian_sums_to_one(self):
        k = create_2d_kernel(5, kernel_type="gaussian")
        self.assertAlmostEqual(np.sum(k), 1.0, places=5)

    def test_2d_edges_kernel_centre_differs_from_rest(self):
        k = create_2d_kernel(3, kernel_type="edges")
        cy, cx = 1, 1
        self.assertGreater(k[cy, cx], 0)
        self.assertLess(k[0, 0], 0)

    def test_kernel_too_small_raises(self):
        with self.assertRaises(ValueError):
            create_1d_kernel(1, direction="horizontal")

    def test_2d_kernel_too_small_raises(self):
        with self.assertRaises(ValueError):
            create_2d_kernel(1, kernel_type="box")


class ConvolutionApplicationTests(unittest.TestCase):

    def test_box_blur_uniform_image_unchanged(self):
        """Convolving a flat image with a box filter keeps all pixels the same."""
        image = make_gray(64, 64, 100)
        kernel = create_2d_kernel(5, kernel_type="box")
        output = apply_convolution(image, kernel)
        arr = np.asarray(output)
        self.assertTrue(np.all(arr == 100))

    def test_output_size_equals_input_size(self):
        image = make_gray(64, 32)
        kernel = create_2d_kernel(3, kernel_type="box")
        output = apply_convolution(image, kernel)
        self.assertEqual(output.size, image.size)

    def test_1d_horizontal_kernel_on_grayscale(self):
        image = make_gray(64, 64, 80)
        kernel = create_1d_kernel(4, direction="horizontal", kernel_type="box")
        output = apply_convolution(image, kernel)
        self.assertEqual(output.size, image.size)
        arr = np.asarray(output)
        self.assertTrue(np.all(arr == 80))

    def test_kernel_larger_than_image_raises(self):
        image = make_gray(10, 10)
        kernel = create_2d_kernel(2, kernel_type="box")  # valid 2x2
        # Now make a kernel bigger than the image
        big_kernel = np.ones((20, 20), dtype=np.float64) / 400.0
        with self.assertRaises(ValueError):
            apply_convolution(image, big_kernel)

    def test_rgb_image_convolution_preserves_mode(self):
        image = make_rgb(64, 64)
        kernel = create_2d_kernel(3, kernel_type="gaussian")
        output = apply_convolution(image, kernel)
        self.assertEqual(output.mode, "RGB")
        self.assertEqual(output.size, image.size)

    def test_edge_kernel_produces_nontrivial_output_on_gradient(self):
        """An edge-detection kernel applied to a gradient image should produce non-zero output."""
        image = make_gradient_gray()
        kernel = create_2d_kernel(3, kernel_type="edges")
        output = apply_convolution(image, kernel)
        arr = np.asarray(output)
        # The boundary between light and dark regions should have bright pixels
        self.assertGreater(arr.max(), 0)


# ── 2. Scikit-image filters ───────────────────────────────────────────────────

from forensics_app.tools.skimage_filters import (
    apply_gaussian,
    apply_median,
    apply_prewitt,
    apply_sobel,
    from_float_array,
    to_float_array,
    to_grayscale_float,
)


class SkimageFilterTests(unittest.TestCase):

    def test_gaussian_uniform_image_unchanged(self):
        """Gaussian blur of a flat image leaves all pixels identical."""
        image = make_gray(64, 64, 120)
        output = apply_gaussian(image, sigma=2.0)
        arr = np.asarray(output)
        self.assertTrue(np.all(arr == 120))

    def test_gaussian_blurs_edge(self):
        """Gaussian blur should soften the sharp edge in a gradient image."""
        image = make_gradient_gray()
        output = apply_gaussian(image, sigma=3.0)
        # At the boundary column, blurred values should differ from the original
        orig_arr = np.asarray(image)
        blur_arr = np.asarray(output)
        mid = image.width // 2
        # The blurred column near the edge should be an intermediate value
        self.assertGreater(int(blur_arr[0, mid - 1]), 50)
        self.assertLess(int(blur_arr[0, mid - 1]), 200)

    def test_gaussian_sigma_zero_raises(self):
        with self.assertRaises(ValueError):
            apply_gaussian(make_gray(), sigma=0.0)

    def test_gaussian_output_same_size_and_mode_for_rgb(self):
        image = make_rgb(32, 32)
        output = apply_gaussian(image, sigma=1.5)
        self.assertEqual(output.size, image.size)
        self.assertEqual(output.mode, "RGB")

    def test_median_uniform_image_unchanged(self):
        image = make_gray(32, 32, 80)
        output = apply_median(image, radius=2)
        arr = np.asarray(output)
        self.assertTrue(np.all(arr == 80))

    def test_median_output_same_size(self):
        image = make_rgb(32, 32)
        output = apply_median(image, radius=1)
        self.assertEqual(output.size, image.size)

    def test_sobel_all_black_image_produces_zero_edges(self):
        """A completely uniform image has zero gradient everywhere."""
        image = make_gray(32, 32, 0)
        output = apply_sobel(image)
        arr = np.asarray(output)
        self.assertTrue(np.all(arr == 0))

    def test_sobel_detects_edge_in_gradient(self):
        image = make_gradient_gray()
        output = apply_sobel(image)
        arr = np.asarray(output)
        mid = image.width // 2
        # Column at the edge should have a non-zero response
        self.assertGreater(arr[:, mid].max(), 0)

    def test_prewitt_detects_edge_in_gradient(self):
        image = make_gradient_gray()
        output = apply_prewitt(image)
        arr = np.asarray(output)
        mid = image.width // 2
        self.assertGreater(arr[:, mid].max(), 0)

    def test_to_float_array_range(self):
        image = make_rgb(4, 4, (0, 128, 255))
        arr, is_rgb = to_float_array(image)
        self.assertTrue(is_rgb)
        self.assertGreaterEqual(arr.min(), 0.0)
        self.assertLessEqual(arr.max(), 1.0)

    def test_from_float_array_clips_correctly(self):
        arr = np.array([[-0.5, 0.5, 1.5]], dtype=np.float64)
        img = from_float_array(arr)
        px = np.asarray(img)
        self.assertEqual(px[0, 0], 0)
        self.assertEqual(px[0, 1], 128)
        self.assertEqual(px[0, 2], 255)


# ── 3. Canny edge detection ───────────────────────────────────────────────────

from forensics_app.tools.canny_edges import apply_canny, edges_to_image


class CannyTests(unittest.TestCase):

    def test_returns_boolean_array(self):
        image = make_gradient_gray()
        edges = apply_canny(image, sigma=1.0)
        self.assertEqual(edges.dtype, bool)

    def test_shape_matches_input(self):
        image = make_gray(48, 64)
        edges = apply_canny(image, sigma=1.0)
        self.assertEqual(edges.shape, (64, 48))

    def test_uniform_image_has_no_edges(self):
        """A completely uniform image has no intensity variation → no edges."""
        image = make_gray(64, 64, 128)
        edges = apply_canny(image, sigma=1.0)
        self.assertEqual(np.sum(edges), 0)

    def test_gradient_image_has_edges(self):
        """A gradient image should produce some edge pixels."""
        image = make_gradient_gray(64, 64)
        edges = apply_canny(image, sigma=1.0)
        self.assertGreater(np.sum(edges), 0)

    def test_sigma_zero_raises(self):
        with self.assertRaises(ValueError):
            apply_canny(make_gray(), sigma=0.0)

    def test_invalid_thresholds_raise(self):
        with self.assertRaises(ValueError):
            apply_canny(make_gradient_gray(), sigma=1.0,
                        low_threshold=0.5, high_threshold=0.3)

    def test_edges_to_image_white_where_true(self):
        mask = np.array([[True, False], [False, True]], dtype=bool)
        img = edges_to_image(mask)
        arr = np.asarray(img)
        self.assertEqual(arr[0, 0], 255)
        self.assertEqual(arr[0, 1], 0)
        self.assertEqual(arr[1, 1], 255)

    def test_higher_sigma_fewer_edges(self):
        """Stronger pre-smoothing suppresses high-frequency noise edges."""
        rng = np.random.RandomState(42)
        noisy = Image.fromarray(rng.randint(0, 100, (64, 64), dtype=np.uint8), mode="L")
        edges_sharp = apply_canny(noisy, sigma=1.0)
        edges_smooth = apply_canny(noisy, sigma=3.0)
        self.assertGreater(np.sum(edges_sharp), np.sum(edges_smooth))


# ── 4. Edge visualisation ─────────────────────────────────────────────────────

from forensics_app.tools.edge_visualisation import (
    overlay_colored_edges,
    overlay_dimmed_background,
    overlay_transparent_edges,
)


class EdgeVisualisationTests(unittest.TestCase):

    def setUp(self):
        self.image = make_gradient_gray(64, 64)
        # A simple synthetic edge mask: centre column is True
        self.mask = np.zeros((64, 64), dtype=bool)
        self.mask[:, 32] = True

    def test_colored_overlay_output_size(self):
        output = overlay_colored_edges(self.image, self.mask, edge_color=(255, 0, 0))
        self.assertEqual(output.size, self.image.size)
        self.assertEqual(output.mode, "RGB")

    def test_colored_overlay_edge_pixels_are_red(self):
        output = overlay_colored_edges(self.image, self.mask, edge_color=(255, 0, 0))
        arr = np.asarray(output)
        # All pixels in the edge column must be exactly (255, 0, 0)
        self.assertTrue(np.all(arr[:, 32] == [255, 0, 0]))

    def test_colored_overlay_non_edge_pixels_unchanged(self):
        """Pixels outside the mask must come from the original image."""
        output = overlay_colored_edges(self.image, self.mask)
        orig_rgb = np.asarray(self.image.convert("RGB"))
        out_arr = np.asarray(output)
        # Column 0 is not in the mask – should match original
        np.testing.assert_array_equal(out_arr[:, 0], orig_rgb[:, 0])

    def test_dimmed_background_dims_non_edges(self):
        output = overlay_dimmed_background(self.image, self.mask, dim_factor=0.0)
        arr = np.asarray(output)
        # Non-edge pixels with dim_factor=0 should be black
        self.assertTrue(np.all(arr[:, 0] == 0))

    def test_dimmed_background_edge_pixels_are_bright(self):
        output = overlay_dimmed_background(self.image, self.mask, dim_factor=0.0,
                                           edge_color=(0, 255, 255))
        arr = np.asarray(output)
        # Edge column should be the edge_color
        self.assertTrue(np.all(arr[:, 32] == [0, 255, 255]))

    def test_transparent_overlay_blends_values(self):
        """With alpha=1.0, edge pixels should become exactly the edge_color."""
        output = overlay_transparent_edges(self.image, self.mask, alpha=1.0,
                                           edge_color=(255, 0, 0))
        arr = np.asarray(output)
        self.assertTrue(np.all(arr[:, 32] == [255, 0, 0]))

    def test_transparent_overlay_with_alpha_zero_keeps_original(self):
        """With alpha=0.0, no blending occurs – pixels keep original values."""
        output = overlay_transparent_edges(self.image, self.mask, alpha=0.0,
                                           edge_color=(255, 0, 0))
        orig_rgb = np.asarray(self.image.convert("RGB"))
        out_arr = np.asarray(output)
        np.testing.assert_array_equal(out_arr[:, 32], orig_rgb[:, 32])


# ── 5. Image sharpening ───────────────────────────────────────────────────────

from forensics_app.tools.sharpening import apply_unsharp_mask


class SharpeningTests(unittest.TestCase):

    def test_output_same_size_and_mode(self):
        image = make_rgb(64, 64)
        output = apply_unsharp_mask(image, radius=1.0, amount=1.0)
        self.assertEqual(output.size, image.size)
        self.assertEqual(output.mode, "RGB")

    def test_uniform_image_unchanged_after_sharpening(self):
        """A flat image has no edges/details to sharpen – result stays flat."""
        image = make_rgb(32, 32, (100, 100, 100))
        output = apply_unsharp_mask(image, radius=1.0, amount=1.0)
        arr_out = np.asarray(output)
        # All pixels should still be very close to 100
        self.assertTrue(np.all(np.abs(arr_out.astype(int) - 100) <= 1))

    def test_sharpening_increases_contrast_on_gradient(self):
        """Sharpening should make the edge in a gradient image more pronounced."""
        image = make_gradient_gray(64, 64)
        image_rgb = image.convert("RGB")
        output = apply_unsharp_mask(image_rgb, radius=1.0, amount=2.0)
        orig_arr = np.asarray(image_rgb).astype(int)
        out_arr = np.asarray(output).astype(int)
        # The maximum value in the sharpened image should be >= the original max
        self.assertGreaterEqual(out_arr.max(), orig_arr.max())

    def test_radius_zero_raises(self):
        with self.assertRaises(ValueError):
            apply_unsharp_mask(make_rgb(), radius=0.0, amount=1.0)

    def test_amount_zero_raises(self):
        with self.assertRaises(ValueError):
            apply_unsharp_mask(make_rgb(), radius=1.0, amount=0.0)

    def test_output_values_clipped_to_valid_range(self):
        """Sharpened output must never exceed 0–255."""
        image = make_gradient_gray(64, 64).convert("RGB")
        output = apply_unsharp_mask(image, radius=0.5, amount=5.0)
        arr = np.asarray(output)
        self.assertGreaterEqual(arr.min(), 0)
        self.assertLessEqual(arr.max(), 255)

    def test_grayscale_input_works(self):
        image = make_gray(32, 32, 100)
        image_rgb = image.convert("RGB")
        output = apply_unsharp_mask(image_rgb, radius=1.0, amount=1.0)
        self.assertEqual(output.size, image_rgb.size)


if __name__ == "__main__":
    unittest.main()
