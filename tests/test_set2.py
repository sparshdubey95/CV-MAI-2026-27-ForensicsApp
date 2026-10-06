import unittest

from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools.channel_split import channel_plane, split_all_planes
from forensics_app.tools.channel_swap import swap_channels
from forensics_app.tools.contrast_stretch import intensity_at_percentile, stretch_contrast
from forensics_app.tools.histogram import render_histogram
from forensics_app.tools.masking import apply_mask, threshold_mask


class ChannelSplitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.red = Image.new("RGB", (4, 3), (255, 0, 0))

    def test_red_plane_keeps_red_and_zeros_others(self) -> None:
        plane = channel_plane(self.red, "R")
        self.assertEqual(plane.getpixel((0, 0)), (255, 0, 0))
        self.assertEqual(channel_plane(self.red, "G").getpixel((0, 0)), (0, 0, 0))
        self.assertEqual(channel_plane(self.red, "B").getpixel((0, 0)), (0, 0, 0))

    def test_all_planes_are_side_by_side(self) -> None:
        strip = split_all_planes(self.red, gap=8)
        self.assertEqual(strip.size, (4 * 3 + 8 * 2, 3))

    def test_does_not_mutate_source(self) -> None:
        channel_plane(self.red, "R")
        self.assertEqual(self.red.getpixel((0, 0)), (255, 0, 0))


class ChannelSwapTests(unittest.TestCase):
    def test_bgr_turns_red_into_blue(self) -> None:
        red = Image.new("RGB", (2, 2), (255, 0, 0))
        swapped = swap_channels(red, "BGR")
        self.assertEqual(swapped.getpixel((0, 0)), (0, 0, 255))
        self.assertEqual(red.getpixel((0, 0)), (255, 0, 0))

    def test_rejects_invalid_order(self) -> None:
        with self.assertRaises(ValueError):
            swap_channels(Image.new("RGB", (1, 1), "white"), "RRB")


class MaskingTests(unittest.TestCase):
    def test_above_threshold_keeps_bright_pixels(self) -> None:
        image = Image.new("L", (2, 1), 0)
        image.putpixel((0, 0), 200)
        image.putpixel((1, 0), 10)
        mask = threshold_mask(image, 128, "above")
        self.assertEqual(mask.getpixel((0, 0)), 255)
        self.assertEqual(mask.getpixel((1, 0)), 0)

    def test_apply_mask_blacks_out_rejected_pixels(self) -> None:
        colour = Image.new("RGB", (2, 1), (10, 20, 30))
        colour.putpixel((0, 0), (1, 2, 3))
        mask = Image.new("L", (2, 1), 0)
        mask.putpixel((0, 0), 255)
        masked = apply_mask(colour, mask)
        self.assertEqual(masked.getpixel((0, 0)), (1, 2, 3))
        self.assertEqual(masked.getpixel((1, 0)), (0, 0, 0))


class HistogramTests(unittest.TestCase):
    def test_chart_has_expected_width(self) -> None:
        chart = render_histogram(Image.new("RGB", (5, 5), (12, 34, 56)))
        self.assertEqual(chart.size[0], 640)
        self.assertGreater(chart.size[1], 100)


class ContrastStretchTests(unittest.TestCase):
    def test_percentile_zero_and_hundred_are_min_and_max(self) -> None:
        histogram = [0] * 256
        histogram[40] = 3
        histogram[90] = 1
        self.assertEqual(intensity_at_percentile(histogram, 0), 40)
        self.assertEqual(intensity_at_percentile(histogram, 100), 90)

    def test_stretch_uses_full_range_on_low_contrast_gray(self) -> None:
        image = Image.new("L", (4, 1), 0)
        for x, value in enumerate((60, 70, 80, 90)):
            image.putpixel((x, 0), value)
        stretched, notes = stretch_contrast(image, 0, 100)
        self.assertEqual(stretched.getpixel((0, 0)), 0)
        self.assertEqual(stretched.getpixel((3, 0)), 255)
        self.assertEqual(notes["L in"], "60–90")

    def test_does_not_mutate_document_current(self) -> None:
        document = ImageDocument()
        document.current = Image.new("RGB", (3, 2), (40, 40, 40))
        stretch_contrast(document.current, 0, 100)
        self.assertEqual(document.current.getpixel((0, 0)), (40, 40, 40))


if __name__ == "__main__":
    unittest.main()
