import unittest

from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools.channel_split import channel_plane, split_all_planes
from forensics_app.tools.channel_swap import swap_channels
from forensics_app.tools.contrast_stretch import stretch_band, stretch_contrast
from forensics_app.tools.histogram import _band_stats, render_histogram
from forensics_app.tools.masking import apply_texture, composite_jacket, create_jacket_mask


class ChannelSplitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.red = Image.new("RGB", (4, 3), (255, 0, 0))

    def test_red_plane_keeps_red_and_zeros_others(self) -> None:
        plane = channel_plane(self.red, "R")
        self.assertEqual(plane.mode, "L")
        self.assertEqual(plane.getpixel((0, 0)), 255)
        self.assertEqual(channel_plane(self.red, "G").getpixel((0, 0)), 0)
        self.assertEqual(channel_plane(self.red, "B").getpixel((0, 0)), 0)

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
    def setUp(self) -> None:
        self.base = Image.new("RGB", (3, 3), (255, 255, 255))
        self.jacket = Image.new("RGB", (3, 3), (0, 0, 0))
        self.jacket.putpixel((1, 1), (255, 255, 0))

    def test_create_jacket_mask_excludes_black_background(self) -> None:
        mask = create_jacket_mask(self.jacket)
        self.assertEqual(mask.getpixel((0, 0)), 0)
        self.assertEqual(mask.getpixel((1, 1)), 255)

    def test_composite_jacket_places_jacket_over_base(self) -> None:
        mask = create_jacket_mask(self.jacket)
        composited = composite_jacket(self.base, self.jacket, mask)
        self.assertEqual(composited.getpixel((0, 0)), (255, 255, 255))
        self.assertEqual(composited.getpixel((1, 1)), (255, 255, 0))

    def test_apply_texture_changes_only_jacket(self) -> None:
        mask = create_jacket_mask(self.jacket)
        texture = Image.new("RGB", (3, 3), (50, 100, 200))
        textured = apply_texture(self.base, texture, mask)
        self.assertEqual(textured.getpixel((0, 0)), (255, 255, 255))
        self.assertEqual(textured.getpixel((1, 1)), (50, 100, 200))


class HistogramTests(unittest.TestCase):
    def test_chart_has_expected_width(self) -> None:
        chart = render_histogram(Image.new("RGB", (5, 5), (12, 34, 56)))
        self.assertEqual(chart.size[0], 640)
        self.assertGreater(chart.size[1], 100)

    def test_band_stats_calculates_min_max_mean_peak(self) -> None:
        band = Image.new("L", (3, 1), 0)
        band.putpixel((0, 0), 10)
        band.putpixel((1, 0), 50)
        band.putpixel((2, 0), 50)
        stats = _band_stats(band)
        self.assertEqual(stats["min"], 10)
        self.assertEqual(stats["max"], 50)
        self.assertEqual(stats["peak"], 50)
        self.assertEqual(stats["mean"], round((10 + 50 + 50) / 3, 2))


class ContrastStretchTests(unittest.TestCase):
    def test_stretch_band_maps_min_to_zero_and_max_to_255(self) -> None:
        band = Image.new("L", (2, 1), 0)
        band.putpixel((0, 0), 50)
        band.putpixel((1, 0), 200)
        stretched, minimum, maximum = stretch_band(band)
        self.assertEqual(minimum, 50)
        self.assertEqual(maximum, 200)
        self.assertEqual(stretched.getpixel((0, 0)), 0)
        self.assertEqual(stretched.getpixel((1, 0)), 255)

    def test_stretch_uses_full_range_on_low_contrast_gray(self) -> None:
        image = Image.new("L", (4, 1), 0)
        for x, value in enumerate((60, 70, 80, 90)):
            image.putpixel((x, 0), value)
        stretched, notes = stretch_contrast(image)
        self.assertEqual(stretched.getpixel((0, 0)), 0)
        self.assertEqual(stretched.getpixel((3, 0)), 255)
        self.assertEqual(notes["L in"], "60–90")

    def test_does_not_mutate_document_current(self) -> None:
        document = ImageDocument()
        document.current = Image.new("RGB", (3, 2), (40, 40, 40))
        stretch_contrast(document.current)
        self.assertEqual(document.current.getpixel((0, 0)), (40, 40, 40))


if __name__ == "__main__":
    unittest.main()
