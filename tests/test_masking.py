"""Tests for image masking and jacket texture replacement."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools.masking import (
    apply_texture,
    composite_jacket,
    create_jacket_mask,
    MaskingTool,
)


class MaskingImageProcessingTests(unittest.TestCase):
    def setUp(self) -> None:
        # Base image: 4x4 image, representing model (e.g. white background with dark shirt)
        self.base_image = Image.new("RGB", (4, 4), (240, 240, 240))
        for y in range(2, 4):
            for x in range(1, 3):
                self.base_image.putpixel((x, y), (20, 20, 20))

        # Jacket image: 4x4 image, pure black background (0,0,0) and yellow jacket (255,255,0)
        self.jacket_image = Image.new("RGB", (4, 4), (0, 0, 0))
        self.jacket_pixels = [(1, 1), (1, 2), (2, 1), (2, 2)]
        for pt in self.jacket_pixels:
            self.jacket_image.putpixel(pt, (255, 255, 0))

        # Texture image 1: full-color red texture
        self.texture1 = Image.new("RGB", (4, 4), (200, 30, 30))

        # Texture image 2: full-color blue texture
        self.texture2 = Image.new("RGB", (4, 4), (30, 50, 220))

    def test_black_background_pixels_excluded_from_jacket_mask(self) -> None:
        """Requirement 3: Test that black background pixels are excluded from the jacket mask."""
        mask = create_jacket_mask(self.jacket_image)
        self.assertEqual(mask.size, (4, 4))
        self.assertEqual(mask.mode, "L")

        # Background pixels (pure black) must have value 0 (excluded)
        for y in range(4):
            for x in range(4):
                if (x, y) not in self.jacket_pixels:
                    self.assertEqual(
                        mask.getpixel((x, y)),
                        0,
                        f"Pixel at ({x}, {y}) should be excluded (0)",
                    )

        # Jacket pixels must have value 255 (selected)
        for pt in self.jacket_pixels:
            self.assertEqual(
                mask.getpixel(pt),
                255,
                f"Pixel at {pt} should be selected (255)",
            )

    def test_jacket_pixels_placed_over_base_image(self) -> None:
        """Requirement 4: Test that jacket pixels are placed over the base image."""
        mask = create_jacket_mask(self.jacket_image)
        composited = composite_jacket(self.base_image, self.jacket_image, mask)

        self.assertEqual(composited.size, (4, 4))
        # Jacket pixels placed over base image
        for pt in self.jacket_pixels:
            self.assertEqual(
                composited.getpixel(pt),
                (255, 255, 0),
                f"Jacket pixel at {pt} should be yellow",
            )

        # Background pixels must match the base image
        for y in range(4):
            for x in range(4):
                if (x, y) not in self.jacket_pixels:
                    self.assertEqual(
                        composited.getpixel((x, y)),
                        self.base_image.getpixel((x, y)),
                        f"Non-jacket pixel at ({x}, {y}) must remain unchanged",
                    )

    def test_texture_replacement_changes_only_jacket_region(self) -> None:
        """Requirement 5: Test that texture replacement changes only the jacket region."""
        mask = create_jacket_mask(self.jacket_image)
        textured = apply_texture(self.base_image, self.texture1, mask)

        # Jacket region should have the texture color
        for pt in self.jacket_pixels:
            self.assertEqual(
                textured.getpixel(pt),
                (200, 30, 30),
                f"Pixel at {pt} should have texture 1 color",
            )

    def test_pixels_outside_mask_remain_unchanged(self) -> None:
        """Requirement 6: Test that pixels outside the mask remain unchanged."""
        mask = create_jacket_mask(self.jacket_image)
        textured = apply_texture(self.base_image, self.texture1, mask)

        # All pixels outside the mask must be strictly identical to base image
        for y in range(4):
            for x in range(4):
                if (x, y) not in self.jacket_pixels:
                    self.assertEqual(
                        textured.getpixel((x, y)),
                        self.base_image.getpixel((x, y)),
                        f"Pixel at ({x}, {y}) outside mask must match base image",
                    )

    def test_subsequent_texture_replaces_previous_texture_without_accumulation(self) -> None:
        """Feature B: Switching textures replaces previous texture inside jacket area."""
        mask = create_jacket_mask(self.jacket_image)
        first_textured = apply_texture(self.base_image, self.texture1, mask)
        second_textured = apply_texture(self.base_image, self.texture2, mask)

        # Inside jacket area must be exactly texture 2, not a mixture/accumulation of texture 1 and 2
        for pt in self.jacket_pixels:
            self.assertEqual(second_textured.getpixel(pt), (30, 50, 220))

        # Outside jacket area must still match original base image
        for y in range(4):
            for x in range(4):
                if (x, y) not in self.jacket_pixels:
                    self.assertEqual(
                        second_textured.getpixel((x, y)),
                        self.base_image.getpixel((x, y)),
                    )

    def test_invalid_dimensions_rejected_by_composite_jacket(self) -> None:
        """Requirement 7: Test that invalid dimensions are rejected when compositing jacket."""
        wrong_jacket = Image.new("RGB", (6, 6), (255, 255, 0))
        mask = Image.new("L", (4, 4), 255)
        with self.assertRaises(ValueError):
            composite_jacket(self.base_image, wrong_jacket, mask)

    def test_invalid_dimensions_rejected_by_apply_texture(self) -> None:
        """Requirement 7: Test that invalid dimensions are rejected when applying texture."""
        wrong_texture = Image.new("RGB", (8, 8), (100, 100, 100))
        mask = Image.new("L", (4, 4), 255)
        with self.assertRaises(ValueError):
            apply_texture(self.base_image, wrong_texture, mask)



class MaskingToolStateAndWorkflowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)

        # Create physical test files
        self.model_path = Path(self.temp_dir.name) / "model.png"
        self.jacket_path = Path(self.temp_dir.name) / "jacket.png"
        self.texture1_path = Path(self.temp_dir.name) / "texture1.png"
        self.texture2_path = Path(self.temp_dir.name) / "texture2.png"
        self.mismatched_path = Path(self.temp_dir.name) / "mismatched.png"
        self.corrupt_path = Path(self.temp_dir.name) / "corrupt.png"

        model = Image.new("RGB", (4, 4), (240, 240, 240))
        model.save(self.model_path)

        jacket = Image.new("RGB", (4, 4), (0, 0, 0))
        jacket.putpixel((1, 1), (255, 255, 0))
        jacket.putpixel((1, 2), (255, 255, 0))
        jacket.save(self.jacket_path)

        Image.new("RGB", (4, 4), (200, 30, 30)).save(self.texture1_path)
        Image.new("RGB", (4, 4), (30, 50, 220)).save(self.texture2_path)
        Image.new("RGB", (8, 8), (100, 100, 100)).save(self.mismatched_path)

        with open(self.corrupt_path, "wb") as f:
            f.write(b"NOT_A_VALID_IMAGE_DATA")

        self.document = ImageDocument()
        self.document.load(self.model_path)
        self.tool = MaskingTool()

    def test_texture_replacement_cannot_run_before_jacket_applied(self) -> None:
        """Requirement 8: Test that texture replacement cannot run before jacket has been applied."""
        self.assertFalse(self.tool.is_jacket_applied(self.document))

        # Run tool with "texture" option selected
        with patch("forensics_app.tools.masking.ask_choice", return_value="texture"), \
             patch("forensics_app.tools.masking.show_error") as mock_error:
            result = self.tool.run(None, self.document)
            self.assertIsNone(result)
            mock_error.assert_called_once()
            self.assertIn("put the jacket on the model", mock_error.call_args[0][2].lower())

    def test_put_jacket_then_apply_texture_success(self) -> None:
        """Test happy path: applying jacket then applying texture."""
        # 1. Apply jacket
        with patch("forensics_app.tools.masking.ask_choice", return_value="jacket"), \
             patch("forensics_app.tools.masking.ask_image_file", return_value=str(self.jacket_path)):
            result_jacket = self.tool.run(None, self.document)
            self.assertIsNotNone(result_jacket)
            self.assertIsNotNone(result_jacket.image)
            self.document.apply(result_jacket.image)

        self.assertTrue(self.tool.is_jacket_applied(self.document))

        # 2. Apply texture
        with patch("forensics_app.tools.masking.ask_choice", return_value="texture"), \
             patch("forensics_app.tools.masking.ask_image_file", return_value=str(self.texture1_path)):
            result_texture = self.tool.run(None, self.document)
            self.assertIsNotNone(result_texture)
            self.assertIsNotNone(result_texture.image)
            self.document.apply(result_texture.image)

        self.assertTrue(self.tool.is_jacket_applied(self.document))
        self.assertEqual(self.document.current.getpixel((1, 1)), (200, 30, 30))

    def test_texture_replacement_cannot_run_on_unrelated_image(self) -> None:
        """State management: changing to an unrelated image invalidates jacket state."""
        # Apply jacket on first document
        with patch("forensics_app.tools.masking.ask_choice", return_value="jacket"), \
             patch("forensics_app.tools.masking.ask_image_file", return_value=str(self.jacket_path)):
            result = self.tool.run(None, self.document)
            self.document.apply(result.image)

        self.assertTrue(self.tool.is_jacket_applied(self.document))

        # Load unrelated image
        unrelated_path = Path(self.temp_dir.name) / "unrelated.png"
        Image.new("RGB", (4, 4), (50, 50, 50)).save(unrelated_path)
        self.document.load(unrelated_path)

        self.assertFalse(self.tool.is_jacket_applied(self.document))

    def test_reject_jacket_with_incompatible_dimensions(self) -> None:
        """Requirement 7: Tool shows error and returns None when jacket dimensions do not match."""
        original_current = self.document.current.copy()
        with patch("forensics_app.tools.masking.ask_choice", return_value="jacket"), \
             patch("forensics_app.tools.masking.ask_image_file", return_value=str(self.mismatched_path)), \
             patch("forensics_app.tools.masking.show_error") as mock_error:
            result = self.tool.run(None, self.document)
            self.assertIsNone(result)
            mock_error.assert_called_once()
            self.assertIn("dimensions", mock_error.call_args[0][2].lower())
            self.assertEqual(self.document.current, original_current)



    def test_resize_texture_with_incompatible_dimensions(self) -> None:
        """Textures with different dimensions should be resized automatically."""

        with patch(
                "forensics_app.tools.masking.ask_choice",
                return_value="jacket",
        ), patch(
            "forensics_app.tools.masking.ask_image_file",
            return_value=str(self.jacket_path),
        ):
            result_jacket = self.tool.run(None, self.document)

        self.assertIsNotNone(result_jacket)
        self.document.apply(result_jacket.image)

        with patch(
                "forensics_app.tools.masking.ask_choice",
                return_value="texture",
        ), patch(
            "forensics_app.tools.masking.ask_image_file",
            return_value=str(self.mismatched_path),
        ), patch(
            "forensics_app.tools.masking.show_error",
        ) as mock_error:
            result_texture = self.tool.run(None, self.document)

        self.assertIsNotNone(result_texture)
        self.assertEqual(result_texture.image.size, (4, 4))
        mock_error.assert_not_called()

        self.assertEqual(
            result_texture.image.getpixel((1, 1)),
            (100, 100, 100),
        )
        self.assertEqual(
            result_texture.image.getpixel((0, 0)),
            (240, 240, 240),
        )

        self.document.apply(result_texture.image)
        self.assertTrue(self.tool.is_jacket_applied(self.document))


    def test_corrupted_file_handling(self) -> None:
        """Requirement 4: Non-image file displays error and leaves image unchanged."""
        original = self.document.current.copy()

        with (
            patch(
                "forensics_app.tools.masking.ask_choice",
                return_value="jacket",
            ),
            patch(
                "forensics_app.tools.masking.ask_image_file",
                return_value=str(self.corrupt_path),
            ),
            patch("forensics_app.tools.masking.show_error") as mock_error,
        ):
            result = self.tool.run(None, self.document)

            self.assertIsNone(result)
            mock_error.assert_called_once()
            self.assertEqual(self.document.current, original)

    def test_cancellation_leaves_image_unchanged_and_shows_no_error(self) -> None:
        """Requirement 4: Cancellation does not show error and does not modify image."""
        original = self.document.current.copy()

        # 1. User cancels the choice dialog.
        with (
            patch(
                "forensics_app.tools.masking.ask_choice",
                return_value=None,
            ),
            patch("forensics_app.tools.masking.show_error") as mock_error,
        ):
            result = self.tool.run(None, self.document)

            self.assertIsNone(result)
            mock_error.assert_not_called()
            self.assertEqual(self.document.current, original)

        # 2. User cancels file selection.
        with (
            patch(
                "forensics_app.tools.masking.ask_choice",
                return_value="jacket",
            ),
            patch(
                "forensics_app.tools.masking.ask_image_file",
                return_value=None,
            ),
            patch("forensics_app.tools.masking.show_error") as mock_error,
        ):
            result = self.tool.run(None, self.document)

            self.assertIsNone(result)
            mock_error.assert_not_called()
            self.assertEqual(self.document.current, original)



    def test_undo_redo_workflow(self) -> None:
        """Requirement 9: Check that existing undo/redo workflow still works."""
        base_pixels = self.document.current.getpixel((1, 1))

        # 1. Put jacket on model
        with patch("forensics_app.tools.masking.ask_choice", return_value="jacket"), \
             patch("forensics_app.tools.masking.ask_image_file", return_value=str(self.jacket_path)):
            result1 = self.tool.run(None, self.document)
            self.document.apply(result1.image)

        self.assertEqual(self.document.current.getpixel((1, 1)), (255, 255, 0))
        self.assertTrue(self.tool.is_jacket_applied(self.document))

        # 2. Apply texture 1
        with patch("forensics_app.tools.masking.ask_choice", return_value="texture"), \
             patch("forensics_app.tools.masking.ask_image_file", return_value=str(self.texture1_path)):
            result2 = self.tool.run(None, self.document)
            self.document.apply(result2.image)

        self.assertEqual(self.document.current.getpixel((1, 1)), (200, 30, 30))
        self.assertTrue(self.tool.is_jacket_applied(self.document))

        # 3. Apply texture 2
        with patch("forensics_app.tools.masking.ask_choice", return_value="texture"), \
             patch("forensics_app.tools.masking.ask_image_file", return_value=str(self.texture2_path)):
            result3 = self.tool.run(None, self.document)
            self.document.apply(result3.image)

        self.assertEqual(self.document.current.getpixel((1, 1)), (30, 50, 220))
        self.assertTrue(self.tool.is_jacket_applied(self.document))

        # 4. Undo back to texture 1
        self.assertTrue(self.document.undo())
        self.assertEqual(self.document.current.getpixel((1, 1)), (200, 30, 30))
        self.assertTrue(self.tool.is_jacket_applied(self.document))

        # 5. Undo back to jacket
        self.assertTrue(self.document.undo())
        self.assertEqual(self.document.current.getpixel((1, 1)), (255, 255, 0))
        self.assertTrue(self.tool.is_jacket_applied(self.document))

        # 6. Undo back to base model (before jacket)
        self.assertTrue(self.document.undo())
        self.assertEqual(self.document.current.getpixel((1, 1)), base_pixels)
        self.assertFalse(self.tool.is_jacket_applied(self.document))

        # 7. Redo back to jacket
        self.assertTrue(self.document.redo())
        self.assertEqual(self.document.current.getpixel((1, 1)), (255, 255, 0))
        self.assertTrue(self.tool.is_jacket_applied(self.document))

        # 8. Redo back to texture 1
        self.assertTrue(self.document.redo())
        self.assertEqual(self.document.current.getpixel((1, 1)), (200, 30, 30))
        self.assertTrue(self.tool.is_jacket_applied(self.document))

        # 9. Redo back to texture 2
        self.assertTrue(self.document.redo())
        self.assertEqual(self.document.current.getpixel((1, 1)), (30, 50, 220))
        self.assertTrue(self.tool.is_jacket_applied(self.document))


if __name__ == "__main__":
    unittest.main()
