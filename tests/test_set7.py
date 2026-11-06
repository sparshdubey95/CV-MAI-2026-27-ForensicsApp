"""Comprehensive tests for Set 7: Bag of Visual Words (BoVW) and Image Classification."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools import build_tool_registry
from forensics_app.tools.bovw_core import (
    build_codebook,
    classify_single_image,
    compute_bovw_histogram,
    compute_dataset_histograms,
    evaluate_bovw_classifier,
    extract_descriptors,
    load_bovw_bundle,
    render_codebook_visualization,
    render_confusion_matrix_figure,
    render_image_and_histogram,
    save_bovw_bundle,
    scan_dataset,
    set_active_bovw_model,
    set_active_bovw_split,
    split_dataset,
    train_bovw_classifier,
)
from forensics_app.tools.bovw_tools import (
    BoVWClassifierTool,
    BoVWCodebookTool,
    BoVWHistogramTool,
)


def create_pattern_image(category_seed: int, sample_seed: int, size: int = 120) -> Image.Image:
    """Create a textured synthetic image with distinctive local gradients for SIFT detection."""
    img = np.full((size, size), 128, dtype=np.uint8)
    rng = np.random.RandomState(category_seed * 100 + sample_seed)

    # Category-specific geometric patterns
    if category_seed == 0:  # butterfly: diagonal symmetric wings
        for i in range(15, size - 15, 12):
            img[i : i + 6, i : i + 6] = 240
            img[i : i + 6, size - i - 6 : size - i] = 240
            img[i + 2, :] = 30
    elif category_seed == 1:  # kangaroo: vertical gradient bars
        for j in range(10, size - 10, 16):
            img[:, j : j + 8] = 210
            img[j : j + 4, :] = 50
    else:  # dalmatian: dotted circle spots
        for _ in range(12):
            cx, cy = rng.randint(15, size - 15, size=2)
            rad = rng.randint(4, 9)
            y, x = np.ogrid[:size, :size]
            mask = (x - cx) ** 2 + (y - cy) ** 2 <= rad ** 2
            img[mask] = 20

    # Add localized noise/texture
    noise = rng.randint(-20, 20, size=(size, size)).astype(np.int16)
    final_arr = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(final_arr)


class BoVWCoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dataset_dir = Path(self.temp_dir.name)

        # Create 3 category folders: butterfly, kangaroo, dalmatian
        self.classes = ["butterfly", "kangaroo", "dalmatian"]
        for cat_idx, cat_name in enumerate(self.classes):
            cat_folder = self.dataset_dir / cat_name
            cat_folder.mkdir(parents=True, exist_ok=True)
            for s_idx in range(5):
                img = create_pattern_image(cat_idx, s_idx, size=100)
                img.save(cat_folder / f"sample_{s_idx}.jpg")

    def tearDown(self) -> None:
        set_active_bovw_model(None)
        set_active_bovw_split(None)
        self.temp_dir.cleanup()

    def test_scan_dataset(self) -> None:
        image_paths, labels, class_names = scan_dataset(self.dataset_dir)
        self.assertEqual(len(image_paths), 15)
        self.assertEqual(len(labels), 15)
        self.assertEqual(class_names, sorted(self.classes))
        self.assertEqual(set(labels), set(self.classes))

    def test_split_dataset(self) -> None:
        image_paths, labels, _ = scan_dataset(self.dataset_dir)
        train_p, train_l, test_p, test_l = split_dataset(
            image_paths, labels, test_size=0.33, random_state=42
        )
        self.assertEqual(len(train_p) + len(test_p), 15)
        self.assertEqual(len(train_l) + len(test_l), 15)
        # Ensure training and test sets are strictly disjoint
        self.assertEqual(len(set(train_p).intersection(set(test_p))), 0)
        # Check all 3 classes exist in train set
        self.assertEqual(set(train_l), set(self.classes))

    def test_extract_sift_descriptors(self) -> None:
        test_img = create_pattern_image(0, 0, size=120)
        desc = extract_descriptors(test_img, method="SIFT")
        self.assertEqual(desc.ndim, 2)
        self.assertEqual(desc.shape[1], 128)
        self.assertGreater(desc.shape[0], 0)

    def test_build_codebook_and_histogram(self) -> None:
        image_paths, labels, _ = scan_dataset(self.dataset_dir)
        train_p, _, _, _ = split_dataset(image_paths, labels, test_size=0.25, random_state=42)

        # Build codebook with K = 12 visual words for speed in unit test
        k = 12
        codebook, kmeans, total_desc = build_codebook(
            train_paths=train_p,
            k=k,
            method="SIFT",
            random_state=42,
        )
        self.assertEqual(codebook.shape, (k, 128))
        self.assertGreater(total_desc, k)

        # Test histogram computation
        test_img = create_pattern_image(1, 2)
        desc = extract_descriptors(test_img, method="SIFT")
        hist = compute_bovw_histogram(desc, codebook, normalize=True)

        self.assertEqual(hist.shape, (k,))
        self.assertAlmostEqual(float(np.sum(hist)), 1.0, places=4)
        self.assertTrue(np.all(hist >= 0.0))

    def test_train_and_evaluate_classifier(self) -> None:
        image_paths, labels, class_names = scan_dataset(self.dataset_dir)
        train_p, train_l, test_p, test_l = split_dataset(image_paths, labels, test_size=0.33, random_state=42)

        k = 10
        codebook, _, _ = build_codebook(train_p, k=k, method="SIFT", random_state=42)

        X_train, _ = compute_dataset_histograms(train_p, codebook, method="SIFT")
        clf, scaler = train_bovw_classifier(X_train, train_l, class_names, random_state=42)

        X_test, _ = compute_dataset_histograms(test_p, codebook, method="SIFT")
        accuracy, report, conf_matrix = evaluate_bovw_classifier(
            clf, scaler, X_test, test_l, class_names
        )

        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)
        self.assertEqual(conf_matrix.shape, (3, 3))
        self.assertIn("accuracy", report)

    def test_classify_single_image_with_probabilities(self) -> None:
        image_paths, labels, class_names = scan_dataset(self.dataset_dir)
        train_p, train_l, _, _ = split_dataset(image_paths, labels, test_size=0.25, random_state=42)

        k = 10
        codebook, _, _ = build_codebook(train_p, k=k, method="SIFT", random_state=42)
        X_train, _ = compute_dataset_histograms(train_p, codebook, method="SIFT")
        clf, scaler = train_bovw_classifier(X_train, train_l, class_names, random_state=42)

        test_img = create_pattern_image(0, 4)
        pred_class, probs, hist = classify_single_image(
            test_img, codebook, clf, scaler, class_names, method="SIFT"
        )

        self.assertIn(pred_class, class_names)
        self.assertEqual(len(probs), 3)
        self.assertAlmostEqual(sum(probs.values()), 1.0, places=3)
        for c in class_names:
            self.assertGreaterEqual(probs[c], 0.0)
            self.assertLessEqual(probs[c], 1.0)
        self.assertEqual(hist.shape, (k,))

    def test_save_and_load_bovw_bundle(self) -> None:
        codebook = np.random.randn(30, 128).astype(np.float32)
        class_names = ["butterfly", "dalmatian", "kangaroo"]
        save_path = self.dataset_dir / "test_model.bovw"

        save_bovw_bundle(
            save_path,
            codebook=codebook,
            class_names=class_names,
            method="SIFT",
        )
        self.assertTrue(save_path.is_file())

        loaded = load_bovw_bundle(save_path)
        np.testing.assert_allclose(codebook, loaded["codebook"])
        self.assertEqual(loaded["class_names"], class_names)
        self.assertEqual(loaded["k"], 30)

    def test_visualizations_render_images(self) -> None:
        codebook = np.random.randn(15, 128).astype(np.float32)
        vis_cb = render_codebook_visualization(codebook, self.classes, total_descriptors=150)
        self.assertIsInstance(vis_cb, Image.Image)

        hist = np.ones(15, dtype=np.float32) / 15.0
        vis_hist = render_image_and_histogram(
            image=create_pattern_image(0, 0),
            histogram=hist,
            class_name="butterfly",
            image_title="Test Butterfly",
            probabilities={"butterfly": 0.85, "kangaroo": 0.10, "dalmatian": 0.05},
            predicted_class="butterfly",
        )
        self.assertIsInstance(vis_hist, Image.Image)

        cm = np.array([[5, 0, 0], [1, 4, 0], [0, 1, 4]])
        vis_cm = render_confusion_matrix_figure(cm, self.classes, accuracy=0.87)
        self.assertIsInstance(vis_cm, Image.Image)


class Set7RegistrationTests(unittest.TestCase):
    def test_set7_tools_registered(self) -> None:
        registry = build_tool_registry()
        categories = dict(registry.categories())

        self.assertIn("Set 7: Bag of Visual Words & Classification", categories)
        set7_tools = categories["Set 7: Bag of Visual Words & Classification"]
        tool_ids = {tool.tool_id for tool in set7_tools}

        expected = {"bovw_codebook", "bovw_histogram", "bovw_classifier"}
        self.assertEqual(tool_ids, expected)


if __name__ == "__main__":
    unittest.main()
