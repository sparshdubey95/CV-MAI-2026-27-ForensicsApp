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
    extract_keypoints_and_descriptors,
    load_bovw_bundle,
    map_features_to_codebook,
    render_codebook_visualization,
    render_confusion_matrix_figure,
    render_image_and_histogram,
    render_test_image_histogram_mapping,
    save_bovw_bundle,
    scan_dataset,
    set_active_bovw_model,
    set_active_bovw_split,
    split_dataset,
    train_bovw_classifier,
)
from forensics_app.tools.dialogs import ask_bovw_codebook_parameters
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
        codebook, kmeans, total_desc, vw_patches = build_codebook(
            train_paths=train_p,
            k=k,
            method="SIFT",
            random_state=42,
        )
        self.assertEqual(codebook.shape, (k, 128))
        self.assertEqual(vw_patches.shape, (k, 32, 32))
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
        codebook, _, _, _ = build_codebook(train_p, k=k, method="SIFT", random_state=42)

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
        codebook, _, _, _ = build_codebook(train_p, k=k, method="SIFT", random_state=42)
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

    def test_ask_bovw_codebook_parameters_locked_defaults(self) -> None:
        """Section 7.1: UI locks visual words to 30 and method to SIFT."""
        res = ask_bovw_codebook_parameters(None, ["butterfly", "kangaroo", "dalmatian"], default_split=0.25)
        self.assertIsNotNone(res)
        k, method, split = res
        self.assertEqual(k, 30)
        self.assertEqual(method, "SIFT")
        self.assertEqual(split, 0.25)

    def test_extract_keypoints_and_descriptors(self) -> None:
        """Section 7.1: Clean keypoint and SIFT descriptor extraction."""
        test_img = create_pattern_image(0, 0, size=120)
        kps, descs = extract_keypoints_and_descriptors(test_img, method="SIFT", max_keypoints=100)
        self.assertGreater(len(kps), 0)
        self.assertEqual(len(kps), len(descs))
        self.assertEqual(descs.shape[1], 128)

    def test_map_features_to_codebook_explicit_mapping(self) -> None:
        """Section 7.2: Explicit feature-to-closest-centroid mapping and K-dim histogram."""
        test_img = create_pattern_image(1, 1, size=120)
        k = 30
        codebook = np.random.randn(k, 128).astype(np.float32)
        kps_coords, assigned, hist, kps = map_features_to_codebook(
            test_img, codebook, method="SIFT", max_keypoints=80
        )
        self.assertEqual(kps_coords.shape[0], len(assigned))
        self.assertEqual(kps_coords.shape[1], 2)
        self.assertEqual(len(hist), k)
        self.assertAlmostEqual(float(np.sum(hist)), 1.0, places=4)
        self.assertTrue(np.all(assigned >= 0) and np.all(assigned < k))

    def test_render_test_image_histogram_mapping(self) -> None:
        """Section 7.2: Visualization layout with test image on left, histogram on right, and mapping lines."""
        test_img = create_pattern_image(2, 0, size=120)
        k = 30
        n_pts = 25
        kps_coords = np.random.uniform(10, 110, size=(n_pts, 2)).astype(np.float32)
        assigned = np.random.randint(0, k, size=n_pts)
        counts = np.bincount(assigned, minlength=k).astype(np.float32)
        hist = counts / np.sum(counts)

        vis = render_test_image_histogram_mapping(
            image=test_img,
            kps_coords=kps_coords,
            assigned_centroids=assigned,
            histogram=hist,
            class_name="dalmatian",
            image_title="test_dalmatian.jpg",
        )
        self.assertIsInstance(vis, Image.Image)

    def test_bovw_classifier_tool_new_image_inference(self) -> None:
        """Section 7.3: Image classifier for new/external images displaying top probability."""
        image_paths, labels, class_names = scan_dataset(self.dataset_dir)
        train_p, train_l, _, _ = split_dataset(image_paths, labels, test_size=0.25, random_state=42)

        codebook, _, _, _ = build_codebook(train_p, k=15, method="SIFT", random_state=42)
        X_train, _ = compute_dataset_histograms(train_p, codebook, method="SIFT")
        clf, scaler = train_bovw_classifier(X_train, train_l, class_names, random_state=42)

        model = {
            "codebook": codebook,
            "classifier": clf,
            "scaler": scaler,
            "class_names": class_names,
            "method": "SIFT",
            "k": 15,
        }
        set_active_bovw_model(model)

        doc = ImageDocument()
        new_img = create_pattern_image(1, 4, size=100)
        new_path = self.dataset_dir / "query.jpg"
        new_img.save(new_path)
        doc.load(new_path)

        tool = BoVWClassifierTool()
        result = tool.run(None, doc)
        self.assertIsNotNone(result)
        self.assertIn("Top Probability Score", result.details)
        self.assertIn("p(", result.message)
        self.assertIsInstance(result.image, Image.Image)

    def test_section_7_2_and_7_3_histogram_pipeline_identical(self) -> None:
        """Section 7.3 must use the exact same feature extraction and mapping pipeline as 7.2."""
        image = create_pattern_image(0, 3, size=110)
        codebook = np.random.RandomState(42).randn(30, 128).astype(np.float32)

        kps_72, assigned_72, hist_72, _ = map_features_to_codebook(image, codebook, method="SIFT", max_keypoints=150)
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
        clf = SVC(probability=True).fit(np.eye(30)[:3], ["a", "b", "c"])
        scaler = StandardScaler().fit(np.eye(30)[:3])
        _, _, hist_classify = classify_single_image(
            image, codebook,
            classifier=clf,
            scaler=scaler,
            class_names=["a", "b", "c"],
            method="SIFT",
        )
        np.testing.assert_allclose(hist_72, hist_classify, rtol=1e-5, atol=1e-5)

    def test_bovw_histogram_tool_test_images_only(self) -> None:
        """Section 7.2: Histogram visualizer strictly restricted to test images."""
        image_paths, labels, class_names = scan_dataset(self.dataset_dir)
        train_p, train_l, test_p, test_l = split_dataset(image_paths, labels, test_size=0.33, random_state=42)

        codebook, _, _, _ = build_codebook(train_p, k=15, method="SIFT", random_state=42)
        split_info = {
            "train_paths": [str(p) for p in train_p],
            "train_labels": train_l,
            "test_paths": [str(p) for p in test_p],
            "test_labels": test_l,
            "dataset_root": str(self.dataset_dir),
        }
        model = {
            "codebook": codebook,
            "class_names": class_names,
            "method": "SIFT",
            "k": 15,
            "split_info": split_info,
        }
        set_active_bovw_model(model)
        set_active_bovw_split(split_info)

        tool = BoVWHistogramTool()
        doc = ImageDocument()
        result = tool.run(None, doc)

        self.assertIsNotNone(result)
        self.assertIn("Keypoints mapped", result.details)
        self.assertIn("Target test image", result.details)
        self.assertIsInstance(result.image, Image.Image)


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
