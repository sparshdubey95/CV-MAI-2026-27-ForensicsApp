"""Comprehensive tests for Set 6: Face Detection, Eigenfaces, and Recognition."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
from skimage import data

from forensics_app.core import ImageDocument
from forensics_app.tools import build_tool_registry
from forensics_app.tools.eigenface_builder import (
    EigenfaceBuilderTool,
    get_active_model,
    load_eigenface_model,
    load_face_dataset,
    render_eigenfaces_montage,
    save_eigenface_model,
    set_active_model,
    train_eigenfaces,
)
from forensics_app.tools.face_detection import (
    FaceDetectionTool,
    detect_faces_yunet,
    find_yunet_model,
)
from forensics_app.tools.face_recognition import (
    FaceRecognitionTool,
    preprocess_face_image,
    project_test_face,
    recognize_face,
    render_recognition_result,
)


def create_synthetic_face(
    seed: int,
    size: tuple[int, int] = (32, 32),
    tone: int = 150,
) -> np.ndarray:
    """Generate a simple reproducible synthetic face pattern."""
    w, h = size
    img = np.full((h, w), tone, dtype=np.uint8)
    rng = np.random.RandomState(seed)

    # Eyes
    img[h // 4 : h // 3, w // 4 : w // 3] = 40
    img[h // 4 : h // 3, 2 * w // 3 : 3 * w // 4] = 40

    # Nose
    img[h // 2 : h // 2 + 3, w // 2 - 1 : w // 2 + 2] = 60

    # Mouth
    img[3 * h // 4 : 3 * h // 4 + 2, w // 3 : 2 * w // 3] = 30

    # Individual variation
    noise = rng.randint(-15, 15, size=(h, w)).astype(np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


class FaceDetectionTests(unittest.TestCase):
    def test_model_file_found(self) -> None:
        model_path = find_yunet_model()
        self.assertIsNotNone(model_path, "YuNet ONNX model file must be located.")
        self.assertTrue(Path(model_path).is_file())

    def test_detect_faces_on_astronaut(self) -> None:
        astro_rgb = data.astronaut()
        annotated, faces, details = detect_faces_yunet(
            astro_rgb,
            score_threshold=0.6,
            draw_landmarks=True,
        )
        self.assertIsInstance(annotated, Image.Image)
        self.assertGreaterEqual(len(faces), 1)
        self.assertEqual(details["Detected faces"], len(faces))

        first_face = faces[0]
        self.assertIn("bbox", first_face)
        self.assertIn("score", first_face)
        self.assertIn("landmarks", first_face)
        self.assertGreater(first_face["score"], 0.7)
        self.assertEqual(len(first_face["landmarks"]), 5)

    def test_zero_faces_on_blank_image(self) -> None:
        blank = np.zeros((200, 200, 3), dtype=np.uint8)
        annotated, faces, details = detect_faces_yunet(blank)
        self.assertEqual(len(faces), 0)
        self.assertEqual(details["Detected faces"], 0)
        self.assertEqual(details["Status"], "No faces detected")

    def test_missing_model_raises_filenotfound(self) -> None:
        blank = np.zeros((100, 100, 3), dtype=np.uint8)
        with self.assertRaises(FileNotFoundError):
            detect_faces_yunet(blank, model_path="completely_missing_weights.onnx")

    def test_invalid_input_arrays(self) -> None:
        with self.assertRaises(ValueError):
            detect_faces_yunet(np.array([]))

        with self.assertRaises(ValueError):
            detect_faces_yunet(np.zeros((50, 50, 5), dtype=np.uint8))

    def test_face_detection_tool_execution(self) -> None:
        tool = FaceDetectionTool()
        doc = ImageDocument()
        doc.current = Image.fromarray(data.astronaut())
        doc.original = doc.current.copy()

        result = tool.run(parent=None, document=doc)
        self.assertIsNotNone(result)
        self.assertIsNotNone(result.image)
        self.assertIn("Face detection complete", result.message)
        self.assertIn("Detected faces", result.details)


class EigenfaceBuilderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp_dir.name)

        # Create 2 persons with 3 synthetic images each
        for person_name, tone, base_seed in [("Alice_Smith", 180, 10), ("Bob_Jones", 120, 50)]:
            p_dir = self.folder / person_name
            p_dir.mkdir(parents=True, exist_ok=True)
            for i in range(3):
                img_arr = create_synthetic_face(seed=base_seed + i, size=(32, 32), tone=tone)
                Image.fromarray(img_arr).save(p_dir / f"face_{i}.png")

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_load_dataset_from_folder(self) -> None:
        X, labels, unique_names, thumbnails = load_face_dataset(
            source="folder",
            folder_path=self.folder,
            target_size=(32, 32),
        )
        self.assertEqual(X.shape, (6, 1024))
        self.assertEqual(len(labels), 6)
        self.assertEqual(set(unique_names), {"Alice Smith", "Bob Jones"})
        self.assertEqual(thumbnails.shape, (6, 32, 32))

    def test_pca_eigenface_training(self) -> None:
        X, labels, unique_names, thumbnails = load_face_dataset(
            source="folder",
            folder_path=self.folder,
            target_size=(32, 32),
        )
        mean_face, eigenfaces, coefficients, pca = train_eigenfaces(X, n_components=3)

        self.assertEqual(mean_face.shape, (1024,))
        self.assertEqual(eigenfaces.shape, (3, 1024))
        self.assertEqual(coefficients.shape, (6, 3))
        self.assertEqual(len(pca.explained_variance_ratio_), 3)

    def test_save_and_load_eigenface_model(self) -> None:
        X, labels, unique_names, thumbnails = load_face_dataset(
            source="folder",
            folder_path=self.folder,
            target_size=(32, 32),
        )
        mean_face, eigenfaces, coefficients, pca = train_eigenfaces(X, n_components=3)

        save_path = self.folder / "test_model.npz"
        save_eigenface_model(
            save_path,
            mean_face=mean_face,
            eigenfaces=eigenfaces,
            coefficients=coefficients,
            labels=labels,
            image_shape=(32, 32),
            thumbnails=thumbnails,
            explained_variance_ratio=pca.explained_variance_ratio_,
        )
        self.assertTrue(save_path.is_file())

        loaded = load_eigenface_model(save_path)
        np.testing.assert_allclose(mean_face, loaded["mean_face"])
        np.testing.assert_allclose(eigenfaces, loaded["eigenfaces"])
        np.testing.assert_allclose(coefficients, loaded["coefficients"])
        self.assertEqual(list(labels), list(loaded["labels"]))
        self.assertEqual(loaded["image_shape"], (32, 32))
        self.assertIsNotNone(loaded["thumbnails"])

    def test_render_eigenfaces_montage(self) -> None:
        mean_face = np.zeros(1024, dtype=np.float32)
        eigenfaces = np.random.randn(4, 1024).astype(np.float32)
        var_ratios = np.array([0.4, 0.3, 0.2, 0.1], dtype=np.float32)

        montage = render_eigenfaces_montage(
            mean_face=mean_face,
            eigenfaces=eigenfaces,
            image_shape=(32, 32),
            max_eigenfaces=4,
            explained_variance_ratio=var_ratios,
        )
        self.assertIsInstance(montage, Image.Image)
        self.assertGreater(montage.width, 100)
        self.assertGreater(montage.height, 100)


class FaceRecognitionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp_dir.name)

        # Build synthetic face dataset with 3 distinct individuals
        self.faces: dict[str, list[np.ndarray]] = {}
        for person_name, tone, base_seed in [
            ("Alice Smith", 200, 100),
            ("Bob Jones", 100, 200),
            ("Charlie Brown", 50, 300),
        ]:
            p_dir = self.folder / person_name.replace(" ", "_")
            p_dir.mkdir(parents=True, exist_ok=True)
            self.faces[person_name] = []
            for i in range(3):
                img_arr = create_synthetic_face(seed=base_seed + i, size=(32, 32), tone=tone)
                self.faces[person_name].append(img_arr)
                Image.fromarray(img_arr).save(p_dir / f"face_{i}.png")

        X, labels, unique_names, thumbnails = load_face_dataset(
            source="folder",
            folder_path=self.folder,
            target_size=(32, 32),
        )
        mean_face, eigenfaces, coefficients, pca = train_eigenfaces(X, n_components=4)
        self.model = {
            "mean_face": mean_face,
            "eigenfaces": eigenfaces,
            "coefficients": coefficients,
            "labels": labels,
            "image_shape": (32, 32),
            "thumbnails": thumbnails,
            "explained_variance_ratio": pca.explained_variance_ratio_,
        }
        set_active_model(self.model)

    def tearDown(self) -> None:
        set_active_model(None)
        self.temp_dir.cleanup()

    def test_preprocessing_and_projection(self) -> None:
        sample_img = self.faces["Alice Smith"][0]
        x_test, thumb = preprocess_face_image(sample_img, (32, 32))
        self.assertEqual(x_test.shape, (1024,))
        self.assertEqual(thumb.shape, (32, 32))

        c_test = project_test_face(x_test, self.model["mean_face"], self.model["eigenfaces"])
        self.assertEqual(c_test.shape, (4,))

    def test_recognize_face_matches_correct_identity(self) -> None:
        # Query with an exact Alice image
        query_face = self.faces["Alice Smith"][0]
        top_matches, c_test, thumb = recognize_face(query_face, self.model, top_k=3)

        self.assertEqual(len(top_matches), 3)
        # Rank 1 match should be Alice Smith with near zero distance
        self.assertEqual(top_matches[0]["name"], "Alice Smith")
        self.assertAlmostEqual(top_matches[0]["distance"], 0.0, places=3)

        # Distances should be ordered ascending
        self.assertLessEqual(top_matches[0]["distance"], top_matches[1]["distance"])
        self.assertLessEqual(top_matches[1]["distance"], top_matches[2]["distance"])

    def test_model_reuse_without_retraining(self) -> None:
        orig_coeffs = self.model["coefficients"].copy()
        orig_eigenfaces = self.model["eigenfaces"].copy()

        query_face = self.faces["Bob Jones"][0]
        recognize_face(query_face, self.model, top_k=2)

        np.testing.assert_array_equal(self.model["coefficients"], orig_coeffs)
        np.testing.assert_array_equal(self.model["eigenfaces"], orig_eigenfaces)

    def test_render_recognition_result(self) -> None:
        query_face = self.faces["Charlie Brown"][1]
        top_matches, _, thumb = recognize_face(query_face, self.model, top_k=3)
        result_img = render_recognition_result(thumb, top_matches, query_title="Test Query")

        self.assertIsInstance(result_img, Image.Image)
        self.assertGreater(result_img.width, 200)

    def test_recognition_tool_headless(self) -> None:
        tool = FaceRecognitionTool()
        doc = ImageDocument()
        doc.current = Image.fromarray(self.faces["Bob Jones"][1])

        result = tool.run(parent=None, document=doc)
        self.assertIsNotNone(result)
        self.assertIn("Recognition complete", result.message)
        self.assertIn("Bob Jones", result.message)
        self.assertEqual(result.details["Predicted identity"], "Bob Jones")


class Set6RegistrationTests(unittest.TestCase):
    def test_set6_tools_registered(self) -> None:
        registry = build_tool_registry()
        categories = dict(registry.categories())

        self.assertIn("Set 6: Face Detection & Recognition", categories)
        set6_tools = categories["Set 6: Face Detection & Recognition"]
        registered_ids = {tool.tool_id for tool in set6_tools}

        expected_ids = {"face_detection", "eigenface_builder", "face_recognition"}
        self.assertEqual(registered_ids, expected_ids)


if __name__ == "__main__":
    unittest.main()
