"""Focused tests for Set 8 architectures, passport loading, and UI guards."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import tensorflow as tf
from PIL import Image

from forensics_app.ml.cv_models import (
    BUILDERS,
    CLASS_NAMES,
    build_custom_cnn,
    build_inceptionv3_transfer,
)
from forensics_app.tools import build_tool_registry
from forensics_app.tools.base import ForensicsTool, ToolResult
from forensics_app.tools.cnn_core import (
    DEFAULT_OUTPUT_DIR,
    discover_passports,
    extract_feature_embedding,
    load_features_npz,
    load_model_from_passport,
    prepare_image,
)
from forensics_app.tools.cnn_tools import CNNClassifyTool, CNNFeatureProjectionTool, CNNTrainTool
from forensics_app.ui.main_window import MainWindow
from train_set8 import (
    _compute_tsne,
    _record_feature_subset_metrics,
    _save_feature_archive,
    _stratified_feature_subset,
    _stratified_indices,
)


class Set8ArchitectureTests(unittest.TestCase):
    def test_only_the_two_supported_builders_are_registered(self) -> None:
        self.assertEqual(set(BUILDERS), {"custom_cnn", "inceptionv3_transfer"})

    def test_custom_cnn_is_compact_and_exposes_a_128d_embedding(self) -> None:
        model = build_custom_cnn()
        self.assertGreaterEqual(model.count_params(), 60000)
        self.assertLessEqual(model.count_params(), 90000)
        self.assertEqual(
            [layer.filters for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D)],
            [32, 64, 96],
        )
        self.assertEqual(model.get_layer("features").units, 128)
        self.assertEqual(model.get_layer("predictions").units, 10)

    def test_inception_matches_colab_head_and_preprocessing(self) -> None:
        model = build_inceptionv3_transfer(weights=None)
        self.assertEqual(model.input_shape, (None, 32, 32, 3))
        self.assertEqual(model.count_params(), 22_329_898)
        self.assertEqual(model.get_layer("feature_vector").units, 256)
        self.assertEqual(model.get_layer("predictions").units, 10)
        self.assertEqual(model.get_layer("resize_75").output.shape[1:3], (75, 75))
        self.assertEqual(model.get_layer("inception_preprocess").scale, 2.0)
        self.assertEqual(model.get_layer("inception_preprocess").offset, -1.0)
        self.assertEqual(model.get_layer("dropout").rate, 0.3)
        base = model.get_layer("inception_v3")
        self.assertFalse(base.trainable)

    def test_available_colab_inception_weights_load(self) -> None:
        passport_path = DEFAULT_OUTPUT_DIR / "inceptionv3_cifar10_passport.json"
        weights_path = DEFAULT_OUTPUT_DIR / "inceptionv3_cifar10.weights.h5"
        if not passport_path.is_file() or not weights_path.is_file():
            self.skipTest("Colab Inception model artifacts are not available.")
        model, passport = load_model_from_passport(passport_path)
        self.assertEqual(model.get_layer("feature_vector").units, 256)
        self.assertEqual(passport["model_architecture"]["feature_dim"], 256)
        prediction = model.predict(prepare_image(Image.new("RGB", (32, 32)), "inceptionv3_transfer"), verbose=0)
        self.assertEqual(prediction.shape, (1, 10))


class Set8ArtifactTests(unittest.TestCase):
    def test_passport_discovery_is_independent_of_filename_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "inceptionv3_cifar10_passport.json"
            second = root / "custom_cnn_passport.json"
            first.write_text("{}", encoding="utf-8")
            second.write_text("{}", encoding="utf-8")
            self.assertEqual(discover_passports(root), [second, first])

    def test_legacy_passport_loads_weights_relative_to_its_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = build_custom_cnn()
            sample = np.full((1, 32, 32, 3), 127, dtype=np.float32)
            expected = source(sample, training=False).numpy()
            source.save_weights(root / "custom_fixture.weights.h5")
            passport_path = root / "custom_fixture_passport.json"
            passport_path.write_text(
                json.dumps({
                    "model_type": "custom_cnn",
                    "model_name": "custom_fixture",
                    "class_names": CLASS_NAMES,
                    "model_architecture": {"feature_layer": "features", "feature_dim": 128},
                }),
                encoding="utf-8",
            )
            rebuilt, _ = load_model_from_passport(passport_path)
            np.testing.assert_allclose(rebuilt(sample, training=False).numpy(), expected, rtol=1e-5, atol=1e-6)

    def test_feature_archive_validates_dynamic_embedding_dimensions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive_path = Path(directory) / "inceptionv3_cifar10_features.npz"
            np.savez(
                archive_path,
                features=np.zeros((20, 256), dtype=np.float32),
                labels=np.tile(np.arange(10), 2),
                indices=np.arange(20),
                class_names=np.asarray(CLASS_NAMES),
                tsne_2d=np.zeros((20, 2), dtype=np.float32),
                umap_2d=np.zeros((20, 2), dtype=np.float32),
                sample_images=np.zeros((20, 32, 32, 3), dtype=np.uint8),
            )
            data = load_features_npz(archive_path)
            self.assertEqual(data["features"].shape, (20, 256))
            self.assertEqual(data["umap_2d"].shape, (20, 2))

    def test_model_specific_image_scaling_and_custom_embedding(self) -> None:
        image = Image.new("RGBA", (90, 40), color=(255, 80, 20, 128))
        raw_input = prepare_image(image, "custom_cnn")
        inception_input = prepare_image(image, "inceptionv3_transfer")
        self.assertEqual(raw_input.shape, (1, 32, 32, 3))
        self.assertGreater(float(raw_input.max()), 1)
        self.assertGreaterEqual(float(inception_input.min()), 0)
        self.assertLessEqual(float(inception_input.max()), 1)
        model = build_custom_cnn()
        embedding = extract_feature_embedding(model, image, "custom_cnn")
        self.assertEqual(embedding.shape, (128,))

    def test_feature_archive_writer_extracts_stratified_embeddings(self) -> None:
        labels = np.repeat(np.arange(10), 2)
        images = np.zeros((len(labels), 32, 32, 3), dtype=np.uint8)
        model = build_custom_cnn()
        passport = {"model_architecture": {"feature_layer": "features"}}
        with tempfile.TemporaryDirectory() as directory:
            archive_path = Path(directory) / "custom_cnn_features.npz"
            with patch("train_set8._compute_tsne", return_value=np.zeros((20, 2), dtype=np.float32)):
                _save_feature_archive(
                    model, "custom_cnn", passport, images, labels, archive_path,
                    sample_count=20, batch_size=10, seed=42, skip_umap=True,
                )
            data = load_features_npz(archive_path)
            self.assertEqual(data["features"].shape, (20, 128))
            self.assertEqual(data["sample_images"].shape, (20, 32, 32, 3))
            self.assertEqual(data["umap_2d"].shape, (0, 2))

    def test_feature_extraction_records_missing_passport_metrics(self) -> None:
        model = build_custom_cnn()
        features = np.zeros((4, 128), dtype=np.float32)
        labels = np.array([0, 1, 2, 3], dtype=np.int32)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            passport_path = root / "model_passport.json"
            feature_path = root / "model_features.npz"
            passport = {"test_metrics": None}
            passport_path.write_text(json.dumps(passport), encoding="utf-8")
            np.savez(feature_path, features=features, labels=labels)

            _record_feature_subset_metrics(model, passport_path, passport, feature_path)

            persisted = json.loads(passport_path.read_text(encoding="utf-8"))
            self.assertEqual(persisted["test_metrics"]["evaluated_samples"], 4)
            self.assertEqual(
                persisted["test_metrics"]["evaluation_source"],
                "Stratified CIFAR-10 test subset",
            )
            self.assertGreaterEqual(persisted["test_metrics"]["loss"], 0)


class Set8PipelineTests(unittest.TestCase):
    def test_stratified_selection_and_projection(self) -> None:
        labels = np.repeat(np.arange(10), 20)
        selected = _stratified_indices(labels, 50, 42)
        self.assertEqual(len(selected), 50)
        self.assertEqual(set(labels[selected]), set(range(10)))
        images = np.zeros((len(labels), 32, 32, 3), dtype=np.uint8)
        subset_images, subset_labels, subset_indices = _stratified_feature_subset(images, labels, 40, 42)
        self.assertEqual(subset_images.shape[0], 40)
        self.assertEqual(len(subset_labels), len(subset_indices))
        projection = _compute_tsne(np.random.default_rng(42).normal(size=(30, 256)).astype(np.float32), 42)
        self.assertEqual(projection.shape, (30, 2))
        self.assertTrue(np.all(np.isfinite(projection)))


class Set8ToolGuardTests(unittest.TestCase):
    def test_tool_image_requirements_are_explicit(self) -> None:
        self.assertFalse(CNNTrainTool.requires_image)
        self.assertFalse(CNNTrainTool.modifies_image)
        self.assertFalse(CNNClassifyTool.requires_image)
        self.assertFalse(CNNClassifyTool.modifies_image)
        self.assertFalse(CNNFeatureProjectionTool.requires_image)
        self.assertFalse(CNNFeatureProjectionTool.modifies_image)

    def test_missing_image_uses_warning_instead_of_running_tool(self) -> None:
        class Document:
            is_loaded = False

        class Tool(ForensicsTool):
            tool_id = "requires_image_test"
            title = "Requires image"

            def run(self, parent, document):
                raise AssertionError("Tool must not run without an image.")

        window = MainWindow.__new__(MainWindow)
        window.root = object()
        window.document = Document()
        with patch("forensics_app.ui.main_window.messagebox.showwarning") as warning:
            window.run_tool(Tool())
        warning.assert_called_once_with(
            "No Image Loaded",
            "Please open an image first (File -> Open).",
            parent=window.root,
        )

    def test_non_modifying_tool_never_applies_its_result_image(self) -> None:
        class Document:
            is_loaded = False

            def apply(self, image):
                raise RuntimeError("Load an image before applying a result.")

        class Status:
            def set(self, value):
                self.value = value

        class Tool(ForensicsTool):
            tool_id = "no_image_write_test"
            title = "Non-modifying"
            requires_image = False
            modifies_image = False

            def run(self, parent, document):
                return ToolResult("completed", Image.new("RGB", (4, 4)))

        window = MainWindow.__new__(MainWindow)
        window.root = object()
        window.document = Document()
        window.status = Status()
        window._show_details = lambda _details: None
        window._refresh = lambda: None
        window.run_tool(Tool())
        self.assertEqual(window.status.value, "completed")

    def test_registered_tools_without_image_never_apply_visualizations(self) -> None:
        class Document:
            is_loaded = False

            def apply(self, image):
                raise RuntimeError("Load an image before applying a result.")

        class Status:
            def set(self, value):
                self.value = value

        window = MainWindow.__new__(MainWindow)
        window.root = object()
        window.document = Document()
        window.status = Status()
        window._show_details = lambda _details: None
        window._refresh = lambda: None
        non_image_tools = [tool for tool in build_tool_registry().all() if not tool.requires_image]
        for tool in non_image_tools:
            with self.subTest(tool=tool.tool_id):
                self.assertFalse(tool.modifies_image)
                tool.run = lambda _parent, _document: ToolResult("completed", Image.new("RGB", (4, 4)))
                window.run_tool(tool)
                self.assertEqual(window.status.value, "completed")
