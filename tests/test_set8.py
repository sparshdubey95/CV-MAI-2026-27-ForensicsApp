"""Tests for Set 8: Image Classification using CNN."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
from sklearn.metrics import silhouette_score

from forensics_app.core import ImageDocument
from forensics_app.tools import (
    CNNFeatureProjectionTool,
    CNNInferenceTool,
    CNNTrainingTool,
    build_tool_registry,
)
from forensics_app.tools.cnn_core import (
    CIFAR10_CLASSES,
    CIFAR10_REQUIRED_FILES,
    DEFAULT_PREPROCESSING,
    TENSORFLOW_AVAILABLE,
    UMAP_AVAILABLE,
    Cifar10DataError,
    ModelPassport,
    PassportCompatibilityError,
    build_deeper_vgg_cnn,
    build_shallow_cnn,
    build_transfer_learning_model,
    compute_tsne_projection,
    compute_umap_projection,
    extract_cifar10_features,
    find_cifar10_batches_dir,
    find_default_cnn_passport,
    find_default_cnn_weights,
    find_default_features_path,
    find_sample_cifar_images,
    get_cifar10_data_dir,
    load_cifar10_data,
    load_cifar10_dataset,
    load_extracted_features,
    load_pretrained_cnn,
    preprocess_image_for_cifar,
    prepare_features_for_projection,
    render_classification_visualization,
    render_feature_projections,
    render_training_curves,
    run_cnn_inference,
    stratified_train_val_split,
    train_cnn_model,
    validate_training_hparams,
)
from forensics_app.tools.dialogs import (
    ask_cnn_inference_parameters,
    ask_cnn_projection_parameters,
    ask_cnn_training_parameters,
    show_cnn_passport_dialog,
)


class Set8RegistryTests(unittest.TestCase):
    def test_set8_tools_registered(self) -> None:
        registry = build_tool_registry()
        tool_ids = [t.tool_id for t in registry.all()]
        self.assertIn("cnn_training", tool_ids)
        self.assertIn("cnn_inference", tool_ids)
        self.assertIn("cnn_feature_projection", tool_ids)

    def test_set8_category_and_attributes(self) -> None:
        registry = build_tool_registry()
        categories = dict(registry.categories())
        self.assertIn("Set 8: Image Classification using CNN", categories)
        tools = categories["Set 8: Image Classification using CNN"]
        self.assertEqual(len(tools), 3)
        for tool in tools:
            self.assertEqual(tool.category, "Set 8: Image Classification using CNN")
            self.assertFalse(tool.requires_image)
            self.assertTrue(len(tool.title) > 0)
            self.assertTrue(len(tool.description) > 0)


class Set81DatasetAndSplitTests(unittest.TestCase):
    def test_load_official_local_cifar10(self) -> None:
        data = load_cifar10_dataset(sample_limit=100, seed=42, normalize=True)
        self.assertEqual(data.x_train.shape[1:], (32, 32, 3))
        self.assertEqual(data.x_test.shape[1:], (32, 32, 3))
        self.assertEqual(len(data.class_names), 10)
        self.assertEqual(data.class_names, list(CIFAR10_CLASSES))
        self.assertEqual(len(np.unique(data.y_train)), 10)
        self.assertEqual(len(np.unique(data.y_test)), 10)
        self.assertEqual(data.x_train.shape[0], 100)
        self.assertEqual(data.x_test.shape[0], 10000)
        self.assertGreaterEqual(float(data.x_train.min()), 0.0)
        self.assertLessEqual(float(data.x_train.max()), 1.0)
        self.assertTrue(Path(data.source_path).is_dir())
        self.assertIn("pickle", data.provenance.lower())
        self.assertNotIn("synthetic", data.provenance.lower())
        for name in CIFAR10_REQUIRED_FILES:
            self.assertTrue((Path(data.source_path) / name).is_file())

    def test_load_cifar10_data_wrapper(self) -> None:
        (x_tr, y_tr), (x_te, y_te) = load_cifar10_data(sample_limit=50, seed=0)
        self.assertEqual(x_tr.shape, (50, 32, 32, 3))
        self.assertEqual(len(np.unique(y_tr)), 10)
        self.assertEqual(x_te.shape[0], 10000)

    def test_missing_dataset_directory_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(Cifar10DataError) as ctx:
                load_cifar10_dataset(dataset_dir=tmp)
            self.assertIn("missing required files", str(ctx.exception).lower())

    def test_download_and_synthetic_flags_rejected(self) -> None:
        with self.assertRaises(Cifar10DataError):
            load_cifar10_data(sample_limit=20, allow_download=True)
        with self.assertRaises(Cifar10DataError):
            load_cifar10_data(sample_limit=20, use_synthetic_fallback=True)

    def test_stratified_split_keeps_all_classes(self) -> None:
        data = load_cifar10_dataset(sample_limit=100, seed=7, normalize=False)
        x_tr, y_tr, x_val, y_val = stratified_train_val_split(data.x_train, data.y_train, val_fraction=0.1, seed=7)
        self.assertEqual(len(np.unique(y_tr)), 10)
        self.assertEqual(len(np.unique(y_val)), 10)
        self.assertEqual(len(x_tr) + len(x_val), 100)
        self.assertTrue(np.isclose(len(x_val) / 100.0, 0.1, atol=0.02) or len(x_val) == 10)

    def test_hparams_validation(self) -> None:
        validate_training_hparams(100, 2, 16, 0.1)
        with self.assertRaises(ValueError):
            validate_training_hparams(10, 2, 16, 0.1)
        with self.assertRaises(ValueError):
            validate_training_hparams(100, 0, 16, 0.1)
        with self.assertRaises(ValueError):
            validate_training_hparams(40, 1, 64, 0.1)

    def test_find_batches_dir_is_project_relative(self) -> None:
        found = find_cifar10_batches_dir()
        self.assertTrue(found.is_dir())
        self.assertEqual(found.name, "cifar-10-batches-py")


class Set81TrainingAndPassportTests(unittest.TestCase):
    @unittest.skipUnless(TENSORFLOW_AVAILABLE, "TensorFlow required for CNN tests")
    def test_shallow_cnn_architecture(self) -> None:
        model = build_shallow_cnn(input_shape=(32, 32, 3), num_classes=10)
        self.assertEqual(model.input_shape, (None, 32, 32, 3))
        self.assertEqual(model.output_shape, (None, 10))
        layer_names = [layer.name for layer in model.layers]
        self.assertTrue(any("conv" in n for n in layer_names))
        self.assertTrue(any("pool" in n for n in layer_names))
        self.assertTrue(any("flatten" in n for n in layer_names))
        self.assertTrue(any("dense" in n for n in layer_names))

    @unittest.skipUnless(TENSORFLOW_AVAILABLE, "TensorFlow required for CNN tests")
    def test_deeper_vgg_architecture(self) -> None:
        model = build_deeper_vgg_cnn(input_shape=(32, 32, 3), num_classes=10)
        self.assertEqual(model.input_shape, (None, 32, 32, 3))
        self.assertEqual(model.output_shape, (None, 10))
        self.assertGreater(model.count_params(), 50000)

    @unittest.skipUnless(TENSORFLOW_AVAILABLE, "TensorFlow required for CNN tests")
    def test_transfer_learning_inceptionv3(self) -> None:
        try:
            model = build_transfer_learning_model(input_shape=(32, 32, 3), num_classes=10)
        except RuntimeError as exc:
            self.assertIn("ImageNet", str(exc))
            self.skipTest(f"InceptionV3 ImageNet weights unavailable: {exc}")
        self.assertEqual(model.input_shape, (None, 32, 32, 3))
        self.assertEqual(model.output_shape, (None, 10))
        self.assertIn("inception", model.name.lower())
        nested = [layer for layer in model.layers if layer.__class__.__name__ == "Functional" or "inception" in layer.name.lower()]
        self.assertTrue(any(not getattr(layer, "trainable", True) for layer in model.layers))
        dummy = np.zeros((1, 32, 32, 3), dtype=np.float32)
        out = model.predict(dummy, verbose=0)
        self.assertEqual(out.shape, (1, 10))
        del nested

    def test_passport_dataclass_methods(self) -> None:
        passport = ModelPassport(
            model_name="test_model",
            dataset="CIFAR-10",
            sample_size=500,
            class_names=list(CIFAR10_CLASSES),
            preprocessing_instructions=dict(DEFAULT_PREPROCESSING),
            architecture={"type": "shallow"},
            training_history={"loss": [1.0, 0.5], "accuracy": [0.6, 0.9]},
            metrics={"final_accuracy": 0.9},
            created_at="2026-11-21T12:00:00",
            weights_path="dummy.weights.h5",
            dataset_source="data/cifar-10-batches-py",
            train_samples=450,
            val_samples=50,
            test_samples=10000,
            random_seed=42,
        )
        reconstructed = ModelPassport.from_dict(passport.to_dict())
        self.assertEqual(reconstructed.model_name, "test_model")
        self.assertEqual(reconstructed.sample_size, 500)
        self.assertEqual(len(reconstructed.class_names), 10)
        self.assertEqual(reconstructed.train_samples, 450)

    def test_render_training_curves(self) -> None:
        history = {
            "loss": [2.3, 1.5, 0.8, 0.4],
            "val_loss": [2.4, 1.7, 1.0, 0.6],
            "accuracy": [0.2, 0.5, 0.7, 0.9],
            "val_accuracy": [0.15, 0.45, 0.65, 0.85],
        }
        img = render_training_curves(history)
        self.assertIsInstance(img, Image.Image)
        self.assertGreater(img.size[0], 200)
        self.assertGreater(img.size[1], 200)

    def _train_tiny(self, architecture: str, tmp_path: Path) -> tuple[object, dict, Path, Path]:
        weights_file = tmp_path / f"{architecture}.weights.h5"
        passport_file = tmp_path / f"{architecture}_passport.json"
        return train_cnn_model(
            architecture=architecture,
            sample_size=40,
            epochs=1,
            batch_size=8,
            weights_path=weights_file,
            passport_path=passport_file,
            random_seed=42,
            max_test_eval=40,
        )

    @unittest.skipUnless(TENSORFLOW_AVAILABLE, "TensorFlow required for CNN tests")
    def test_train_shallow_and_vgg_save_passport(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            for arch in ("shallow", "vgg"):
                model, passport, w_path, p_path = self._train_tiny(arch, tmp_path / arch)
                self.assertTrue(w_path.is_file())
                self.assertTrue(p_path.is_file())
                saved = json.loads(p_path.read_text(encoding="utf-8"))
                self.assertEqual(saved["dataset"], "CIFAR-10")
                self.assertEqual(saved["sample_size"], 40)
                self.assertEqual(saved["class_names"], list(CIFAR10_CLASSES))
                self.assertGreaterEqual(saved["train_samples"], 10)
                self.assertGreaterEqual(saved["val_samples"], 10)
                self.assertEqual(saved["test_samples"], 10000)
                self.assertEqual(
                    saved["test_metrics"]["evaluated_on"],
                    "official CIFAR-10 test_batch (full 10,000 images)",
                )
                self.assertIn("cifar-10-batches-py", saved["dataset_source"].replace("\\", "/"))
                self.assertIn("loss", saved["training_history"])
                self.assertIn("val_accuracy", saved["training_history"])
                self.assertIn("test_accuracy", saved["metrics"])
                self.assertIn("accuracy", saved["test_metrics"])
                self.assertEqual(saved["random_seed"], 42)
                self.assertEqual(saved["epochs"], 1)
                self.assertEqual(saved["batch_size"], 8)
                self.assertEqual(saved["weights_filename"], w_path.name)
                self.assertEqual(len(saved["confusion_matrix"]), 10)
                self.assertIn("macro avg", saved["classification_report"])
                reloaded, loaded_p = load_pretrained_cnn(w_path, p_path)
                self.assertEqual(loaded_p["model_name"], saved["model_name"])
                dummy = np.zeros((1, 32, 32, 3), dtype=np.float32)
                pred = reloaded.predict(dummy, verbose=0)
                self.assertEqual(pred.shape, (1, 10))
                del model

    @unittest.skipUnless(TENSORFLOW_AVAILABLE, "TensorFlow required for CNN tests")
    def test_train_transfer_inception_small(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            try:
                _model, passport, w_path, p_path = self._train_tiny("transfer", tmp_path)
            except RuntimeError as exc:
                self.assertIn("ImageNet", str(exc))
                self.skipTest(str(exc))
            self.assertEqual(passport["architecture"]["type"], "transfer")
            self.assertEqual(passport["architecture"]["base_name"], "InceptionV3")
            self.assertEqual(passport["pretrained_base_weights"], "imagenet")
            self.assertIn("inception_v3", passport["preprocessing_instructions"]["normalization"])
            reloaded, _ = load_pretrained_cnn(w_path, p_path)
            pred = reloaded.predict(np.zeros((1, 32, 32, 3), dtype=np.float32), verbose=0)
            self.assertEqual(pred.shape, (1, 10))

    @unittest.skipUnless(TENSORFLOW_AVAILABLE, "TensorFlow required for CNN tests")
    def test_cnn_training_tool_headless(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            w_file = tmp_path / "tool_model.weights.h5"
            p_file = tmp_path / "tool_passport.json"
            tool = CNNTrainingTool()
            doc = ImageDocument()
            mock_params = {
                "architecture": "shallow",
                "sample_size": 40,
                "epochs": 1,
                "batch_size": 8,
                "optimizer": "sgd",
                "learning_rate": 0.01,
                "weights_path": str(w_file),
                "passport_path": str(p_file),
                "dataset_dir": None,
                "random_seed": 1,
                "max_test_eval": 40,
            }
            with patch("forensics_app.tools.cnn_tools.ask_cnn_training_parameters", return_value=mock_params):
                res = tool.run(parent=None, document=doc)
            self.assertIsNotNone(res)
            self.assertIn("Successfully trained", res.message)
            self.assertIn("Test Acc", res.message)
            self.assertIsInstance(res.image, Image.Image)
            self.assertTrue(w_file.is_file())
            self.assertTrue(p_file.is_file())


class Set82LoadModelAndInferenceTests(unittest.TestCase):
    def test_preprocess_image_for_cifar(self) -> None:
        raw_img = Image.new("RGB", (200, 150), color=(128, 64, 32))
        processed = preprocess_image_for_cifar(raw_img, target_size=(32, 32))
        self.assertEqual(processed.shape, (1, 32, 32, 3))
        self.assertEqual(processed.dtype, np.float32)
        self.assertGreaterEqual(processed.min(), 0.0)
        self.assertLessEqual(processed.max(), 1.0)

    def test_preprocess_rejects_unknown_normalization(self) -> None:
        raw_img = Image.new("RGB", (32, 32), color=(10, 20, 30))
        with self.assertRaises(ValueError):
            preprocess_image_for_cifar(raw_img, normalization="unknown_scheme")

    @unittest.skipUnless(TENSORFLOW_AVAILABLE, "TensorFlow required for CNN tests")
    def test_load_pretrained_model_and_inference(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            w_file = tmp_path / "temp.weights.h5"
            p_file = tmp_path / "temp_passport.json"
            train_cnn_model(
                architecture="shallow",
                sample_size=40,
                epochs=1,
                batch_size=8,
                weights_path=w_file,
                passport_path=p_file,
                max_test_eval=40,
            )
            model, passport = load_pretrained_cnn(w_file, p_file)
            self.assertIsNotNone(model)
            self.assertEqual(len(passport["class_names"]), 10)
            test_img = Image.new("RGB", (64, 64), color=(50, 150, 200))
            result = run_cnn_inference(model, test_img, passport)
            self.assertIn("predicted_class", result)
            self.assertIn("confidence", result)
            self.assertEqual(len(result["probabilities"]), 10)
            self.assertAlmostEqual(sum(result["probabilities"].values()), 1.0, places=4)
            self.assertIn("32", result["note"])

    @unittest.skipUnless(TENSORFLOW_AVAILABLE, "TensorFlow required for CNN tests")
    def test_mismatched_weights_and_passport_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            w_a = tmp_path / "a.weights.h5"
            p_a = tmp_path / "a_passport.json"
            w_b = tmp_path / "b.weights.h5"
            p_b = tmp_path / "b_passport.json"
            train_cnn_model(architecture="shallow", sample_size=40, epochs=1, batch_size=8, weights_path=w_a, passport_path=p_a, max_test_eval=40)
            train_cnn_model(architecture="vgg", sample_size=40, epochs=1, batch_size=8, weights_path=w_b, passport_path=p_b, max_test_eval=40)
            with self.assertRaises(PassportCompatibilityError):
                load_pretrained_cnn(w_a, p_b)

    @unittest.skipUnless(TENSORFLOW_AVAILABLE, "TensorFlow required for CNN tests")
    def test_missing_weights_error(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_pretrained_cnn("Z:/definitely/missing/cnn.weights.h5", "Z:/definitely/missing/passport.json")

    def test_render_classification_visualization(self) -> None:
        test_img = Image.new("RGB", (32, 32), color=(100, 200, 50))
        probs = {c: 0.1 for c in CIFAR10_CLASSES}
        probs["frog"] = 0.7
        probs["cat"] = 0.05
        vis = render_classification_visualization(
            test_img,
            predicted_class="frog",
            confidence=0.70,
            probabilities=probs,
            passport={"model_name": "TestModel", "sample_size": 1000},
        )
        self.assertIsInstance(vis, Image.Image)
        self.assertGreater(vis.size[0], 400)

    @unittest.skipUnless(TENSORFLOW_AVAILABLE, "TensorFlow required for CNN tests")
    def test_cnn_inference_tool_headless(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            w_file = tmp_path / "inf.weights.h5"
            p_file = tmp_path / "inf_passport.json"
            train_cnn_model(architecture="shallow", sample_size=40, epochs=1, batch_size=8, weights_path=w_file, passport_path=p_file, max_test_eval=40)
            tool = CNNInferenceTool()
            doc = ImageDocument()
            samples = find_sample_cifar_images()
            if samples and os.path.isfile(samples[0]):
                doc.load(samples[0])
            else:
                doc.original = Image.new("RGB", (32, 32), color=(255, 0, 0))
                doc.current = doc.original
            mock_params = {
                "weights_path": str(w_file),
                "passport_path": str(p_file),
                "use_current_image": True,
                "custom_image_path": None,
                "show_passport": False,
            }
            with patch("forensics_app.tools.cnn_tools.ask_cnn_inference_parameters", return_value=mock_params):
                res = tool.run(parent=None, document=doc)
            self.assertIsNotNone(res)
            self.assertIn("Classification Inference Result", res.message)
            self.assertIn("32", res.message)
            self.assertIn("predicted_class", res.details)
            self.assertEqual(len(res.details["probabilities"]), 10)

    @unittest.skipUnless(TENSORFLOW_AVAILABLE, "TensorFlow required for CNN tests")
    def test_inference_tool_requires_image(self) -> None:
        tool = CNNInferenceTool()
        doc = ImageDocument()
        mock_params = {
            "weights_path": find_default_cnn_weights(),
            "passport_path": find_default_cnn_passport(),
            "use_current_image": False,
            "custom_image_path": None,
            "show_passport": False,
        }
        with patch("forensics_app.tools.cnn_tools.ask_cnn_inference_parameters", return_value=mock_params):
            res = tool.run(parent=None, document=doc)
        self.assertIsNone(res)


class Set83FeatureProjectionTests(unittest.TestCase):
    def test_genuine_feature_file_alignment_and_projection_comparison(self) -> None:
        feature_path = find_default_features_path()
        self.assertIsNotNone(feature_path)
        bundle = load_extracted_features(feature_path)
        data = load_cifar10_dataset(sample_limit=None, seed=42, normalize=True)
        self.assertTrue(np.array_equal(data.y_test[bundle["indices"]], bundle["labels"]))
        self.assertEqual(np.bincount(bundle["labels"], minlength=10).tolist(), [50] * 10)

        prepared, labels, keep = prepare_features_for_projection(
            bundle["features"], bundle["labels"], random_state=42
        )
        self.assertIsNotNone(labels)
        self.assertEqual(np.bincount(labels, minlength=10).tolist(), [int(keep.sum() / 10)] * 10)
        self.assertTrue(np.isfinite(prepared).all())

        raw = bundle["features"][keep]
        raw_labels = bundle["labels"][keep]
        baseline_tsne = compute_tsne_projection(raw, perplexity=40.0, n_iter=300, prepared=True)
        improved_tsne = compute_tsne_projection(
            prepared, perplexity=40.0, n_iter=1000, random_state=42, prepared=True
        )
        self.assertEqual(improved_tsne.shape, (len(labels), 2))
        self.assertTrue(np.isfinite(improved_tsne).all())
        self.assertTrue(np.isfinite(silhouette_score(baseline_tsne, raw_labels)))
        self.assertTrue(np.isfinite(silhouette_score(improved_tsne, labels)))

        if UMAP_AVAILABLE:
            baseline_umap = compute_umap_projection(
                raw, n_neighbors=15, min_dist=0.1, random_state=42, prepared=True
            )
            improved_umap = compute_umap_projection(
                prepared, n_neighbors=15, min_dist=0.1, metric="euclidean", random_state=42, prepared=True
            )
            self.assertEqual(improved_umap.shape, (len(labels), 2))
            self.assertTrue(np.isfinite(improved_umap).all())
            self.assertTrue(np.isfinite(silhouette_score(baseline_umap, raw_labels)))
            self.assertTrue(np.isfinite(silhouette_score(improved_umap, labels)))

    @unittest.skipUnless(TENSORFLOW_AVAILABLE, "TensorFlow required for CNN tests")
    def test_extract_requires_trained_weights(self) -> None:
        with self.assertRaises(RuntimeError) as ctx:
            extract_cifar10_features(model=None, samples_per_class=2, require_trained_weights=True)
        self.assertIn("trained", str(ctx.exception).lower())

    @unittest.skipUnless(TENSORFLOW_AVAILABLE, "TensorFlow required for CNN tests")
    def test_extract_and_save_feature_vectors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            w_file = tmp_path / "feat.weights.h5"
            p_file = tmp_path / "feat_passport.json"
            feat_file = tmp_path / "test_features.npz"
            train_cnn_model(architecture="shallow", sample_size=40, epochs=1, batch_size=8, weights_path=w_file, passport_path=p_file, max_test_eval=40)
            bundle = extract_cifar10_features(
                samples_per_class=5,
                save_path=feat_file,
                weights_path=w_file,
                passport_path=p_file,
            )
            self.assertTrue(feat_file.is_file())
            self.assertEqual(bundle["features"].shape[0], 50)
            self.assertEqual(bundle["labels"].shape[0], 50)
            self.assertEqual(len(np.unique(bundle["labels"])), 10)
            official = load_cifar10_dataset(sample_limit=None, seed=42, normalize=True)
            self.assertTrue(np.array_equal(official.y_test[bundle["indices"]], bundle["labels"]))
            self.assertTrue(bundle["extractor_layer"])
            self.assertIn("cifar-10-batches-py", str(bundle["dataset_source"]).replace("\\", "/"))
            loaded = load_extracted_features(feat_file)
            self.assertEqual(loaded["features"].shape, bundle["features"].shape)
            self.assertEqual(loaded["extractor_layer"], bundle["extractor_layer"])
            self.assertEqual(loaded["dataset_source"], bundle["dataset_source"])

    def test_compute_tsne_projection(self) -> None:
        rng = np.random.RandomState(42)
        feats = rng.randn(60, 30).astype(np.float32)
        coords_2d = compute_tsne_projection(feats, perplexity=15.0, n_iter=250)
        self.assertEqual(coords_2d.shape, (60, 2))
        self.assertFalse(np.isnan(coords_2d).any())

    @unittest.skipUnless(UMAP_AVAILABLE, "umap-learn required for UMAP tests")
    def test_compute_umap_projection(self) -> None:
        rng = np.random.RandomState(42)
        feats = rng.randn(60, 30).astype(np.float32)
        coords_2d = compute_umap_projection(feats, n_neighbors=10, min_dist=0.1)
        self.assertEqual(coords_2d.shape, (60, 2))
        self.assertFalse(np.isnan(coords_2d).any())

    def test_umap_missing_dependency_message(self) -> None:
        if UMAP_AVAILABLE:
            self.skipTest("umap-learn is installed")
        with self.assertRaises(RuntimeError) as ctx:
            compute_umap_projection(np.random.randn(20, 8).astype(np.float32))
        self.assertIn("UMAP", str(ctx.exception))

    def test_render_feature_projections_visualization(self) -> None:
        rng = np.random.RandomState(42)
        vis = render_feature_projections(
            rng.randn(100, 2),
            rng.randn(100, 2),
            np.array([i % 10 for i in range(100)]),
            class_names=list(CIFAR10_CLASSES),
        )
        self.assertIsInstance(vis, Image.Image)
        self.assertGreater(vis.size[0], 600)

    @unittest.skipUnless(TENSORFLOW_AVAILABLE, "TensorFlow required for CNN tests")
    def test_cnn_feature_projection_tool_headless(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            feat_file = Path(tmp_dir) / "proj.npz"
            rng = np.random.RandomState(0)
            np.savez_compressed(
                feat_file,
                features=rng.randn(40, 16).astype(np.float32),
                labels=np.array([i % 10 for i in range(40)]),
                class_names=np.array(CIFAR10_CLASSES),
                indices=np.arange(40),
                dataset_source=np.array("test"),
                model_name=np.array("unit"),
                weights_path=np.array("unit.weights.h5"),
                extractor_layer=np.array("dense_feature"),
            )
            tool = CNNFeatureProjectionTool()
            mock_params = {
                "action": "load",
                "features_path": str(feat_file),
                "samples_per_class": 5,
                "perplexity": 10.0,
                "umap_neighbors": 5,
                "method": "tsne",
                "weights_path": None,
                "passport_path": None,
            }
            with patch("forensics_app.tools.cnn_tools.ask_cnn_projection_parameters", return_value=mock_params):
                res = tool.run(parent=None, document=ImageDocument())
            self.assertIsNotNone(res)
            self.assertIn("Feature Projection Complete", res.message)
            self.assertIn("t-SNE", res.details["methods"])


class Set8DialogHeadlessTests(unittest.TestCase):
    def test_ask_cnn_training_parameters_headless(self) -> None:
        params = ask_cnn_training_parameters(None)
        self.assertIsNotNone(params)
        self.assertIn("architecture", params)
        self.assertIn("sample_size", params)
        self.assertIn("dataset_dir", params)
        self.assertIn("random_seed", params)

    def test_ask_cnn_inference_parameters_headless(self) -> None:
        params = ask_cnn_inference_parameters(None)
        self.assertIsNotNone(params)
        self.assertIn("weights_path", params)
        self.assertIn("use_current_image", params)
        self.assertIn("show_passport", params)

    def test_ask_cnn_projection_parameters_headless(self) -> None:
        params = ask_cnn_projection_parameters(None)
        self.assertIsNotNone(params)
        self.assertIn("action", params)
        self.assertIn("weights_path", params)
        self.assertIn("method", params)

    def test_show_cnn_passport_dialog_headless(self) -> None:
        passport = {
            "model_name": "TestModel",
            "dataset": "CIFAR-10",
            "sample_size": 1000,
            "class_names": list(CIFAR10_CLASSES),
            "preprocessing_instructions": {"description": "32x32"},
            "metrics": {"final_accuracy": 0.95},
        }
        show_cnn_passport_dialog(None, passport)


if __name__ == "__main__":
    unittest.main()
