"""Unit and integration tests for Set 11: U-Net Binary Food Segmentation."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

from forensics_app.tools import build_tool_registry
from forensics_app.tools.unet_core import (
    UNetCheckpointError,
    binarize_food_mask,
    compute_binary_dice,
    evaluate_unet_on_test_set,
    find_default_tray_dataset,
    find_tray_dataset_dirs,
    load_resized_pairs,
    load_unet_checkpoint,
    pair_images_and_masks,
    render_segmentation_visualization,
    render_test_set_grid,
    render_unet_training_report,
    segment_single_image,
    train_unet_model,
)
from forensics_app.tools.unet_tools import UNetSegmentationTool, UNetTrainingTool


def _create_synthetic_tray_dataset(
    base_dir: Path,
    n_train: int = 6,
    n_test: int = 2,
    include_unmatched: bool = False,
) -> Path:
    """Create a minimal synthetic TrayDataset with XTrain/yTrain/XTest/yTest."""
    root = base_dir / "TrayDataset"
    for sub in ("XTrain", "yTrain", "XTest", "yTest"):
        (root / sub).mkdir(parents=True, exist_ok=True)

    def _write_pair(img_dir: Path, mask_dir: Path, stem: str, food_class: int) -> None:
        img = np.zeros((48, 64, 3), dtype=np.uint8)
        img[12:36, 16:48] = [200, 120, 60]
        Image.fromarray(img).save(img_dir / f"{stem}.jpg")

        mask = np.ones((48, 64), dtype=np.uint8)  # 1 = tray (not food)
        mask[0:8, :] = 0  # background (not food)
        mask[12:36, 16:48] = food_class  # >= 5 -> food
        Image.fromarray(mask).save(mask_dir / f"{stem}.png")

    for i in range(n_train):
        _write_pair(root / "XTrain", root / "yTrain", f"train_{i:02d}", food_class=5 + (i % 5))

    if include_unmatched:
        Image.new("RGB", (32, 32), (10, 10, 10)).save(root / "XTrain" / "orphan_img.jpg")
        Image.new("L", (32, 32), 0).save(root / "yTrain" / "orphan_mask.png")

    for i in range(n_test):
        _write_pair(root / "XTest", root / "yTest", f"test_{i:02d}", food_class=7)

    return root


class Set11RegistryTests(unittest.TestCase):
    def test_unet_tools_registered(self) -> None:
        registry = build_tool_registry()
        cat_map = dict(registry.categories())
        self.assertIn("Set 11: U-Net Segmentation", cat_map)
        tools = cat_map["Set 11: U-Net Segmentation"]
        self.assertEqual([t.tool_id for t in tools], ["unet_training", "unet_segmentation"])
        self.assertIsInstance(tools[0], UNetTrainingTool)
        self.assertIsInstance(tools[1], UNetSegmentationTool)


class Set11DatasetAndMaskTests(unittest.TestCase):
    def test_bundled_tray_dataset_discovered_and_paired(self) -> None:
        ds = find_default_tray_dataset()
        if ds is None:
            self.skipTest("Bundled TrayDataset not present.")
        dirs = find_tray_dataset_dirs(ds, require_train=True, require_test=True)
        train_pairs, train_info = pair_images_and_masks(dirs["XTrain"], dirs["yTrain"])
        test_pairs, test_info = pair_images_and_masks(dirs["XTest"], dirs["yTest"])
        self.assertGreater(len(train_pairs), 100)
        self.assertGreater(len(test_pairs), 0)
        self.assertEqual(train_info["matched_count"], len(train_pairs))
        self.assertEqual(test_info["matched_count"], len(test_pairs))

    def test_unmatched_pairs_warning_and_zero_match_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ds = _create_synthetic_tray_dataset(Path(tmp), n_train=3, include_unmatched=True)
            dirs = find_tray_dataset_dirs(ds)
            pairs, info = pair_images_and_masks(dirs["XTrain"], dirs["yTrain"])
            self.assertEqual(len(pairs), 3)
            self.assertEqual(info["unmatched_images"], ["orphan_img.jpg"])
            self.assertEqual(info["unmatched_masks"], ["orphan_mask.png"])
            self.assertIn("Skipped 1 unmatched", info["warning"])

            # Directory with zero matching stems
            bad_x = Path(tmp) / "bad_x"
            bad_y = Path(tmp) / "bad_y"
            bad_x.mkdir()
            bad_y.mkdir()
            Image.new("RGB", (16, 16)).save(bad_x / "a.jpg")
            Image.new("L", (16, 16)).save(bad_y / "b.png")
            with self.assertRaises(ValueError):
                pair_images_and_masks(bad_x, bad_y)

    def test_binary_food_mapping_and_nearest_resize(self) -> None:
        raw = np.array([[0, 1, 2, 3, 4], [5, 6, 10, 20, 42]], dtype=np.uint8)
        binary = binarize_food_mask(raw)
        np.testing.assert_array_equal(binary[0], [0, 0, 0, 0, 0])
        np.testing.assert_array_equal(binary[1], [1, 1, 1, 1, 1])

        with tempfile.TemporaryDirectory() as tmp:
            ds = _create_synthetic_tray_dataset(Path(tmp), n_train=2)
            dirs = find_tray_dataset_dirs(ds)
            pairs, _ = pair_images_and_masks(dirs["XTrain"], dirs["yTrain"])
            X, Y = load_resized_pairs(pairs, height=32, width=48)
            self.assertEqual(X.shape, (2, 32, 48, 3))
            self.assertEqual(Y.shape, (2, 32, 48))
            self.assertTrue(set(np.unique(Y)).issubset({0, 1}))


class Set11TrainLoadEvaluateTests(unittest.TestCase):
    def test_train_save_load_evaluate_and_visualize(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            ds = _create_synthetic_tray_dataset(tmp_path, n_train=6, n_test=2)
            ckpt_path = tmp_path / "unet_test.pt"

            summary = train_unet_model(
                dataset_root=ds,
                checkpoint_path=ckpt_path,
                epochs=2,
                batch_size=2,
                val_fraction=0.2,
                input_height=32,
                input_width=48,
                base_channels=8,
                device="cpu",
            )
            self.assertTrue(ckpt_path.is_file())
            self.assertEqual(summary["epochs_trained"], 2)
            self.assertGreater(summary["val_samples"], 0)
            report_img = render_unet_training_report(summary)
            self.assertEqual(report_img.mode, "RGB")

            model, config, metadata, dev = load_unet_checkpoint(ckpt_path, device="cpu")
            self.assertEqual(dev, "cpu")
            self.assertEqual(config["input_height"], 32)
            self.assertEqual(config["input_width"], 48)
            self.assertIn("best_val_dice", metadata)

            # Single image WITHOUT ground truth -> dice_score must be None
            test_img_path = ds / "XTest" / "test_00.jpg"
            test_gt_path = ds / "yTest" / "test_00.png"
            seg_no_gt = segment_single_image(model, test_img_path, config, device="cpu")
            self.assertIsNone(seg_no_gt["dice_score"])
            self.assertIsNone(seg_no_gt["gt_mask"])
            vis_no_gt = render_segmentation_visualization(seg_no_gt)
            self.assertEqual(vis_no_gt.mode, "RGB")

            # Single image WITH ground truth -> dice_score must be float
            seg_with_gt = segment_single_image(
                model, test_img_path, config, device="cpu", gt_mask_path=test_gt_path
            )
            self.assertIsInstance(seg_with_gt["dice_score"], float)
            self.assertIsNotNone(seg_with_gt["gt_mask"])
            vis_with_gt = render_segmentation_visualization(seg_with_gt)
            self.assertEqual(vis_with_gt.mode, "RGB")

            # Test set evaluation
            test_eval = evaluate_unet_on_test_set(model, config, device="cpu", dataset_root=ds)
            self.assertEqual(test_eval["n_test"], 2)
            grid_img = render_test_set_grid(test_eval)
            self.assertEqual(grid_img.mode, "RGB")

    def test_missing_corrupted_and_incompatible_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)

            # Missing
            with self.assertRaises(UNetCheckpointError):
                load_unet_checkpoint(tmp_path / "does_not_exist.pt")

            # Corrupted bytes
            corrupt = tmp_path / "corrupt.pt"
            corrupt.write_bytes(b"corrupted-checkpoint-data")
            with self.assertRaises(UNetCheckpointError):
                load_unet_checkpoint(corrupt)

            # Incompatible dict (missing keys / mismatched state_dict)
            incompat = tmp_path / "incompatible.pt"
            torch.save({"foo": "bar"}, incompat)
            with self.assertRaises(UNetCheckpointError):
                load_unet_checkpoint(incompat)

            bad_weights = tmp_path / "bad_weights.pt"
            torch.save(
                {
                    "model_state_dict": {"wrong.weight": torch.zeros(1)},
                    "config": {
                        "in_ch": 3,
                        "out_ch": 1,
                        "base": 8,
                        "input_height": 32,
                        "input_width": 48,
                    },
                },
                bad_weights,
            )
            with self.assertRaises(UNetCheckpointError):
                load_unet_checkpoint(bad_weights)

    def test_compute_binary_dice_exact(self) -> None:
        p = np.array([[0, 1, 1], [0, 1, 1]], dtype=np.uint8)
        g = np.array([[0, 0, 1], [0, 0, 1]], dtype=np.uint8)
        # |P cap G| = 2, |P| = 4, |G| = 2 -> 4 / 6 = 0.6667
        self.assertAlmostEqual(compute_binary_dice(p, g), 4.0 / 6.0, places=4)


if __name__ == "__main__":
    unittest.main()
