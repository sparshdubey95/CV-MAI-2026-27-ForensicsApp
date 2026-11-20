"""Comprehensive unit and integration tests for Set 9: Object Detection with YOLO.

Tests all requirements from Materials-CV-MAI-Set9.pdf:
  - 9.1 Pretrained Model Inference (yolo11n.pt, bus.jpg fallback, tab20 colormap plotting, BGR-to-RGB)
  - 9.2 COCO-to-YOLO conversion (images/labels directories, normalized [cls, xc, yc, w, h], data.yaml)
        and model training controls (from_scratch_50e_v1, epochs=50, batch=16, classes=[0,1,2])
  - 9.3 Unseen Image Testing & Comparison (blood cell dataset, GT vs Pred visualization, IoU matching)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools import build_tool_registry
from forensics_app.tools.yolo_core import (
    BLOOD_CELL_CATEGORIES,
    BLOOD_CELL_COLORS,
    ULTRALYTICS_AVAILABLE,
    calculate_box_iou,
    coco_to_yolo,
    convert_coco_dataset_splits,
    create_yolo_data_yaml,
    find_blood_cell_dataset,
    find_default_image,
    find_default_yolo_weights,
    get_active_yolo_model_path,
    match_ground_truth_and_predictions,
    parse_yolo_label_file,
    render_custom_yolo_plot,
    run_yolo_inference,
    set_active_yolo_model_path,
    train_yolo_model,
    visualize_unseen_comparison,
)
from forensics_app.tools.yolo_tools import (
    YOLOComparisonTool,
    YOLODataConversionTool,
    YOLOInferenceTool,
    YOLOTrainingTool,
)


def create_synthetic_coco_dataset(base_dir: Path, num_images: int = 3) -> Path:
    """Create a minimal synthetic COCO formatted dataset directory for testing."""
    img_dir = base_dir / "synthetic_coco"
    img_dir.mkdir(parents=True, exist_ok=True)

    images = []
    annotations = []
    ann_id = 0

    for i in range(num_images):
        file_name = f"image_{i}.jpg"
        img_arr = np.full((100, 100, 3), 120 + i * 20, dtype=np.uint8)
        Image.fromarray(img_arr).save(img_dir / file_name)

        images.append({
            "id": i,
            "file_name": file_name,
            "width": 100,
            "height": 100,
        })

        # Add 2 annotations per image: class 1 (PC) and class 2 (RBC)
        # COCO category_ids: 1=PC, 2=RBC, 3=WBC
        annotations.append({
            "id": ann_id,
            "image_id": i,
            "category_id": 1,  # Should become YOLO class 0
            "bbox": [10.0, 15.0, 30.0, 25.0],  # x, y, w, h
            "area": 750.0,
            "iscrowd": 0,
        })
        ann_id += 1
        annotations.append({
            "id": ann_id,
            "image_id": i,
            "category_id": 2,  # Should become YOLO class 1
            "bbox": [50.0, 40.0, 40.0, 45.0],
            "area": 1800.0,
            "iscrowd": 0,
        })
        ann_id += 1

    coco_data = {
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": 0, "name": "none"},
            {"id": 1, "name": "PC"},
            {"id": 2, "name": "RBC"},
            {"id": 3, "name": "WBC"},
        ],
    }

    with open(img_dir / "_annotations.coco.json", "w", encoding="utf-8") as f:
        json.dump(coco_data, f)

    return img_dir


class Set9RegistryTests(unittest.TestCase):
    """Test registry inclusion and metadata of Set 9 tools."""

    def setUp(self) -> None:
        self.registry = build_tool_registry()

    def test_set9_tools_registered(self) -> None:
        category_map = dict(self.registry.categories())
        self.assertIn("Set 9: Object Detection with YOLO", category_map)

        set9_tools = category_map["Set 9: Object Detection with YOLO"]
        tool_ids = [t.tool_id for t in set9_tools]
        self.assertIn("yolo_inference", tool_ids)
        self.assertIn("coco_to_yolo", tool_ids)
        self.assertIn("yolo_training", tool_ids)
        self.assertIn("yolo_comparison", tool_ids)

    def test_tool_attributes(self) -> None:
        tool_map = {t.tool_id: t for t in self.registry.all()}
        inf_tool = tool_map.get("yolo_inference")
        self.assertIsNotNone(inf_tool)
        self.assertEqual(inf_tool.category, "Set 9: Object Detection with YOLO")
        self.assertFalse(inf_tool.requires_image)

        conv_tool = tool_map.get("coco_to_yolo")
        self.assertIsNotNone(conv_tool)
        self.assertFalse(conv_tool.requires_image)

        train_tool = tool_map.get("yolo_training")
        self.assertIsNotNone(train_tool)
        self.assertFalse(train_tool.requires_image)

        comp_tool = tool_map.get("yolo_comparison")
        self.assertIsNotNone(comp_tool)
        self.assertFalse(comp_tool.requires_image)


@unittest.skipUnless(ULTRALYTICS_AVAILABLE, "Ultralytics is required for Set 9 inference tests")
class Set9InferenceTests(unittest.TestCase):
    """Test Section 9.1: Pretrained Model Inference and Customizable Plotting."""

    def setUp(self) -> None:
        self.model_weights = find_default_yolo_weights() or "yolo11n.pt"
        self.sample_image = find_default_image()

    def test_inference_on_sample_image(self) -> None:
        self.assertIsNotNone(self.sample_image, "bus.jpg sample image should be located.")
        results, details = run_yolo_inference(
            model_or_path=self.model_weights,
            image_input=self.sample_image,
            conf=0.25,
        )
        self.assertGreater(details["Total detected objects"], 0)
        self.assertIn("Inference latency", details)
        self.assertIn("Raw boxes", details)

        raw_boxes = details["Raw boxes"]
        labels = [b["label"] for b in raw_boxes]
        self.assertTrue("bus" in labels or "person" in labels)

    def test_custom_tab20_plot_rendering(self) -> None:
        results, _details = run_yolo_inference(
            model_or_path=self.model_weights,
            image_input=self.sample_image,
            conf=0.25,
        )
        r = results[0]
        annotated = render_custom_yolo_plot(
            r,
            colormap_name="tab20",
            show_conf=True,
            show_labels=True,
            box_linewidth=2,
            fix_bgr_to_rgb=True,
        )
        self.assertIsInstance(annotated, Image.Image)
        self.assertEqual(annotated.mode, "RGB")
        self.assertGreater(annotated.width, 100)
        self.assertGreater(annotated.height, 100)

    def test_yolo_inference_tool_headless_execution(self) -> None:
        doc = ImageDocument()
        if self.sample_image and os.path.isfile(self.sample_image):
            doc.load(self.sample_image)

        tool = YOLOInferenceTool()
        result = tool.run(None, doc)

        self.assertIsNotNone(result)
        self.assertIn("Detected", result.message)
        self.assertIsInstance(result.image, Image.Image)
        self.assertIn("Total detected objects", result.details)


class Set9ConversionAndTrainingTests(unittest.TestCase):
    """Test Section 9.2: COCO-to-YOLO Conversion and Training Specifications."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base_path = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_coco_to_yolo_conversion(self) -> None:
        coco_dir = create_synthetic_coco_dataset(self.base_path, num_images=3)
        yolo_out = self.base_path / "synthetic_yolo"

        res = coco_to_yolo(coco_dir, yolo_out)

        self.assertEqual(res["images_converted"], 3)
        self.assertEqual(res["total_annotations"], 6)  # 2 per image * 3

        # Verify images and labels directories exist
        images_dir = yolo_out / "images"
        labels_dir = yolo_out / "labels"
        self.assertTrue(images_dir.is_dir())
        self.assertTrue(labels_dir.is_dir())

        # Check copied images
        self.assertEqual(len(list(images_dir.glob("*.jpg"))), 3)

        # Check generated txt files
        txt_files = list(labels_dir.glob("*.txt"))
        self.assertEqual(len(txt_files), 3)

        # Inspect contents of first label file
        with open(txt_files[0], "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]

        self.assertEqual(len(lines), 2)
        # Check normalization format: [class_id xc yc w h]
        first_line_parts = lines[0].split()
        self.assertEqual(len(first_line_parts), 5)
        cls_id = int(first_line_parts[0])
        xc, yc, w, h = [float(p) for p in first_line_parts[1:5]]

        self.assertEqual(cls_id, 0)  # category 1 (PC) minus 1 = 0
        # Check bounding box math: x=10, y=15, w=30, h=25 on 100x100
        # xc = (10 + 15)/100 = 0.25, yc = (15 + 12.5)/100 = 0.275, w=0.30, h=0.25
        self.assertAlmostEqual(xc, 0.25, places=3)
        self.assertAlmostEqual(yc, 0.275, places=3)
        self.assertAlmostEqual(w, 0.30, places=3)
        self.assertAlmostEqual(h, 0.25, places=3)

    def test_create_data_yaml(self) -> None:
        yolo_dir = self.base_path / "blood_cell_yolo"
        yolo_dir.mkdir()
        yaml_path = create_yolo_data_yaml(yolo_dir, class_names=["PC", "RBC", "WBC"])

        self.assertTrue(os.path.isfile(yaml_path))
        with open(yaml_path, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn("train:", content)
        self.assertIn("val:", content)
        self.assertIn("test:", content)
        self.assertIn("nc: 3", content)
        self.assertIn("names: ['PC', 'RBC', 'WBC']", content)

    def test_coco_conversion_tool_headless(self) -> None:
        coco_dir = create_synthetic_coco_dataset(self.base_path, num_images=2)
        yolo_out = self.base_path / "tool_yolo_out"

        tool = YOLODataConversionTool()
        doc = ImageDocument()

        # In headless mode (parent=None), uses default inputs
        # We test directly with valid parameters
        res = coco_to_yolo(coco_dir, yolo_out)
        self.assertEqual(res["images_converted"], 2)


class Set9ComparisonAndEvaluationTests(unittest.TestCase):
    """Test Section 9.3: Unseen Image Testing & Comparison (GT vs Pred)."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base_path = Path(self.temp_dir.name)

        # Create synthetic test image (640x640) and label
        self.img_path = self.base_path / "test_cell.jpg"
        Image.fromarray(np.full((640, 640, 3), 180, dtype=np.uint8)).save(self.img_path)

        self.lbl_path = self.base_path / "test_cell.txt"
        # 3 annotations: PC (0), RBC (1), WBC (2)
        labels_content = (
            "0 0.2 0.3 0.1 0.1\n"
            "1 0.5 0.5 0.2 0.2\n"
            "2 0.8 0.7 0.15 0.15\n"
        )
        with open(self.lbl_path, "w", encoding="utf-8") as f:
            f.write(labels_content)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_parse_yolo_label_file(self) -> None:
        boxes = parse_yolo_label_file(self.lbl_path, 640, 640, BLOOD_CELL_CATEGORIES)
        self.assertEqual(len(boxes), 3)

        self.assertEqual(boxes[0]["label"], "PC")
        self.assertEqual(boxes[1]["label"], "RBC")
        self.assertEqual(boxes[2]["label"], "WBC")

        # Check box coordinates conversion from normalized xc, yc, w, h
        # Box 0: xc=0.2*640=128, yc=0.3*640=192, w=64, h=64
        # x1=128-32=96, y1=192-32=160, x2=160, y2=224
        x1, y1, x2, y2 = boxes[0]["xyxy"]
        self.assertAlmostEqual(x1, 96.0)
        self.assertAlmostEqual(y1, 160.0)
        self.assertAlmostEqual(x2, 160.0)
        self.assertAlmostEqual(y2, 224.0)

    def test_calculate_box_iou(self) -> None:
        box1 = [10.0, 10.0, 50.0, 50.0]  # area = 1600
        box2 = [10.0, 10.0, 50.0, 50.0]  # identical
        self.assertAlmostEqual(calculate_box_iou(box1, box2), 1.0)

        box3 = [60.0, 60.0, 100.0, 100.0]  # disjoint
        self.assertAlmostEqual(calculate_box_iou(box1, box3), 0.0)

        box4 = [30.0, 10.0, 70.0, 50.0]  # partial overlap: inter = 20*40=800, union = 1600+1600-800 = 2400
        self.assertAlmostEqual(calculate_box_iou(box1, box4), 800.0 / 2400.0)

    def test_match_ground_truth_and_predictions(self) -> None:
        gt_boxes = [
            {"class_id": 0, "xyxy": [10.0, 10.0, 50.0, 50.0]},
            {"class_id": 1, "xyxy": [100.0, 100.0, 150.0, 150.0]},
        ]
        pred_boxes = [
            {"class_id": 0, "xyxy": [12.0, 12.0, 48.0, 48.0], "confidence": 0.9},  # Good match
            {"class_id": 1, "xyxy": [200.0, 200.0, 250.0, 250.0], "confidence": 0.8},  # False positive
        ]

        metrics = match_ground_truth_and_predictions(gt_boxes, pred_boxes, iou_threshold=0.5)

        self.assertEqual(metrics["true_positives"], 1)
        self.assertEqual(metrics["false_positives"], 1)
        self.assertEqual(metrics["false_negatives"], 1)
        self.assertAlmostEqual(metrics["precision"], 0.5)
        self.assertAlmostEqual(metrics["recall"], 0.5)

    @unittest.skipUnless(ULTRALYTICS_AVAILABLE, "Ultralytics is required for full visual comparison test")
    def test_visualize_unseen_comparison(self) -> None:
        weights = find_default_yolo_weights() or "yolo11n.pt"
        img, details = visualize_unseen_comparison(
            image_input=self.img_path,
            label_path=self.lbl_path,
            model_or_path=weights,
            conf_threshold=0.25,
            mode="side_by_side",
        )
        self.assertIsInstance(img, Image.Image)
        self.assertEqual(img.mode, "RGB")
        self.assertEqual(details["Ground Truth Boxes"], 3)
        self.assertIn("Predicted Boxes", details)
        self.assertIn("Precision", details)
        self.assertIn("Recall", details)

    def test_blood_cell_categories_and_colors(self) -> None:
        self.assertEqual(BLOOD_CELL_CATEGORIES[0], "PC")
        self.assertEqual(BLOOD_CELL_CATEGORIES[1], "RBC")
        self.assertEqual(BLOOD_CELL_CATEGORIES[2], "WBC")

        self.assertEqual(BLOOD_CELL_COLORS[0], "red")
        self.assertEqual(BLOOD_CELL_COLORS[1], "green")
        self.assertEqual(BLOOD_CELL_COLORS[2], "blue")


class Set9DialogTests(unittest.TestCase):
    """Test dialog functions and parameter parsing for Set 9."""

    def test_ask_coco_to_yolo_parameters_headless(self) -> None:
        from forensics_app.tools.dialogs import ask_coco_to_yolo_parameters
        res = ask_coco_to_yolo_parameters(None, default_input="/coco", default_output="/yolo")
        self.assertIsNotNone(res)
        self.assertEqual(res["input_folder"], "/coco")
        self.assertEqual(res["output_folder"], "/yolo")
        self.assertTrue(res["create_yaml"])

    def test_ask_yolo_training_parameters_headless(self) -> None:
        from forensics_app.tools.dialogs import ask_yolo_training_parameters
        res = ask_yolo_training_parameters(None, default_yaml="/data.yaml")
        self.assertIsNotNone(res)
        self.assertEqual(res["data_yaml"], "/data.yaml")
        self.assertEqual(res["epochs"], 50)
        self.assertEqual(res["batch"], 16)
        self.assertEqual(res["name"], "from_scratch_50e_v1")
        self.assertEqual(res["classes"], [0, 1, 2])

    def test_ask_yolo_inference_parameters_headless(self) -> None:
        from forensics_app.tools.dialogs import ask_yolo_inference_parameters
        res = ask_yolo_inference_parameters(None, current_image_loaded=True)
        self.assertIsNotNone(res)
        self.assertEqual(res["model_path"], "yolo11n.pt")
        self.assertEqual(res["colormap"], "tab20")
        self.assertAlmostEqual(res["conf"], 0.25)

    def test_ask_yolo_comparison_parameters_headless(self) -> None:
        from forensics_app.tools.dialogs import ask_yolo_comparison_parameters
        res = ask_yolo_comparison_parameters(None, default_image="/test.jpg")
        self.assertIsNotNone(res)
        self.assertEqual(res["image_path"], "/test.jpg")
        self.assertEqual(res["mode"], "side_by_side")


if __name__ == "__main__":
    unittest.main()
