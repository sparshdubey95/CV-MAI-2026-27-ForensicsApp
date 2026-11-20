"""Core YOLO object detection, dataset conversion, training, and evaluation module (Set 9).

This module implements the requirements from Materials-CV-MAI-Set9.pdf:
  - 9.1 Pretrained Model Inference (yolo11n.pt, customizable tab20 Matplotlib/OpenCV plotting,
        handling BGR-to-RGB conversion to solve 'Why people are blue?').
  - 9.2 COCO-to-YOLO conversion (coco_to_yolo, data.yaml generation) and model training controls
        (model.train() with data.yaml, epochs=50, batch=16, classes=[0, 1, 2], plots=True,
        named from_scratch_50e_v1).
  - 9.3 Unseen Image Testing & Comparison (evaluating on unseen blood cell images,
        visualizing and comparing ground truth vs prediction bounding boxes, IoU matching,
        and quantitative evaluation curves).
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any, Callable

import cv2
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend safe for GUI and background threads
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    YOLO = None  # type: ignore


# Default blood cell categories per coursework specification
BLOOD_CELL_CATEGORIES: dict[int, str] = {
    0: "PC",   # Platelets / Thrombocytes
    1: "RBC",  # Red Blood Cells / Erythrocytes
    2: "WBC",  # White Blood Cells / Leukocytes
}

# Distinct category colors matching notebook cell 35
BLOOD_CELL_COLORS: dict[int, str] = {
    0: "red",    # PC
    1: "green",  # RBC
    2: "blue",   # WBC
}

# Global tracked active YOLO model path across application sessions
_ACTIVE_YOLO_MODEL_PATH: str | None = None


def get_active_yolo_model_path() -> str | None:
    """Return the path to the currently active/trained YOLO model weights."""
    global _ACTIVE_YOLO_MODEL_PATH
    if _ACTIVE_YOLO_MODEL_PATH and os.path.exists(_ACTIVE_YOLO_MODEL_PATH):
        return _ACTIVE_YOLO_MODEL_PATH

    # Check common locations for trained runs or pretrained weights
    candidates = [
        os.path.abspath("runs/detect/from_scratch_50e_v1/weights/best.pt"),
        os.path.abspath("runs/detect/from_scratch_50e_v1/weights/last.pt"),
        os.path.abspath("yolo11n.pt"),
    ]
    for cand in candidates:
        if os.path.isfile(cand):
            _ACTIVE_YOLO_MODEL_PATH = cand
            return cand
    return None


def set_active_yolo_model_path(path: str | Path | None) -> None:
    """Set the active YOLO model weights path."""
    global _ACTIVE_YOLO_MODEL_PATH
    _ACTIVE_YOLO_MODEL_PATH = str(path) if path else None


def find_default_yolo_weights() -> str | None:
    """Locate yolo11n.pt in project root or current working directory."""
    root = Path(__file__).resolve().parents[2]
    candidates = [
        root / "yolo11n.pt",
        Path.cwd() / "yolo11n.pt",
        Path("yolo11n.pt"),
    ]
    for cand in candidates:
        if cand.is_file():
            return str(cand)
    return "yolo11n.pt"  # Let Ultralytics resolve or download if needed


def find_default_image() -> str | None:
    """Locate fallback sample image (bus.jpg)."""
    root = Path(__file__).resolve().parents[2]
    candidates = [
        root / "bus.jpg",
        Path.cwd() / "bus.jpg",
        Path("bus.jpg"),
    ]
    for cand in candidates:
        if cand.is_file():
            return str(cand)
    return None


def find_blood_cell_dataset() -> tuple[Path | None, Path | None]:
    """Find local blood cell datasets (coco and yolov11) if present in the workspace."""
    root = Path(__file__).resolve().parents[2]
    coco_candidates = [
        root / "blood_cell.v1i.coco",
        Path.cwd() / "blood_cell.v1i.coco",
    ]
    yolo_candidates = [
        root / "blood cell.v1i.yolov11",
        root / "blood_cell.v1i.yolov11",
        Path.cwd() / "blood cell.v1i.yolov11",
        Path.cwd() / "blood_cell.v1i.yolov11",
    ]

    coco_path = next((p for p in coco_candidates if p.is_dir()), None)
    yolo_path = next((p for p in yolo_candidates if p.is_dir()), None)
    return coco_path, yolo_path


# ============================================================================
# Section 9.1: Pretrained Model Inference & Customizable Plotting
# ============================================================================


def run_yolo_inference(
    model_or_path: Any,
    image_input: str | Path | Image.Image | np.ndarray,
    conf: float = 0.25,
    iou: float = 0.45,
    imgsz: int = 640,
    device: str | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Run YOLO inference on an image input and return results and parsed details.

    Parameters
    ----------
    model_or_path : YOLO model instance or filepath string to model weights.
    image_input : File path, PIL Image, or numpy array.
    conf : Confidence threshold for filtering detections.
    iou : IoU threshold for NMS.
    imgsz : Inference image size.
    device : Device string ('cpu', '0', etc.) or None for automatic.

    Returns
    -------
    results : Ultralytics Results list or object.
    details : Dictionary with detection statistics, bounding box details, and speed.
    """
    if not ULTRALYTICS_AVAILABLE:
        raise ImportError("Ultralytics is not installed. Please install ultralytics to use YOLO.")

    if isinstance(model_or_path, (str, Path)):
        model = YOLO(str(model_or_path))
    else:
        model = model_or_path

    # Prepare image source
    if isinstance(image_input, Image.Image):
        # Convert PIL RGB to numpy array
        img_np = np.array(image_input)
    elif isinstance(image_input, np.ndarray):
        img_np = image_input
    else:
        img_path = str(image_input)
        if not os.path.isfile(img_path):
            raise FileNotFoundError(f"Image file not found: {img_path}")
        img_np = img_path

    start_time = time.perf_counter()
    predict_kwargs: dict[str, Any] = {
        "conf": conf,
        "iou": iou,
        "imgsz": imgsz,
        "verbose": False,
    }
    if device:
        predict_kwargs["device"] = device

    results = model(img_np, **predict_kwargs)
    inference_duration_ms = (time.perf_counter() - start_time) * 1000.0

    r = results[0]
    boxes = r.boxes
    labels_map = r.names if hasattr(r, "names") else {}

    parsed_boxes: list[dict[str, Any]] = []
    class_counts: dict[str, int] = {}

    if boxes is not None and len(boxes) > 0:
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)

        for i in range(len(xyxy)):
            c_id = int(classes[i])
            c_name = labels_map.get(c_id, f"class_{c_id}")
            score = float(confs[i])
            box_coords = [float(coord) for coord in xyxy[i]]

            parsed_boxes.append({
                "class_id": c_id,
                "label": c_name,
                "confidence": score,
                "xyxy": box_coords,
            })
            class_counts[c_name] = class_counts.get(c_name, 0) + 1

    # Speed metrics
    speed = r.speed if hasattr(r, "speed") and r.speed else {}
    preprocess_ms = speed.get("preprocess", 0.0)
    infer_ms = speed.get("inference", inference_duration_ms)
    postprocess_ms = speed.get("postprocess", 0.0)

    summary_str = ", ".join(f"{count} {cls}" for cls, count in class_counts.items()) if class_counts else "None"

    details: dict[str, Any] = {
        "Total detected objects": len(parsed_boxes),
        "Detections by class": summary_str,
        "Confidence threshold": f"{conf:.2f}",
        "Inference latency": f"{infer_ms:.1f} ms",
        "Preprocess latency": f"{preprocess_ms:.1f} ms",
        "Postprocess latency": f"{postprocess_ms:.1f} ms",
        "Raw boxes": parsed_boxes,
    }

    # Add top 5 individual detection summaries to details for treeview inspection
    for idx, b in enumerate(parsed_boxes[:5]):
        x1, y1, x2, y2 = [int(v) for v in b["xyxy"]]
        details[f"Box #{idx + 1}"] = f"{b['label']} ({b['confidence']:.2f}) at [{x1}, {y1}, {x2}, {y2}]"
    if len(parsed_boxes) > 5:
        details["Remaining boxes"] = f"{len(parsed_boxes) - 5} additional objects detected"

    return results, details


def render_custom_yolo_plot(
    result: Any,
    colormap_name: str = "tab20",
    show_conf: bool = True,
    show_labels: bool = True,
    box_linewidth: int = 2,
    title: str = "Detected Objects",
    font_size: int = 10,
    fix_bgr_to_rgb: bool = True,
) -> Image.Image:
    """Render customizable Matplotlib plot using results[0].boxes and colormap.

    Matches code in notebook cell 9 and slide 4 of Materials-CV-MAI-Set9.pdf.
    Solves the 'Why people are blue?' problem by converting OpenCV BGR to RGB
    via cv2.cvtColor(result.orig_img, cv2.COLOR_BGR2RGB).

    Parameters
    ----------
    result : Ultralytics single Results object (results[0]).
    colormap_name : Matplotlib colormap (e.g. 'tab20', 'viridis', 'hsv').
    show_conf : Whether to show confidence score in text label.
    show_labels : Whether to show class name in text label.
    box_linewidth : Bounding box border line thickness.
    title : Title for the plot.
    font_size : Label text font size.
    fix_bgr_to_rgb : Whether to convert orig_img from BGR to RGB.

    Returns
    -------
    Image.Image : Rendered PIL RGB Image.
    """
    orig_img = result.orig_img
    if orig_img is None:
        raise ValueError("Results object has no orig_img attribute.")

    # Convert BGR to RGB to fix 'Why people are blue?' issue from slide 3/4
    if fix_bgr_to_rgb and len(orig_img.shape) == 3 and orig_img.shape[2] == 3:
        img_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = orig_img

    boxes = result.boxes
    labels = result.names if hasattr(result, "names") else {}

    # Get colormap
    try:
        colors = matplotlib.colormaps.get_cmap(colormap_name)
    except (AttributeError, ValueError):
        colors = plt.get_cmap("tab20")

    h, w = img_rgb.shape[:2]
    # Maintain aspect ratio for matplotlib figure
    figsize_w = 10
    figsize_h = max(4, int(figsize_w * (h / max(w, 1))))

    fig, ax = plt.subplots(1, 1, figsize=(figsize_w, figsize_h), dpi=120)
    ax.imshow(img_rgb)

    if boxes is not None and len(boxes) > 0:
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)

        for i in range(len(xyxy)):
            x1, y1, x2, y2 = xyxy[i]
            conf = float(confs[i])
            cls_id = int(classes[i])
            label_name = labels.get(cls_id, str(cls_id))

            # Pick color from colormap (cycling through 20 bins for tab20)
            box_color = colors(cls_id % 20 if "tab20" in colormap_name else (cls_id / max(len(labels), 1)))

            # Create rectangle patch
            rect = plt.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=box_linewidth,
                edgecolor=box_color,
                facecolor="none",
            )
            ax.add_patch(rect)

            # Build label text
            parts = []
            if show_labels:
                parts.append(label_name)
            if show_conf:
                parts.append(f"{conf:.2f}")
            text_str = " ".join(parts)

            if text_str:
                ax.text(
                    x1,
                    max(12, y1 - 6),
                    text_str,
                    color="white",
                    fontsize=font_size,
                    weight="bold",
                    backgroundcolor=box_color,
                    bbox=dict(
                        boxstyle="square,pad=0.2",
                        facecolor=box_color,
                        edgecolor=box_color,
                        alpha=0.85,
                    ),
                )

    ax.axis("off")
    if title:
        box_count = len(boxes) if boxes is not None else 0
        ax.set_title(f"{title} ({box_count} objects detected)", fontsize=13, pad=10)

    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


# ============================================================================
# Section 9.2: COCO-to-YOLO Conversion & Model Training
# ============================================================================


def coco_to_yolo(input_folder: str | Path, output_folder: str | Path) -> dict[str, Any]:
    """Convert COCO JSON annotations into normalized YOLO text files.

    Exact implementation matching notebook cell 19 and slide 11 of Materials-CV-MAI-Set9.pdf:
    - Creates 'images' and 'labels' directories inside output_folder.
    - Copies image files (.jpg, .jpeg, .png, .bmp, .tif, .tiff) from input_folder to images/.
    - Loads _annotations.coco.json (or any .json file in input_folder).
    - Converts bounding boxes [x, y, width, height] to normalized
      [class_id x_center y_center width height] text files in labels/.
      where class_id = category_id - 1 (COCO 1-indexed -> YOLO 0-indexed).

    Parameters
    ----------
    input_folder : Directory containing images and _annotations.coco.json.
    output_folder : Destination directory to create images/ and labels/ inside.

    Returns
    -------
    dict : Summary statistics with number of images, annotations, and category counts.
    """
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)

    images_dir = output_folder / "images"
    labels_dir = output_folder / "labels"

    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    # 1. Copy images
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    copied_images = 0
    for f in os.listdir(input_folder):
        src = input_folder / f
        if src.is_file() and src.suffix.lower() in exts:
            dst = images_dir / f
            if not dst.exists() or dst.stat().st_size != src.stat().st_size:
                shutil.copy2(src, dst)
            copied_images += 1

    # 2. Locate COCO annotations JSON
    coco_path = input_folder / "_annotations.coco.json"
    if not coco_path.is_file():
        # Fallback to any json file in input_folder
        json_candidates = list(input_folder.glob("*.json"))
        if json_candidates:
            coco_path = json_candidates[0]
        else:
            raise FileNotFoundError(
                f"No COCO JSON file found in {input_folder} (expected _annotations.coco.json)"
            )

    with open(coco_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    images = {img["id"]: img for img in coco.get("images", [])}
    anns_by_img: dict[int, list[dict[str, Any]]] = {}
    for ann in coco.get("annotations", []):
        iid = ann["image_id"]
        anns_by_img.setdefault(iid, []).append(ann)

    categories = {cat["id"]: cat["name"] for cat in coco.get("categories", [])}
    cat_counts: dict[str, int] = {}
    total_annotations = 0

    # 3. Generate YOLO txt files
    for img_id, img_info in images.items():
        file_name = img_info["file_name"]
        width = float(img_info["width"])
        height = float(img_info["height"])

        stem = Path(file_name).stem
        yolo_path = labels_dir / f"{stem}.txt"

        lines = []
        for ann in anns_by_img.get(img_id, []):
            cid = ann["category_id"] - 1  # COCO starts with 1, YOLO needs [0, N-1]
            cat_name = categories.get(ann["category_id"], str(cid))
            cat_counts[cat_name] = cat_counts.get(cat_name, 0) + 1
            total_annotations += 1

            x, y, w, h = ann["bbox"]
            xc = (x + w / 2.0) / width
            yc = (y + h / 2.0) / height
            wn = w / width
            hn = h / height

            # Bound between 0.0 and 1.0
            xc = max(0.0, min(1.0, xc))
            yc = max(0.0, min(1.0, yc))
            wn = max(0.0, min(1.0, wn))
            hn = max(0.0, min(1.0, hn))

            lines.append(f"{cid} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")

        with open(yolo_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    return {
        "images_converted": len(images),
        "images_copied": copied_images,
        "total_annotations": total_annotations,
        "category_counts": cat_counts,
        "categories": categories,
        "images_dir": str(images_dir),
        "labels_dir": str(labels_dir),
    }


def convert_coco_dataset_splits(
    coco_root: str | Path,
    yolo_root: str | Path,
    splits: tuple[str, ...] = ("train", "valid", "test"),
    class_names: list[str] | None = None,
) -> dict[str, Any]:
    """Convert all COCO splits (train, valid, test) and create data.yaml.

    Matches the workflow in notebook cells 19-21.
    """
    coco_root = Path(coco_root)
    yolo_root = Path(yolo_root)
    os.makedirs(yolo_root, exist_ok=True)

    summary: dict[str, Any] = {}
    all_categories: dict[int, str] = {}

    for split in splits:
        input_split = coco_root / split
        output_split = yolo_root / split
        if input_split.is_dir():
            res = coco_to_yolo(input_split, output_split)
            summary[split] = res
            if res.get("categories"):
                all_categories.update(res["categories"])

    # Determine class names
    if class_names is None:
        if all_categories:
            # Sorted by category id
            # COCO category ids are 1, 2, 3 -> names: [categories[1], categories[2], ...]
            # Filter out id 0 if dummy supercategory
            valid_ids = sorted([cid for cid in all_categories.keys() if cid > 0])
            class_names = [all_categories[cid] for cid in valid_ids]
        else:
            class_names = ["PC", "RBC", "WBC"]

    data_yaml_path = create_yolo_data_yaml(yolo_root, class_names)
    summary["data_yaml"] = data_yaml_path
    summary["class_names"] = class_names
    return summary


def create_yolo_data_yaml(
    yolo_dir: str | Path,
    class_names: list[str] | None = None,
    output_path: str | Path | None = None,
) -> str:
    """Generate the data.yaml file required for YOLO training.

    Matches notebook cell 21:
      train: <root>/train/images
      val: <root>/valid/images
      test: <root>/test/images
      nc: 3
      names: ['PC', 'RBC', 'WBC']
    """
    yolo_dir = Path(yolo_dir).resolve()
    if class_names is None:
        class_names = ["PC", "RBC", "WBC"]

    # Use forward slashes for cross-platform YOLO compatibility
    train_dir = (yolo_dir / "train" / "images").as_posix()
    val_dir = (yolo_dir / "valid" / "images").as_posix()
    test_dir = (yolo_dir / "test" / "images").as_posix()

    names_formatted = "[" + ", ".join(f"'{name}'" for name in class_names) + "]"
    content = f"""train: {train_dir}
val: {val_dir}
test: {test_dir}

nc: {len(class_names)}
names: {names_formatted}
"""
    if output_path is None:
        output_path = yolo_dir / "data.yaml"
    else:
        output_path = Path(output_path).resolve()

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)

    return str(output_path)


def train_yolo_model(
    data_yaml_path: str | Path,
    model_cfg_or_weights: str = "yolo11n.yaml",
    name: str = "from_scratch_50e_v1",
    epochs: int = 50,
    batch: int = 16,
    classes: list[int] | None = None,
    plots: bool = True,
    device: str | None = None,
    project: str | None = None,
    progress_callback: Callable[[str], None] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Train YOLO model with parameters matching Section 9.2 of the coursework.

    Matches notebook cell 27 & 29 and slide 12 of Materials-CV-MAI-Set9.pdf:
      model = YOLO("yolo11n.yaml")
      model.train(
          data=f"{root_path}/data.yaml",
          name="from_scratch_50e_v1",
          epochs=50,
          batch=16,
          classes=[0, 1, 2],
          plots=True
      )

    Parameters
    ----------
    data_yaml_path : Path to data.yaml.
    model_cfg_or_weights : "yolo11n.yaml" for scratch training or "yolo11n.pt" for fine-tuning.
    name : Run name (default "from_scratch_50e_v1").
    epochs : Number of training epochs (default 50).
    batch : Batch size (default 16).
    classes : List of class indices (default [0, 1, 2]).
    plots : Whether to save plots (default True).
    device : Device to run on ('cpu', '0', etc.).
    project : Directory for training runs (defaults to 'runs/detect').
    progress_callback : Optional status notification callback.

    Returns
    -------
    dict : Training summary containing best weights path, metrics, and plots.
    """
    if not ULTRALYTICS_AVAILABLE:
        raise ImportError("Ultralytics is not installed. Please install ultralytics to train YOLO.")

    data_yaml_path = Path(data_yaml_path).resolve()
    if not data_yaml_path.is_file():
        raise FileNotFoundError(f"data.yaml not found at: {data_yaml_path}")

    if classes is None:
        classes = [0, 1, 2]

    if progress_callback:
        progress_callback(f"Initializing YOLO model ({model_cfg_or_weights})...")

    model = YOLO(model_cfg_or_weights)

    train_args: dict[str, Any] = {
        "data": str(data_yaml_path),
        "name": name,
        "epochs": epochs,
        "batch": batch,
        "classes": classes,
        "plots": plots,
        "verbose": False,
    }
    if device is not None:
        train_args["device"] = device
    if project is not None:
        train_args["project"] = project

    train_args.update(kwargs)

    if progress_callback:
        progress_callback(f"Starting training: {name} (epochs={epochs}, batch={batch})...")

    results = model.train(**train_args)

    # Determine save directory and best weights
    save_dir = Path(getattr(model.trainer, "save_dir", Path("runs/detect") / name))
    best_weights = save_dir / "weights" / "best.pt"
    last_weights = save_dir / "weights" / "last.pt"

    trained_weights_path = str(best_weights if best_weights.is_file() else last_weights)
    if os.path.isfile(trained_weights_path):
        set_active_yolo_model_path(trained_weights_path)

    # Collect available training plot images
    plot_names = [
        "results.png",
        "confusion_matrix.png",
        "BoxF1_curve.png",
        "BoxP_curve.png",
        "BoxPR_curve.png",
    ]
    available_plots = [str(save_dir / p) for p in plot_names if (save_dir / p).is_file()]

    summary: dict[str, Any] = {
        "name": name,
        "save_dir": str(save_dir),
        "best_weights": trained_weights_path if os.path.isfile(trained_weights_path) else None,
        "epochs_completed": epochs,
        "plots": available_plots,
        "results": results,
    }

    if progress_callback:
        progress_callback(f"Training completed! Weights saved to: {trained_weights_path}")

    return summary


# ============================================================================
# Section 9.3: Unseen Image Testing, Comparison & Evaluation
# ============================================================================


def parse_yolo_label_file(
    label_path: str | Path,
    img_width: int,
    img_height: int,
    category_names: dict[int, str] | None = None,
) -> list[dict[str, Any]]:
    """Parse a normalized YOLO text label file into absolute pixel bounding boxes."""
    label_path = Path(label_path)
    if not label_path.is_file():
        return []

    if category_names is None:
        category_names = BLOOD_CELL_CATEGORIES

    boxes = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                c_id = int(float(parts[0]))
                xc, yc, wn, hn = [float(p) for p in parts[1:5]]

                # Convert normalized center coordinates to absolute pixel coordinates
                w = wn * img_width
                h = hn * img_height
                x1 = (xc * img_width) - (w / 2.0)
                y1 = (yc * img_height) - (h / 2.0)
                x2 = x1 + w
                y2 = y1 + h

                boxes.append({
                    "class_id": c_id,
                    "label": category_names.get(c_id, f"class_{c_id}"),
                    "xyxy": [x1, y1, x2, y2],
                    "xywh": [xc * img_width, yc * img_height, w, h],
                })
    return boxes


def calculate_box_iou(box1: list[float], box2: list[float]) -> float:
    """Calculate Intersection over Union (IoU) of two bounding boxes in [x1, y1, x2, y2] format."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])

    union_area = area1 + area2 - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def match_ground_truth_and_predictions(
    gt_boxes: list[dict[str, Any]],
    pred_boxes: list[dict[str, Any]],
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    """Match predictions to ground truth bounding boxes using single-match policy."""
    used_gt = [False] * len(gt_boxes)
    matches = []
    false_positives = []

    # Sort predictions descending by confidence
    sorted_preds = sorted(pred_boxes, key=lambda b: b.get("confidence", 0.0), reverse=True)

    for p in sorted_preds:
        best_iou = 0.0
        best_gt_idx = -1

        for idx, g in enumerate(gt_boxes):
            if used_gt[idx]:
                continue
            if p["class_id"] == g["class_id"]:
                iou = calculate_box_iou(p["xyxy"], g["xyxy"])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

        if best_iou >= iou_threshold and best_gt_idx >= 0:
            used_gt[best_gt_idx] = True
            matches.append({
                "prediction": p,
                "ground_truth": gt_boxes[best_gt_idx],
                "iou": best_iou,
            })
        else:
            false_positives.append(p)

    false_negatives = [gt_boxes[i] for i, used in enumerate(used_gt) if not used]

    tp = len(matches)
    fp = len(false_positives)
    fn = len(false_negatives)

    precision = tp / max(tp + fp, 1) if (tp + fp) > 0 else 0.0
    recall = tp / max(tp + fn, 1) if (tp + fn) > 0 else 0.0
    mean_iou = float(np.mean([m["iou"] for m in matches])) if matches else 0.0

    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "mean_iou": mean_iou,
        "matches": matches,
        "unmatched_predictions": false_positives,
        "unmatched_ground_truth": false_negatives,
    }


def visualize_unseen_comparison(
    image_input: str | Path | Image.Image | np.ndarray,
    label_path: str | Path | None = None,
    model_or_path: Any = None,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.5,
    category_names: dict[int, str] | None = None,
    mode: str = "side_by_side",
) -> tuple[Image.Image, dict[str, Any]]:
    """Visualize and compare Ground Truth versus Predicted bounding boxes.

    Matches notebook cell 35 and slide 15 of Materials-CV-MAI-Set9.pdf:
      - Blood cell categories: PC (0), RBC (1), WBC (2)
      - Category colors: 0=red, 1=green, 2=blue
      - Ground truth boxes: green rectangle, linewidth=2
      - Predicted boxes: blue rectangle, linewidth=1

    Parameters
    ----------
    image_input : Image path, PIL Image, or numpy array.
    label_path : Path to corresponding YOLO .txt label file (optional).
    model_or_path : Model instance or path to weights.
    conf_threshold : Confidence threshold.
    iou_threshold : IoU matching threshold.
    category_names : Mapping of class index to name.
    mode : "side_by_side" or "overlay".

    Returns
    -------
    tuple[Image.Image, dict[str, Any]] : Annotated PIL comparison image and metrics dict.
    """
    if category_names is None:
        category_names = BLOOD_CELL_CATEGORIES

    # Load image
    if isinstance(image_input, (str, Path)):
        img_path = Path(image_input)
        img_rgb = np.array(Image.open(img_path).convert("RGB"))
        if label_path is None:
            # Auto-infer label path by replacing 'images' with 'labels' and extension with .txt
            candidate_label = Path(str(img_path).replace("images", "labels")).with_suffix(".txt")
            if candidate_label.is_file():
                label_path = candidate_label
    elif isinstance(image_input, Image.Image):
        img_rgb = np.array(image_input.convert("RGB"))
    else:
        img_rgb = image_input
        if len(img_rgb.shape) == 2:
            img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_GRAY2RGB)

    h, w = img_rgb.shape[:2]

    # Parse Ground Truth
    gt_boxes: list[dict[str, Any]] = []
    if label_path and os.path.isfile(label_path):
        gt_boxes = parse_yolo_label_file(label_path, w, h, category_names)

    # Run Prediction
    if model_or_path is None:
        active_model = get_active_yolo_model_path() or find_default_yolo_weights()
    else:
        active_model = model_or_path

    start_time = time.perf_counter()
    results, _details = run_yolo_inference(
        active_model,
        img_rgb,
        conf=conf_threshold,
        iou=iou_threshold,
    )
    latency_ms = (time.perf_counter() - start_time) * 1000.0

    pred_boxes: list[dict[str, Any]] = []
    r = results[0]
    if r.boxes is not None and len(r.boxes) > 0:
        xyxy = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy().astype(int)

        for i in range(len(xyxy)):
            cid = int(classes[i])
            pred_boxes.append({
                "class_id": cid,
                "label": category_names.get(cid, str(cid)),
                "confidence": float(confs[i]),
                "xyxy": [float(c) for c in xyxy[i]],
            })

    # Compute matching metrics
    eval_metrics = match_ground_truth_and_predictions(gt_boxes, pred_boxes, iou_threshold)

    # Render visualization
    if mode == "side_by_side":
        fig, axes = plt.subplots(1, 2, figsize=(14, 7), dpi=120)
        ax_gt, ax_pred = axes

        # 1. Left: Ground Truth
        ax_gt.imshow(img_rgb)
        ax_gt.set_title(f"Ground Truth ({len(gt_boxes)} annotations)", fontsize=12, weight="bold")
        ax_gt.axis("off")

        for b in gt_boxes:
            x1, y1, x2, y2 = b["xyxy"]
            cid = b["class_id"]
            color = BLOOD_CELL_COLORS.get(cid, "green")
            rect = plt.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                fill=False,
                edgecolor="green",
                linewidth=2,
            )
            ax_gt.add_patch(rect)
            ax_gt.text(
                x1,
                max(10, y1 - 4),
                b["label"],
                color=color,
                fontsize=9,
                weight="bold",
                bbox=dict(boxstyle="square,pad=0.15", facecolor="white", edgecolor=color, alpha=0.8),
            )

        # 2. Right: Predictions
        ax_pred.imshow(img_rgb)
        ax_pred.set_title(
            f"YOLO Predictions ({len(pred_boxes)} detections, {latency_ms:.1f}ms)",
            fontsize=12,
            weight="bold",
        )
        ax_pred.axis("off")

        for b in pred_boxes:
            x1, y1, x2, y2 = b["xyxy"]
            cid = b["class_id"]
            color = BLOOD_CELL_COLORS.get(cid, "blue")
            rect = plt.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                fill=False,
                edgecolor="blue",
                linewidth=1.5,
            )
            ax_pred.add_patch(rect)
            ax_pred.text(
                x1,
                max(10, y1 - 4),
                f"{b['label']} {b['confidence']:.2f}",
                color=color,
                fontsize=9,
                weight="bold",
                bbox=dict(boxstyle="square,pad=0.15", facecolor="white", edgecolor=color, alpha=0.8),
            )

    else:
        # Overlay mode matching cell 35 exactly
        fig, ax = plt.subplots(1, 1, figsize=(10, 8), dpi=120)
        ax.imshow(img_rgb)

        # Draw GT boxes (green border, linewidth=2)
        for b in gt_boxes:
            x1, y1, x2, y2 = b["xyxy"]
            cid = b["class_id"]
            color = BLOOD_CELL_COLORS.get(cid, "green")
            rect = plt.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                fill=False,
                edgecolor="green",
                linewidth=2,
            )
            ax.add_patch(rect)
            ax.text(
                x1,
                max(10, y1 - 4),
                f"GT:{b['label']}",
                color=color,
                fontsize=9,
                weight="bold",
                bbox=dict(boxstyle="square,pad=0.15", facecolor="white", edgecolor="green", alpha=0.8),
            )

        # Draw Predicted boxes (blue border, linewidth=1.5)
        for b in pred_boxes:
            x1, y1, x2, y2 = b["xyxy"]
            cid = b["class_id"]
            color = BLOOD_CELL_COLORS.get(cid, "blue")
            rect = plt.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                fill=False,
                edgecolor="blue",
                linewidth=1.5,
            )
            ax.add_patch(rect)
            ax.text(
                x1,
                min(h - 5, y2 + 10),
                f"Pred:{b['label']} {b['confidence']:.2f}",
                color=color,
                fontsize=9,
                weight="bold",
                bbox=dict(boxstyle="square,pad=0.15", facecolor="white", edgecolor="blue", alpha=0.8),
            )

        ax.axis("off")
        ax.set_title(
            f"Ground Truth (Green, {len(gt_boxes)}) vs Predictions (Blue, {len(pred_boxes)})",
            fontsize=12,
            weight="bold",
        )

    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    buf.seek(0)
    annotated_image = Image.open(buf).convert("RGB")

    # Build details dictionary for the UI Results inspector
    details: dict[str, Any] = {
        "Ground Truth Boxes": len(gt_boxes),
        "Predicted Boxes": len(pred_boxes),
        "True Positives (IoU >= 0.5)": eval_metrics["true_positives"],
        "False Positives": eval_metrics["false_positives"],
        "False Negatives": eval_metrics["false_negatives"],
        "Precision": f"{eval_metrics['precision']:.3f}",
        "Recall": f"{eval_metrics['recall']:.3f}",
        "Mean Match IoU": f"{eval_metrics['mean_iou']:.3f}",
        "Inference Latency": f"{latency_ms:.1f} ms",
        "Active Model": Path(str(active_model)).name if active_model else "None",
    }

    return annotated_image, details


# ============================================================================
# Quantitative Evaluation (Notebook cells 39 - 41)
# ============================================================================


def run_inference_and_collect(
    model_path: str | Path,
    image_folder: str | Path,
    label_folder: str | Path,
    iou_thr: float = 0.5,
    imgsz: int = 640,
) -> Any:
    """Run inference over a folder of images and collect TP/FP records.

    Matches cell 39 of object_detection_with_YOLO.ipynb.
    """
    if not ULTRALYTICS_AVAILABLE:
        raise ImportError("Ultralytics is required for run_inference_and_collect.")

    import pandas as pd

    model = YOLO(str(model_path))
    rows: list[list[float]] = []

    for r in model.predict(str(image_folder), stream=True, imgsz=imgsz, verbose=False):
        img_name = os.path.basename(r.path)
        stem = os.path.splitext(img_name)[0]
        lab = os.path.join(str(label_folder), stem + ".txt")
        if not os.path.exists(lab):
            continue

        img_width, img_height = r.orig_shape[1], r.orig_shape[0]

        try:
            gt = np.loadtxt(lab).reshape(-1, 5)
        except Exception:
            continue

        gt_abs = []
        for c, xc, yc, w, h in gt:
            x1 = (xc - w / 2.0) * img_width
            y1 = (yc - h / 2.0) * img_height
            x2 = (xc + w / 2.0) * img_width
            y2 = (yc + h / 2.0) * img_height
            gt_abs.append([c, x1, y1, x2, y2])
        gt_abs = np.array(gt_abs)

        if r.boxes is None or len(r.boxes) == 0:
            continue

        det = r.boxes
        det_xyxy = det.xyxy.cpu().numpy()
        det_conf = det.conf.cpu().numpy()

        used = np.zeros(len(gt_abs), dtype=bool)

        for i in range(len(det_xyxy)):
            dx1, dy1, dx2, dy2 = det_xyxy[i]
            dconf = float(det_conf[i])

            ious = []
            for j, g in enumerate(gt_abs):
                gx1, gy1, gx2, gy2 = g[1], g[2], g[3], g[4]
                ix1 = max(dx1, gx1)
                iy1 = max(dy1, gy1)
                ix2 = min(dx2, gx2)
                iy2 = min(dy2, gy2)
                inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                ua = (dx2 - dx1) * (dy2 - dy1) + (gx2 - gx1) * (gy2 - gy1) - inter
                iou = inter / ua if ua > 0 else 0.0
                ious.append(iou)

            if len(ious) == 0:
                rows.append([dconf, 0.0, 1.0])
                continue

            jmax = int(np.argmax(ious))
            best = ious[jmax]

            if best >= iou_thr and not used[jmax]:
                used[jmax] = True
                rows.append([dconf, 1.0, 0.0])
            else:
                rows.append([dconf, 0.0, 1.0])

    return pd.DataFrame(rows, columns=["conf", "tp", "fp"])


def render_pr_curve_figure(df: Any) -> Image.Image:
    """Plot Precision-Recall curve matching notebook cell 39."""
    df_sorted = df.sort_values("conf", ascending=False)
    tp_cum = df_sorted["tp"].cumsum().values
    fp_cum = df_sorted["fp"].cumsum().values

    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
    recall = tp_cum / np.maximum(df_sorted["tp"].sum(), 1e-9)

    fig, ax = plt.subplots(figsize=(6, 5), dpi=120)
    ax.plot(recall, precision, color="#0055aa", linewidth=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.set_xlim([0.0, 1.05])
    ax.set_ylim([0.0, 1.05])
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")
