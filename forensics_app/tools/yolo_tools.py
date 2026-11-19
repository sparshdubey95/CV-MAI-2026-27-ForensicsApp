"""Tkinter tools for YOLO Object Detection, Dataset Conversion, and Evaluation (Set 9).

Integrates:
  - 9.1 Pretrained Model Inference (YOLOInferenceTool)
  - 9.2 COCO-to-YOLO Conversion & Training (YOLODataConversionTool, YOLOTrainingTool)
  - 9.3 Unseen Image Testing & Comparison (YOLOComparisonTool)
"""

from __future__ import annotations

import os
from pathlib import Path
import threading
import tkinter as tk
from tkinter import ttk
from typing import Any

from PIL import Image

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
from .dialogs import (
    ask_coco_to_yolo_parameters,
    ask_yolo_comparison_parameters,
    ask_yolo_inference_parameters,
    ask_yolo_training_parameters,
    show_error,
)
from .yolo_core import (
    BLOOD_CELL_CATEGORIES,
    ULTRALYTICS_AVAILABLE,
    convert_coco_dataset_splits,
    coco_to_yolo,
    find_blood_cell_dataset,
    find_default_image,
    find_default_yolo_weights,
    get_active_yolo_model_path,
    render_custom_yolo_plot,
    run_yolo_inference,
    set_active_yolo_model_path,
    train_yolo_model,
    visualize_unseen_comparison,
)


class YOLOInferenceTool(ForensicsTool):
    """Set 9.1: Load pretrained YOLO model and run inference with customizable colormap plotting."""

    tool_id = "yolo_inference"
    title = "YOLO pretrained inference"
    category = "Set 9: Object Detection with YOLO"
    description = "Run pretrained Ultralytics YOLO model (yolo11n.pt) with customizable tab20 colormap plotting."
    requires_image = False

    def run(self, parent: tk.Misc | None, document: ImageDocument) -> ToolResult | None:
        if not ULTRALYTICS_AVAILABLE:
            if parent is not None:
                show_error(
                    parent,
                    "Ultralytics Required",
                    "The 'ultralytics' package is required for YOLO tools.\nPlease install it via 'pip install ultralytics'.",
                )
            return None

        # Determine default model and image
        default_model = get_active_yolo_model_path() or find_default_yolo_weights() or "yolo11n.pt"
        has_loaded_image = document.is_loaded

        params = ask_yolo_inference_parameters(
            parent,
            current_image_loaded=has_loaded_image,
            default_model=str(default_model),
            default_conf=0.25,
        )
        if params is None:
            return None

        # Determine image source
        img_source: Any = None
        source_name = "image"
        if params["custom_image_path"]:
            img_source = params["custom_image_path"]
            source_name = Path(params["custom_image_path"]).name
        elif params["use_fallback_image"] or not has_loaded_image:
            fallback = find_default_image()
            if fallback and os.path.isfile(fallback):
                img_source = fallback
                source_name = Path(fallback).name
            else:
                img_source = "https://ultralytics.com/images/bus.jpg"
                source_name = "bus.jpg (online fallback)"
        else:
            img_source = document.current
            source_name = document.path.name if document.path else "current_document.png"

        try:
            results, details = run_yolo_inference(
                model_or_path=params["model_path"],
                image_input=img_source,
                conf=params["conf"],
            )
        except Exception as err:
            if parent is not None:
                show_error(parent, "YOLO Inference Failed", str(err))
            return None

        # Render customizable Matplotlib plot (tab20 by default, solves 'Why people are blue?')
        try:
            annotated_image = render_custom_yolo_plot(
                results[0],
                colormap_name=params["colormap"],
                box_linewidth=params["box_linewidth"],
                title="YOLO Object Detection",
                fix_bgr_to_rgb=True,
            )
        except Exception as err:
            if parent is not None:
                show_error(parent, "Plotting Failed", f"Could not render custom plot: {err}")
            return None

        total_objs = details.get("Total detected objects", 0)
        message = (
            f"Detected {total_objs} objects in {source_name} "
            f"using {Path(params['model_path']).name} ({details['Inference latency']})"
        )

        details["Model Used"] = Path(params["model_path"]).name
        details["Image Source"] = source_name
        details["Colormap"] = params["colormap"]

        return ToolResult(
            message=message,
            image=annotated_image,
            details=details,
        )


class YOLODataConversionTool(ForensicsTool):
    """Set 9.2: Convert COCO JSON annotations to YOLO format (images & labels directories)."""

    tool_id = "coco_to_yolo"
    title = "COCO to YOLO converter"
    category = "Set 9: Object Detection with YOLO"
    description = "Convert COCO JSON annotations into normalized YOLO text files (images & labels)."
    requires_image = False

    def run(self, parent: tk.Misc | None, document: ImageDocument) -> ToolResult | None:
        coco_dir, yolo_dir = find_blood_cell_dataset()
        default_input = str(coco_dir) if coco_dir else ""
        default_output = str(yolo_dir) if yolo_dir else (
            str(coco_dir.parent / "blood_cell_yolo") if coco_dir else ""
        )

        params = ask_coco_to_yolo_parameters(
            parent,
            default_input=default_input,
            default_output=default_output,
        )
        if params is None:
            return None

        input_folder = Path(params["input_folder"])
        output_folder = Path(params["output_folder"])

        try:
            # Check if input has split directories (train/valid/test)
            has_splits = any((input_folder / s).is_dir() for s in ("train", "valid", "test"))

            if has_splits and params["create_yaml"]:
                summary = convert_coco_dataset_splits(input_folder, output_folder)
                total_imgs = sum(s.get("images_converted", 0) for s in summary.values() if isinstance(s, dict))
                total_anns = sum(s.get("total_annotations", 0) for s in summary.values() if isinstance(s, dict))
                yaml_path = summary.get("data_yaml", "")
                cat_names = summary.get("class_names", [])

                details: dict[str, Any] = {
                    "Mode": "Multi-split conversion (train/valid/test)",
                    "Total Images": total_imgs,
                    "Total Annotations": total_anns,
                    "Classes": ", ".join(cat_names),
                    "Dataset Config": str(yaml_path),
                    "Output Directory": str(output_folder),
                }
                for split in ("train", "valid", "test"):
                    if split in summary:
                        s_info = summary[split]
                        details[f"Split '{split}' Images"] = s_info.get("images_converted", 0)
                        details[f"Split '{split}' Annotations"] = s_info.get("total_annotations", 0)

                message = f"Successfully converted dataset ({total_imgs} images, {total_anns} annotations) to YOLO format."
            else:
                res = coco_to_yolo(input_folder, output_folder)
                cat_summary = ", ".join(f"{k}: {v}" for k, v in res.get("category_counts", {}).items())
                details = {
                    "Images Converted": res["images_converted"],
                    "Images Copied": res["images_copied"],
                    "Total Annotations": res["total_annotations"],
                    "Category Distribution": cat_summary or "None",
                    "Images Directory": res["images_dir"],
                    "Labels Directory": res["labels_dir"],
                }
                message = f"Converted {res['images_converted']} images and {res['total_annotations']} annotations."

        except Exception as err:
            if parent is not None:
                show_error(parent, "Conversion Failed", str(err))
            return None

        return ToolResult(
            message=message,
            image=None,
            details=details,
        )


class YOLOTrainingTool(ForensicsTool):
    """Set 9.2: Model training controls using model.train() with parameters from PDF."""

    tool_id = "yolo_training"
    title = "YOLO model trainer"
    category = "Set 9: Object Detection with YOLO"
    description = "Train YOLO model on custom dataset (data.yaml, epochs=50, batch=16, from_scratch_50e_v1)."
    requires_image = False

    def run(self, parent: tk.Misc | None, document: ImageDocument) -> ToolResult | None:
        if not ULTRALYTICS_AVAILABLE:
            if parent is not None:
                show_error(parent, "Ultralytics Required", "Please install ultralytics to train YOLO.")
            return None

        _coco_dir, yolo_dir = find_blood_cell_dataset()
        default_yaml = ""
        if yolo_dir:
            cand = yolo_dir / "data.yaml"
            if cand.is_file():
                default_yaml = str(cand)

        params = ask_yolo_training_parameters(
            parent,
            default_yaml=default_yaml,
            default_epochs=50,
            default_batch=16,
            default_name="from_scratch_50e_v1",
        )
        if params is None:
            return None

        # Build progress popup dialog to keep UI responsive
        progress_data: dict[str, Any] = {"result": None, "error": None, "done": False}
        status_var = tk.StringVar(value="Initializing training process...")

        prog_win: tk.Toplevel | None = None
        if parent is not None:
            prog_win = tk.Toplevel(parent)
            prog_win.title("Training YOLO Model")
            prog_win.transient(parent)
            prog_win.resizable(False, False)
            prog_win.geometry("450x160")

            pf = ttk.Frame(prog_win, padding=16)
            pf.pack(fill="both", expand=True)

            ttk.Label(
                pf,
                text=f"Training YOLO: {params['name']}",
                font=("TkDefaultFont", 10, "bold"),
                foreground="#0055aa",
            ).pack(anchor="w", pady=(0, 6))

            ttk.Label(
                pf,
                text=f"Epochs: {params['epochs']}  |  Batch: {params['batch']}  |  Model: {params['model_cfg']}",
                font=("TkDefaultFont", 9),
            ).pack(anchor="w", pady=(0, 10))

            pbar = ttk.Progressbar(pf, mode="indeterminate")
            pbar.pack(fill="x", pady=(0, 8))
            pbar.start(15)

            status_lbl = ttk.Label(pf, textvariable=status_var, font=("TkDefaultFont", 9, "italic"))
            status_lbl.pack(anchor="w")

        def _worker() -> None:
            try:
                res = train_yolo_model(
                    data_yaml_path=params["data_yaml"],
                    model_cfg_or_weights=params["model_cfg"],
                    name=params["name"],
                    epochs=params["epochs"],
                    batch=params["batch"],
                    classes=params["classes"],
                    plots=params["plots"],
                    progress_callback=lambda msg: status_var.set(msg),
                )
                progress_data["result"] = res
            except Exception as err:
                progress_data["error"] = err
            finally:
                progress_data["done"] = True
                if prog_win is not None:
                    try:
                        prog_win.after(100, prog_win.destroy)
                    except Exception:
                        pass

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()

        if prog_win is not None:
            prog_win.grab_set()
            prog_win.wait_window()
        else:
            thread.join()

        if progress_data["error"] is not None:
            if parent is not None:
                show_error(parent, "Training Error", str(progress_data["error"]))
            return None

        summary = progress_data["result"]
        if summary is None:
            return None

        # Load results plot or confusion matrix if available
        result_image: Image.Image | None = None
        plots = summary.get("plots", [])
        for p in plots:
            if p.endswith("results.png") or p.endswith("confusion_matrix.png"):
                try:
                    result_image = Image.open(p).convert("RGB")
                    break
                except Exception:
                    pass

        best_w = summary.get("best_weights")
        save_dir = summary.get("save_dir", "")
        message = (
            f"Training completed successfully ({params['epochs']} epochs)! "
            f"Weights saved to: {Path(best_w).name if best_w else save_dir}"
        )

        details: dict[str, Any] = {
            "Run Name": params["name"],
            "Epochs Completed": params["epochs"],
            "Batch Size": params["batch"],
            "Classes Trained": str(params["classes"]),
            "Best Weights": best_w or "weights/best.pt",
            "Output Directory": save_dir,
            "Plots Generated": len(plots),
        }

        return ToolResult(
            message=message,
            image=result_image,
            details=details,
        )


class YOLOComparisonTool(ForensicsTool):
    """Set 9.3: Test model on unseen validation/test images and compare Ground Truth vs Predictions."""

    tool_id = "yolo_comparison"
    title = "YOLO test & GT comparison"
    category = "Set 9: Object Detection with YOLO"
    description = "Test trained YOLO model on unseen images and visualize Ground Truth vs Predictions (RBC/WBC/PC)."
    requires_image = False

    def run(self, parent: tk.Misc | None, document: ImageDocument) -> ToolResult | None:
        if not ULTRALYTICS_AVAILABLE:
            if parent is not None:
                show_error(parent, "Ultralytics Required", "Please install ultralytics for YOLO evaluation.")
            return None

        _coco_dir, yolo_dir = find_blood_cell_dataset()
        default_img = ""
        default_lbl = ""
        if yolo_dir:
            test_img_dir = yolo_dir / "test" / "images"
            if test_img_dir.is_dir():
                test_imgs = list(test_img_dir.glob("*.jpg")) + list(test_img_dir.glob("*.png"))
                if test_imgs:
                    default_img = str(test_imgs[0])
                    cand_lbl = Path(str(test_imgs[0]).replace("images", "labels")).with_suffix(".txt")
                    if cand_lbl.is_file():
                        default_lbl = str(cand_lbl)

        default_model = get_active_yolo_model_path() or find_default_yolo_weights() or "yolo11n.pt"

        params = ask_yolo_comparison_parameters(
            parent,
            default_image=default_img,
            default_label=default_lbl,
            default_model=str(default_model),
        )
        if params is None:
            return None

        image_path = params["image_path"]
        label_path = params["label_path"]
        model_path = params["model_path"] or default_model

        try:
            annotated_img, details = visualize_unseen_comparison(
                image_input=image_path,
                label_path=label_path,
                model_or_path=model_path,
                conf_threshold=params["conf"],
                iou_threshold=params["iou"],
                category_names=BLOOD_CELL_CATEGORIES,
                mode=params["mode"],
            )
        except Exception as err:
            if parent is not None:
                show_error(parent, "Comparison Failed", str(err))
            return None

        gt_count = details.get("Ground Truth Boxes", 0)
        pred_count = details.get("Predicted Boxes", 0)
        prec = details.get("Precision", "0.0")
        rec = details.get("Recall", "0.0")

        message = (
            f"Compared {Path(image_path).name}: {gt_count} Ground Truth vs {pred_count} Predictions "
            f"(Precision: {prec}, Recall: {rec})"
        )

        return ToolResult(
            message=message,
            image=annotated_img,
            details=details,
        )


__all__ = [
    "YOLOInferenceTool",
    "YOLODataConversionTool",
    "YOLOTrainingTool",
    "YOLOComparisonTool",
]
