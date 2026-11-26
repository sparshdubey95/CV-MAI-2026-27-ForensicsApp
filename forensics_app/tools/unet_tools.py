"""Tkinter tools for Set 11: U-Net Binary Food Segmentation."""

from __future__ import annotations

from pathlib import Path
import threading
import tkinter as tk
from tkinter import ttk
from typing import Any

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
from .dialogs import (
    ask_unet_segmentation_parameters,
    ask_unet_training_parameters,
    show_error,
)
from .unet_core import (
    DEFAULT_INPUT_HEIGHT,
    DEFAULT_INPUT_WIDTH,
    DEFAULT_THRESHOLD,
    UNetTrainingCancelled,
    evaluate_unet_on_test_set,
    find_default_tray_dataset,
    get_active_unet_checkpoint_path,
    get_default_unet_checkpoint_path,
    load_unet_checkpoint,
    render_segmentation_visualization,
    render_test_set_grid,
    render_unet_training_report,
    segment_single_image,
    train_unet_model,
    unet_dependencies_available,
)


def _run_unet_with_progress(
    parent: tk.Misc | None,
    title: str,
    subtitle: str,
    worker: Any,
    allow_cancel: bool = True,
) -> tuple[Any, BaseException | None, bool]:
    """Run a U-Net training or evaluation worker with a Tk progress dialog."""
    state: dict[str, Any] = {"result": None, "error": None, "cancelled": False}
    cancel_event = threading.Event()
    prog_win: tk.Toplevel | None = None
    status_var: tk.StringVar | None = None

    if parent is not None:
        prog_win = tk.Toplevel(parent)
        prog_win.title(title)
        prog_win.transient(parent)
        prog_win.resizable(False, False)
        prog_win.geometry("500x165")

        frame = ttk.Frame(prog_win, padding=16)
        frame.pack(fill="both", expand=True)
        ttk.Label(
            frame,
            text=subtitle,
            font=("TkDefaultFont", 10, "bold"),
            foreground="#0055aa",
        ).pack(anchor="w", pady=(0, 8))
        pbar = ttk.Progressbar(frame, mode="indeterminate")
        pbar.pack(fill="x", pady=(0, 8))
        pbar.start(15)
        status_var = tk.StringVar(value="Starting...")
        ttk.Label(
            frame,
            textvariable=status_var,
            font=("TkDefaultFont", 9, "italic"),
            wraplength=460,
        ).pack(anchor="w")

        def _cancel() -> None:
            state["cancelled"] = True
            cancel_event.set()
            if status_var is not None:
                status_var.set("Cancelling...")

        if allow_cancel:
            ttk.Button(frame, text="Cancel", command=_cancel).pack(anchor="e", pady=(10, 0))
            prog_win.protocol("WM_DELETE_WINDOW", _cancel)

    def _on_progress(message: str) -> None:
        if status_var is not None and prog_win is not None:
            try:
                prog_win.after(0, lambda m=message: status_var.set(m))
            except Exception:
                pass

    def _thread_main() -> None:
        try:
            state["result"] = worker(_on_progress, cancel_event)
        except UNetTrainingCancelled as exc:
            state["error"] = exc
            state["cancelled"] = True
        except Exception as exc:
            state["error"] = exc
        finally:
            if prog_win is not None:
                try:
                    prog_win.after(60, prog_win.destroy)
                except Exception:
                    pass

    thread = threading.Thread(target=_thread_main, daemon=True)
    thread.start()
    if prog_win is not None:
        prog_win.grab_set()
        prog_win.wait_window()
    else:
        thread.join()

    return state["result"], state["error"], bool(state["cancelled"])


class UNetTrainingTool(ForensicsTool):
    """Set 11.1: Train a binary food-vs-not-food U-Net on TrayDataset and save a checkpoint."""

    tool_id = "unet_training"
    title = "U-Net model trainer"
    category = "Set 11: U-Net Segmentation"
    description = "Train a PyTorch U-Net for binary food segmentation on TrayDataset and save a checkpoint."
    requires_image = False

    def run(self, parent: tk.Misc | None, document: ImageDocument) -> ToolResult | None:
        if not unet_dependencies_available():
            if parent is not None:
                show_error(parent, "PyTorch Required", "PyTorch is required to train the U-Net model.")
            return None

        default_dataset = find_default_tray_dataset()
        default_ckpt = get_default_unet_checkpoint_path()

        params = ask_unet_training_parameters(
            parent,
            default_dataset_dir=str(default_dataset) if default_dataset else "",
            default_checkpoint_path=str(default_ckpt),
            default_epochs=10,
            default_batch_size=16,
            default_lr=1e-3,
            default_height=DEFAULT_INPUT_HEIGHT,
            default_width=DEFAULT_INPUT_WIDTH,
        )
        if params is None:
            return None

        def _worker(progress_cb: Any, cancel_event: threading.Event) -> dict[str, Any]:
            summary = train_unet_model(
                dataset_root=params["dataset_dir"],
                checkpoint_path=params["checkpoint_path"],
                epochs=params["epochs"],
                batch_size=params["batch_size"],
                learning_rate=params["learning_rate"],
                val_fraction=params["val_fraction"],
                input_height=params["input_height"],
                input_width=params["input_width"],
                progress_callback=progress_cb,
                cancel_event=cancel_event,
            )
            summary["report_image"] = render_unet_training_report(summary)
            return summary

        summary, error, cancelled = _run_unet_with_progress(
            parent,
            title="U-Net Food Segmentation Training",
            subtitle="Training U-Net on TrayDataset (XTrain / yTrain)",
            worker=_worker,
            allow_cancel=True,
        )
        if cancelled:
            return None
        if error is not None:
            if parent is not None:
                show_error(parent, "U-Net Training Failed", str(error))
            return None

        pairing = summary["pairing_info"]
        details: dict[str, Any] = {
            "Checkpoint": summary["checkpoint_path"],
            "Device": summary["device"],
            "Epochs": summary["epochs_trained"],
            "Train Samples": summary["train_samples"],
            "Val Samples": summary["val_samples"],
            "Best Val Dice": f"{summary['best_val_dice']:.4f}",
            "Final Train Loss": f"{summary['final_train_loss']:.4f}",
            "Input Size": f"{summary['config']['input_height']} x {summary['config']['input_width']}",
        }
        if pairing.get("warning"):
            details["Pairing Note"] = pairing["warning"]

        return ToolResult(
            message=(
                f"Trained U-Net for {summary['epochs_trained']} epochs "
                f"(Best Val Dice = {summary['best_val_dice']:.3f}). "
                f"Saved checkpoint to {Path(summary['checkpoint_path']).name}."
            ),
            image=summary["report_image"],
            details=details,
        )


class UNetSegmentationTool(ForensicsTool):
    """Set 11.2: Load a saved U-Net checkpoint and segment an image or the TrayDataset test set."""

    tool_id = "unet_segmentation"
    title = "U-Net segment & test"
    category = "Set 11: U-Net Segmentation"
    description = "Load a saved U-Net checkpoint, run food segmentation, and visualize masks/Dice."
    requires_image = False

    def run(self, parent: tk.Misc | None, document: ImageDocument) -> ToolResult | None:
        if not unet_dependencies_available():
            if parent is not None:
                show_error(parent, "PyTorch Required", "PyTorch is required to run U-Net segmentation.")
            return None

        default_ckpt = get_active_unet_checkpoint_path() or str(get_default_unet_checkpoint_path())
        default_dataset = find_default_tray_dataset()

        params = ask_unet_segmentation_parameters(
            parent,
            current_image_loaded=document.is_loaded,
            default_checkpoint_path=default_ckpt,
            default_dataset_dir=str(default_dataset) if default_dataset else "",
            default_threshold=DEFAULT_THRESHOLD,
        )
        if params is None:
            return None

        try:
            model, config, metadata, device = load_unet_checkpoint(params["checkpoint_path"])
        except Exception as exc:
            if parent is not None:
                show_error(parent, "Could Not Load Checkpoint", str(exc))
            return None

        threshold = float(params.get("threshold", DEFAULT_THRESHOLD))
        mode = params.get("mode", "image")

        if mode == "test_set":
            def _test_worker(progress_cb: Any, _cancel: threading.Event) -> dict[str, Any]:
                eval_res = evaluate_unet_on_test_set(
                    model=model,
                    config=config,
                    device=device,
                    dataset_root=params["dataset_dir"],
                    threshold=threshold,
                    progress_callback=progress_cb,
                )
                eval_res["grid_image"] = render_test_set_grid(eval_res)
                return eval_res

            eval_res, error, _ = _run_unet_with_progress(
                parent,
                title="U-Net Test Set Evaluation",
                subtitle="Evaluating U-Net on TrayDataset (XTest / yTest)",
                worker=_test_worker,
                allow_cancel=False,
            )
            if error is not None:
                if parent is not None:
                    show_error(parent, "Test Set Evaluation Failed", str(error))
                return None

            details: dict[str, Any] = {
                "Checkpoint": Path(params["checkpoint_path"]).name,
                "Device": device,
                "Test Images": eval_res["n_test"],
                "Mean Test Dice": f"{eval_res['mean_dice']:.4f}",
                "Std Test Dice": f"{eval_res['std_dice']:.4f}",
                "Min / Max Dice": f"{eval_res['min_dice']:.4f} / {eval_res['max_dice']:.4f}",
                "Threshold": f"{threshold:.2f}",
            }
            if eval_res["pairing_info"].get("warning"):
                details["Pairing Note"] = eval_res["pairing_info"]["warning"]
            if "best_val_dice" in metadata:
                details["Checkpoint Val Dice"] = f"{float(metadata['best_val_dice']):.4f}"

            return ToolResult(
                message=(
                    f"Test set evaluation ({eval_res['n_test']} images): "
                    f"Mean Dice = {eval_res['mean_dice']:.3f} ± {eval_res['std_dice']:.3f}."
                ),
                image=eval_res["grid_image"],
                details=details,
            )

        # Single image mode
        image_source: Any
        if params.get("custom_image_path"):
            image_source = params["custom_image_path"]
        elif document.is_loaded:
            image_source = document.current
        else:
            if parent is not None:
                show_error(parent, "No Image Selected", "Please load an image or select an image file.")
            return None

        try:
            seg_res = segment_single_image(
                model=model,
                image_input=image_source,
                config=config,
                device=device,
                gt_mask_path=params.get("gt_mask_path"),
                threshold=threshold,
            )
        except Exception as exc:
            if parent is not None:
                show_error(parent, "Segmentation Failed", str(exc))
            return None

        source_name = (
            document.path.name
            if params.get("use_current_image") and document.path is not None
            else seg_res["image_source"]
        )
        seg_res["image_source"] = source_name
        rendered = render_segmentation_visualization(
            seg_res,
            title=f"U-Net Food Segmentation — {source_name}",
        )

        details = {
            "Checkpoint": Path(params["checkpoint_path"]).name,
            "Device": device,
            "Image": source_name,
            "Model Input Size": f"{config['input_height']} x {config['input_width']}",
            "Threshold": f"{threshold:.2f}",
            "Predicted Food Area": f"{seg_res['food_fraction'] * 100:.1f}%",
        }
        if seg_res["dice_score"] is not None:
            details["Dice Score"] = f"{seg_res['dice_score']:.4f}"
            message = (
                f"Segmented {source_name}: {seg_res['food_fraction'] * 100:.1f}% food "
                f"(Dice = {seg_res['dice_score']:.3f})."
            )
        else:
            message = f"Segmented {source_name}: {seg_res['food_fraction'] * 100:.1f}% predicted food."

        return ToolResult(
            message=message,
            image=rendered,
            details=details,
        )
