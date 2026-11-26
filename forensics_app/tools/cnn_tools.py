"""Tkinter tools for Image Classification using CNN (Set 8).

Integrates:
  - 8.1 CNN Model Training, Saving Weights & Generating Passport (CNNTrainingTool)
  - 8.2 Loading Pretrained Model & Passport, Inference (CNNInferenceTool)
  - 8.3 Feature Vector Extraction, Saving/Loading & 2D Projection via t-SNE & UMAP (CNNFeatureProjectionTool)
"""

from __future__ import annotations

import os
from pathlib import Path
import threading
import tkinter as tk
from tkinter import ttk
from typing import Any

import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
from .cnn_core import (
    CIFAR10_CLASSES,
    TENSORFLOW_AVAILABLE,
    UMAP_AVAILABLE,
    TrainingCancelled,
    compute_tsne_projection,
    compute_umap_projection,
    extract_cifar10_features,
    find_cifar10_batches_dir,
    find_default_cnn_passport,
    find_default_cnn_weights,
    find_default_features_path,
    find_sample_cifar_images,
    get_active_cnn_model,
    get_active_cnn_passport_path,
    get_active_cnn_weights_path,
    get_default_user_passport_path,
    get_default_user_weights_path,
    is_bundled_model_path,
    load_extracted_features,
    load_pretrained_cnn,
    prepare_features_for_projection,
    render_classification_visualization,
    render_feature_projections,
    render_training_report,
    run_cnn_inference,
    train_cnn_model,
    umap_unavailable_message,
)
from .dialogs import (
    ask_cnn_inference_parameters,
    ask_cnn_projection_parameters,
    ask_cnn_training_parameters,
    show_cnn_passport_dialog,
    show_error,
)


def _run_with_progress(
    parent: tk.Misc | None,
    title: str,
    subtitle: str,
    worker: Any,
    allow_cancel: bool = True,
) -> tuple[Any, BaseException | None, bool]:
    """Run worker(progress_callback, cancel_event) with an optional Tk progress dialog."""
    progress_data: dict[str, Any] = {"result": None, "error": None, "done": False, "cancelled": False}
    cancel_event = threading.Event()
    status_holder = {"text": "Starting..."}

    prog_win: tk.Toplevel | None = None
    status_var: tk.StringVar | None = None
    if parent is not None:
        prog_win = tk.Toplevel(parent)
        prog_win.title(title)
        prog_win.transient(parent)
        prog_win.resizable(False, False)
        prog_win.geometry("480x170")

        pf = ttk.Frame(prog_win, padding=16)
        pf.pack(fill="both", expand=True)
        ttk.Label(pf, text=subtitle, font=("TkDefaultFont", 10, "bold"), foreground="#0055aa").pack(
            anchor="w", pady=(0, 8)
        )
        pbar = ttk.Progressbar(pf, mode="indeterminate")
        pbar.pack(fill="x", pady=(0, 8))
        pbar.start(15)
        status_var = tk.StringVar(value="Starting...")
        ttk.Label(pf, textvariable=status_var, font=("TkDefaultFont", 9, "italic"), wraplength=440).pack(
            anchor="w"
        )

        def _cancel() -> None:
            progress_data["cancelled"] = True
            cancel_event.set()
            if status_var is not None:
                status_var.set("Cancelling...")

        if allow_cancel:
            ttk.Button(pf, text="Cancel", command=_cancel).pack(anchor="e", pady=(12, 0))
            prog_win.protocol("WM_DELETE_WINDOW", _cancel)

    def _on_progress(message: str) -> None:
        status_holder["text"] = message
        if status_var is not None and prog_win is not None:
            try:
                prog_win.after(0, lambda m=message: status_var.set(m))
            except Exception:
                pass

    def _thread_main() -> None:
        try:
            progress_data["result"] = worker(_on_progress, cancel_event)
        except TrainingCancelled as err:
            progress_data["error"] = err
            progress_data["cancelled"] = True
        except Exception as err:
            progress_data["error"] = err
        finally:
            progress_data["done"] = True
            if prog_win is not None:
                try:
                    prog_win.after(100, prog_win.destroy)
                except Exception:
                    pass

    thread = threading.Thread(target=_thread_main, daemon=True)
    thread.start()
    if prog_win is not None:
        prog_win.grab_set()
        prog_win.wait_window()
    else:
        thread.join()
    return progress_data["result"], progress_data["error"], bool(progress_data["cancelled"])


class CNNTrainingTool(ForensicsTool):
    """Set 8.1: Train a Keras CNN on CIFAR-10, save weights, and generate a model passport."""

    tool_id = "cnn_training"
    title = "CNN model trainer"
    category = "Set 8: Image Classification using CNN"
    description = "Train a Keras CNN on CIFAR-10, save model weights, and generate a model passport."
    requires_image = False

    def run(self, parent: tk.Misc | None, document: ImageDocument) -> ToolResult | None:
        if not TENSORFLOW_AVAILABLE:
            if parent is not None:
                show_error(
                    parent,
                    "TensorFlow Required",
                    "TensorFlow is required for CNN training.\nPlease install tensorflow (see requirements.txt).",
                )
            return None

        try:
            default_dataset = str(find_cifar10_batches_dir())
        except Exception:
            default_dataset = str(Path("data") / "cifar-10-batches-py")

        params = ask_cnn_training_parameters(
            parent,
            default_arch="shallow",
            default_samples=2000,
            default_epochs=5,
            default_batch=32,
            default_weights=str(get_default_user_weights_path()),
            default_passport=str(get_default_user_passport_path()),
            default_dataset=default_dataset,
        )
        if params is None:
            return None

        weights_path = params.get("weights_path")
        passport_path = params.get("passport_path")
        if is_bundled_model_path(weights_path) or is_bundled_model_path(passport_path):
            if parent is not None:
                show_error(
                    parent,
                    "Bundled Model Protected",
                    "The default bundled CIFAR-10 model is write-protected from this dialog.\n"
                    "Choose a path under cifar10/user_runs/ (the default) instead of "
                    "cifar10_cnn_model.weights.h5 / cifar10_cnn_passport.json.",
                )
            return None

        def _worker(progress_callback: Any, cancel_event: Any) -> tuple[Any, dict[str, Any], Path, Path]:
            return train_cnn_model(
                architecture=params.get("architecture", "shallow"),
                sample_size=params.get("sample_size", 2000),
                epochs=params.get("epochs", 5),
                batch_size=params.get("batch_size", 32),
                optimizer_name=params.get("optimizer", "sgd"),
                learning_rate=params.get("learning_rate", 0.01),
                weights_path=weights_path,
                passport_path=passport_path,
                dataset_dir=params.get("dataset_dir"),
                random_seed=params.get("random_seed", 42),
                progress_callback=progress_callback,
                cancel_event=cancel_event,
                max_test_eval=params.get("max_test_eval"),
            )

        result, error, cancelled = _run_with_progress(
            parent,
            title="Training CNN",
            subtitle=f"Training {params.get('architecture', 'shallow')} on official CIFAR-10",
            worker=_worker,
        )
        if cancelled:
            if parent is not None:
                show_error(parent, "Training Cancelled", "CNN training was cancelled. No bundled files were replaced.")
            return None
        if error is not None:
            if parent is not None:
                show_error(parent, "Training Error", f"Failed to train CNN model: {error}")
            raise error

        model, passport, w_path, p_path = result
        del model

        report_img = render_training_report(
            passport["training_history"],
            passport.get("confusion_matrix"),
            class_names=passport.get("class_names", list(CIFAR10_CLASSES)),
            title=f"CIFAR-10 {params.get('architecture', 'shallow').upper()} Training History",
            test_accuracy=passport.get("metrics", {}).get("test_accuracy"),
        )

        metrics = passport.get("metrics", {})
        acc_pct = float(metrics.get("final_accuracy", 0.0)) * 100.0
        val_acc = metrics.get("val_accuracy")
        test_acc = metrics.get("test_accuracy")
        val_loss = metrics.get("val_loss")
        test_loss = metrics.get("test_loss")
        val_str = (
            f" | Val Loss: {val_loss:.4f} | Val Acc: {val_acc * 100:.1f}%"
            if val_acc is not None and val_loss is not None else ""
        )
        test_str = (
            f" | Test Loss: {test_loss:.4f} | Test Acc: {test_acc * 100:.1f}%"
            if test_acc is not None and test_loss is not None else ""
        )

        message = (
            f"Successfully trained {passport.get('model_name', 'CNN')} on official CIFAR-10 "
            f"({passport.get('dataset_source', 'local batches')}).\n"
            f"Train/val/test samples: {passport.get('train_samples')}/"
            f"{passport.get('val_samples')}/{passport.get('test_samples')} | "
            f"Seed: {passport.get('random_seed')}\n"
            f"Epochs: {params.get('epochs')} | Final Loss: {metrics.get('final_loss', 0.0):.4f} | "
            f"Train Acc: {acc_pct:.1f}%{val_str}{test_str}\n"
            f"Weights saved to: {w_path}\n"
            f"Model passport saved to: {p_path}"
        )

        return ToolResult(
            message=message,
            image=report_img,
            details={
                "passport": passport,
                "weights_path": str(w_path),
                "passport_path": str(p_path),
                "metrics": metrics,
            },
        )


class CNNInferenceTool(ForensicsTool):
    """Set 8.2: Load pretrained CNN model & passport, view passport, and classify an image."""

    tool_id = "cnn_inference"
    title = "CNN model inference & passport"
    category = "Set 8: Image Classification using CNN"
    description = "Load pretrained model & passport, view passport, and classify an image using correct preprocessing."
    requires_image = False

    def run(self, parent: tk.Misc | None, document: ImageDocument) -> ToolResult | None:
        if not TENSORFLOW_AVAILABLE:
            if parent is not None:
                show_error(
                    parent,
                    "TensorFlow Required",
                    "TensorFlow is required for CNN inference.\nPlease install tensorflow (see requirements.txt).",
                )
            return None

        active_w = get_active_cnn_weights_path()
        if active_w and os.path.isfile(active_w):
            default_weights = active_w
        else:
            default_weights = find_default_cnn_weights()

        active_p = get_active_cnn_passport_path()
        default_passport = active_p if active_p and os.path.isfile(active_p) else find_default_cnn_passport()
        sample_images = find_sample_cifar_images()

        params = ask_cnn_inference_parameters(
            parent,
            current_image_loaded=document.is_loaded,
            default_weights=default_weights,
            default_passport=default_passport,
            sample_images=sample_images,
        )
        if params is None:
            return None

        weights_path = params.get("weights_path")
        if not weights_path or not os.path.isfile(weights_path):
            if parent is not None:
                show_error(
                    parent,
                    "Model Not Found",
                    "Select an existing weights file (.weights.h5). Train a model in 8.1 if you do not have one yet.",
                )
            return None

        passport_path = params.get("passport_path")
        if not passport_path or not os.path.isfile(passport_path):
            if parent is not None:
                show_error(
                    parent,
                    "Passport Not Found",
                    "Select the JSON passport saved with these weights. Inference will not guess an architecture.",
                )
            return None

        try:
            model, passport = load_pretrained_cnn(weights_path, passport_path)
        except Exception as e:
            if parent is not None:
                show_error(parent, "Load Error", f"Failed to load pretrained model: {e}")
            raise e

        if params.get("show_passport") and parent is not None and passport:
            show_cnn_passport_dialog(parent, passport)

        input_image: Image.Image | None = None
        if params.get("use_current_image") and document.is_loaded and document.current is not None:
            input_image = document.current
        elif params.get("custom_image_path") and os.path.isfile(params["custom_image_path"]):
            input_image = Image.open(params["custom_image_path"]).convert("RGB")

        if input_image is None:
            if parent is not None:
                show_error(
                    parent,
                    "Image Required",
                    "Select an image to classify. The tool will not silently use a random or bundled sample.",
                )
            return None

        pred_result = run_cnn_inference(model, input_image, passport)
        pred_class = pred_result["predicted_class"]
        confidence = pred_result["confidence"]
        probabilities = pred_result["probabilities"]

        vis_img = render_classification_visualization(
            input_image,
            predicted_class=pred_class,
            confidence=confidence,
            probabilities=probabilities,
            passport=passport,
        )

        prob_lines = ", ".join(f"{name}={prob * 100:.1f}%" for name, prob in probabilities.items())
        message = (
            f"Classification Inference Result:\n"
            f"Predicted Class: {pred_class.upper()} (Confidence: {confidence * 100.0:.1f}%)\n"
            f"All class probabilities: {prob_lines}\n"
            f"Model: {passport.get('model_name', 'CNN')} | Preprocessing: "
            f"{passport.get('preprocessing_instructions', {}).get('description', '32x32 RGB / 255.0')}\n"
            "Note: CIFAR-10 models expect small 32×32 images. Arbitrary photographs are resized "
            "to that resolution and may be classified poorly."
        )

        return ToolResult(
            message=message,
            image=vis_img,
            details={
                "predicted_class": pred_class,
                "confidence": confidence,
                "probabilities": probabilities,
                "passport": passport,
                "weights_path": str(weights_path),
            },
        )


class CNNFeatureProjectionTool(ForensicsTool):
    """Set 8.3: Feature extraction from CIFAR-10 test set, save/reload, and 2D projections via t-SNE & UMAP."""

    tool_id = "cnn_feature_projection"
    title = "CNN feature projection (t-SNE & UMAP)"
    category = "Set 8: Image Classification using CNN"
    description = "Extract feature vectors from CIFAR-10 test set, save/reload features, and project in 2D with t-SNE & UMAP."
    requires_image = False

    def run(self, parent: tk.Misc | None, document: ImageDocument) -> ToolResult | None:
        default_features = find_default_features_path()
        default_weights = get_active_cnn_weights_path() or find_default_cnn_weights()
        active_p = get_active_cnn_passport_path()
        default_passport = active_p if active_p and os.path.isfile(active_p) else find_default_cnn_passport()

        params = ask_cnn_projection_parameters(
            parent,
            default_features_path=default_features,
            default_samples_per_class=50,
            default_perplexity=40.0,
            default_weights=default_weights,
            default_passport=default_passport,
            umap_available=UMAP_AVAILABLE,
        )
        if params is None:
            return None

        action = params.get("action", "load")
        features_path = params.get("features_path") or default_features
        bundle: dict[str, Any] | None = None
        notes: list[str] = []

        if action == "load":
            if not features_path or not os.path.isfile(features_path):
                if parent is not None:
                    show_error(parent, "Features Not Found", "Select an existing .npz feature file or choose Extract.")
                return None
            try:
                bundle = load_extracted_features(features_path)
            except Exception as e:
                if parent is not None:
                    show_error(parent, "Load Error", f"Failed to load features: {e}")
                raise e
        else:
            if not TENSORFLOW_AVAILABLE:
                if parent is not None:
                    show_error(parent, "TensorFlow Required", "TensorFlow is required to extract CNN features.")
                return None
            model = get_active_cnn_model()
            weights_path = params.get("weights_path")
            passport_path = params.get("passport_path")
            if model is None and (not weights_path or not os.path.isfile(weights_path)):
                if parent is not None:
                    show_error(
                        parent,
                        "Trained Model Required",
                        "Feature extraction needs trained CNN weights. Train a model or load a weights file. "
                        "A randomly initialised network is not used.",
                    )
                return None
            try:
                bundle = extract_cifar10_features(
                    model=model,
                    samples_per_class=params.get("samples_per_class", 50),
                    save_path=features_path,
                    weights_path=weights_path,
                    passport_path=passport_path,
                )
            except Exception as e:
                if parent is not None:
                    show_error(parent, "Extraction Error", f"Failed to extract features: {e}")
                raise e

        features = bundle["features"]
        labels = bundle["labels"]
        class_names = [str(c) for c in bundle.get("class_names", list(CIFAR10_CLASSES))]
        method = params.get("method", "both")

        original_feature_count = len(features)
        features, prepared_labels, retained_mask = prepare_features_for_projection(
            features,
            labels=labels,
            random_state=42,
        )
        if prepared_labels is None:
            raise ValueError("CIFAR-10 labels are required to project features by class.")
        labels = prepared_labels
        counts = [int(np.sum(labels == class_id)) for class_id in range(10)]
        if original_feature_count != int(np.sum(retained_mask)):
            notes.append(
                f"Removed {original_feature_count - int(np.sum(retained_mask))} invalid or excess unbalanced feature rows."
            )

        tsne_2d = None
        umap_2d = None

        if method in ("both", "tsne"):
            tsne_2d = compute_tsne_projection(
                features,
                perplexity=params.get("perplexity", 40.0),
                n_iter=1000,
                random_state=42,
                prepared=True,
            )

        if method in ("both", "umap"):
            if UMAP_AVAILABLE:
                try:
                    umap_2d = compute_umap_projection(
                        features,
                        n_neighbors=params.get("umap_neighbors", 15),
                        min_dist=0.1,
                        metric="euclidean",
                        random_state=42,
                        prepared=True,
                    )
                except Exception as exc:
                    notes.append(f"UMAP failed ({exc}). Showing t-SNE only." if tsne_2d is not None else f"UMAP failed ({exc}).")
                    umap_2d = None
                    if method == "umap" and tsne_2d is None:
                        if parent is not None:
                            show_error(parent, "UMAP Failed", str(exc))
                        return None
            else:
                notes.append(umap_unavailable_message())
                if method == "umap":
                    if parent is not None:
                        show_error(parent, "UMAP Unavailable", umap_unavailable_message())
                    return None

        vis_img = render_feature_projections(
            tsne_2d=tsne_2d,
            umap_2d=umap_2d,
            labels=labels,
            class_names=class_names,
            title=f"Set 8.3: CIFAR-10 Feature Space Projections ({len(labels)} test samples)",
        )

        methods_used = []
        if tsne_2d is not None:
            methods_used.append("t-SNE")
        if umap_2d is not None:
            methods_used.append("UMAP")

        model_accuracy = bundle.get("model_test_accuracy")
        accuracy_line = (
            f"Recorded official test accuracy: {model_accuracy * 100:.1f}%.\n"
            if model_accuracy is not None else ""
        )

        message = (
            f"CIFAR-10 Feature Projection Complete:\n"
            f"Extracted/loaded {len(labels)} balanced feature vectors ({bundle.get('feature_dim', features.shape[1])} dims) "
            f"from layer {bundle.get('extractor_layer', 'n/a')}.\n"
            f"Model: {bundle.get('model_name', 'n/a')} | Weights: {bundle.get('weights_path', 'n/a')}\n"
            f"{accuracy_line}"
            f"Dataset: {bundle.get('dataset_source', 'n/a')}\n"
            f"Samples per class: {counts[0]} (all ten classes balanced).\n"
            f"Projected into 2D via {', '.join(methods_used) or 'no method'} using standardized CNN features.\n"
            f"Features source: {bundle.get('saved_path', 'memory')}"
        )
        if notes:
            message += "\n" + "\n".join(notes)

        return ToolResult(
            message=message,
            image=vis_img,
            details={
                "num_samples": len(labels),
                "feature_dim": int(features.shape[1]),
                "model_test_accuracy": model_accuracy,
                "samples_per_class": counts[0],
                "class_counts": {class_names[i]: counts[i] for i in range(10)},
                "methods": methods_used,
                "projection_preprocessing": "drop non-finite/zero vectors; deterministic class balancing; StandardScaler",
                "tsne_settings": {"perplexity": params.get("perplexity", 40.0), "iterations": 1000, "learning_rate": "auto", "seed": 42} if tsne_2d is not None else None,
                "umap_settings": {"neighbors": params.get("umap_neighbors", 15), "min_dist": 0.1, "metric": "euclidean", "seed": 42} if umap_2d is not None else None,
                "class_names": class_names,
                "saved_path": bundle.get("saved_path"),
                "notes": notes,
            },
        )
