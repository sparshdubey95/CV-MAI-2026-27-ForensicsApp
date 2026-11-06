"""Tkinter tools for Bag of Visual Words (BoVW) and Image Classification (Set 7).

Integrates:
  - 7.1 Codebook Construction (BoVWCodebookTool)
  - 7.2 Histogram Construction (BoVWHistogramTool)
  - 7.3 Classification & Evaluation (BoVWClassifierTool)
"""

from __future__ import annotations

from pathlib import Path
import tkinter as tk
from typing import Any

import cv2
import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
from .bovw_core import (
    build_codebook,
    classify_single_image,
    compute_bovw_histogram,
    compute_dataset_histograms,
    evaluate_bovw_classifier,
    extract_descriptors,
    get_active_bovw_model,
    get_active_bovw_split,
    load_bovw_bundle,
    render_codebook_visualization,
    render_confusion_matrix_figure,
    render_image_and_histogram,
    save_bovw_bundle,
    scan_dataset,
    set_active_bovw_model,
    set_active_bovw_split,
    split_dataset,
    train_bovw_classifier,
)
from .dialogs import (
    ask_bovw_codebook_parameters,
    ask_bovw_model_file,
    ask_bovw_select_test_image,
    ask_choice,
    ask_image_file,
    ask_image_folder,
    ask_save_bovw_model_file,
    show_error,
)


class BoVWCodebookTool(ForensicsTool):
    """Set 7.1: Build visual vocabulary codebook from training images using SIFT and K-Means."""

    tool_id = "bovw_codebook"
    title = "BoVW codebook builder"
    category = "Set 7: Bag of Visual Words & Classification"
    description = "Build and save visual word codebook using SIFT and K-Means from dataset."
    requires_image = False

    def run(self, parent: tk.Misc | None, document: ImageDocument) -> ToolResult | None:
        # 1. Ask for dataset parent folder
        folder = ask_image_folder(
            parent,
            title="Select Dataset Parent Folder (e.g. containing butterfly/, kangaroo/, dalmatian/)",
        )
        if not folder:
            return None

        # 2. Scan dataset folders
        try:
            image_paths, labels, class_names = scan_dataset(folder)
        except Exception as err:
            if parent is not None:
                show_error(parent, "Dataset Error", str(err))
            return None

        # 3. Configure parameters (K = 10 * num_classes, method, split)
        params = ask_bovw_codebook_parameters(parent, class_names, default_split=0.25)
        if params is None:
            return None
        k_val, method, test_ratio = params

        # 4. Perform stratified train/test split (only training images are used for codebook)
        try:
            train_paths, train_labels, test_paths, test_labels = split_dataset(
                image_paths,
                labels,
                test_size=test_ratio,
                random_state=42,
            )
        except Exception as err:
            if parent is not None:
                show_error(parent, "Split Error", f"Failed to split dataset: {err}")
            return None

        # 5. Build codebook using ONLY training images
        try:
            codebook, kmeans, total_desc, visual_word_patches = build_codebook(
                train_paths=train_paths,
                k=k_val,
                method=method,
                random_state=42,
                sample_limit_per_image=300,
            )
        except Exception as err:
            if parent is not None:
                show_error(parent, "Codebook Training Error", str(err))
            return None

        # Extract 1 representative sample image with detected keypoints per category
        category_samples: dict[str, tuple[Image.Image, np.ndarray]] = {}
        for c_name in class_names:
            for p, lbl in zip(train_paths, train_labels):
                if lbl == c_name:
                    try:
                        with Image.open(p) as opened:
                            sample_img = opened.convert("RGB")
                        gray_sample = np.array(sample_img.convert("L"), dtype=np.uint8)
                        if method.upper() == "SIFT":
                            det = cv2.SIFT_create(nfeatures=400)
                        else:
                            det = cv2.ORB_create(nfeatures=400)
                        kps, _ = det.detectAndCompute(gray_sample, None)
                        coords = np.array([kp.pt for kp in kps]) if kps else np.empty((0, 2))
                        category_samples[c_name] = (sample_img, coords)
                    except Exception:
                        pass
                    break

        # 6. Ask save location for the model bundle
        save_path = ask_save_bovw_model_file(
            parent,
            title="Save Trained BoVW Codebook Model (.bovw)",
            initialfile="bovw_model.bovw",
        )

        split_info = {
            "train_paths": [str(p) for p in train_paths],
            "train_labels": train_labels,
            "test_paths": [str(p) for p in test_paths],
            "test_labels": test_labels,
            "dataset_root": str(folder),
        }

        if save_path:
            try:
                save_bovw_bundle(
                    save_path,
                    codebook=codebook,
                    class_names=class_names,
                    classifier=None,
                    scaler=None,
                    method=method,
                    split_info=split_info,
                    visual_word_patches=visual_word_patches,
                )
            except Exception as err:
                if parent is not None:
                    show_error(parent, "Save Error", f"Could not save codebook bundle: {err}")

        # Cache active model and split in memory for immediate reuse
        bundle = {
            "codebook": codebook,
            "class_names": class_names,
            "method": method,
            "k": k_val,
            "classifier": None,
            "scaler": None,
            "split_info": split_info,
            "visual_word_patches": visual_word_patches,
            "filepath": str(save_path) if save_path else "In-memory session",
        }
        set_active_bovw_model(bundle)
        set_active_bovw_split(split_info)

        # 7. Render Codebook Visualisation (Visual Word Patch Grid & Detected Feature Examples)
        vis_image = render_codebook_visualization(
            codebook=codebook,
            class_names=class_names,
            visual_word_patches=visual_word_patches,
            category_samples=category_samples,
            total_descriptors=total_desc,
        )

        details = {
            "Operation": "BoVW Codebook Construction (Set 7.1)",
            "Dataset folder": Path(folder).name,
            "Detected classes": ", ".join(class_names),
            "Total images": len(image_paths),
            "Training images": len(train_paths),
            "Held-out test images": len(test_paths),
            "Descriptor method": method,
            "Visual words (K)": k_val,
            "Training descriptors pooled": total_desc,
            "Saved model": Path(save_path).name if save_path else "Active Session Memory",
        }

        message = (
            f"Codebook built successfully: K={k_val} visual words from {len(train_paths)} training images "
            f"across {len(class_names)} classes ({', '.join(class_names)})."
        )

        return ToolResult(
            image=vis_image,
            message=message,
            details=details,
        )


class BoVWHistogramTool(ForensicsTool):
    """Set 7.2: Generate and visualize BoVW frequency histograms for held-out testing or query images."""

    tool_id = "bovw_histogram"
    title = "BoVW histogram visualizer"
    category = "Set 7: Bag of Visual Words & Classification"
    description = "Generate and inspect BoVW visual word histograms for test or query images."
    requires_image = False

    def run(self, parent: tk.Misc | None, document: ImageDocument) -> ToolResult | None:
        # 1. Acquire Model / Codebook
        model = get_active_bovw_model()
        if model is None:
            if parent is not None:
                load_file = ask_bovw_model_file(parent, title="Select Saved BoVW Model File")
                if not load_file:
                    show_error(parent, "Model Required", "Please build or select a BoVW codebook model first.")
                    return None
                try:
                    model = load_bovw_bundle(load_file)
                    set_active_bovw_model(model)
                    if model.get("split_info"):
                        set_active_bovw_split(model["split_info"])
                except Exception as err:
                    show_error(parent, "Load Error", str(err))
                    return None
            else:
                return None

        codebook = model["codebook"]
        method = model.get("method", "SIFT")
        split_info = get_active_bovw_split() or model.get("split_info")

        target_image: Image.Image | None = None
        true_class = "Unknown"
        image_name = "Query Image"

        # 2. Offer selection source: Held-out Test Set vs Browse / Open Image
        if split_info and split_info.get("test_paths") and parent is not None:
            choice = ask_choice(
                parent,
                "Select Histogram Target Image",
                "Choose which image to generate a visual word histogram for:",
                [
                    ("test_set", "Select from held-out testing set (Set 7.2 requirement)"),
                    ("current", "Use currently opened image in application"),
                    ("browse", "Browse for an image file from disk..."),
                ],
                initial="test_set",
            )
            if choice is None:
                return None

            if choice == "test_set":
                test_paths = [Path(p) for p in split_info["test_paths"]]
                test_labels = split_info.get("test_labels", ["Unknown"] * len(test_paths))
                test_items = [(test_labels[i], test_paths[i].name) for i in range(len(test_paths))]

                chosen_idx = ask_bovw_select_test_image(parent, test_items)
                if chosen_idx is None:
                    return None

                img_path = test_paths[chosen_idx]
                true_class = test_labels[chosen_idx]
                image_name = img_path.name
                try:
                    with Image.open(img_path) as opened:
                        target_image = opened.copy()
                except Exception as err:
                    show_error(parent, "Open Error", f"Could not open image: {err}")
                    return None

            elif choice == "current":
                if document.is_loaded and document.current is not None:
                    target_image = document.current
                    if document.path:
                        image_name = document.path.name
                else:
                    show_error(parent, "No Image Loaded", "No image is currently loaded. Opening file picker...")
                    path = ask_image_file(parent, title="Select image for BoVW histogram")
                    if not path:
                        return None
                    with Image.open(path) as opened:
                        target_image = opened.copy()
                    image_name = Path(path).name
            else:
                path = ask_image_file(parent, title="Select image for BoVW histogram")
                if not path:
                    return None
                with Image.open(path) as opened:
                    target_image = opened.copy()
                image_name = Path(path).name
        else:
            # Fallback when no active split is present or headless
            if document.is_loaded and document.current is not None:
                target_image = document.current
                if document.path:
                    image_name = document.path.name
            else:
                path = ask_image_file(parent, title="Select image for BoVW histogram")
                if not path:
                    return None
                try:
                    with Image.open(path) as opened:
                        target_image = opened.copy()
                    image_name = Path(path).name
                except Exception as err:
                    if parent is not None:
                        show_error(parent, "Open Error", str(err))
                    return None

        assert target_image is not None

        # 3. Extract descriptors and compute BoVW histogram
        try:
            desc = extract_descriptors(target_image, method=method)
            histogram = compute_bovw_histogram(desc, codebook, normalize=True)
        except Exception as err:
            if parent is not None:
                show_error(parent, "Histogram Error", str(err))
            return None

        # 4. Render 2-panel figure showing the image + its BoVW histogram
        rendered_fig = render_image_and_histogram(
            image=target_image,
            histogram=histogram,
            class_name=true_class,
            image_title=image_name,
            visual_word_patches=model.get("visual_word_patches"),
        )

        top_word_indices = np.argsort(histogram)[::-1][:3]
        top_words_str = ", ".join(f"Word #{idx+1} ({histogram[idx]*100:.1f}%)" for idx in top_word_indices)

        details = {
            "Operation": "BoVW Histogram Generation (Set 7.2)",
            "Target image": image_name,
            "Class label": true_class,
            "Descriptors extracted": len(desc),
            "Visual words (K)": len(codebook),
            "Active visual words": int(np.count_nonzero(histogram)),
            "Dominant visual words": top_words_str,
            "Codebook source": Path(model.get("filepath", "In-memory")).name,
        }

        message = (
            f"Histogram generated for {image_name} ({len(desc)} descriptors, "
            f"{details['Active visual words']}/{len(codebook)} active words)."
        )

        return ToolResult(
            image=rendered_fig,
            message=message,
            details=details,
        )


class BoVWClassifierTool(ForensicsTool):
    """Set 7.3: Train/evaluate classifier on test set, or classify a single query image with calibrated probabilities."""

    tool_id = "bovw_classifier"
    title = "BoVW image classifier"
    category = "Set 7: Bag of Visual Words & Classification"
    description = "Train/evaluate classifier on BoVW histograms, or classify a single image with calibrated probabilities."
    requires_image = False

    def run(self, parent: tk.Misc | None, document: ImageDocument) -> ToolResult | None:
        # 1. Acquire Model
        model = get_active_bovw_model()
        if model is None:
            if parent is not None:
                load_file = ask_bovw_model_file(parent, title="Select Saved BoVW Model File")
                if not load_file:
                    show_error(parent, "Model Required", "Please build or load a BoVW model first.")
                    return None
                try:
                    model = load_bovw_bundle(load_file)
                    set_active_bovw_model(model)
                    if model.get("split_info"):
                        set_active_bovw_split(model["split_info"])
                except Exception as err:
                    show_error(parent, "Load Error", str(err))
                    return None
            else:
                return None

        has_trained_clf = model.get("classifier") is not None
        split_info = get_active_bovw_split() or model.get("split_info")

        # 2. Ask user action: Train & Evaluate vs Classify Single Image
        if parent is not None:
            options: list[tuple[str, str]] = []
            if split_info and split_info.get("train_paths"):
                options.append(("train_eval", "Train & Evaluate Classifier on Held-Out Test Set (Set 7.3.1 - 7.3.3)"))
            options.append(("classify", "Classify New Image with Calibrated Probability (Set 7.3.4 - 7.3.10)"))

            action = ask_choice(
                parent,
                "BoVW Classification Action",
                "Select classification task:",
                options,
                initial="train_eval" if not has_trained_clf and split_info else "classify",
            )
            if action is None:
                return None
        else:
            action = "train_eval" if not has_trained_clf and split_info else "classify"

        # ── ACTION A: Train & Evaluate Classifier ─────────────────────────────
        if action == "train_eval":
            if not split_info or not split_info.get("train_paths"):
                if parent is not None:
                    show_error(parent, "Dataset Split Missing", "No dataset split available to train classifier.")
                return None

            train_paths = [Path(p) for p in split_info["train_paths"]]
            train_labels = split_info["train_labels"]
            test_paths = [Path(p) for p in split_info["test_paths"]]
            test_labels = split_info["test_labels"]
            class_names = model["class_names"]
            codebook = model["codebook"]
            method = model.get("method", "SIFT")

            try:
                # 1. Compute training set histograms
                X_train, _ = compute_dataset_histograms(train_paths, codebook, method=method)
                # 2. Train RBF SVM with probability calibration
                classifier, scaler = train_bovw_classifier(X_train, train_labels, class_names)
                # 3. Compute test set histograms
                X_test, _ = compute_dataset_histograms(test_paths, codebook, method=method)
                # 4. Evaluate on held-out test images
                accuracy, report, conf_matrix = evaluate_bovw_classifier(
                    classifier, scaler, X_test, test_labels, class_names
                )
            except Exception as err:
                if parent is not None:
                    show_error(parent, "Training/Evaluation Error", str(err))
                return None

            # Update model and save
            model["classifier"] = classifier
            model["scaler"] = scaler
            model["evaluation"] = {
                "accuracy": accuracy,
                "report": report,
                "conf_matrix": conf_matrix,
            }
            set_active_bovw_model(model)

            save_path = ask_save_bovw_model_file(
                parent,
                title="Save Fully Trained BoVW Classifier Model (.bovw)",
                initialfile="bovw_trained_classifier.bovw",
            )
            if save_path:
                try:
                    save_bovw_bundle(
                        save_path,
                        codebook=codebook,
                        class_names=class_names,
                        classifier=classifier,
                        scaler=scaler,
                        method=method,
                        split_info=split_info,
                        evaluation=model["evaluation"],
                        visual_word_patches=model.get("visual_word_patches"),
                    )
                except Exception as err:
                    if parent is not None:
                        show_error(parent, "Save Error", str(err))

            # Render Confusion Matrix Figure
            cm_fig = render_confusion_matrix_figure(conf_matrix, class_names, accuracy=accuracy)

            details = {
                "Operation": "Classifier Training & Test Evaluation",
                "Classifier": "RBF Support Vector Machine (SVC, Calibrated)",
                "Classes": ", ".join(class_names),
                "Training samples": len(train_paths),
                "Test samples evaluated": len(test_paths),
                "Test Accuracy": f"{accuracy * 100.0:.2f}%",
            }
            for c_name in class_names:
                if c_name in report:
                    details[f"{c_name} (Prec/Recall)"] = (
                        f"{report[c_name]['precision']:.2f} / {report[c_name]['recall']:.2f}"
                    )

            message = (
                f"Classifier trained & evaluated successfully! Test Accuracy: {accuracy * 100.0:.1f}% "
                f"across {len(test_paths)} held-out testing images."
            )

            return ToolResult(
                image=cm_fig,
                message=message,
                details=details,
            )

        # ── ACTION B: Classify Single Image ───────────────────────────────────
        else:
            if model.get("classifier") is None or model.get("scaler") is None:
                if parent is not None:
                    show_error(
                        parent,
                        "Classifier Required",
                        "The classifier has not been trained yet. Please select 'Train & Evaluate Classifier' first.",
                    )
                return None

            classifier = model["classifier"]
            scaler = model["scaler"]
            codebook = model["codebook"]
            class_names = model["class_names"]
            method = model.get("method", "SIFT")

            target_image: Image.Image | None = None
            source_name = "Query Image"

            if document.is_loaded and document.current is not None and parent is not None:
                img_choice = ask_choice(
                    parent,
                    "Select Image to Classify",
                    "Choose image to classify:",
                    [
                        ("current", "Use currently opened image in application"),
                        ("browse", "Browse for a new image file..."),
                    ],
                    initial="current",
                )
                if img_choice is None:
                    return None
                if img_choice == "current":
                    target_image = document.current
                    if document.path:
                        source_name = document.path.name
                else:
                    path = ask_image_file(parent, title="Select image to classify")
                    if not path:
                        return None
                    with Image.open(path) as opened:
                        target_image = opened.copy()
                    source_name = Path(path).name
            else:
                if document.is_loaded and document.current is not None:
                    target_image = document.current
                    if document.path:
                        source_name = document.path.name
                else:
                    path = ask_image_file(parent, title="Select image to classify")
                    if not path:
                        return None
                    try:
                        with Image.open(path) as opened:
                            target_image = opened.copy()
                        source_name = Path(path).name
                    except Exception as err:
                        if parent is not None:
                            show_error(parent, "Open Error", str(err))
                        return None

            assert target_image is not None

            # Perform single image classification
            try:
                pred_class, probabilities, hist = classify_single_image(
                    image=target_image,
                    codebook=codebook,
                    classifier=classifier,
                    scaler=scaler,
                    class_names=class_names,
                    method=method,
                )
            except Exception as err:
                if parent is not None:
                    show_error(parent, "Classification Error", str(err))
                return None

            # Render 2-panel figure: image + histogram + calibrated probabilities
            rendered_fig = render_image_and_histogram(
                image=target_image,
                histogram=hist,
                class_name="Unknown",
                image_title=source_name,
                probabilities=probabilities,
                predicted_class=pred_class,
                visual_word_patches=model.get("visual_word_patches"),
            )

            p_pred = probabilities.get(pred_class, 0.0)
            details = {
                "Operation": "BoVW Single Image Inference",
                "Query Image": source_name,
                "Predicted Category": pred_class,
                f"Probability p({pred_class})": f"{p_pred:.3f} ({p_pred * 100.0:.1f}%)",
            }
            for c_name in class_names:
                p_c = probabilities.get(c_name, 0.0)
                details[f"p({c_name})"] = f"{p_c:.3f} ({p_c * 100.0:.1f}%)"

            details["Visual words (K)"] = len(codebook)
            details["Classifier"] = "Calibrated RBF Support Vector Machine"

            message = (
                f"Classification result: p({pred_class}) = {p_pred:.2f} "
                f"({', '.join(f'p({c})={p:.2f}' for c, p in sorted(probabilities.items()))})."
            )

            return ToolResult(
                image=rendered_fig,
                message=message,
                details=details,
            )
