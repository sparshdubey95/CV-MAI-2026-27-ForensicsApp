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
    map_features_to_codebook,
    render_codebook_visualization,
    render_confusion_matrix_figure,
    render_image_and_histogram,
    render_test_image_histogram_mapping,
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
    show_bovw_classification_dialog,
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

        # 3. Configure parameters (K locked to 30, method locked to SIFT per coursework requirements)
        params = ask_bovw_codebook_parameters(parent, class_names, default_split=0.25)
        if params is None:
            return None
        k_val, method, test_ratio = 30, "SIFT", params[2]

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

        # 5. Build codebook using ONLY training images (K=30 visual words, SIFT)
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

        # Extract 1 representative sample image with clean, properly localized keypoints per category
        category_samples: dict[str, tuple[Image.Image, np.ndarray]] = {}
        for c_name in class_names:
            for p, lbl in zip(train_paths, train_labels):
                if lbl == c_name:
                    try:
                        with Image.open(p) as opened:
                            sample_img = opened.convert("RGB")
                        gray_sample = np.array(sample_img.convert("L"), dtype=np.uint8)
                        det = cv2.SIFT_create(
                            nfeatures=250,
                            contrastThreshold=0.04,
                            edgeThreshold=10.0,
                        )
                        kps, _ = det.detectAndCompute(gray_sample, None)
                        if kps:
                            kps = sorted(kps, key=lambda kp: kp.response, reverse=True)[:100]
                            coords = np.array([kp.pt for kp in kps], dtype=np.float32)
                        else:
                            coords = np.empty((0, 2), dtype=np.float32)
                        category_samples[c_name] = (sample_img, coords)
                    except Exception:
                        pass
                    break

        # Train classifier and evaluate on test set so model bundle is fully usable in 7.2 & 7.3
        try:
            X_train, _ = compute_dataset_histograms(train_paths, codebook, method="SIFT")
            classifier, scaler = train_bovw_classifier(X_train, train_labels, class_names)
            X_test, _ = compute_dataset_histograms(test_paths, codebook, method="SIFT")
            accuracy, report, conf_matrix = evaluate_bovw_classifier(
                classifier, scaler, X_test, test_labels, class_names
            )
            eval_info = {"accuracy": accuracy, "report": report, "conf_matrix": conf_matrix}
        except Exception:
            classifier = None
            scaler = None
            eval_info = None

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
                    classifier=classifier,
                    scaler=scaler,
                    method=method,
                    split_info=split_info,
                    evaluation=eval_info,
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
            "classifier": classifier,
            "scaler": scaler,
            "split_info": split_info,
            "evaluation": eval_info,
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
    """Set 7.2: Generate and visualize BoVW frequency histograms for held-out testing images only."""

    tool_id = "bovw_histogram"
    title = "BoVW histogram visualizer"
    category = "Set 7: Bag of Visual Words & Classification"
    description = "Inspect BoVW visual word histogram with feature-to-centroid mapping lines for test images only."
    requires_image = False

    def run(self, parent: tk.Misc | None, document: ImageDocument) -> ToolResult | None:
        # 1. Require user to select/load a saved codebook model (from Section 7.1)
        model = get_active_bovw_model()
        if parent is not None:
            if model is not None and model.get("split_info") and model["split_info"].get("test_paths"):
                source_choice = ask_choice(
                    parent,
                    "Select BoVW Codebook Model",
                    "Choose codebook model source for test image histogram visualization:",
                    [
                        ("active", f"Use active codebook model ({Path(model.get('filepath', 'Active Session')).name})"),
                        ("load", "Select/load a saved codebook model file (.bovw) from Section 7.1..."),
                    ],
                    initial="active",
                )
                if source_choice is None:
                    return None
                if source_choice == "load":
                    load_file = ask_bovw_model_file(
                        parent,
                        title="Select Saved BoVW Codebook Model (.bovw) from Section 7.1",
                    )
                    if not load_file:
                        return None
                    try:
                        model = load_bovw_bundle(load_file)
                        set_active_bovw_model(model)
                        set_active_bovw_split(model.get("split_info"))
                    except Exception as err:
                        show_error(parent, "Load Error", str(err))
                        return None
            else:
                load_file = ask_bovw_model_file(
                    parent,
                    title="Select Saved BoVW Codebook Model (.bovw) from Section 7.1",
                )
                if not load_file:
                    show_error(
                        parent,
                        "Saved Codebook Required",
                        "Please select or load a saved BoVW codebook model (from Section 7.1) "
                        "before running the histogram visualizer.",
                    )
                    return None
                try:
                    model = load_bovw_bundle(load_file)
                    set_active_bovw_model(model)
                    set_active_bovw_split(model.get("split_info"))
                except Exception as err:
                    show_error(parent, "Load Error", str(err))
                    return None

        if model is None:
            return None

        codebook = model["codebook"]
        method = model.get("method", "SIFT")
        split_info = model.get("split_info") or get_active_bovw_split()

        # 2. Restrict this section strictly to test images only
        if not split_info or not split_info.get("test_paths"):
            if parent is not None:
                show_error(
                    parent,
                    "Test Images Required",
                    "The selected codebook model does not contain held-out test images.\n"
                    "Section 7.2 is strictly restricted to test images only.",
                )
            return None

        test_paths = [Path(p) for p in split_info["test_paths"]]
        test_labels = split_info.get("test_labels", ["Unknown"] * len(test_paths))

        # Check existing test images on disk
        valid_indices: list[int] = []
        resolved_paths: list[Path] = []
        for i, p in enumerate(test_paths):
            if p.is_file():
                valid_indices.append(i)
                resolved_paths.append(p)
            elif split_info.get("dataset_root"):
                alt_p = Path(split_info["dataset_root"]) / p.parent.name / p.name
                if alt_p.is_file():
                    valid_indices.append(i)
                    resolved_paths.append(alt_p)

        if not resolved_paths:
            if parent is not None:
                show_error(
                    parent,
                    "Test Images Not Found",
                    "Could not locate the held-out test images on disk. Please verify the dataset location.",
                )
            return None

        test_items = [(test_labels[orig_i], resolved_paths[idx].name) for idx, orig_i in enumerate(valid_indices)]

        if parent is not None:
            chosen_idx = ask_bovw_select_test_image(parent, test_items)
            if chosen_idx is None:
                return None
        else:
            chosen_idx = 0

        target_path = resolved_paths[chosen_idx]
        true_class = test_items[chosen_idx][0]
        image_name = target_path.name
        try:
            with Image.open(target_path) as opened:
                target_image = opened.copy()
        except Exception as err:
            if parent is not None:
                show_error(parent, "Open Error", f"Could not open test image: {err}")
            return None

        # 3. Explicit mapping logic: map each feature to its closest centroid
        try:
            kps_coords, assigned_centroids, histogram, kps = map_features_to_codebook(
                image=target_image,
                codebook=codebook,
                method=method,
                max_keypoints=150,
            )
        except Exception as err:
            if parent is not None:
                show_error(parent, "Feature Mapping Error", str(err))
            return None

        # 4. Render layout: Test image with keypoints on left, mapping lines connecting to histogram bins on right
        rendered_fig = render_test_image_histogram_mapping(
            image=target_image,
            kps_coords=kps_coords,
            assigned_centroids=assigned_centroids,
            histogram=histogram,
            class_name=true_class,
            image_title=image_name,
            max_mapping_lines=120,
        )

        top_word_indices = np.argsort(histogram)[::-1][:3]
        top_words_str = ", ".join(f"Word #{idx} ({histogram[idx]*100:.1f}%)" for idx in top_word_indices)

        details = {
            "Operation": "BoVW Test Image Histogram Construction (Set 7.2)",
            "Target test image": image_name,
            "Class label": true_class,
            "Keypoints mapped": len(kps_coords),
            "Visual words (K)": len(codebook),
            "Active visual words": int(np.count_nonzero(histogram)),
            "Dominant visual words": top_words_str,
            "Codebook source": Path(model.get("filepath", "Active session")).name,
        }

        message = (
            f"Explicit mapping completed for {image_name}: {len(kps_coords)} features mapped to closest centroids "
            f"({details['Active visual words']}/{len(codebook)} active bins in histogram)."
        )

        return ToolResult(
            image=rendered_fig,
            message=message,
            details=details,
        )


class BoVWClassifierTool(ForensicsTool):
    """Set 7.3: Image classification designed specifically for new/external images."""

    tool_id = "bovw_classifier"
    title = "BoVW image classifier"
    category = "Set 7: Bag of Visual Words & Classification"
    description = "Classify new/external images: inspect computed histogram first, then run classifier for top probability."
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
                    set_active_bovw_split(model.get("split_info"))
                except Exception as err:
                    show_error(parent, "Load Error", str(err))
                    return None
            else:
                return None

        # Ensure classifier is trained
        if model.get("classifier") is None or model.get("scaler") is None:
            split_info = model.get("split_info") or get_active_bovw_split()
            if split_info and split_info.get("train_paths"):
                try:
                    train_paths = [Path(p) for p in split_info["train_paths"]]
                    X_tr, _ = compute_dataset_histograms(train_paths, model["codebook"], method=model.get("method", "SIFT"))
                    clf, sc = train_bovw_classifier(X_tr, split_info["train_labels"], model["class_names"])
                    model["classifier"] = clf
                    model["scaler"] = sc
                    set_active_bovw_model(model)
                except Exception as err:
                    if parent is not None:
                        show_error(parent, "Classifier Training Error", str(err))
                    return None
            else:
                if parent is not None:
                    show_error(parent, "Classifier Required", "This model does not contain a trained classifier.")
                return None

        # 2. Select new / external image
        if parent is not None:
            path = ask_image_file(parent, title="Select New / External Image to Classify (Set 7.3)")
            if not path:
                return None
            try:
                with Image.open(path) as opened:
                    target_image = opened.copy()
                image_name = Path(path).name
            except Exception as err:
                show_error(parent, "Open Error", f"Could not open image: {err}")
                return None
        else:
            if document.is_loaded and document.current is not None:
                target_image = document.current
                image_name = document.path.name if document.path else "query_image.jpg"
            else:
                return None

        codebook = model["codebook"]
        classifier = model["classifier"]
        scaler = model["scaler"]
        class_names = model["class_names"]
        method = model.get("method", "SIFT")

        # 3. Use the exact same feature extraction, mapping, and histogram pipeline as Section 7.2
        try:
            kps_coords, assigned_centroids, hist, kps = map_features_to_codebook(
                image=target_image,
                codebook=codebook,
                method=method,
                max_keypoints=150,
            )
        except Exception as err:
            if parent is not None:
                show_error(parent, "Feature Extraction / Mapping Error", str(err))
            return None

        # Display this exact 7.2 histogram visualizer first before the user clicks "Classify"
        initial_fig = render_test_image_histogram_mapping(
            image=target_image,
            kps_coords=kps_coords,
            assigned_centroids=assigned_centroids,
            histogram=hist,
            class_name="",
            image_title=image_name,
            max_mapping_lines=120,
        )

        # 4. Display computed histogram first, with 'Classify' button at bottom to execute classifier
        if parent is not None:
            dialog_res = show_bovw_classification_dialog(
                parent=parent,
                image_name=image_name,
                initial_fig=initial_fig,
                target_image=target_image,
                codebook=codebook,
                classifier=classifier,
                scaler=scaler,
                class_names=class_names,
                histogram=hist,
                kps_coords=kps_coords,
                assigned_centroids=assigned_centroids,
                visual_word_patches=model.get("visual_word_patches"),
            )
            if dialog_res is None:
                return None
            pred_class, probabilities, final_fig = dialog_res
        else:
            hist_scaled = scaler.transform(hist.reshape(1, -1))
            pred_class = str(classifier.predict(hist_scaled)[0])
            proba_vector = classifier.predict_proba(hist_scaled)[0]
            clf_classes = list(classifier.classes_)
            probabilities = {
                c_name: float(proba_vector[clf_classes.index(c_name)]) if c_name in clf_classes else 0.0
                for c_name in class_names
            }
            top_prob = probabilities.get(pred_class, 0.0)
            final_fig = render_test_image_histogram_mapping(
                image=target_image,
                kps_coords=kps_coords,
                assigned_centroids=assigned_centroids,
                histogram=hist,
                class_name=pred_class,
                image_title=image_name,
                max_mapping_lines=120,
                prediction_text=f"p({pred_class}) = {top_prob:.2f}",
                probabilities=probabilities,
            )

        top_prob = probabilities.get(pred_class, 0.0)
        details = {
            "Operation": "BoVW Classification for New Images (Set 7.3)",
            "Query Image": image_name,
            "Predicted Class": pred_class,
            "Top Probability Score": f"p({pred_class}) = {top_prob:.2f}",
        }
        for c_name in class_names:
            details[f"p({c_name})"] = f"{probabilities.get(c_name, 0.0):.2f}"
        details["Visual Words (K)"] = len(codebook)
        details["Classifier"] = "Calibrated RBF Support Vector Machine"

        message = f"Classification result: p({pred_class}) = {top_prob:.2f} ({image_name})"

        return ToolResult(
            image=final_fig,
            message=message,
            details=details,
        )
