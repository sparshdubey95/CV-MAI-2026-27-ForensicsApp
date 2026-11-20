"""Set 8 model training, passport classification, and feature exploration."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, UnidentifiedImageError

from forensics_app.core import ImageDocument
from forensics_app.tools.base import ForensicsTool, ToolResult
from forensics_app.tools.cnn_core import (
    DEFAULT_OUTPUT_DIR,
    discover_passports,
    extract_feature_embedding,
    load_features_npz,
    load_model_from_passport,
    model_type_from_passport,
    prepare_image,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_TRAIN_SCRIPT = _PROJECT_ROOT / "train_set8.py"
_MODEL_TYPES = ("custom_cnn", "inceptionv3_transfer")
_CLASS_NAMES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)


def _show_error(parent: tk.Misc, title: str, error: object) -> None:
    messagebox.showerror(title, str(error), parent=parent)


def _choose_passport(parent: tk.Misc, output_dir: Path = DEFAULT_OUTPUT_DIR) -> Path | None:
    """Choose from all discovered passports or browse to another passport."""
    passports = discover_passports(output_dir)
    if not passports:
        selected = filedialog.askopenfilename(
            parent=parent,
            title="Select a model passport",
            filetypes=[("Model passport", "*_passport.json"), ("JSON", "*.json")],
        )
        return Path(selected) if selected else None

    choice: dict[str, Path | None] = {"path": None}
    window = tk.Toplevel(parent)
    window.title("Step 1 — Load pretrained model")
    window.transient(parent)
    window.resizable(False, False)
    frame = ttk.Frame(window, padding=14)
    frame.pack(fill="both", expand=True)
    ttk.Label(frame, text="Select a saved model passport").pack(anchor="w", pady=(0, 6))
    selected_name = tk.StringVar(value=passports[0].name)
    selected_custom_path = tk.StringVar()
    ttk.Combobox(
        frame,
        textvariable=selected_name,
        values=[path.name for path in passports],
        state="readonly",
        width=42,
    ).pack(fill="x")

    def browse() -> None:
        filename = filedialog.askopenfilename(
            parent=window,
            title="Browse for a model passport",
            filetypes=[("Model passport", "*_passport.json"), ("JSON", "*.json")],
        )
        if filename:
            selected_custom_path.set(filename)

    buttons = ttk.Frame(frame)
    buttons.pack(fill="x", pady=(12, 0))

    def load_selected() -> None:
        choice["path"] = (
            Path(selected_custom_path.get())
            if selected_custom_path.get()
            else output_dir / selected_name.get()
        )
        window.destroy()

    ttk.Button(buttons, text="Browse…", command=browse).pack(side="left")
    ttk.Label(frame, textvariable=selected_custom_path, wraplength=360).pack(anchor="w", pady=(6, 0))
    ttk.Button(buttons, text="Cancel", command=window.destroy).pack(side="right")
    ttk.Button(buttons, text="Load Model", command=load_selected).pack(side="right", padx=(0, 6))
    window.protocol("WM_DELETE_WINDOW", window.destroy)
    window.grab_set()
    window.wait_window()
    return choice["path"]


class CNNTrainTool(ForensicsTool):
    """Train a selected model in an offline child process."""

    tool_id = "cnn_train"
    title = "Model trainer"
    category = "Set 8: CNN Classification & Feature Projection"
    description = "Train custom CNN or InceptionV3 on CIFAR-10 and save weights, passport, and projections."
    requires_image = False
    modifies_image = False

    def __init__(self) -> None:
        self._process: subprocess.Popen[bytes] | None = None
        self._log_path: Path | None = None
        self._progress_window: tk.Toplevel | None = None

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        if self._process is not None and self._process.poll() is None:
            messagebox.showwarning("Training in progress", "A Set 8 training job is already running.", parent=parent)
            return None

        parameters = self._ask_parameters(parent)
        if parameters is None:
            return None
        model_name, epochs, n_train, batch_size = parameters

        DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        log_file = tempfile.NamedTemporaryFile(prefix="set8-training-", suffix=".log", delete=False)
        self._log_path = Path(log_file.name)
        log_file.close()
        command = [
            sys.executable, str(_TRAIN_SCRIPT),
            "--model", model_name,
            "--epochs", str(epochs),
            "--n-train", str(n_train),
            "--batch-size", str(batch_size),
            "--output-dir", str(DEFAULT_OUTPUT_DIR),
        ]
        try:
            with self._log_path.open("wb") as training_log:
                self._process = subprocess.Popen(
                    command,
                    cwd=_PROJECT_ROOT,
                    stdout=training_log,
                    stderr=subprocess.STDOUT,
                    env={**os.environ, "TF_CPP_MIN_LOG_LEVEL": "2"},
                )
        except OSError:
            self._log_path.unlink(missing_ok=True)
            self._log_path = None
            raise

        progress = tk.Toplevel(parent)
        self._progress_window = progress
        progress.title("Set 8 training")
        progress.transient(parent)
        progress.resizable(False, False)
        frame = ttk.Frame(progress, padding=18)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text=f"Training {model_name}; the app remains available.").pack(anchor="w")
        bar = ttk.Progressbar(frame, mode="indeterminate")
        bar.pack(fill="x", pady=(10, 0))
        bar.start(12)
        progress.protocol("WM_DELETE_WINDOW", lambda: None)
        self._poll_training(parent)
        return ToolResult(
            message=f"Started {model_name} training in the background.",
            details={"Model": model_name, "Output directory": str(DEFAULT_OUTPUT_DIR)},
        )

    def _ask_parameters(self, parent: tk.Misc) -> tuple[str, int, int, int] | None:
        result: dict[str, tuple[str, int, int, int] | None] = {"value": None}
        window = tk.Toplevel(parent)
        window.title("Set 8 training configuration")
        window.transient(parent)
        window.resizable(False, False)
        frame = ttk.Frame(window, padding=16)
        frame.pack(fill="both", expand=True)

        model = tk.StringVar(value="custom_cnn")
        epochs = tk.StringVar(value="2")
        count = tk.StringVar(value="1000")
        batch = tk.StringVar(value="128")
        fields = (
            ("Architecture", model, _MODEL_TYPES),
            ("Epochs", epochs, None),
            ("Training samples (20–50,000)", count, None),
            ("Batch size", batch, None),
        )
        for row, (label, variable, options) in enumerate(fields):
            ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", pady=4)
            if options is None:
                ttk.Entry(frame, textvariable=variable, width=18).grid(row=row, column=1, sticky="e", padx=(12, 0))
            else:
                ttk.Combobox(frame, textvariable=variable, values=options, state="readonly", width=25).grid(
                    row=row, column=1, sticky="e", padx=(12, 0)
                )
        error = ttk.Label(frame, text="", foreground="red")
        error.grid(row=len(fields), column=0, columnspan=2, sticky="w", pady=(6, 0))

        def confirm() -> None:
            try:
                epoch_count, sample_count, batch_size = int(epochs.get()), int(count.get()), int(batch.get())
                if epoch_count < 1 or batch_size < 1 or not 20 <= sample_count <= 50000:
                    raise ValueError("Use positive epochs/batch size and 20–50,000 training samples.")
            except ValueError as exception:
                error.configure(text=str(exception))
                return
            result["value"] = model.get(), epoch_count, sample_count, batch_size
            window.destroy()

        buttons = ttk.Frame(frame)
        buttons.grid(row=len(fields) + 1, column=0, columnspan=2, sticky="e", pady=(12, 0))
        ttk.Button(buttons, text="Cancel", command=window.destroy).pack(side="right")
        ttk.Button(buttons, text="Start", command=confirm).pack(side="right", padx=(0, 6))
        window.protocol("WM_DELETE_WINDOW", window.destroy)
        window.grab_set()
        window.wait_window()
        return result["value"]

    def _poll_training(self, parent: tk.Misc) -> None:
        if self._process is None or self._process.poll() is None:
            parent.after(250, lambda: self._poll_training(parent))
            return

        return_code = self._process.returncode
        if self._progress_window is not None and self._progress_window.winfo_exists():
            self._progress_window.destroy()
        self._progress_window = None
        log_text = ""
        if self._log_path is not None:
            log_text = self._log_path.read_text(encoding="utf-8", errors="replace")
            self._log_path.unlink(missing_ok=True)
            self._log_path = None
        self._process = None
        if return_code == 0:
            messagebox.showinfo("Training complete", f"Artifacts are ready in:\n{DEFAULT_OUTPUT_DIR}", parent=parent)
        else:
            messagebox.showerror(
                "Training failed",
                f"Training exited with code {return_code}.\n\n{log_text[-3000:]}",
                parent=parent,
            )


class CNNClassifyTool(ForensicsTool):
    """Load a pretrained model, show its passport, and classify a chosen image."""

    tool_id = "cnn_classify"
    title = "Load pretrained model & classify"
    category = "Set 8: CNN Classification & Feature Projection"
    description = "Load a model passport, inspect its details, then classify a selected image."
    requires_image = False
    modifies_image = False

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        passport_path = _choose_passport(parent)
        if passport_path is None:
            return None
        try:
            model, passport = load_model_from_passport(passport_path)
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
            _show_error(parent, "Could not load model", error)
            return None

        self._show_classifier(parent, document, model, passport)
        model_name = passport.get("model_name", passport_path.stem.removesuffix("_passport"))
        return ToolResult(
            message=f"Loaded {model_name}; choose an image to classify.",
            details={"Model": model_name, "Passport": passport_path.name},
        )

    def _show_classifier(
        self,
        parent: tk.Misc,
        document: ImageDocument,
        model: Any,
        passport: dict[str, Any],
    ) -> None:
        model_type = model_type_from_passport(passport)
        class_names = [str(name) for name in passport.get("class_names", _CLASS_NAMES)]
        if len(class_names) != 10:
            raise ValueError("The model passport must contain all 10 CIFAR-10 class names.")

        window = tk.Toplevel(parent)
        window.title("Model passport and image classification")
        window.geometry("1050x790")
        window.transient(parent)

        passport_frame = ttk.LabelFrame(window, text="Step 2 — Model passport", padding=8)
        passport_frame.pack(fill="x", padx=10, pady=(10, 5))
        passport_text = tk.Text(passport_frame, height=10, width=54, wrap="word")
        passport_text.pack(side="left", fill="both", expand=True)
        passport_text.insert("1.0", _format_passport(passport))
        passport_text.configure(state="disabled")

        history_frame = ttk.Frame(passport_frame)
        history_frame.pack(side="left", fill="both", expand=True, padx=(10, 0))
        history = passport.get("history", passport.get("training_history", {}))
        history_figure, (accuracy_axis, loss_axis) = plt.subplots(1, 2, figsize=(7, 2.5), dpi=90)
        _plot_history(accuracy_axis, loss_axis, history)
        history_canvas = FigureCanvasTkAgg(history_figure, master=history_frame)
        history_canvas.draw()
        history_canvas.get_tk_widget().pack(fill="both", expand=True)

        controls = ttk.LabelFrame(window, text="Step 3 — Load image and customize", padding=8)
        controls.pack(fill="x", padx=10, pady=5)
        selected: dict[str, Image.Image | None] = {"image": None}
        top_k = tk.StringVar(value="5")
        preview_32 = tk.BooleanVar(value=True)
        ttk.Button(
            controls,
            text="Load Image from File...",
            command=lambda: self._load_image_from_file(window, selected, update_results),
        ).pack(side="left")
        use_open_button = ttk.Button(
            controls,
            text="Use Open MainWindow Image",
            command=lambda: self._use_open_image(document, selected, update_results),
            state="normal" if document.current is not None else "disabled",
        )
        use_open_button.pack(side="left", padx=(6, 14))
        ttk.Label(controls, text="Top-K classes").pack(side="left")
        ttk.Spinbox(
            controls,
            from_=1,
            to=10,
            textvariable=top_k,
            state="readonly",
            width=4,
        ).pack(side="left", padx=(5, 14))
        ttk.Checkbutton(
            controls,
            text="Show 32x32 model input preview",
            variable=preview_32,
        ).pack(side="left")

        result_frame = ttk.LabelFrame(window, text="Step 4 — Classification results", padding=8)
        result_frame.pack(fill="both", expand=True, padx=10, pady=(5, 10))
        prediction_label = ttk.Label(result_frame, text="Load an image to see the prediction.")
        prediction_label.pack(anchor="w", pady=(0, 5))
        result_figure, (image_axis, probability_axis) = plt.subplots(
            1, 2, figsize=(9, 4), gridspec_kw={"width_ratios": [1, 2]}
        )
        result_canvas = FigureCanvasTkAgg(result_figure, master=result_frame)
        result_canvas.get_tk_widget().pack(fill="both", expand=True)

        def update_results(*_args: object) -> None:
            image = selected["image"]
            if image is None:
                return
            try:
                model_input = prepare_image(image, model_type)
                probabilities = np.asarray(model.predict(model_input, verbose=0)[0])
                if probabilities.shape != (10,):
                    raise ValueError(f"Expected 10 class probabilities, got {probabilities.shape}.")
                top_index = int(np.argmax(probabilities))
                top_k_count = min(max(int(top_k.get()), 1), len(class_names))
            except Exception as error:
                _show_error(window, "Classification failed", error)
                return

            preview = image.convert("RGB").resize((32, 32), Image.Resampling.BILINEAR) if preview_32.get() else image
            image_axis.clear()
            image_axis.imshow(preview)
            image_axis.set_title("32×32 model input" if preview_32.get() else "Selected image")
            image_axis.axis("off")

            probability_axis.clear()
            ordered = np.argsort(probabilities)[::-1][:top_k_count]
            shown_names = [class_names[index] for index in ordered]
            shown_probabilities = probabilities[ordered]
            colors = ["#2e8b57" if index == top_index else "#4775a8" for index in ordered]
            probability_axis.barh(shown_names, shown_probabilities, color=colors)
            probability_axis.invert_yaxis()
            probability_axis.set_xlim(0, 1)
            probability_axis.set_xlabel("Probability")
            probability_axis.set_title(f"Top {top_k_count} classes")
            for row, probability in enumerate(shown_probabilities):
                probability_axis.text(float(probability) + 0.01, row, f"{probability:.1%}", va="center")
            prediction_label.configure(
                text=f"Predicted: {class_names[top_index]} — confidence {probabilities[top_index]:.1%}"
            )
            result_figure.tight_layout()
            result_canvas.draw_idle()

        top_k.trace_add("write", update_results)
        preview_32.trace_add("write", update_results)
        window.protocol(
            "WM_DELETE_WINDOW",
            lambda: (plt.close(history_figure), plt.close(result_figure), window.destroy()),
        )

    @staticmethod
    def _load_image_from_file(
        parent: tk.Misc,
        selected: dict[str, Image.Image | None],
        update_results: Any,
    ) -> None:
        filename = filedialog.askopenfilename(
            parent=parent,
            title="Load image for classification",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.webp"),
                ("All files", "*.*"),
            ],
        )
        if not filename:
            return
        try:
            with Image.open(filename) as image:
                selected["image"] = image.convert("RGB").copy()
        except (OSError, UnidentifiedImageError) as error:
            _show_error(parent, "Could not open image", error)
            return
        update_results()

    @staticmethod
    def _use_open_image(
        document: ImageDocument,
        selected: dict[str, Image.Image | None],
        update_results: Any,
    ) -> None:
        if document.current is not None:
            selected["image"] = document.current.copy()
            update_results()


def _format_passport(passport: dict[str, Any]) -> str:
    dataset = passport.get("dataset", {})
    sample_size = passport.get("sample_size")
    if sample_size is None and isinstance(dataset, dict):
        sample_size = {
            key: dataset.get(key)
            for key in ("n_train", "n_val", "n_test")
            if dataset.get(key) is not None
        }
    architecture = passport.get("model_architecture", {})
    training = passport.get("training", {})
    test_metrics = passport.get("test_metrics", {})
    fields = (
        ("Model", passport.get("model_name", "Not recorded")),
        ("Architecture", passport.get("architecture_builder", passport.get("model_type", "Not recorded"))),
        ("Parameters", architecture.get("total_params", "Not recorded")),
        ("Dataset", dataset.get("name", "Not recorded") if isinstance(dataset, dict) else dataset),
        ("Sample size", sample_size or "Not recorded"),
        ("Classes", ", ".join(passport.get("class_names", [])) or "Not recorded"),
        ("Preprocessing", passport.get("preprocessing", "Not recorded")),
        ("Training parameters", training or "Not recorded"),
        ("Test accuracy", test_metrics.get("accuracy", "Not recorded")),
        ("Test loss", test_metrics.get("loss", "Not recorded")),
        ("Test metric samples", test_metrics.get("evaluated_samples", "Not recorded")),
    )
    return "\n\n".join(f"{label}:\n{value}" for label, value in fields)


def _plot_history(accuracy_axis: Any, loss_axis: Any, history: dict[str, Any]) -> None:
    for key, label in (("accuracy", "train"), ("val_accuracy", "validation")):
        values = history.get(key, [])
        if values:
            accuracy_axis.plot(range(1, len(values) + 1), values, label=label)
    for key, label in (("loss", "train"), ("val_loss", "validation")):
        values = history.get(key, [])
        if values:
            loss_axis.plot(range(1, len(values) + 1), values, label=label)
    accuracy_axis.set_title("Accuracy")
    accuracy_axis.set_xlabel("Epoch")
    loss_axis.set_title("Loss")
    loss_axis.set_xlabel("Epoch")
    if accuracy_axis.lines:
        accuracy_axis.legend(fontsize="small")
    if loss_axis.lines:
        loss_axis.legend(fontsize="small")


class CNNFeatureProjectionTool(ForensicsTool):
    """Explore saved t-SNE and UMAP projections without recomputing them."""

    tool_id = "cnn_feature_projection"
    title = "Feature projection: t-SNE & UMAP"
    category = "Set 8: CNN Classification & Feature Projection"
    description = "Explore precomputed t-SNE and UMAP projections and nearest neighbors."
    requires_image = False
    modifies_image = False

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        candidates = sorted(DEFAULT_OUTPUT_DIR.glob("*_features.npz"))
        if not candidates:
            messagebox.showwarning(
                "No feature projections",
                f"No precomputed feature archives were found in {DEFAULT_OUTPUT_DIR}.",
                parent=parent,
            )
            return None
        selected = filedialog.askopenfilename(
            parent=parent,
            title="Select precomputed feature archive",
            initialdir=str(DEFAULT_OUTPUT_DIR),
            filetypes=[("Feature archive", "*_features.npz")],
        )
        if not selected:
            return None
        try:
            data = load_features_npz(selected)
        except (OSError, ValueError) as error:
            _show_error(parent, "Could not load feature archive", error)
            return None
        self._show_projection(parent, Path(selected), data, document)
        return ToolResult(
            message=f"Loaded {len(data['features'])} precomputed feature vectors.",
            details={
                "Feature dimension": data["features"].shape[1],
                "Samples": len(data["features"]),
                "Projection": Path(selected).name,
            },
        )

    def _show_projection(
        self,
        parent: tk.Misc,
        feature_path: Path,
        data: dict[str, np.ndarray],
        document: ImageDocument,
    ) -> None:
        names = [str(name) for name in data["class_names"]]
        labels = data["labels"].reshape(-1).astype(int)
        features = data["features"]
        projections = {"t-SNE": data["tsne_2d"], "UMAP": data["umap_2d"]}
        passport_path = feature_path.with_name(
            feature_path.name.removesuffix("_features.npz") + "_passport.json"
        )
        nearest: np.ndarray | None = None
        if document.current is not None and passport_path.is_file():
            try:
                model, passport = load_model_from_passport(passport_path)
                model_type = model_type_from_passport(passport)
                embedding = extract_feature_embedding(model, document.current, model_type, passport)
                if embedding.shape[0] != features.shape[1]:
                    raise ValueError("Open-image embedding dimension does not match the saved feature archive.")
                nearest = np.argsort(np.linalg.norm(features - embedding, axis=1))[:5]
            except (OSError, ValueError, KeyError) as error:
                _show_error(parent, "Could not project the open image", error)

        window = tk.Toplevel(parent)
        window.title("Feature space explorer — t-SNE / UMAP")
        window.geometry("900x720")
        window.transient(parent)
        toolbar = ttk.Frame(window, padding=8)
        toolbar.pack(fill="x")
        projection_var = tk.StringVar(value="t-SNE")
        for label in ("t-SNE", "UMAP"):
            ttk.Radiobutton(toolbar, text=label, variable=projection_var, value=label).pack(side="left", padx=5)
        if not projections["UMAP"].size:
            ttk.Label(toolbar, text="UMAP coordinates are unavailable in this archive.").pack(side="left", padx=12)

        figure, axis = plt.subplots(figsize=(9, 6), dpi=100)
        canvas = FigureCanvasTkAgg(figure, master=window)
        canvas.get_tk_widget().pack(fill="both", expand=True)

        def draw() -> None:
            axis.clear()
            method = projection_var.get()
            coordinates = projections[method]
            if not coordinates.size:
                axis.text(0.5, 0.5, "UMAP projection was not saved.", ha="center", va="center")
                axis.set_axis_off()
            else:
                for class_index, class_name in enumerate(names):
                    mask = labels == class_index
                    if np.any(mask):
                        axis.scatter(
                            coordinates[mask, 0],
                            coordinates[mask, 1],
                            s=15,
                            alpha=0.7,
                            label=class_name,
                        )
                if nearest is not None:
                    position = coordinates[nearest].mean(axis=0)
                    axis.scatter(
                        coordinates[nearest, 0],
                        coordinates[nearest, 1],
                        marker="x",
                        s=90,
                        linewidths=2,
                        color="black",
                        label="Nearest neighbors",
                    )
                    axis.scatter(
                        [position[0]], [position[1]],
                        marker="*", s=230, color="red", edgecolors="black",
                        label="Open image (neighbor centroid)",
                    )
                axis.set_title(f"{method} projection of CIFAR-10 CNN features")
                axis.set_xlabel(f"{method} component 1")
                axis.set_ylabel(f"{method} component 2")
                axis.legend(loc="best", fontsize="small", ncol=2)
            figure.tight_layout()
            canvas.draw_idle()

        projection_var.trace_add("write", lambda *_args: draw())
        draw()
        window.protocol("WM_DELETE_WINDOW", lambda: (plt.close(figure), window.destroy()))
