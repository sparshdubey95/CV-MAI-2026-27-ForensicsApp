"""Eigenface model builder using PCA (Set 6.2).

This module implements:
  - Face dataset loading with strict Train/Test separation (LFW or local folder).
  - Background threaded LFW dataset download (UI never freezes).
  - Automatic test dataset export to 'LFW_Test_Images' with identity manifest.
  - Correct mathematical Eigenfaces workflow:
      1. Calculate mean face μ = (1/N) ∑ x_i
      2. Centre training images: x_i - μ
      3. Calculate first K principal components via PCA/SVD (U_K)
      4. Project training images: w_i = U_K^T (x_i - μ)
      5. Store coefficients with corresponding identity labels
  - Interactive GUI Controls:
      - Fetch LFW Dataset
      - Select K (default 12)
      - Build Eigenfaces Model
      - Visualise Eigenfaces (first 12, indexed 0–11)
      - Save Model (.npy)
  - Full model serialization as .npy (NumPy binary format) and .npz
"""

from __future__ import annotations

import io
from pathlib import Path
import threading
import tkinter as tk
from tkinter import ttk, messagebox
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageTk

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
from .dialogs import (
    ask_eigenface_builder_parameters,
    ask_image_folder,
    ask_npy_file,
    ask_save_npy_file,
    show_error,
)
from .eigenfaces import (
    SUPPORTED_IMAGE_EXTS,
    compute_mean_face,
    center_faces,
    compute_principal_components,
    export_test_dataset,
    extract_person_name_from_filename,
    fetch_lfw_people,
    get_active_model,
    load_eigenface_model,
    load_face_dataset,
    project_faces,
    render_eigenfaces_montage,
    save_eigenface_model,
    set_active_model,
    train_eigenfaces,
)


class EigenfaceBuilderWindow(tk.Toplevel):
    """Interactive GUI window for Set 6.2 Eigenfaces Builder."""

    def __init__(self, parent: tk.Misc, document: ImageDocument) -> None:
        super().__init__(parent)
        self.title("Set 6.2 – Eigenfaces Builder")
        self.transient(parent)
        self.geometry("920x680")
        self.minsize(800, 540)

        self.document = document
        self.result: ToolResult | None = None

        # State
        self.dataset_X: np.ndarray | None = None
        self.dataset_labels: np.ndarray | None = None
        self.dataset_unique_names: list[str] = []
        self.dataset_thumbnails: np.ndarray | None = None
        self.dataset_source_desc: str = "None"
        self.target_size: tuple[int, int] = (64, 64)

        # Model state
        self.mean_face: np.ndarray | None = None
        self.eigenfaces: np.ndarray | None = None
        self.coefficients: np.ndarray | None = None
        self.pca: Any = None
        self.k: int = 12
        self.visualization_img: Image.Image | None = None

        self._build_ui()

    def _build_ui(self) -> None:
        main_frame = ttk.Frame(self, padding=14)
        main_frame.pack(fill="both", expand=True)

        # Header
        ttk.Label(
            main_frame,
            text="Set 6.2: Eigenfaces Model Builder",
            font=("TkDefaultFont", 12, "bold"),
            foreground="#003366",
        ).pack(anchor="w", pady=(0, 2))

        ttk.Label(
            main_frame,
            text="Extract PCA Eigenfaces, mean face μ, and projection weights w_i = U_K^T(x_i - μ) from training faces.",
            font=("TkDefaultFont", 9),
            foreground="#555555",
        ).pack(anchor="w", pady=(0, 10))

        # Workflow Panel
        workflow_frame = ttk.LabelFrame(main_frame, text="Model Construction Controls", padding=10)
        workflow_frame.pack(fill="x", pady=(0, 10))

        # 1. Dataset Controls
        row1 = ttk.Frame(workflow_frame)
        row1.pack(fill="x", pady=3)

        self.btn_fetch_lfw = ttk.Button(
            row1,
            text="1. Fetch LFW Dataset",
            command=self._on_fetch_lfw,
        )
        self.btn_fetch_lfw.pack(side="left", padx=(0, 6))

        self.btn_load_folder = ttk.Button(
            row1,
            text="Load Local Folder...",
            command=self._on_load_folder,
        )
        self.btn_load_folder.pack(side="left", padx=(0, 10))

        self.data_status_var = tk.StringVar(value="No dataset loaded. Click 'Fetch LFW Dataset' or select a folder.")
        self.data_status_label = ttk.Label(
            row1,
            textvariable=self.data_status_var,
            font=("TkDefaultFont", 9, "italic"),
            foreground="#666666",
        )
        self.data_status_label.pack(side="left", fill="x", expand=True)

        # 2. Select K
        row2 = ttk.Frame(workflow_frame)
        row2.pack(fill="x", pady=4)

        ttk.Label(row2, text="2. Principal Components (K):", font=("TkDefaultFont", 9, "bold")).pack(side="left", padx=(0, 6))
        self.k_var = tk.IntVar(value=12)
        self.k_spin = ttk.Spinbox(row2, from_=1, to=200, textvariable=self.k_var, width=8)
        self.k_spin.pack(side="left", padx=(0, 16))

        # 3. Build Eigenfaces Model
        self.btn_build_model = ttk.Button(
            row2,
            text="3. Build Eigenfaces Model",
            command=self._on_build_model,
            state="disabled",
        )
        self.btn_build_model.pack(side="left", padx=(0, 10))

        # 4. Visualise Eigenfaces
        self.btn_visualise = ttk.Button(
            row2,
            text="4. Visualise Eigenfaces (0–11)",
            command=self._on_visualise,
            state="disabled",
        )
        self.btn_visualise.pack(side="left", padx=(0, 10))

        # 5. Save Model
        self.btn_save_model = ttk.Button(
            row2,
            text="5. Save Model (.npy)",
            command=self._on_save_model,
            state="disabled",
        )
        self.btn_save_model.pack(side="left", padx=(0, 6))

        # Status text
        self.build_status_var = tk.StringVar(value="")
        self.build_status_label = ttk.Label(
            workflow_frame,
            textvariable=self.build_status_var,
            font=("TkDefaultFont", 9, "bold"),
            foreground="#0055aa",
        )
        self.build_status_label.pack(anchor="w", pady=(4, 0))

        # Visualization Preview Container
        preview_frame = ttk.LabelFrame(main_frame, text="Eigenfaces Visualization (First 12, Indexed 0–11)", padding=10)
        preview_frame.pack(fill="both", expand=True, pady=(0, 8))

        self.canvas_label = ttk.Label(preview_frame, text="Eigenface montage will be displayed here after training.")
        self.canvas_label.pack(fill="both", expand=True)

        # Bottom Bar
        bottom_bar = ttk.Frame(main_frame)
        bottom_bar.pack(fill="x", pady=(4, 0))

        ttk.Button(bottom_bar, text="Apply to ForensicsApp View & Close", command=self._on_apply).pack(side="right")
        ttk.Button(bottom_bar, text="Close", command=self.destroy).pack(side="right", padx=(0, 6))

    def _on_fetch_lfw(self) -> None:
        """Download LFW in a background thread to keep GUI responsive."""
        self.btn_fetch_lfw.configure(state="disabled")
        self.btn_load_folder.configure(state="disabled")
        self.data_status_var.set("Fetching LFW dataset in background... (Please wait)")
        self.data_status_label.configure(foreground="#b8860b")

        def worker() -> None:
            try:
                # Fetch train subset, automatically exports test images to LFW_Test_Images
                X, labels, unique_names, thumbnails = load_face_dataset(
                    source="lfw",
                    target_size=self.target_size,
                    min_faces_per_person=20,
                    subset="train",
                    export_test_images=True,
                    export_dir="LFW_Test_Images",
                )
                self.after(0, self._on_dataset_loaded, X, labels, unique_names, thumbnails, "LFW People (Train subset)")
            except Exception as err:
                self.after(0, self._on_dataset_error, str(err))

        threading.Thread(target=worker, daemon=True).start()

    def _on_load_folder(self) -> None:
        folder = ask_image_folder(self, title="Select folder of face images")
        if not folder:
            return

        self.btn_fetch_lfw.configure(state="disabled")
        self.btn_load_folder.configure(state="disabled")
        self.data_status_var.set("Loading local folder faces...")

        def worker() -> None:
            try:
                X, labels, unique_names, thumbnails = load_face_dataset(
                    source="folder",
                    folder_path=folder,
                    target_size=self.target_size,
                )
                self.after(0, self._on_dataset_loaded, X, labels, unique_names, thumbnails, f"Folder: {Path(folder).name}")
            except Exception as err:
                self.after(0, self._on_dataset_error, str(err))

        threading.Thread(target=worker, daemon=True).start()

    def _on_dataset_loaded(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        unique_names: list[str],
        thumbnails: np.ndarray,
        desc: str,
    ) -> None:
        self.btn_fetch_lfw.configure(state="normal")
        self.btn_load_folder.configure(state="normal")
        self.dataset_X = X
        self.dataset_labels = labels
        self.dataset_unique_names = unique_names
        self.dataset_thumbnails = thumbnails
        self.dataset_source_desc = desc

        msg = f"Dataset Ready: {len(X)} training images, {len(unique_names)} identities ({self.target_size[0]}x{self.target_size[1]})."
        if "LFW" in desc:
            msg += " Test images exported to 'LFW_Test_Images/'."
        self.data_status_var.set(msg)
        self.data_status_label.configure(foreground="#155724")

        self.btn_build_model.configure(state="normal")

    def _on_dataset_error(self, err_msg: str) -> None:
        self.btn_fetch_lfw.configure(state="normal")
        self.btn_load_folder.configure(state="normal")
        self.data_status_var.set(f"Dataset load failed: {err_msg}")
        self.data_status_label.configure(foreground="#cc0000")
        show_error(self, "Dataset Error", f"Failed to load dataset:\n{err_msg}")

    def _on_build_model(self) -> None:
        if self.dataset_X is None or self.dataset_labels is None:
            return

        try:
            k_val = int(self.k_var.get())
        except ValueError:
            k_val = 12

        self.btn_build_model.configure(state="disabled")
        self.build_status_var.set("Computing PCA Eigenfaces (mean face, components, projections)...")

        def worker() -> None:
            try:
                mean_face, eigenfaces, coefficients, pca = train_eigenfaces(
                    self.dataset_X,
                    n_components=k_val,
                )
                self.after(0, self._on_model_built, mean_face, eigenfaces, coefficients, pca, k_val)
            except Exception as err:
                self.after(0, self._on_build_error, str(err))

        threading.Thread(target=worker, daemon=True).start()

    def _on_model_built(
        self,
        mean_face: np.ndarray,
        eigenfaces: np.ndarray,
        coefficients: np.ndarray,
        pca: Any,
        k_val: int,
    ) -> None:
        self.btn_build_model.configure(state="normal")
        self.mean_face = mean_face
        self.eigenfaces = eigenfaces
        self.coefficients = coefficients
        self.pca = pca
        self.k = len(eigenfaces)

        var_pct = float(np.sum(pca.explained_variance_ratio_)) * 100.0
        self.build_status_var.set(
            f"Eigenfaces Model Built: K={self.k} components, total explained variance: {var_pct:.2f}%."
        )
        self.build_status_label.configure(foreground="#155724")

        # Cache active session model
        target_h, target_w = self.target_size
        model_dict = {
            "mean_face": mean_face,
            "eigenfaces": eigenfaces,
            "components": eigenfaces,
            "coefficients": coefficients,
            "training_coefficients": coefficients,
            "labels": self.dataset_labels,
            "target_names": self.dataset_unique_names,
            "image_shape": (target_h, target_w),
            "dimensions": (target_h, target_w),
            "thumbnails": self.dataset_thumbnails,
            "training_images": self.dataset_thumbnails,
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "k": self.k,
            "filepath": "In-memory session model",
        }
        set_active_model(model_dict)

        self.btn_visualise.configure(state="normal")
        self.btn_save_model.configure(state="normal")

        # Auto-visualise
        self._on_visualise()

    def _on_build_error(self, err_msg: str) -> None:
        self.btn_build_model.configure(state="normal")
        self.build_status_var.set(f"Model build error: {err_msg}")
        self.build_status_label.configure(foreground="#cc0000")
        show_error(self, "Build Error", f"PCA computation failed:\n{err_msg}")

    def _on_visualise(self) -> None:
        if self.mean_face is None or self.eigenfaces is None or self.pca is None:
            return

        target_h, target_w = self.target_size
        montage = render_eigenfaces_montage(
            mean_face=self.mean_face,
            eigenfaces=self.eigenfaces,
            image_shape=(target_h, target_w),
            max_eigenfaces=12,
            explained_variance_ratio=self.pca.explained_variance_ratio_,
            include_mean=True,
        )
        self.visualization_img = montage

        # Preview on canvas
        max_w, max_h = 860, 360
        w, h = montage.size
        scale = min(max_w / w, max_h / h, 1.0)
        disp_w, disp_h = max(1, int(w * scale)), max(1, int(h * scale))
        disp_img = montage.resize((disp_w, disp_h), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(disp_img)
        self.canvas_label.configure(image=photo, text="")
        self.canvas_label.image = photo

        # Prepare result
        var_pct = float(np.sum(self.pca.explained_variance_ratio_)) * 100.0
        details = {
            "Operation": "PCA Eigenfaces Model Building (Set 6.2)",
            "Dataset source": self.dataset_source_desc,
            "Training samples": len(self.dataset_labels) if self.dataset_labels is not None else 0,
            "Unique identities": len(self.dataset_unique_names),
            "Principal components (K)": self.k,
            "Total explained variance": f"{var_pct:.2f}%",
            "Image dimensions": f"{target_w}x{target_h}",
        }
        self.result = ToolResult(
            image=montage,
            message=f"Eigenface model built: K={self.k} ({var_pct:.1f}% variance explained).",
            details=details,
        )

    def _on_save_model(self) -> None:
        if self.mean_face is None or self.eigenfaces is None:
            return

        save_path = ask_save_npy_file(
            self,
            title="Save Eigenface Model (.npy)",
            initialfile="eigenface_model.npy",
        )
        if not save_path:
            return

        try:
            target_h, target_w = self.target_size
            save_eigenface_model(
                filepath=save_path,
                mean_face=self.mean_face,
                eigenfaces=self.eigenfaces,
                coefficients=self.coefficients if self.coefficients is not None else np.empty(0),
                labels=self.dataset_labels if self.dataset_labels is not None else [],
                image_shape=(target_h, target_w),
                target_names=self.dataset_unique_names,
                thumbnails=self.dataset_thumbnails,
                explained_variance_ratio=self.pca.explained_variance_ratio_ if self.pca else None,
                k=self.k,
            )
            self.build_status_var.set(f"Model saved successfully to: {Path(save_path).name}")
            self.build_status_label.configure(foreground="#155724")
        except Exception as err:
            show_error(self, "Save Error", f"Failed to save model:\n{err}")

    def _on_apply(self) -> None:
        self.destroy()


# ---------------------------------------------------------------------------
# ForensicsTool Class
# ---------------------------------------------------------------------------

class EigenfaceBuilderTool(ForensicsTool):
    """Set 6.2: Build and save PCA Eigenfaces from LFW dataset or local folder."""

    tool_id = "eigenface_builder"
    title = "Eigenface builder"
    category = "Set 6: Face Detection & Recognition"
    description = "Compute PCA mean face, eigenfaces and coefficients from face dataset and save model."
    requires_image = False

    def run(self, parent: tk.Misc | None, document: ImageDocument) -> ToolResult | None:
        if parent is not None:
            dialog = EigenfaceBuilderWindow(parent, document)
            dialog.grab_set()
            dialog.wait_window()
            return dialog.result

        # Headless mode (automated testing)
        params = ask_eigenface_builder_parameters(
            parent=None,
            initial_source="folder",
            initial_k=12,
            initial_size="64, 64",
        )
        if params is None:
            source, k_req, (target_w, target_h) = "folder", 12, (64, 64)
        else:
            source, k_req, (target_w, target_h) = params

        X, labels, unique_names, thumbnails = load_face_dataset(
            source=source,
            folder_path="tests/data/faces" if Path("tests/data/faces").is_dir() else None,
            target_size=(target_w, target_h),
            min_faces_per_person=1,
            subset="train",
        )

        mean_face, eigenfaces, coefficients, pca = train_eigenfaces(X, n_components=k_req)
        model_dict = {
            "mean_face": mean_face,
            "eigenfaces": eigenfaces,
            "components": eigenfaces,
            "coefficients": coefficients,
            "labels": labels,
            "target_names": unique_names,
            "image_shape": (target_h, target_w),
            "thumbnails": thumbnails,
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "k": len(eigenfaces),
        }
        set_active_model(model_dict)

        montage = render_eigenfaces_montage(
            mean_face=mean_face,
            eigenfaces=eigenfaces,
            image_shape=(target_h, target_w),
            max_eigenfaces=12,
            explained_variance_ratio=pca.explained_variance_ratio_,
            include_mean=True,
        )

        return ToolResult(
            image=montage,
            message=f"Eigenface model built: {len(X)} images, K={len(eigenfaces)}.",
            details={"Operation": "PCA Eigenface Training", "Components (K)": len(eigenfaces)},
        )
