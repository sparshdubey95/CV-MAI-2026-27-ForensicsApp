"""Face recognition using saved Eigenface PCA model (Set 6.3).

This module implements:
  - Loading pre-trained Eigenface models (.npy or .npz) without downloading LFW or retraining PCA.
  - Applying identical preprocessing (grayscale conversion, resizing to model shape).
  - Flattening and centering using the saved mean face: x_test - μ.
  - Projecting into PCA space: w_test = U_K (x_test - μ).
  - Calculating Euclidean distances against all training coefficients:
        d_i = ||w_test - w_i||_2
  - Finding nearest training image using argmin and retrieving its identity.
  - Looking up true identity from test manifest (labels.csv / manifest.json) if available.
  - Displaying:
      - Test image and filename
      - Predicted identity
      - True identity from test manifest (if available)
      - Nearest Euclidean distance
      - Correct/incorrect status when ground truth is known
      - Closest 2-3 training images with thumbnails, names and distances
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
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
from .dialogs import ask_image_file, ask_npy_file, show_error
from .eigenfaces import (
    get_active_model,
    load_eigenface_model,
    set_active_model,
    SUPPORTED_IMAGE_EXTS,
)


def preprocess_face_image(
    image: Image.Image | np.ndarray,
    target_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Preprocess a test image identically to the training pipeline.

    Parameters
    ----------
    image : Image.Image | np.ndarray
        Input image.
    target_size : tuple[int, int]
        (width, height) expected by the trained model.

    Returns
    -------
    x_test : np.ndarray
        (D,) float32 flattened vector.
    thumbnail : np.ndarray
        (H, W) uint8 grayscale image.
    """
    target_w, target_h = target_size

    if isinstance(image, Image.Image):
        pil_img = image.convert("L")
    elif isinstance(image, np.ndarray):
        if image.ndim == 2:
            pil_img = Image.fromarray(image.astype(np.uint8))
        elif image.ndim == 3:
            pil_img = Image.fromarray(image.astype(np.uint8)).convert("L")
        else:
            raise ValueError(f"Invalid image array dimension: {image.ndim}")
    else:
        raise TypeError(f"Expected PIL Image or np.ndarray, got {type(image)}")

    resized = pil_img.resize((target_w, target_h), Image.Resampling.LANCZOS)
    thumbnail = np.array(resized, dtype=np.uint8)
    x_test = thumbnail.astype(np.float32).reshape(-1)

    return x_test, thumbnail


def project_test_face(
    x_test: np.ndarray,
    mean_face: np.ndarray,
    eigenfaces: np.ndarray,
) -> np.ndarray:
    """Project centered test face vector onto eigenfaces: w_test = U_K (x_test - μ).

    Parameters
    ----------
    x_test : np.ndarray
        (D,) preprocessed face vector.
    mean_face : np.ndarray
        (D,) mean face vector μ.
    eigenfaces : np.ndarray
        (K, D) orthonormal eigenface matrix U_K.

    Returns
    -------
    w_test : np.ndarray
        (K,) float32 PCA projection coefficients.
    """
    centered = x_test.astype(np.float32) - mean_face.astype(np.float32)
    # (K, D) @ (D,) -> (K,)
    w_test = np.dot(eigenfaces, centered)
    return w_test.astype(np.float32)


def recognize_face(
    test_image: Image.Image | np.ndarray,
    model: dict[str, Any],
    top_k: int = 3,
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray]:
    """Identify the closest matching training faces for a given test image.

    Parameters
    ----------
    test_image : Image.Image | np.ndarray
        Query face image.
    model : dict[str, Any]
        Loaded Eigenface model containing mean_face, eigenfaces, coefficients, labels, etc.
    top_k : int
        Number of closest matches to return (default 3).

    Returns
    -------
    top_matches : list[dict[str, Any]]
        List of dictionaries for the top matching candidates.
    w_test : np.ndarray
        (K,) PCA projection coefficients for query image.
    test_thumbnail : np.ndarray
        (H, W) preprocessed grayscale query face.
    """
    target_h, target_w = model["image_shape"]
    mean_face = model["mean_face"]
    eigenfaces = model["eigenfaces"]
    coefficients = model["coefficients"]  # (N, K)
    labels = model["labels"]              # (N,)
    thumbnails = model.get("thumbnails")  # (N, H, W) or None

    x_test, test_thumbnail = preprocess_face_image(test_image, (target_w, target_h))
    w_test = project_test_face(x_test, mean_face, eigenfaces)

    # Euclidean distance against all training coefficients:
    # d_i = ||w_test - w_i||_2
    diffs = coefficients - w_test.reshape(1, -1)
    distances = np.linalg.norm(diffs, axis=1)

    # argmin nearest neighbor
    n_candidates = len(distances)
    k_retrieve = min(max(1, top_k), n_candidates)
    sorted_indices = np.argsort(distances)[:k_retrieve]

    top_matches: list[dict[str, Any]] = []
    for rank, idx in enumerate(sorted_indices):
        name = str(labels[idx])
        dist = float(distances[idx])
        thumb = thumbnails[idx] if thumbnails is not None and idx < len(thumbnails) else None
        top_matches.append({
            "rank": rank + 1,
            "index": int(idx),
            "name": name,
            "distance": dist,
            "thumbnail": thumb,
        })

    return top_matches, w_test, test_thumbnail


def lookup_ground_truth(
    image_path_or_name: str | Path | None,
    search_dirs: list[str | Path] | None = None,
) -> tuple[str | None, bool]:
    """Look up the actual ground truth identity from labels.csv or manifest.json.

    Parameters
    ----------
    image_path_or_name : str | Path | None
        Filename or path to the test image.
    search_dirs : list[str | Path] | None
        Optional directories to inspect for manifests.

    Returns
    -------
    true_identity : str | None
        Identity if found in manifest, otherwise None.
    is_known : bool
        True if ground truth was found, False otherwise.
    """
    if image_path_or_name is None:
        return None, False

    path = Path(image_path_or_name)
    fname = path.name

    # Candidate directories containing manifest
    candidate_dirs: list[Path] = []
    if path.is_file():
        candidate_dirs.append(path.parent)
        candidate_dirs.append(path.parent.parent)
    candidate_dirs.append(Path("LFW_Test_Images"))
    candidate_dirs.append(Path.cwd() / "LFW_Test_Images")
    if search_dirs:
        for d in search_dirs:
            candidate_dirs.append(Path(d))

    for d in candidate_dirs:
        if not d.is_dir():
            continue

        # 1. Try manifest.json
        json_file = d / "manifest.json"
        if json_file.is_file():
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
                if fname in manifest:
                    return str(manifest[fname]), True
                # Match relative path
                for k, v in manifest.items():
                    if Path(k).name == fname:
                        return str(v), True
            except Exception:
                pass

        # 2. Try labels.csv
        csv_file = d / "labels.csv"
        if csv_file.is_file():
            try:
                with open(csv_file, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        row_fname = row.get("filename", "")
                        rel_path = row.get("relative_path", "")
                        if row_fname == fname or Path(rel_path).name == fname:
                            identity = row.get("identity")
                            if identity:
                                return str(identity), True
            except Exception:
                pass

    # If folder name itself represents identity in LFW_Test_Images
    if path.is_file() and ("lfw_test_images" in [p.lower() for p in path.parts]):
        parent_name = path.parent.name.replace("_", " ").strip()
        if parent_name and parent_name.lower() != "lfw_test_images":
            return parent_name.title(), True

    return None, False


def render_recognition_result(
    test_thumbnail: np.ndarray,
    top_matches: list[dict[str, Any]],
    query_title: str = "Query Face",
    true_identity: str | None = None,
    is_correct: bool | None = None,
) -> Image.Image:
    """Render a side-by-side figure showing the test face and top matches.

    Includes:
      - Test image and filename
      - Predicted identity
      - True identity from test manifest (if available)
      - Correct/incorrect status when ground truth is known
      - Nearest Euclidean distance
      - Closest 2-3 training images with thumbnails, names and distances
    """
    n_matches = len(top_matches)
    total_panels = 1 + n_matches

    fig, axes = plt.subplots(1, total_panels, figsize=(total_panels * 2.9, 3.8), dpi=100)
    fig.patch.set_facecolor("#f8f9fa")

    if total_panels == 1:
        axes = [axes]

    # Panel 0: Query Face
    ax_query = axes[0]
    ax_query.imshow(test_thumbnail, cmap="gray")

    best_match = top_matches[0] if top_matches else {"name": "None", "distance": 0.0}
    pred_name = best_match["name"]

    status_str = ""
    status_color = "#0066cc"
    if true_identity is not None and is_correct is not None:
        status_str = f"Status: [{'CORRECT' if is_correct else 'INCORRECT'}]\n"
        status_color = "#28a745" if is_correct else "#dc3545"
        title_text = (
            f"{query_title}\n"
            f"Predicted: {pred_name}\n"
            f"True: {true_identity}\n"
            f"{status_str}"
            f"Dist: {best_match['distance']:.2f}"
        )
    else:
        title_text = (
            f"{query_title}\n"
            f"Predicted: {pred_name}\n"
            f"True: (Unknown/External)\n"
            f"Dist: {best_match['distance']:.2f}"
        )

    ax_query.set_title(title_text, fontsize=9, fontweight="bold", pad=8)
    ax_query.axis("off")
    for spine in ax_query.spines.values():
        spine.set_edgecolor(status_color)
        spine.set_linewidth(3)
        spine.set_visible(True)

    # Panels 1..N: Top Matches
    for rank, match in enumerate(top_matches):
        ax = axes[rank + 1]
        thumb = match.get("thumbnail")
        if thumb is not None:
            ax.imshow(thumb, cmap="gray")
        else:
            ax.text(
                0.5, 0.5, match["name"], ha="center", va="center", fontsize=10, wrap=True
            )
            ax.set_facecolor("#e9ecef")

        border_color = "#28a745" if rank == 0 else "#6c757d"
        for spine in ax.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(3 if rank == 0 else 1.5)
            spine.set_visible(True)

        ax.set_title(
            f"Match #{rank + 1}: {match['name']}\nEuclidean Dist: {match['distance']:.2f}",
            fontsize=9,
            fontweight="bold" if rank == 0 else "normal",
            pad=8,
        )
        ax.axis("off")

    plt.suptitle("Eigenfaces Face Recognition (Set 6.3)", fontsize=13, fontweight="bold", y=0.99)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


# ---------------------------------------------------------------------------
# Interactive Tkinter Recognition Window
# ---------------------------------------------------------------------------

class FaceRecognitionWindow(tk.Toplevel):
    """Interactive GUI dialog for Set 6.3 Face Recognition."""

    def __init__(
        self,
        parent: tk.Misc,
        document: ImageDocument,
        initial_model: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(parent)
        self.title("Set 6.3 – Face Recognition (Eigenfaces)")
        self.transient(parent)
        self.geometry("900x640")
        self.minsize(780, 520)

        self.document = document
        self.model: dict[str, Any] | None = initial_model or get_active_model()
        self.test_image: Image.Image | None = None
        self.test_image_path: Path | None = None
        self.result: ToolResult | None = None

        self._build_ui()
        self._check_initial_state()

    def _build_ui(self) -> None:
        main_frame = ttk.Frame(self, padding=14)
        main_frame.pack(fill="both", expand=True)

        # Title
        title_label = ttk.Label(
            main_frame,
            text="Set 6.3: Face Recognition (Eigenfaces Nearest-Neighbour)",
            font=("TkDefaultFont", 12, "bold"),
            foreground="#003366",
        )
        title_label.pack(anchor="w", pady=(0, 4))

        desc_label = ttk.Label(
            main_frame,
            text="Select a pre-trained model and test face image to find nearest training identities using PCA projection.",
            font=("TkDefaultFont", 9),
            foreground="#555555",
        )
        desc_label.pack(anchor="w", pady=(0, 10))

        # Controls Toolbar
        toolbar = ttk.LabelFrame(main_frame, text="Recognition Workflow Controls", padding=10)
        toolbar.pack(fill="x", pady=(0, 10))

        # 1. Load Model
        self.btn_load_model = ttk.Button(
            toolbar,
            text="1. Load Model (.npy)",
            command=self._on_load_model,
        )
        self.btn_load_model.grid(row=0, column=0, padx=4, pady=4, sticky="w")

        self.model_status_var = tk.StringVar(value="No model loaded. Click 'Load Model' to begin.")
        self.model_status_label = ttk.Label(
            toolbar,
            textvariable=self.model_status_var,
            font=("TkDefaultFont", 9, "italic"),
            foreground="#666666",
        )
        self.model_status_label.grid(row=0, column=1, columnspan=2, padx=8, pady=4, sticky="w")

        # 2. Select Test Image
        self.btn_select_image = ttk.Button(
            toolbar,
            text="2. Select Test Image",
            command=self._on_select_test_image,
        )
        self.btn_select_image.grid(row=1, column=0, padx=4, pady=4, sticky="w")

        self.image_status_var = tk.StringVar(value="No test image selected.")
        self.image_status_label = ttk.Label(
            toolbar,
            textvariable=self.image_status_var,
            font=("TkDefaultFont", 9, "italic"),
            foreground="#666666",
        )
        self.image_status_label.grid(row=1, column=1, columnspan=2, padx=8, pady=4, sticky="w")

        # 3. Recognise
        self.btn_recognise = ttk.Button(
            toolbar,
            text="3. Recognise Face",
            command=self._on_recognise,
            state="disabled",
        )
        self.btn_recognise.grid(row=2, column=0, padx=4, pady=6, sticky="w")

        self.recog_info_var = tk.StringVar(value="")
        self.recog_info_label = ttk.Label(
            toolbar,
            textvariable=self.recog_info_var,
            font=("TkDefaultFont", 9, "bold"),
            foreground="#0055aa",
        )
        self.recog_info_label.grid(row=2, column=1, columnspan=2, padx=8, pady=6, sticky="w")

        # Result Display Area
        preview_frame = ttk.LabelFrame(main_frame, text="Recognition Result & Matches", padding=10)
        preview_frame.pack(fill="both", expand=True, pady=(0, 8))

        self.canvas_label = ttk.Label(preview_frame, text="Results will appear here after clicking 'Recognise Face'.")
        self.canvas_label.pack(fill="both", expand=True)

        # Bottom Bar
        bottom_bar = ttk.Frame(main_frame)
        bottom_bar.pack(fill="x", pady=(4, 0))

        ttk.Button(bottom_bar, text="Apply to ForensicsApp View & Close", command=self._on_apply).pack(side="right")
        ttk.Button(bottom_bar, text="Close", command=self.destroy).pack(side="right", padx=(0, 6))

    def _check_initial_state(self) -> None:
        if self.model is not None:
            k = len(self.model["eigenfaces"])
            n_samples = len(self.model["coefficients"])
            shape = self.model["image_shape"]
            self.model_status_var.set(f"Model active: K={k}, {n_samples} training faces, shape {shape[1]}x{shape[0]}.")
            self.model_status_label.configure(foreground="#155724")

        # If document has an image loaded, default to it
        if self.document.is_loaded and self.document.current is not None:
            self.test_image = self.document.current.copy()
            self.test_image_path = self.document.path
            name = self.document.path.name if self.document.path else "Current Document Image"
            true_id, has_gt = lookup_ground_truth(self.test_image_path)
            gt_str = f" | True Identity: {true_id}" if has_gt else ""
            self.image_status_var.set(f"Test image: '{name}'{gt_str}")
            self.image_status_label.configure(foreground="#155724")

        self._update_recognise_button()

    def _update_recognise_button(self) -> None:
        if self.model is not None and self.test_image is not None:
            self.btn_recognise.configure(state="normal")
        else:
            self.btn_recognise.configure(state="disabled")

    def _on_load_model(self) -> None:
        path = ask_npy_file(self, title="Load Eigenface model (.npy or .npz)")
        if not path:
            return
        try:
            model = load_eigenface_model(path)
            self.model = model
            set_active_model(model)
            k = len(model["eigenfaces"])
            n_samples = len(model["coefficients"])
            shape = model["image_shape"]
            self.model_status_var.set(f"Loaded '{Path(path).name}': K={k}, {n_samples} training samples ({shape[1]}x{shape[0]}).")
            self.model_status_label.configure(foreground="#155724")
            self._update_recognise_button()
        except Exception as err:
            show_error(self, "Load Error", f"Failed to load model file:\n{err}")

    def _on_select_test_image(self) -> None:
        path = ask_image_file(self, title="Select test face image (JPG, JPEG, PNG, BMP)")
        if not path:
            return
        try:
            with Image.open(path) as opened:
                self.test_image = opened.copy()
            self.test_image_path = Path(path)
            true_id, has_gt = lookup_ground_truth(self.test_image_path)
            gt_str = f" | True: {true_id}" if has_gt else " | True: (Not in manifest)"
            self.image_status_var.set(f"Loaded '{Path(path).name}'{gt_str}")
            self.image_status_label.configure(foreground="#155724")
            self._update_recognise_button()
        except Exception as err:
            show_error(self, "Open Error", f"Could not open image file:\n{err}")

    def _on_recognise(self) -> None:
        if self.model is None or self.test_image is None:
            return

        try:
            top_matches, w_test, test_thumb = recognize_face(
                test_image=self.test_image,
                model=self.model,
                top_k=3,
            )
        except Exception as err:
            show_error(self, "Recognition Error", f"Recognition failed:\n{err}")
            return

        best_match = top_matches[0]
        pred_id = best_match["name"]

        # Ground truth check
        true_id, has_gt = lookup_ground_truth(self.test_image_path)
        is_correct: bool | None = None
        if has_gt and true_id:
            is_correct = (pred_id.strip().lower() == true_id.strip().lower())

        query_name = self.test_image_path.name if self.test_image_path else "Query Image"

        # Render side-by-side figure
        rendered_img = render_recognition_result(
            test_thumbnail=test_thumb,
            top_matches=top_matches,
            query_title=query_name,
            true_identity=true_id,
            is_correct=is_correct,
        )

        # Update GUI preview
        max_w, max_h = 840, 320
        w, h = rendered_img.size
        scale = min(max_w / w, max_h / h, 1.0)
        disp_w, disp_h = max(1, int(w * scale)), max(1, int(h * scale))
        disp_img = rendered_img.resize((disp_w, disp_h), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(disp_img)
        self.canvas_label.configure(image=photo, text="")
        self.canvas_label.image = photo

        # Status text
        status_info = f"Prediction: '{pred_id}' (Distance: {best_match['distance']:.2f})"
        if is_correct is not None:
            status_info += f"  —  Status: {'[CORRECT]' if is_correct else '[INCORRECT]'} (True: '{true_id}')"
        self.recog_info_var.set(status_info)
        color = "#155724" if (is_correct is True or is_correct is None) else "#dc3545"
        self.recog_info_label.configure(foreground=color)

        # Store tool result
        details: dict[str, Any] = {
            "Operation": "Eigenface PCA Recognition (Set 6.3)",
            "Query image": query_name,
            "Predicted identity": pred_id,
            "Rank 1 Match": f"{pred_id} (Distance: {best_match['distance']:.3f})",
        }
        if true_id is not None:
            details["True identity"] = true_id
            details["Ground truth verification"] = "CORRECT" if is_correct else "INCORRECT"
        else:
            details["True identity"] = "Unknown / External (not in manifest)"

        for m in top_matches[1:]:
            details[f"Rank {m['rank']} Match"] = f"{m['name']} (Distance: {m['distance']:.3f})"

        details["Model components (K)"] = len(self.model["eigenfaces"])
        details["Training images compared"] = len(self.model["coefficients"])

        msg = f"Recognition complete: Closest identity is '{pred_id}' (Distance: {best_match['distance']:.2f})."
        if is_correct is not None:
            msg += f" Ground truth verification: {'CORRECT' if is_correct else 'INCORRECT'}."

        self.result = ToolResult(
            image=rendered_img,
            message=msg,
            details=details,
        )

    def _on_apply(self) -> None:
        self.destroy()


# ---------------------------------------------------------------------------
# ForensicsTool Class
# ---------------------------------------------------------------------------

class FaceRecognitionTool(ForensicsTool):
    """Set 6.3: Recognize face using saved Eigenface PCA model."""

    tool_id = "face_recognition"
    title = "Face recognition (Eigenfaces)"
    category = "Set 6: Face Detection & Recognition"
    description = "Match a test face to the closest training faces using PCA projection and Euclidean distance."
    requires_image = False

    def run(self, parent: tk.Misc | None, document: ImageDocument) -> ToolResult | None:
        if parent is not None:
            dialog = FaceRecognitionWindow(parent, document)
            dialog.grab_set()
            dialog.wait_window()
            return dialog.result

        # Headless mode (automated testing)
        model = get_active_model()
        if model is None:
            return None

        if not document.is_loaded or document.current is None:
            return None

        top_matches, w_test, test_thumb = recognize_face(
            test_image=document.current,
            model=model,
            top_k=3,
        )

        query_name = document.path.name if document.path else "Query Image"
        true_id, has_gt = lookup_ground_truth(document.path)
        is_correct = (top_matches[0]["name"].strip().lower() == true_id.strip().lower()) if has_gt and true_id else None

        rendered = render_recognition_result(
            test_thumbnail=test_thumb,
            top_matches=top_matches,
            query_title=query_name,
            true_identity=true_id,
            is_correct=is_correct,
        )

        best_match = top_matches[0]
        details: dict[str, Any] = {
            "Operation": "Eigenface PCA Recognition (Set 6.3)",
            "Query image": query_name,
            "Predicted identity": best_match["name"],
            "Rank 1 Match": f"{best_match['name']} (Distance: {best_match['distance']:.3f})",
        }
        if true_id is not None:
            details["True identity"] = true_id
            details["Ground truth verification"] = "CORRECT" if is_correct else "INCORRECT"

        for m in top_matches[1:]:
            details[f"Rank {m['rank']} Match"] = f"{m['name']} (Distance: {m['distance']:.3f})"

        return ToolResult(
            image=rendered,
            message=f"Recognition complete: Predicted '{best_match['name']}' (Distance: {best_match['distance']:.2f}).",
            details=details,
        )
