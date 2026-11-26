"""Small Tk dialogs used by parameterized tools."""

from __future__ import annotations

import os
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk
from typing import Any

IMAGE_FILETYPES = [
    ("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"),
    ("All files", "*.*"),
]


def ask_image_file(parent: tk.Misc | None, title: str = "Select image") -> str | None:
    """Ask the user to select an image file; return the path or None if cancelled."""
    filename = filedialog.askopenfilename(
        parent=parent,
        title=title,
        filetypes=IMAGE_FILETYPES,
    )
    return filename or None


def ask_image_folder(parent: tk.Misc | None, title: str = "Select image folder") -> str | None:
    """Ask the user to select a folder; return its path or None if cancelled."""
    folder = filedialog.askdirectory(parent=parent, title=title, mustexist=True)
    return folder or None


def show_error(parent: tk.Misc | None, title: str, message: str) -> None:
    """Display an error message dialog to the user."""
    if parent is None:
        messagebox.showerror(title, message)
    else:
        messagebox.showerror(title, message, parent=parent)


def ask_choice(
    parent: tk.Misc | None,
    title: str,
    prompt: str,
    options: list[tuple[str, str]],
    initial: str | None = None,
) -> str | None:
    """Return the selected option value, or ``None`` if the user cancels.

    ``options`` is a list of ``(value, label)`` pairs.
    """
    default = initial if initial is not None else options[0][0]
    if parent is None:
        return default

    chosen: dict[str, str | None] = {"value": None}
    window = tk.Toplevel(parent)
    window.title(title)
    window.transient(parent)
    window.resizable(False, False)

    frame = ttk.Frame(window, padding=12)
    frame.pack(fill="both", expand=True)
    ttk.Label(frame, text=prompt, wraplength=320).pack(anchor="w", pady=(0, 8))

    variable = tk.StringVar(value=default)
    for value, label in options:
        ttk.Radiobutton(frame, text=label, variable=variable, value=value).pack(anchor="w")

    buttons = ttk.Frame(frame)
    buttons.pack(fill="x", pady=(12, 0))

    def confirm() -> None:
        chosen["value"] = variable.get()
        window.destroy()

    def cancel() -> None:
        window.destroy()

    ttk.Button(buttons, text="OK", command=confirm).pack(side="right")
    ttk.Button(buttons, text="Cancel", command=cancel).pack(side="right", padx=(0, 6))
    window.protocol("WM_DELETE_WINDOW", cancel)
    window.grab_set()
    window.wait_window()
    return chosen["value"]


def ask_float(
    parent: tk.Misc | None,
    title: str,
    prompt: str,
    initial: float = 1.0,
    min_val: float | None = None,
    max_val: float | None = None,
) -> float | None:
    """Ask for a floating point number; return None if cancelled."""
    if parent is None:
        return initial
    kwargs: dict[str, object] = {"parent": parent, "initialvalue": initial}
    if min_val is not None:
        kwargs["minvalue"] = min_val
    if max_val is not None:
        kwargs["maxvalue"] = max_val
    return simpledialog.askfloat(title, prompt, **kwargs)


def ask_hog_parameters(
    parent: tk.Misc | None,
    initial_orientations: str = "2, 4, 9",
    initial_pixels_per_cell: str = "6, 6",
    initial_cells_per_block: str = "3, 3",
) -> tuple[list[int], tuple[int, int], tuple[int, int]] | None:
    """Prompt the user for HOG parameters: orientations, pixels_per_cell, and cells_per_block.

    In headless/test mode (parent is None), returns the parsed initial values.
    """
    def _parse(
        ori_str: str, ppc_str: str, cpb_str: str
    ) -> tuple[list[int], tuple[int, int], tuple[int, int]]:
        oris = [int(x.strip()) for x in ori_str.split(",") if x.strip()]
        if not oris:
            raise ValueError("At least one orientation value is required.")
        if any(o < 1 for o in oris):
            raise ValueError("All orientations must be positive integers.")

        ppc_parts = [int(x.strip()) for x in ppc_str.replace("x", ",").split(",") if x.strip()]
        if len(ppc_parts) == 1:
            ppc = (ppc_parts[0], ppc_parts[0])
        elif len(ppc_parts) == 2:
            ppc = (ppc_parts[0], ppc_parts[1])
        else:
            raise ValueError("Pixels per cell must be 1 or 2 positive integers (e.g. '6, 6' or '8').")
        if ppc[0] < 1 or ppc[1] < 1:
            raise ValueError("Pixels per cell dimensions must be positive integers.")

        cpb_parts = [int(x.strip()) for x in cpb_str.replace("x", ",").split(",") if x.strip()]
        if len(cpb_parts) == 1:
            cpb = (cpb_parts[0], cpb_parts[0])
        elif len(cpb_parts) == 2:
            cpb = (cpb_parts[0], cpb_parts[1])
        else:
            raise ValueError("Cells per block must be 1 or 2 positive integers (e.g. '3, 3' or '2').")
        if cpb[0] < 1 or cpb[1] < 1:
            raise ValueError("Cells per block dimensions must be positive integers.")

        return oris, ppc, cpb

    if parent is None:
        return _parse(initial_orientations, initial_pixels_per_cell, initial_cells_per_block)

    result: dict[str, tuple[list[int], tuple[int, int], tuple[int, int]] | None] = {"value": None}
    window = tk.Toplevel(parent)
    window.title("HOG Feature Parameters")
    window.transient(parent)
    window.resizable(False, False)

    frame = ttk.Frame(window, padding=14)
    frame.pack(fill="both", expand=True)

    ori_var = tk.StringVar(value=initial_orientations)
    ppc_var = tk.StringVar(value=initial_pixels_per_cell)
    cpb_var = tk.StringVar(value=initial_cells_per_block)

    ttk.Label(frame, text="Orientations (comma-separated, e.g. 2, 4, 9):").grid(
        row=0, column=0, sticky="w", pady=(0, 6)
    )
    ttk.Entry(frame, textvariable=ori_var, width=16).grid(
        row=0, column=1, sticky="e", padx=(12, 0), pady=(0, 6)
    )

    ttk.Label(frame, text="Pixels per cell (e.g. 6, 6 or 8, 8):").grid(
        row=1, column=0, sticky="w", pady=(0, 6)
    )
    ttk.Entry(frame, textvariable=ppc_var, width=16).grid(
        row=1, column=1, sticky="e", padx=(12, 0), pady=(0, 6)
    )

    ttk.Label(frame, text="Cells per block (e.g. 3, 3 or 2, 2):").grid(
        row=2, column=0, sticky="w"
    )
    ttk.Entry(frame, textvariable=cpb_var, width=16).grid(
        row=2, column=1, sticky="e", padx=(12, 0)
    )

    error_label = ttk.Label(frame, text="", foreground="red")
    error_label.grid(row=3, column=0, columnspan=2, sticky="w", pady=(8, 0))

    buttons = ttk.Frame(frame)
    buttons.grid(row=4, column=0, columnspan=2, sticky="e", pady=(12, 0))

    def confirm() -> None:
        try:
            parsed = _parse(ori_var.get(), ppc_var.get(), cpb_var.get())
        except ValueError as err:
            error_label.configure(text=str(err))
            return
        result["value"] = parsed
        window.destroy()

    def cancel() -> None:
        window.destroy()

    ttk.Button(buttons, text="OK", command=confirm).pack(side="right")
    ttk.Button(buttons, text="Cancel", command=cancel).pack(side="right", padx=(0, 6))
    window.protocol("WM_DELETE_WINDOW", cancel)
    window.grab_set()
    window.wait_window()
    return result["value"]


def ask_template_matching_parameters(
    parent: tk.Misc | None,
    initial_angle: float = 0.0,
    initial_threshold: float = 0.8,
) -> tuple[float, float] | None:
    """Ask for rotation and threshold together, or return slide defaults headlessly."""
    if parent is None:
        return initial_angle, initial_threshold

    result: dict[str, tuple[float, float] | None] = {"value": None}
    window = tk.Toplevel(parent)
    window.title("Template Matching Parameters")
    window.resizable(False, False)

    frame = ttk.Frame(window, padding=12)
    frame.pack(fill="both", expand=True)

    angle_var = tk.StringVar(value=str(initial_angle))
    threshold_var = tk.StringVar(value=str(initial_threshold))
    ttk.Label(frame, text="Rotation angle (-360 to 360 degrees):").grid(
        row=0, column=0, sticky="w", pady=(0, 6)
    )
    ttk.Entry(frame, textvariable=angle_var, width=12).grid(
        row=0, column=1, sticky="e", padx=(12, 0), pady=(0, 6)
    )
    ttk.Label(frame, text="Detection threshold (0 to 1):").grid(
        row=1, column=0, sticky="w"
    )
    ttk.Entry(frame, textvariable=threshold_var, width=12).grid(
        row=1, column=1, sticky="e", padx=(12, 0)
    )
    error_label = ttk.Label(frame, text="", foreground="red")
    error_label.grid(row=2, column=0, columnspan=2, sticky="w", pady=(8, 0))

    buttons = ttk.Frame(frame)
    buttons.grid(row=3, column=0, columnspan=2, sticky="e", pady=(12, 0))

    def confirm() -> None:
        try:
            angle = float(angle_var.get())
            threshold = float(threshold_var.get())
        except ValueError:
            error_label.configure(text="Enter valid numbers for both values.")
            return
        if not -360 <= angle <= 360:
            error_label.configure(text="Angle must be between -360 and 360 degrees.")
            return
        if not 0 <= threshold <= 1:
            error_label.configure(text="Threshold must be between 0 and 1.")
            return
        result["value"] = angle, threshold
        window.destroy()

    def cancel() -> None:
        window.destroy()

    ttk.Button(buttons, text="OK", command=confirm).pack(side="right")
    ttk.Button(buttons, text="Cancel", command=cancel).pack(side="right", padx=(0, 6))
    window.protocol("WM_DELETE_WINDOW", cancel)
    window.grab_set()
    window.wait_window()
    return result["value"]


def ask_int(
    parent: tk.Misc | None,
    title: str,
    prompt: str,
    initial: int = 3,
    min_val: int | None = None,
    max_val: int | None = None,
) -> int | None:
    """Ask for an integer number; return None if cancelled."""
    if parent is None:
        return initial
    kwargs: dict[str, object] = {"parent": parent, "initialvalue": initial}
    if min_val is not None:
        kwargs["minvalue"] = min_val
    if max_val is not None:
        kwargs["maxvalue"] = max_val
    return simpledialog.askinteger(title, prompt, **kwargs)


def ask_template(
    parent: tk.Misc | None,
    current_image: "Image.Image",
    title: str = "Select Template",
) -> "Image.Image | None":
    """Prompt the user for a template image or crop a patch from current image.

    In headless/test mode (parent is None), returns a central patch of current_image.
    """
    from PIL import Image

    w, h = current_image.size
    crop_w = max(8, min(w, w // 3))
    crop_h = max(8, min(h, h // 3))
    left = (w - crop_w) // 2
    top = (h - crop_h) // 2
    default_crop = current_image.crop((left, top, left + crop_w, top + crop_h))

    if parent is None:
        return default_crop

    choice = ask_choice(
        parent,
        title,
        "How would you like to provide the template image?",
        [
            ("file", "Choose template image file from disk..."),
            ("crop", "Crop center region from current image"),
        ],
        initial="crop",
    )
    if choice is None:
        return None
    if choice == "file":
        path = ask_image_file(parent, title="Select template image file")
        if not path:
            return None
        try:
            return Image.open(path)
        except Exception as error:
            show_error(parent, "Template Error", f"Could not open template image: {error}")
            return None
    return default_crop


MODEL_FILETYPES = [
    ("NumPy binary model", "*.npy"),
    ("NumPy compressed archive", "*.npz"),
    ("All files", "*.*"),
]


def ask_npy_file(parent: tk.Misc | None, title: str = "Select Eigenface model (.npy)") -> str | None:
    """Ask user to select an existing model .npy or .npz file."""
    filename = filedialog.askopenfilename(
        parent=parent,
        title=title,
        filetypes=MODEL_FILETYPES,
    )
    return filename or None


def ask_save_npy_file(
    parent: tk.Misc | None,
    title: str = "Save Eigenface model (.npy)",
    initialfile: str = "eigenface_model.npy",
) -> str | None:
    """Ask user to specify destination for saving model .npy file."""
    filename = filedialog.asksaveasfilename(
        parent=parent,
        title=title,
        defaultextension=".npy",
        initialfile=initialfile,
        filetypes=MODEL_FILETYPES,
    )
    return filename or None


def ask_npz_file(parent: tk.Misc | None, title: str = "Select Eigenface model (.npz)") -> str | None:
    """Ask user to select an existing model .npz or .npy file."""
    return ask_npy_file(parent=parent, title=title)


def ask_save_npz_file(
    parent: tk.Misc | None,
    title: str = "Save Eigenface model (.npz)",
    initialfile: str = "eigenface_model.npz",
) -> str | None:
    """Ask user to specify destination for saving model .npz file."""
    filename = filedialog.asksaveasfilename(
        parent=parent,
        title=title,
        defaultextension=".npz",
        initialfile=initialfile,
        filetypes=MODEL_FILETYPES,
    )
    return filename or None


def ask_eigenface_builder_parameters(
    parent: tk.Misc | None,
    initial_source: str = "folder",
    initial_k: int = 50,
    initial_size: str = "64, 64",
) -> tuple[str, int, tuple[int, int]] | None:
    """Dialog to configure dataset source, number of components K, and image dimensions."""
    def _parse(source_val: str, k_val: str, size_str: str) -> tuple[str, int, tuple[int, int]]:
        k = int(k_val.strip())
        if k < 1:
            raise ValueError("K (number of components) must be at least 1.")
        parts = [int(x.strip()) for x in size_str.replace("x", ",").split(",") if x.strip()]
        if len(parts) == 1:
            size = (parts[0], parts[0])
        elif len(parts) == 2:
            size = (parts[0], parts[1])
        else:
            raise ValueError("Image size must be 1 or 2 positive integers (e.g. '64, 64').")
        if size[0] < 8 or size[1] < 8:
            raise ValueError("Target image dimensions must be at least 8x8 pixels.")
        return source_val, k, size

    if parent is None:
        return _parse(initial_source, str(initial_k), initial_size)

    result: dict[str, tuple[str, int, tuple[int, int]] | None] = {"value": None}
    window = tk.Toplevel(parent)
    window.title("Eigenface Builder Setup")
    window.transient(parent)
    window.resizable(False, False)

    frame = ttk.Frame(window, padding=14)
    frame.pack(fill="both", expand=True)

    source_var = tk.StringVar(value=initial_source)
    k_var = tk.StringVar(value=str(initial_k))
    size_var = tk.StringVar(value=initial_size)

    ttk.Label(frame, text="Select Dataset Source:", font=("TkDefaultFont", 9, "bold")).grid(
        row=0, column=0, columnspan=2, sticky="w", pady=(0, 6)
    )
    ttk.Radiobutton(
        frame,
        text="Folder of face images (local folder)",
        variable=source_var,
        value="folder",
    ).grid(row=1, column=0, columnspan=2, sticky="w", padx=(10, 0), pady=(0, 2))
    ttk.Radiobutton(
        frame,
        text="LFW Face Dataset (sklearn.datasets.fetch_lfw_people)",
        variable=source_var,
        value="lfw",
    ).grid(row=2, column=0, columnspan=2, sticky="w", padx=(10, 0), pady=(0, 10))

    ttk.Separator(frame, orient="horizontal").grid(row=3, column=0, columnspan=2, sticky="ew", pady=(0, 10))

    ttk.Label(frame, text="PCA Components (K):").grid(row=4, column=0, sticky="w", pady=(0, 6))
    ttk.Entry(frame, textvariable=k_var, width=14).grid(row=4, column=1, sticky="e", padx=(12, 0), pady=(0, 6))

    ttk.Label(frame, text="Normalized Face Size (WxH):").grid(row=5, column=0, sticky="w")
    ttk.Entry(frame, textvariable=size_var, width=14).grid(row=5, column=1, sticky="e", padx=(12, 0))

    error_label = ttk.Label(frame, text="", foreground="red", wraplength=280)
    error_label.grid(row=6, column=0, columnspan=2, sticky="w", pady=(8, 0))

    buttons = ttk.Frame(frame)
    buttons.grid(row=7, column=0, columnspan=2, sticky="e", pady=(12, 0))

    def confirm() -> None:
        try:
            parsed = _parse(source_var.get(), k_var.get(), size_var.get())
        except ValueError as err:
            error_label.configure(text=str(err))
            return
        result["value"] = parsed
        window.destroy()

    def cancel() -> None:
        window.destroy()

    ttk.Button(buttons, text="Continue", command=confirm).pack(side="right")
    ttk.Button(buttons, text="Cancel", command=cancel).pack(side="right", padx=(0, 6))
    window.protocol("WM_DELETE_WINDOW", cancel)
    window.grab_set()
    window.wait_window()
    return result["value"]


BOVW_FILETYPES = [
    ("BoVW model bundle", "*.bovw *.joblib *.pkl"),
    ("All files", "*.*"),
]


def ask_bovw_model_file(
    parent: tk.Misc | None,
    title: str = "Select BoVW Model Bundle (.bovw / .joblib)",
) -> str | None:
    """Prompt user to open a saved BoVW model bundle."""
    filename = filedialog.askopenfilename(
        parent=parent,
        title=title,
        filetypes=BOVW_FILETYPES,
    )
    return filename or None


def ask_save_bovw_model_file(
    parent: tk.Misc | None,
    title: str = "Save BoVW Model Bundle (.bovw)",
    initialfile: str = "bovw_model.bovw",
) -> str | None:
    """Prompt user to specify save location for a BoVW model bundle."""
    filename = filedialog.asksaveasfilename(
        parent=parent,
        title=title,
        defaultextension=".bovw",
        initialfile=initialfile,
        filetypes=BOVW_FILETYPES,
    )
    return filename or None


def ask_bovw_codebook_parameters(
    parent: tk.Misc | None,
    class_names: list[str],
    default_split: float = 0.25,
) -> tuple[int, str, float] | None:
    """Dialog displaying detected categories and allowing configuration of test split ratio.

    Locks visual words (codebook size) to 30 and descriptor method to SIFT per coursework requirements.
    """
    c_count = len(class_names)
    locked_k = 30
    locked_method = "SIFT"

    def _parse(split_str: str) -> tuple[int, str, float]:
        split = float(split_str.strip())
        if not (0.05 <= split <= 0.5):
            raise ValueError("Test split ratio must be between 0.05 and 0.5 (e.g. 0.25).")
        return locked_k, locked_method, split

    if parent is None:
        return locked_k, locked_method, default_split

    result: dict[str, tuple[int, str, float] | None] = {"value": None}
    window = tk.Toplevel(parent)
    window.title("BoVW Codebook Configuration (Set 7.1)")
    window.transient(parent)
    window.resizable(False, False)

    frame = ttk.Frame(window, padding=16)
    frame.pack(fill="both", expand=True)

    ttk.Label(
        frame,
        text=f"Detected {c_count} Categories: {', '.join(class_names)}",
        font=("TkDefaultFont", 10, "bold"),
        foreground="#0055aa",
    ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 10))

    split_var = tk.StringVar(value=str(default_split))

    # Locked detector & descriptor
    ttk.Label(frame, text="Feature Descriptor:").grid(
        row=1, column=0, sticky="w", pady=(0, 6)
    )
    ttk.Label(
        frame,
        text="SIFT",
        font=("TkDefaultFont", 9, "italic"),
        foreground="#444444",
    ).grid(row=1, column=1, sticky="w", padx=(12, 0), pady=(0, 6))

    # Locked visual words count (30)
    ttk.Label(frame, text="Visual Words (Codebook Size):").grid(
        row=2, column=0, sticky="w", pady=(0, 6)
    )
    ttk.Label(
        frame,
        text="30",
        font=("TkDefaultFont", 9, "italic"),
        foreground="#444444",
    ).grid(row=2, column=1, sticky="w", padx=(12, 0), pady=(0, 6))

    # Test set ratio configuration
    ttk.Label(frame, text="Held-Out Test Set Ratio:").grid(
        row=3, column=0, sticky="w", pady=(0, 6)
    )
    ttk.Entry(frame, textvariable=split_var, width=14).grid(
        row=3, column=1, sticky="w", padx=(12, 0), pady=(0, 6)
    )

    error_label = ttk.Label(frame, text="", foreground="red", wraplength=340)
    error_label.grid(row=4, column=0, columnspan=2, sticky="w", pady=(8, 0))

    buttons = ttk.Frame(frame)
    buttons.grid(row=5, column=0, columnspan=2, sticky="e", pady=(12, 0))

    def confirm() -> None:
        try:
            parsed = _parse(split_var.get())
        except ValueError as err:
            error_label.configure(text=str(err))
            return
        result["value"] = parsed
        window.destroy()

    def cancel() -> None:
        window.destroy()

    ttk.Button(buttons, text="Build Codebook", command=confirm).pack(side="right")
    ttk.Button(buttons, text="Cancel", command=cancel).pack(side="right", padx=(0, 6))
    window.protocol("WM_DELETE_WINDOW", cancel)
    window.grab_set()
    window.wait_window()
    return result["value"]


def ask_bovw_select_test_image(
    parent: tk.Misc | None,
    test_items: list[tuple[str, str]],
) -> int | None:
    """Prompt user to choose one of the held-out test set images to inspect."""
    if not test_items:
        return None
    if parent is None:
        return 0

    chosen: dict[str, int | None] = {"index": None}
    window = tk.Toplevel(parent)
    window.title("Select Held-Out Test Image")
    window.transient(parent)
    window.geometry("450x320")

    frame = ttk.Frame(window, padding=12)
    frame.pack(fill="both", expand=True)

    ttk.Label(
        frame,
        text="Choose a held-out test image to generate & view its BoVW histogram:",
        wraplength=420,
    ).pack(anchor="w", pady=(0, 8))

    list_frame = ttk.Frame(frame)
    list_frame.pack(fill="both", expand=True)

    scroll = ttk.Scrollbar(list_frame, orient="vertical")
    tree = ttk.Treeview(
        list_frame,
        columns=("class", "filename"),
        show="headings",
        selectmode="browse",
        yscrollcommand=scroll.set,
    )
    scroll.configure(command=tree.yview)
    scroll.pack(side="right", fill="y")
    tree.pack(side="left", fill="both", expand=True)

    tree.heading("class", text="Category")
    tree.heading("filename", text="File Name")
    tree.column("class", width=120, stretch=False)
    tree.column("filename", width=260, stretch=True)

    for idx, (c_name, f_name) in enumerate(test_items):
        tree.insert("", "end", iid=str(idx), values=(c_name, f_name))

    tree.selection_set("0")

    buttons = ttk.Frame(frame)
    buttons.pack(fill="x", pady=(10, 0))

    def confirm() -> None:
        selected = tree.selection()
        if selected:
            chosen["index"] = int(selected[0])
            window.destroy()

    def cancel() -> None:
        window.destroy()

    ttk.Button(buttons, text="Inspect Histogram", command=confirm).pack(side="right")
    ttk.Button(buttons, text="Cancel", command=cancel).pack(side="right", padx=(0, 6))
    tree.bind("<Double-1>", lambda _e: confirm())
    window.protocol("WM_DELETE_WINDOW", cancel)
    window.grab_set()
    window.wait_window()
    return chosen["index"]


def show_bovw_classification_dialog(
    parent: tk.Misc | None,
    image_name: str,
    initial_fig: Any,
    target_image: Any,
    codebook: Any,
    classifier: Any,
    scaler: Any,
    class_names: list[str],
    histogram: Any,
    kps_coords: Any = None,
    assigned_centroids: Any = None,
    visual_word_patches: Any = None,
) -> tuple[str, dict[str, float], Any] | None:
    """Section 7.3 interactive classification dialog for new images.

    Displays the computed histogram for the new image first (using the exact same
    histogram pipeline and visualization layout as Section 7.2), places a clear
    'Classify' button at the bottom, and upon clicking executes the classifier
    to display the predicted class and top probability score (e.g. p(kangaroo) = 0.64).
    """
    from PIL import Image, ImageTk
    import numpy as np
    from .bovw_core import render_test_image_histogram_mapping

    if parent is None:
        hist_scaled = scaler.transform(histogram.reshape(1, -1))
        pred = str(classifier.predict(hist_scaled)[0])
        proba_vector = classifier.predict_proba(hist_scaled)[0]
        clf_classes = list(classifier.classes_)
        probs = {
            c_name: float(proba_vector[clf_classes.index(c_name)]) if c_name in clf_classes else 0.0
            for c_name in class_names
        }
        top_p = probs.get(pred, 0.0)
        final_fig = render_test_image_histogram_mapping(
            image=target_image,
            kps_coords=kps_coords if kps_coords is not None else np.empty((0, 2)),
            assigned_centroids=assigned_centroids if assigned_centroids is not None else np.empty((0,)),
            histogram=histogram,
            class_name=pred,
            image_title=image_name,
            max_mapping_lines=120,
            prediction_text=f"p({pred}) = {top_p:.2f}",
            probabilities=probs,
        )
        return pred, probs, final_fig

    result_data: dict[str, tuple[str, dict[str, float], Any] | None] = {"value": None}
    current_fig: list[Any] = [initial_fig]

    window = tk.Toplevel(parent)
    window.title(f"Image Classification (Set 7.3) — {image_name}")
    window.transient(parent)
    window.geometry("980x620")
    window.minsize(800, 500)

    main_frame = ttk.Frame(window, padding=12)
    main_frame.pack(fill="both", expand=True)

    header_frame = ttk.Frame(main_frame)
    header_frame.pack(fill="x", pady=(0, 6))

    ttk.Label(
        header_frame,
        text=f"New Query Image: {image_name}",
        font=("TkDefaultFont", 11, "bold"),
        foreground="#003366",
    ).pack(side="left")

    step_info = ttk.Label(
        header_frame,
        text="[Step 1 of 2: BoVW Histogram Computed. Click 'Classify' below to identify category]",
        font=("TkDefaultFont", 9, "italic"),
        foreground="#555555",
    )
    step_info.pack(side="right")

    img_container = ttk.Frame(main_frame, borderwidth=1, relief="solid")
    img_container.pack(fill="both", expand=True, pady=(4, 8))

    img_label = ttk.Label(img_container)
    img_label.pack(fill="both", expand=True)

    def _update_displayed_image(pil_img: Any) -> None:
        max_w, max_h = 940, 420
        w, h = pil_img.size
        scale = min(max_w / w, max_h / h, 1.0)
        disp_w, disp_h = int(w * scale), int(h * scale)
        resized = pil_img.resize((disp_w, disp_h), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(resized)
        img_label.configure(image=photo)
        img_label.image = photo

    _update_displayed_image(initial_fig)

    badge_frame = ttk.Frame(main_frame, padding=(8, 6))
    badge_frame.pack(fill="x", pady=(0, 8))

    status_var = tk.StringVar(
        value="Histogram of visual words computed using Section 7.2 mapping pipeline. Click the 'Classify' button to execute the classifier."
    )
    status_label = ttk.Label(
        badge_frame,
        textvariable=status_var,
        font=("TkDefaultFont", 10, "bold"),
        foreground="#0055aa",
        wraplength=940,
    )
    status_label.pack(anchor="w")

    prob_details_var = tk.StringVar(value="")
    prob_details_label = ttk.Label(
        badge_frame,
        textvariable=prob_details_var,
        font=("TkDefaultFont", 9),
        foreground="#333333",
        wraplength=940,
    )
    prob_details_label.pack(anchor="w", pady=(2, 0))

    btn_frame = ttk.Frame(main_frame)
    btn_frame.pack(fill="x", pady=(4, 0))

    def on_classify() -> None:
        try:
            hist_scaled = scaler.transform(histogram.reshape(1, -1))
            pred_class = str(classifier.predict(hist_scaled)[0])
            proba_vector = classifier.predict_proba(hist_scaled)[0]
            clf_classes = list(classifier.classes_)
            probabilities: dict[str, float] = {}
            for c_name in class_names:
                if c_name in clf_classes:
                    idx = clf_classes.index(c_name)
                    probabilities[c_name] = float(proba_vector[idx])
                else:
                    probabilities[c_name] = 0.0
        except Exception as err:
            status_var.set(f"Classification failed: {err}")
            status_label.configure(foreground="#cc0000")
            return

        top_prob = probabilities.get(pred_class, 0.0)
        status_var.set(f"Predicted Class: {pred_class}   |   Top Score: p({pred_class}) = {top_prob:.2f}")
        status_label.configure(foreground="#155724")

        prob_str = "   •   ".join(f"p({c}) = {p:.2f}" for c, p in sorted(probabilities.items()))
        prob_details_var.set(f"Class Probabilities: {prob_str}")

        step_info.configure(
            text="[Step 2 of 2: Classification Complete]",
            foreground="#155724",
            font=("TkDefaultFont", 9, "bold"),
        )

        classified_fig = render_test_image_histogram_mapping(
            image=target_image,
            kps_coords=kps_coords if kps_coords is not None else np.empty((0, 2)),
            assigned_centroids=assigned_centroids if assigned_centroids is not None else np.empty((0,)),
            histogram=histogram,
            class_name=pred_class,
            image_title=image_name,
            max_mapping_lines=120,
            prediction_text=f"p({pred_class}) = {top_prob:.2f}",
            probabilities=probabilities,
        )
        current_fig[0] = classified_fig
        _update_displayed_image(classified_fig)

        classify_btn.configure(text="Classified ✓", state="disabled")
        result_data["value"] = (pred_class, probabilities, classified_fig)

    def on_close() -> None:
        window.destroy()

    classify_btn = ttk.Button(
        btn_frame,
        text="Classify Image",
        command=on_classify,
    )
    classify_btn.pack(side="left", padx=(0, 8), ipadx=12, ipady=4)

    ttk.Button(btn_frame, text="Close", command=on_close).pack(side="right")

    window.protocol("WM_DELETE_WINDOW", on_close)
    window.grab_set()
    window.wait_window()

    return result_data["value"]


# ============================================================================
# Set 9 Dialogs: Object Detection with YOLO
# ============================================================================


def ask_pt_file(parent: tk.Misc | None, title: str = "Select YOLO Model Weights (.pt)") -> str | None:
    """Open file dialog for choosing YOLO model weights (.pt)."""
    filename = filedialog.askopenfilename(
        title=title,
        filetypes=[("PyTorch Weights", "*.pt"), ("All files", "*.*")],
        parent=parent,
    )
    return filename or None


def ask_yaml_file(parent: tk.Misc | None, title: str = "Select YOLO Dataset YAML (.yaml)") -> str | None:
    """Open file dialog for choosing dataset configuration (.yaml)."""
    filename = filedialog.askopenfilename(
        title=title,
        filetypes=[("YAML files", "*.yaml *.yml"), ("All files", "*.*")],
        parent=parent,
    )
    return filename or None


def ask_yolo_inference_parameters(
    parent: tk.Misc | None,
    current_image_loaded: bool = True,
    default_model: str = "yolo11n.pt",
    default_conf: float = 0.25,
) -> dict[str, Any] | None:
    """Configuration dialog for Set 9.1: Pretrained YOLO Inference."""
    if parent is None:
        return {
            "model_path": default_model,
            "conf": default_conf,
            "colormap": "tab20",
            "box_linewidth": 2,
            "use_fallback_image": not current_image_loaded,
            "custom_image_path": None,
        }

    dialog_result: dict[str, Any] = {"value": None}
    window = tk.Toplevel(parent)
    window.title("Set 9.1: YOLO Pretrained Model Inference")
    window.transient(parent)
    window.resizable(False, False)

    frame = ttk.Frame(window, padding=16)
    frame.pack(fill="both", expand=True)

    ttk.Label(
        frame,
        text="Ultralytics YOLO Pretrained Inference (yolo11n.pt)",
        font=("TkDefaultFont", 11, "bold"),
        foreground="#0055aa",
    ).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 10))

    # Model path
    ttk.Label(frame, text="Model weights:").grid(row=1, column=0, sticky="w", pady=4)
    model_var = tk.StringVar(value=default_model)
    model_entry = ttk.Entry(frame, textvariable=model_var, width=32)
    model_entry.grid(row=1, column=1, sticky="ew", padx=6, pady=4)

    def _browse_model() -> None:
        f = ask_pt_file(window)
        if f:
            model_var.set(f)

    ttk.Button(frame, text="Browse…", command=_browse_model).grid(row=1, column=2, pady=4)

    # Image source
    ttk.Label(frame, text="Input image:").grid(row=2, column=0, sticky="w", pady=4)
    image_choice_var = tk.StringVar(value="current" if current_image_loaded else "bus")
    img_frame = ttk.Frame(frame)
    img_frame.grid(row=2, column=1, columnspan=2, sticky="w", pady=4)

    if current_image_loaded:
        ttk.Radiobutton(img_frame, text="Current open image", variable=image_choice_var, value="current").pack(anchor="w")
    ttk.Radiobutton(img_frame, text="Default sample (bus.jpg)", variable=image_choice_var, value="bus").pack(anchor="w")
    ttk.Radiobutton(img_frame, text="Browse other image…", variable=image_choice_var, value="browse").pack(anchor="w")

    custom_img_var = tk.StringVar(value="")

    # Colormap selector (tab20 required by coursework)
    ttk.Label(frame, text="Matplotlib colormap:").grid(row=3, column=0, sticky="w", pady=6)
    colormap_var = tk.StringVar(value="tab20")
    cmap_combo = ttk.Combobox(
        frame,
        textvariable=colormap_var,
        values=["tab20", "tab10", "viridis", "plasma", "hsv", "rainbow"],
        state="readonly",
        width=18,
    )
    cmap_combo.grid(row=3, column=1, sticky="w", padx=6, pady=6)

    # Confidence threshold
    ttk.Label(frame, text="Confidence threshold:").grid(row=4, column=0, sticky="w", pady=4)
    conf_var = tk.DoubleVar(value=default_conf)
    conf_scale = ttk.Scale(frame, from_=0.05, to=0.95, variable=conf_var, orient="horizontal")
    conf_scale.grid(row=4, column=1, sticky="ew", padx=6, pady=4)
    conf_lbl = ttk.Label(frame, text=f"{default_conf:.2f}")
    conf_lbl.grid(row=4, column=2, sticky="w", pady=4)

    def _update_conf_label(*_args: Any) -> None:
        conf_lbl.configure(text=f"{conf_var.get():.2f}")

    conf_var.trace_add("write", _update_conf_label)

    # Box line width
    ttk.Label(frame, text="Box border width:").grid(row=5, column=0, sticky="w", pady=4)
    width_var = tk.IntVar(value=2)
    ttk.Spinbox(frame, from_=1, to=8, textvariable=width_var, width=5).grid(row=5, column=1, sticky="w", padx=6, pady=4)

    btn_frame = ttk.Frame(frame)
    btn_frame.grid(row=6, column=0, columnspan=3, sticky="e", pady=(14, 0))

    def on_ok() -> None:
        choice = image_choice_var.get()
        custom_path = None
        if choice == "browse":
            custom_path = ask_image_file(window, title="Select Image for YOLO Inference")
            if not custom_path:
                return

        dialog_result["value"] = {
            "model_path": model_var.get().strip() or "yolo11n.pt",
            "conf": float(conf_var.get()),
            "colormap": colormap_var.get(),
            "box_linewidth": int(width_var.get()),
            "use_fallback_image": choice == "bus",
            "custom_image_path": custom_path,
        }
        window.destroy()

    def on_cancel() -> None:
        window.destroy()

    ttk.Button(btn_frame, text="Cancel", command=on_cancel).pack(side="right", padx=(6, 0))
    ttk.Button(btn_frame, text="Run Inference", command=on_ok).pack(side="right")

    window.protocol("WM_DELETE_WINDOW", on_cancel)
    window.grab_set()
    window.wait_window()
    return dialog_result["value"]


def ask_coco_to_yolo_parameters(
    parent: tk.Misc | None,
    default_input: str = "",
    default_output: str = "",
) -> dict[str, Any] | None:
    """Dialog for Section 9.2 COCO-to-YOLO conversion."""
    if parent is None:
        return {
            "input_folder": default_input,
            "output_folder": default_output,
            "create_yaml": True,
        }

    dialog_result: dict[str, Any] = {"value": None}
    window = tk.Toplevel(parent)
    window.title("Set 9.2: COCO to YOLO Dataset Conversion")
    window.transient(parent)
    window.resizable(False, False)

    frame = ttk.Frame(window, padding=16)
    frame.pack(fill="both", expand=True)

    ttk.Label(
        frame,
        text="Convert COCO JSON Annotations to YOLO Format",
        font=("TkDefaultFont", 11, "bold"),
        foreground="#0055aa",
    ).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 10))

    ttk.Label(
        frame,
        text="Converts [x, y, width, height] COCO bboxes into normalized\n"
             "[class_id x_center y_center width height] text files in labels/.",
        font=("TkDefaultFont", 9),
        foreground="#555555",
    ).grid(row=1, column=0, columnspan=3, sticky="w", pady=(0, 10))

    # Input folder
    ttk.Label(frame, text="Input COCO folder:").grid(row=2, column=0, sticky="w", pady=4)
    input_var = tk.StringVar(value=default_input)
    ttk.Entry(frame, textvariable=input_var, width=38).grid(row=2, column=1, padx=6, pady=4)

    def _browse_input() -> None:
        f = ask_image_folder(window, title="Select Input COCO Folder (containing _annotations.coco.json)")
        if f:
            input_var.set(f)

    ttk.Button(frame, text="Browse…", command=_browse_input).grid(row=2, column=2, pady=4)

    # Output folder
    ttk.Label(frame, text="Output YOLO folder:").grid(row=3, column=0, sticky="w", pady=4)
    output_var = tk.StringVar(value=default_output)
    ttk.Entry(frame, textvariable=output_var, width=38).grid(row=3, column=1, padx=6, pady=4)

    def _browse_output() -> None:
        f = ask_image_folder(window, title="Select Output Folder (will create images/ and labels/)")
        if f:
            output_var.set(f)

    ttk.Button(frame, text="Browse…", command=_browse_output).grid(row=3, column=2, pady=4)

    create_yaml_var = tk.BooleanVar(value=True)
    ttk.Checkbutton(
        frame,
        text="Automatically generate data.yaml for blood cell dataset",
        variable=create_yaml_var,
    ).grid(row=4, column=0, columnspan=3, sticky="w", pady=8)

    btn_frame = ttk.Frame(frame)
    btn_frame.grid(row=5, column=0, columnspan=3, sticky="e", pady=(12, 0))

    def on_ok() -> None:
        inp = input_var.get().strip()
        out = output_var.get().strip()
        if not inp or not os.path.isdir(inp):
            show_error(window, "Invalid Folder", "Please select a valid input COCO directory.")
            return
        if not out:
            show_error(window, "Invalid Folder", "Please specify an output directory.")
            return

        dialog_result["value"] = {
            "input_folder": inp,
            "output_folder": out,
            "create_yaml": create_yaml_var.get(),
        }
        window.destroy()

    def on_cancel() -> None:
        window.destroy()

    ttk.Button(btn_frame, text="Cancel", command=on_cancel).pack(side="right", padx=(6, 0))
    ttk.Button(btn_frame, text="Convert", command=on_ok).pack(side="right")

    window.protocol("WM_DELETE_WINDOW", on_cancel)
    window.grab_set()
    window.wait_window()
    return dialog_result["value"]


def ask_yolo_training_parameters(
    parent: tk.Misc | None,
    default_yaml: str = "",
    default_epochs: int = 50,
    default_batch: int = 16,
    default_name: str = "from_scratch_50e_v1",
) -> dict[str, Any] | None:
    """Dialog for Section 9.2 YOLO model training controls."""
    if parent is None:
        return {
            "data_yaml": default_yaml,
            "model_cfg": "yolo11n.yaml",
            "epochs": default_epochs,
            "batch": default_batch,
            "name": default_name,
            "classes": [0, 1, 2],
            "plots": True,
        }

    dialog_result: dict[str, Any] = {"value": None}
    window = tk.Toplevel(parent)
    window.title("Set 9.2: YOLO Model Training Configuration")
    window.transient(parent)
    window.resizable(False, False)

    frame = ttk.Frame(window, padding=16)
    frame.pack(fill="both", expand=True)

    ttk.Label(
        frame,
        text="YOLO Training Controls (Section 9.2)",
        font=("TkDefaultFont", 11, "bold"),
        foreground="#0055aa",
    ).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 10))

    # Data YAML
    ttk.Label(frame, text="Dataset (data.yaml):").grid(row=1, column=0, sticky="w", pady=4)
    yaml_var = tk.StringVar(value=default_yaml)
    ttk.Entry(frame, textvariable=yaml_var, width=36).grid(row=1, column=1, padx=6, pady=4)

    def _browse_yaml() -> None:
        f = ask_yaml_file(window)
        if f:
            yaml_var.set(f)

    ttk.Button(frame, text="Browse…", command=_browse_yaml).grid(row=1, column=2, pady=4)

    # Architecture / Weights
    ttk.Label(frame, text="Model initialization:").grid(row=2, column=0, sticky="w", pady=4)
    arch_var = tk.StringVar(value="yolo11n.yaml")
    arch_combo = ttk.Combobox(
        frame,
        textvariable=arch_var,
        values=["yolo11n.yaml", "yolo11n.pt", "yolo11s.yaml", "yolo11s.pt"],
        state="normal",
        width=20,
    )
    arch_combo.grid(row=2, column=1, sticky="w", padx=6, pady=4)

    # Epochs (default 50 per PDF)
    ttk.Label(frame, text="Epochs (default 50):").grid(row=3, column=0, sticky="w", pady=4)
    epochs_var = tk.IntVar(value=default_epochs)
    ttk.Spinbox(frame, from_=1, to=300, textvariable=epochs_var, width=8).grid(
        row=3, column=1, sticky="w", padx=6, pady=4
    )

    # Batch size (default 16 per PDF)
    ttk.Label(frame, text="Batch size (default 16):").grid(row=4, column=0, sticky="w", pady=4)
    batch_var = tk.IntVar(value=default_batch)
    ttk.Spinbox(frame, from_=1, to=128, textvariable=batch_var, width=8).grid(
        row=4, column=1, sticky="w", padx=6, pady=4
    )

    # Run name (default from_scratch_50e_v1 per PDF)
    ttk.Label(frame, text="Run name:").grid(row=5, column=0, sticky="w", pady=4)
    name_var = tk.StringVar(value=default_name)
    ttk.Entry(frame, textvariable=name_var, width=24).grid(row=5, column=1, sticky="w", padx=6, pady=4)

    # Classes (default 0, 1, 2 per PDF)
    ttk.Label(frame, text="Classes ([0, 1, 2]):").grid(row=6, column=0, sticky="w", pady=4)
    classes_var = tk.StringVar(value="0, 1, 2")
    ttk.Entry(frame, textvariable=classes_var, width=16).grid(row=6, column=1, sticky="w", padx=6, pady=4)

    plots_var = tk.BooleanVar(value=True)
    ttk.Checkbutton(frame, text="Generate training plots (plots=True)", variable=plots_var).grid(
        row=7, column=0, columnspan=2, sticky="w", pady=6
    )

    btn_frame = ttk.Frame(frame)
    btn_frame.grid(row=8, column=0, columnspan=3, sticky="e", pady=(12, 0))

    def on_ok() -> None:
        y_path = yaml_var.get().strip()
        if not y_path or not os.path.isfile(y_path):
            show_error(window, "Missing data.yaml", "Please select a valid data.yaml file.")
            return

        try:
            cl_list = [int(x.strip()) for x in classes_var.get().split(",") if x.strip()]
        except ValueError:
            cl_list = [0, 1, 2]

        dialog_result["value"] = {
            "data_yaml": y_path,
            "model_cfg": arch_var.get().strip() or "yolo11n.yaml",
            "epochs": int(epochs_var.get()),
            "batch": int(batch_var.get()),
            "name": name_var.get().strip() or "from_scratch_50e_v1",
            "classes": cl_list,
            "plots": plots_var.get(),
        }
        window.destroy()

    def on_cancel() -> None:
        window.destroy()

    ttk.Button(btn_frame, text="Cancel", command=on_cancel).pack(side="right", padx=(6, 0))
    ttk.Button(btn_frame, text="Start Training", command=on_ok).pack(side="right")

    window.protocol("WM_DELETE_WINDOW", on_cancel)
    window.grab_set()
    window.wait_window()
    return dialog_result["value"]


def ask_yolo_comparison_parameters(
    parent: tk.Misc | None,
    default_image: str = "",
    default_label: str = "",
    default_model: str = "",
) -> dict[str, Any] | None:
    """Dialog for Section 9.3: Unseen Image Testing & Comparison."""
    if parent is None:
        return {
            "image_path": default_image,
            "label_path": default_label,
            "model_path": default_model,
            "conf": 0.25,
            "iou": 0.5,
            "mode": "side_by_side",
        }

    dialog_result: dict[str, Any] = {"value": None}
    window = tk.Toplevel(parent)
    window.title("Set 9.3: Unseen Image Testing & GT Comparison")
    window.transient(parent)
    window.resizable(False, False)

    frame = ttk.Frame(window, padding=16)
    frame.pack(fill="both", expand=True)

    ttk.Label(
        frame,
        text="Compare Ground Truth vs YOLO Predictions (Section 9.3)",
        font=("TkDefaultFont", 11, "bold"),
        foreground="#0055aa",
    ).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 10))

    # Test Image
    ttk.Label(frame, text="Test image:").grid(row=1, column=0, sticky="w", pady=4)
    img_var = tk.StringVar(value=default_image)
    ttk.Entry(frame, textvariable=img_var, width=38).grid(row=1, column=1, padx=6, pady=4)

    def _browse_img() -> None:
        f = ask_image_file(window, title="Select Unseen Test Image (e.g. from test/images)")
        if f:
            img_var.set(f)
            # Auto-infer label path
            cand = Path(f.replace("images", "labels")).with_suffix(".txt")
            if cand.is_file():
                lbl_var.set(str(cand))

    ttk.Button(frame, text="Browse…", command=_browse_img).grid(row=1, column=2, pady=4)

    # Label file
    ttk.Label(frame, text="Ground truth (.txt):").grid(row=2, column=0, sticky="w", pady=4)
    lbl_var = tk.StringVar(value=default_label)
    ttk.Entry(frame, textvariable=lbl_var, width=38).grid(row=2, column=1, padx=6, pady=4)

    def _browse_lbl() -> None:
        f = filedialog.askopenfilename(
            title="Select Corresponding Ground Truth Label (.txt)",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            parent=window,
        )
        if f:
            lbl_var.set(f)

    ttk.Button(frame, text="Browse…", command=_browse_lbl).grid(row=2, column=2, pady=4)

    # Model Weights
    ttk.Label(frame, text="Model weights:").grid(row=3, column=0, sticky="w", pady=4)
    model_var = tk.StringVar(value=default_model)
    ttk.Entry(frame, textvariable=model_var, width=38).grid(row=3, column=1, padx=6, pady=4)

    def _browse_model() -> None:
        f = ask_pt_file(window)
        if f:
            model_var.set(f)

    ttk.Button(frame, text="Browse…", command=_browse_model).grid(row=3, column=2, pady=4)

    # Confidence Threshold
    ttk.Label(frame, text="Confidence threshold:").grid(row=4, column=0, sticky="w", pady=4)
    conf_var = tk.DoubleVar(value=0.25)
    conf_scale = ttk.Scale(frame, from_=0.05, to=0.95, variable=conf_var, orient="horizontal")
    conf_scale.grid(row=4, column=1, sticky="ew", padx=6, pady=4)
    conf_lbl = ttk.Label(frame, text="0.25")
    conf_lbl.grid(row=4, column=2, sticky="w", pady=4)
    conf_var.trace_add("write", lambda *_args: conf_lbl.configure(text=f"{conf_var.get():.2f}"))

    # Display Mode
    ttk.Label(frame, text="Visualization style:").grid(row=5, column=0, sticky="w", pady=6)
    mode_var = tk.StringVar(value="side_by_side")
    mode_frame = ttk.Frame(frame)
    mode_frame.grid(row=5, column=1, columnspan=2, sticky="w", pady=6)
    ttk.Radiobutton(mode_frame, text="Side-by-Side (GT left, Pred right)", variable=mode_var, value="side_by_side").pack(anchor="w")
    ttk.Radiobutton(mode_frame, text="Overlaid Comparison (GT green, Pred blue)", variable=mode_var, value="overlay").pack(anchor="w")

    btn_frame = ttk.Frame(frame)
    btn_frame.grid(row=6, column=0, columnspan=3, sticky="e", pady=(14, 0))

    def on_ok() -> None:
        ip = img_var.get().strip()
        if not ip or not os.path.isfile(ip):
            show_error(window, "Missing Image", "Please select a valid unseen test image.")
            return

        dialog_result["value"] = {
            "image_path": ip,
            "label_path": lbl_var.get().strip() or None,
            "model_path": model_var.get().strip() or None,
            "conf": float(conf_var.get()),
            "iou": 0.5,
            "mode": mode_var.get(),
        }
        window.destroy()

    def on_cancel() -> None:
        window.destroy()

    ttk.Button(btn_frame, text="Cancel", command=on_cancel).pack(side="right", padx=(6, 0))
    ttk.Button(btn_frame, text="Compare & Evaluate", command=on_ok).pack(side="right")

    window.protocol("WM_DELETE_WINDOW", on_cancel)
    window.grab_set()
    window.wait_window()
    return dialog_result["value"]





# ── Set 8: Image Classification using CNN Dialogs ──────────────────────────────

def ask_cnn_weights_file(
    parent: tk.Misc | None,
    title: str = "Select CNN Model Weights (.weights.h5 / .h5 / .keras)",
) -> str | None:
    """Ask user to select a CNN weights or model file."""
    filename = filedialog.askopenfilename(
        parent=parent,
        title=title,
        filetypes=[
            ("Keras Model/Weights", "*.weights.h5 *.h5 *.keras"),
            ("All files", "*.*"),
        ],
    )
    return filename or None


def ask_cnn_passport_file(
    parent: tk.Misc | None,
    title: str = "Select Model Passport (.json)",
) -> str | None:
    """Ask user to select a model passport JSON file."""
    filename = filedialog.askopenfilename(
        parent=parent,
        title=title,
        filetypes=[
            ("JSON Passport", "*.json"),
            ("All files", "*.*"),
        ],
    )
    return filename or None


def ask_cnn_features_file(
    parent: tk.Misc | None,
    title: str = "Select Extracted Features (.npz)",
) -> str | None:
    """Ask user to select a saved feature vectors NPZ file."""
    filename = filedialog.askopenfilename(
        parent=parent,
        title=title,
        filetypes=[
            ("NumPy Archive", "*.npz"),
            ("All files", "*.*"),
        ],
    )
    return filename or None


def ask_cnn_training_parameters(
    parent: tk.Misc | None,
    default_arch: str = "shallow",
    default_samples: int = 2000,
    default_epochs: int = 5,
    default_batch: int = 32,
    default_weights: str | None = None,
    default_passport: str | None = None,
    default_dataset: str | None = None,
    default_seed: int = 42,
    default_lr: float | None = None,
) -> dict[str, Any] | None:
    """Configuration dialog for Set 8.1: Train Classification Model and Save Passport."""
    default_learning_rate = 0.01 if default_lr is None else default_lr
    if parent is None:
        return {
            "architecture": default_arch,
            "sample_size": default_samples,
            "epochs": default_epochs,
            "batch_size": default_batch,
            "optimizer": "sgd",
            "learning_rate": default_learning_rate,
            "weights_path": default_weights,
            "passport_path": default_passport,
            "dataset_dir": default_dataset,
            "random_seed": default_seed,
        }

    dialog_result: dict[str, Any] = {"value": None}
    window = tk.Toplevel(parent)
    window.title("Set 8.1: Train Classification Model")
    window.transient(parent)
    window.resizable(False, False)

    frame = ttk.Frame(window, padding=16)
    frame.pack(fill="both", expand=True)

    ttk.Label(
        frame,
        text="Train CNN on CIFAR-10 and Save Model Passport",
        font=("TkDefaultFont", 11, "bold"),
        foreground="#0055aa",
    ).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 10))

    # Architecture
    ttk.Label(frame, text="CNN Architecture:").grid(row=1, column=0, sticky="w", pady=4)
    arch_var = tk.StringVar(value=default_arch)
    arch_combo = ttk.Combobox(
        frame,
        textvariable=arch_var,
        values=["shallow", "vgg", "transfer"],
        state="readonly",
        width=20,
    )
    arch_combo.grid(row=1, column=1, sticky="w", padx=6, pady=4)

    # Sample size
    ttk.Label(frame, text="Training Samples:").grid(row=2, column=0, sticky="w", pady=4)
    samples_var = tk.IntVar(value=default_samples)
    samples_entry = ttk.Entry(frame, textvariable=samples_var, width=12)
    samples_entry.grid(row=2, column=1, sticky="w", padx=6, pady=4)
    ttk.Label(frame, text="(e.g. 1000-50000)", font=("TkDefaultFont", 8), foreground="#666666").grid(row=2, column=2, sticky="w")

    # Epochs
    ttk.Label(frame, text="Epochs:").grid(row=3, column=0, sticky="w", pady=4)
    epochs_var = tk.IntVar(value=default_epochs)
    epochs_entry = ttk.Entry(frame, textvariable=epochs_var, width=12)
    epochs_entry.grid(row=3, column=1, sticky="w", padx=6, pady=4)

    # Batch size
    ttk.Label(frame, text="Batch Size:").grid(row=4, column=0, sticky="w", pady=4)
    batch_var = tk.IntVar(value=default_batch)
    batch_entry = ttk.Entry(frame, textvariable=batch_var, width=12)
    batch_entry.grid(row=4, column=1, sticky="w", padx=6, pady=4)

    # Optimizer
    ttk.Label(frame, text="Optimizer:").grid(row=5, column=0, sticky="w", pady=4)
    opt_var = tk.StringVar(value="sgd")
    opt_combo = ttk.Combobox(frame, textvariable=opt_var, values=["sgd", "adam"], state="readonly", width=12)
    opt_combo.grid(row=5, column=1, sticky="w", padx=6, pady=4)

    ttk.Label(frame, text="Learning Rate (Adam: 0.001, SGD: 0.01):").grid(row=6, column=0, sticky="w", pady=4)
    lr_var = tk.DoubleVar(value=default_learning_rate)
    ttk.Entry(frame, textvariable=lr_var, width=12).grid(row=6, column=1, sticky="w", padx=6, pady=4)
    if default_lr is None:
        def update_default_learning_rate(*_args: Any) -> None:
            lr_var.set(0.001 if opt_var.get().lower() == "adam" else 0.01)

        opt_var.trace_add("write", update_default_learning_rate)

    ttk.Label(frame, text="Random Seed:").grid(row=7, column=0, sticky="w", pady=4)
    seed_var = tk.IntVar(value=default_seed)
    ttk.Entry(frame, textvariable=seed_var, width=12).grid(row=7, column=1, sticky="w", padx=6, pady=4)

    ttk.Label(frame, text="CIFAR-10 Folder:").grid(row=8, column=0, sticky="w", pady=4)
    ds_var = tk.StringVar(value=default_dataset or "")
    ttk.Entry(frame, textvariable=ds_var, width=28).grid(row=8, column=1, sticky="ew", padx=6, pady=4)

    def _browse_ds() -> None:
        p = filedialog.askdirectory(parent=window, title="Select cifar-10-batches-py folder")
        if p:
            ds_var.set(p)

    ttk.Button(frame, text="Browse...", command=_browse_ds).grid(row=8, column=2, padx=4, pady=4)

    # Weights save path
    ttk.Label(frame, text="Save Weights To:").grid(row=9, column=0, sticky="w", pady=4)
    w_var = tk.StringVar(value=default_weights or "")
    w_entry = ttk.Entry(frame, textvariable=w_var, width=28)
    w_entry.grid(row=9, column=1, sticky="ew", padx=6, pady=4)

    def _browse_w() -> None:
        p = filedialog.asksaveasfilename(
            parent=window,
            title="Save Weights As",
            defaultextension=".weights.h5",
            filetypes=[("Keras Weights", "*.weights.h5 *.h5 *.keras"), ("All files", "*.*")],
        )
        if p:
            w_var.set(p)

    ttk.Button(frame, text="Browse...", command=_browse_w).grid(row=9, column=2, padx=4, pady=4)

    # Passport save path
    ttk.Label(frame, text="Save Passport To:").grid(row=10, column=0, sticky="w", pady=4)
    p_var = tk.StringVar(value=default_passport or "")
    p_entry = ttk.Entry(frame, textvariable=p_var, width=28)
    p_entry.grid(row=10, column=1, sticky="ew", padx=6, pady=4)

    def _browse_p() -> None:
        p = filedialog.asksaveasfilename(
            parent=window,
            title="Save Passport As",
            defaultextension=".json",
            filetypes=[("JSON Files", "*.json"), ("All files", "*.*")],
        )
        if p:
            p_var.set(p)

    ttk.Button(frame, text="Browse...", command=_browse_p).grid(row=10, column=2, padx=4, pady=4)
    ttk.Label(
        frame,
        text="Default save paths use cifar10/user_runs/ so the bundled demonstration model is not overwritten.",
        font=("TkDefaultFont", 8),
        foreground="#666666",
        wraplength=420,
    ).grid(row=11, column=0, columnspan=3, sticky="w", pady=(4, 0))

    btn_frame = ttk.Frame(frame)
    btn_frame.grid(row=12, column=0, columnspan=3, sticky="e", pady=(14, 0))

    def on_ok() -> None:
        try:
            sample_size = int(samples_var.get())
            epochs = int(epochs_var.get())
            batch_size = int(batch_var.get())
            seed = int(seed_var.get())
            lr = float(lr_var.get())
        except (tk.TclError, ValueError):
            messagebox.showerror("Invalid Parameters", "Sample size, epochs, batch size, seed and learning rate must be numbers.", parent=window)
            return
        if sample_size < 20 or epochs < 1 or batch_size < 1:
            messagebox.showerror(
                "Invalid Parameters",
                "Use at least 20 training samples, 1 epoch, and a positive batch size so all 10 classes can appear in train and validation.",
                parent=window,
            )
            return
        dialog_result["value"] = {
            "architecture": arch_var.get(),
            "sample_size": sample_size,
            "epochs": epochs,
            "batch_size": batch_size,
            "optimizer": opt_var.get(),
            "learning_rate": lr,
            "weights_path": w_var.get().strip() or None,
            "passport_path": p_var.get().strip() or None,
            "dataset_dir": ds_var.get().strip() or None,
            "random_seed": seed,
        }
        window.destroy()

    def on_cancel() -> None:
        window.destroy()

    ttk.Button(btn_frame, text="Cancel", command=on_cancel).pack(side="right", padx=(6, 0))
    ttk.Button(btn_frame, text="Train Model", command=on_ok).pack(side="right")

    window.protocol("WM_DELETE_WINDOW", on_cancel)
    window.grab_set()
    window.wait_window()
    return dialog_result["value"]


def ask_cnn_inference_parameters(
    parent: tk.Misc | None,
    current_image_loaded: bool = True,
    default_weights: str | None = None,
    default_passport: str | None = None,
    sample_images: list[str] | None = None,
) -> dict[str, Any] | None:
    """Configuration dialog for Set 8.2: Load Pretrained Model and Perform Inference."""
    if parent is None:
        return {
            "weights_path": default_weights,
            "passport_path": default_passport,
            "use_current_image": current_image_loaded,
            "custom_image_path": None,
            "show_passport": True,
        }

    dialog_result: dict[str, Any] = {"value": None}
    window = tk.Toplevel(parent)
    window.title("Set 8.2: Load Model and Classify Image")
    window.transient(parent)
    window.resizable(False, False)

    frame = ttk.Frame(window, padding=16)
    frame.pack(fill="both", expand=True)

    ttk.Label(
        frame,
        text="Load Pretrained Model and Passport for Inference",
        font=("TkDefaultFont", 11, "bold"),
        foreground="#0055aa",
    ).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 10))

    # Weights path
    ttk.Label(frame, text="Model Weights:").grid(row=1, column=0, sticky="w", pady=4)
    w_var = tk.StringVar(value=default_weights or "")
    w_entry = ttk.Entry(frame, textvariable=w_var, width=30)
    w_entry.grid(row=1, column=1, sticky="ew", padx=6, pady=4)

    def _browse_w() -> None:
        p = ask_cnn_weights_file(window)
        if p:
            w_var.set(p)

    ttk.Button(frame, text="Browse...", command=_browse_w).grid(row=1, column=2, padx=4, pady=4)

    # Passport path
    ttk.Label(frame, text="Model Passport:").grid(row=2, column=0, sticky="w", pady=4)
    p_var = tk.StringVar(value=default_passport or "")
    p_entry = ttk.Entry(frame, textvariable=p_var, width=30)
    p_entry.grid(row=2, column=1, sticky="ew", padx=6, pady=4)

    def _browse_p() -> None:
        p = ask_cnn_passport_file(window)
        if p:
            p_var.set(p)

    ttk.Button(frame, text="Browse...", command=_browse_p).grid(row=2, column=2, padx=4, pady=4)

    # Image source
    ttk.Label(frame, text="Image Source:").grid(row=3, column=0, sticky="w", pady=4)
    src_var = tk.StringVar(value="current" if current_image_loaded else "browse")
    src_frame = ttk.Frame(frame)
    src_frame.grid(row=3, column=1, columnspan=2, sticky="w", padx=6, pady=4)

    if current_image_loaded:
        ttk.Radiobutton(src_frame, text="Current loaded image", variable=src_var, value="current").pack(anchor="w")
    ttk.Radiobutton(src_frame, text="Select image file from disk", variable=src_var, value="browse").pack(anchor="w")

    # Custom image selection
    img_var = tk.StringVar()
    img_frame = ttk.Frame(frame)
    img_frame.grid(row=4, column=0, columnspan=3, sticky="ew", pady=4)
    ttk.Label(img_frame, text="Custom image:").pack(side="left")
    img_entry = ttk.Entry(img_frame, textvariable=img_var, width=28)
    img_entry.pack(side="left", padx=6, fill="x", expand=True)

    def _browse_img() -> None:
        p = ask_image_file(window, title="Select Test Image for CNN Classification")
        if p:
            img_var.set(p)
            src_var.set("browse")

    ttk.Button(img_frame, text="Browse...", command=_browse_img).pack(side="left")

    # Sample images quick selector
    samples = sample_images or []
    if samples:
        sample_frame = ttk.Frame(frame)
        sample_frame.grid(row=5, column=0, columnspan=3, sticky="w", pady=4)
        ttk.Label(sample_frame, text="Or quick pick CIFAR sample:", font=("TkDefaultFont", 8, "italic")).pack(side="left")
        sample_var = tk.StringVar(value="")
        sample_names = [os.path.basename(s) for s in samples]

        def _on_pick_sample(event=None):
            sel = sample_var.get()
            for s in samples:
                if os.path.basename(s) == sel:
                    img_var.set(s)
                    src_var.set("browse")
                    break

        sample_cb = ttk.Combobox(sample_frame, textvariable=sample_var, values=sample_names, state="readonly", width=18)
        sample_cb.pack(side="left", padx=6)
        sample_cb.bind("<<ComboboxSelected>>", _on_pick_sample)

    # Show passport popup checkbox
    show_p_var = tk.BooleanVar(value=True)
    ttk.Checkbutton(frame, text="Display model passport details dialog", variable=show_p_var).grid(
        row=6, column=0, columnspan=3, sticky="w", pady=(8, 4)
    )
    ttk.Label(
        frame,
        text="CIFAR-10 models expect 32×32 images. Photographs are resized to that size and may be classified poorly.",
        font=("TkDefaultFont", 8),
        foreground="#666666",
        wraplength=440,
    ).grid(row=7, column=0, columnspan=3, sticky="w", pady=(0, 4))

    btn_frame = ttk.Frame(frame)
    btn_frame.grid(row=8, column=0, columnspan=3, sticky="e", pady=(14, 0))

    def on_ok() -> None:
        use_current = src_var.get() == "current"
        custom_path = img_var.get().strip() or None
        if use_current and not current_image_loaded:
            messagebox.showerror("Image Required", "Load an image in the main window or browse to a file.", parent=window)
            return
        if not use_current and not custom_path:
            messagebox.showerror(
                "Image Required",
                "Select an image to classify. A random or bundled sample will not be used automatically.",
                parent=window,
            )
            return
        if not w_var.get().strip() or not p_var.get().strip():
            messagebox.showerror("Model Required", "Select both a weights file and its matching passport JSON.", parent=window)
            return
        dialog_result["value"] = {
            "weights_path": w_var.get().strip() or None,
            "passport_path": p_var.get().strip() or None,
            "use_current_image": use_current,
            "custom_image_path": custom_path,
            "show_passport": show_p_var.get(),
        }
        window.destroy()

    def on_cancel() -> None:
        window.destroy()

    ttk.Button(btn_frame, text="Cancel", command=on_cancel).pack(side="right", padx=(6, 0))
    ttk.Button(btn_frame, text="Load and Classify", command=on_ok).pack(side="right")

    window.protocol("WM_DELETE_WINDOW", on_cancel)
    window.grab_set()
    window.wait_window()
    return dialog_result["value"]


def ask_cnn_projection_parameters(
    parent: tk.Misc | None,
    default_features_path: str | None = None,
    default_samples_per_class: int = 50,
    default_perplexity: float = 40.0,
    default_weights: str | None = None,
    default_passport: str | None = None,
    umap_available: bool = True,
) -> dict[str, Any] | None:
    """Configuration dialog for Set 8.3: Feature Projection (t-SNE and UMAP)."""
    if parent is None:
        return {
            "action": "load" if default_features_path and os.path.isfile(default_features_path) else "extract",
            "features_path": default_features_path,
            "samples_per_class": default_samples_per_class,
            "perplexity": default_perplexity,
            "umap_neighbors": 15,
            "method": "both" if umap_available else "tsne",
            "weights_path": default_weights,
            "passport_path": default_passport,
        }

    dialog_result: dict[str, Any] = {"value": None}
    window = tk.Toplevel(parent)
    window.title("Set 8.3: Feature Projection (t-SNE and UMAP)")
    window.transient(parent)
    window.resizable(False, False)

    frame = ttk.Frame(window, padding=16)
    frame.pack(fill="both", expand=True)

    ttk.Label(
        frame,
        text="Feature Vector Extraction and 2D Projection (t-SNE / UMAP)",
        font=("TkDefaultFont", 11, "bold"),
        foreground="#0055aa",
    ).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 10))

    # Action choice
    has_file = bool(default_features_path and os.path.isfile(default_features_path))
    act_var = tk.StringVar(value="load" if has_file else "extract")

    act_frame = ttk.LabelFrame(frame, text="Features Source", padding=8)
    act_frame.grid(row=1, column=0, columnspan=3, sticky="ew", pady=4)

    ttk.Radiobutton(act_frame, text="Load pre-extracted features file (.npz)", variable=act_var, value="load").pack(anchor="w")
    ttk.Radiobutton(act_frame, text="Extract fresh features from CIFAR-10 test set", variable=act_var, value="extract").pack(anchor="w")

    # Features file
    file_frame = ttk.Frame(frame)
    file_frame.grid(row=2, column=0, columnspan=3, sticky="ew", pady=4)
    ttk.Label(file_frame, text="Features file:").pack(side="left")
    feat_var = tk.StringVar(value=default_features_path or "")
    feat_entry = ttk.Entry(file_frame, textvariable=feat_var, width=28)
    feat_entry.pack(side="left", padx=6, fill="x", expand=True)

    def _browse_f() -> None:
        p = ask_cnn_features_file(window)
        if p:
            feat_var.set(p)
            act_var.set("load")

    ttk.Button(file_frame, text="Browse...", command=_browse_f).pack(side="left")

    ttk.Label(frame, text="CNN Weights:").grid(row=3, column=0, sticky="w", pady=4)
    w_var = tk.StringVar(value=default_weights or "")
    ttk.Entry(frame, textvariable=w_var, width=28).grid(row=3, column=1, sticky="ew", padx=6, pady=4)

    def _browse_w() -> None:
        p = ask_cnn_weights_file(window)
        if p:
            w_var.set(p)
            act_var.set("extract")

    ttk.Button(frame, text="Browse...", command=_browse_w).grid(row=3, column=2, padx=4, pady=4)

    ttk.Label(frame, text="Model Passport:").grid(row=4, column=0, sticky="w", pady=4)
    p_var = tk.StringVar(value=default_passport or "")
    ttk.Entry(frame, textvariable=p_var, width=28).grid(row=4, column=1, sticky="ew", padx=6, pady=4)

    def _browse_p() -> None:
        p = ask_cnn_passport_file(window)
        if p:
            p_var.set(p)

    ttk.Button(frame, text="Browse...", command=_browse_p).grid(row=4, column=2, padx=4, pady=4)

    # Parameters
    ttk.Label(frame, text="Samples per class:").grid(row=5, column=0, sticky="w", pady=4)
    samples_var = tk.IntVar(value=default_samples_per_class)
    ttk.Entry(frame, textvariable=samples_var, width=12).grid(row=5, column=1, sticky="w", padx=6, pady=4)
    ttk.Label(frame, text="(e.g. 50 per class = 500 total)", font=("TkDefaultFont", 8), foreground="#666666").grid(row=5, column=2, sticky="w")

    ttk.Label(frame, text="t-SNE Perplexity:").grid(row=6, column=0, sticky="w", pady=4)
    perp_var = tk.DoubleVar(value=default_perplexity)
    ttk.Entry(frame, textvariable=perp_var, width=12).grid(row=6, column=1, sticky="w", padx=6, pady=4)

    ttk.Label(frame, text="UMAP Neighbors:").grid(row=7, column=0, sticky="w", pady=4)
    umap_var = tk.IntVar(value=15)
    ttk.Entry(frame, textvariable=umap_var, width=12).grid(row=7, column=1, sticky="w", padx=6, pady=4)

    ttk.Label(frame, text="Projection Methods:").grid(row=8, column=0, sticky="w", pady=4)
    method_var = tk.StringVar(value="both" if umap_available else "tsne")
    method_values = ["both", "tsne", "umap"] if umap_available else ["tsne"]
    method_combo = ttk.Combobox(frame, textvariable=method_var, values=method_values, state="readonly", width=12)
    method_combo.grid(row=8, column=1, sticky="w", padx=6, pady=4)
    umap_note = (
        "UMAP is available."
        if umap_available
        else "UMAP is unavailable (install umap-learn). t-SNE can still be used."
    )
    ttk.Label(frame, text=umap_note, font=("TkDefaultFont", 8), foreground="#666666", wraplength=280).grid(
        row=8, column=2, sticky="w"
    )

    btn_frame = ttk.Frame(frame)
    btn_frame.grid(row=9, column=0, columnspan=3, sticky="e", pady=(14, 0))

    def on_ok() -> None:
        dialog_result["value"] = {
            "action": act_var.get(),
            "features_path": feat_var.get().strip() or None,
            "samples_per_class": max(5, int(samples_var.get())),
            "perplexity": float(perp_var.get()),
            "umap_neighbors": max(2, int(umap_var.get())),
            "method": method_var.get(),
            "weights_path": w_var.get().strip() or None,
            "passport_path": p_var.get().strip() or None,
        }
        window.destroy()

    def on_cancel() -> None:
        window.destroy()

    ttk.Button(btn_frame, text="Cancel", command=on_cancel).pack(side="right", padx=(6, 0))
    ttk.Button(btn_frame, text="Run Projections", command=on_ok).pack(side="right")

    window.protocol("WM_DELETE_WINDOW", on_cancel)
    window.grab_set()
    window.wait_window()
    return dialog_result["value"]


def show_cnn_passport_dialog(
    parent: tk.Misc | None,
    passport: dict[str, Any],
) -> None:
    """Display structured model passport in a dialog window matching Set 8.2."""
    if parent is None:
        return

    window = tk.Toplevel(parent)
    window.title(f"Model Passport: {passport.get('model_name', 'CNN Model')}")
    window.transient(parent)
    window.geometry("600x520")
    window.resizable(True, True)

    frame = ttk.Frame(window, padding=16)
    frame.pack(fill="both", expand=True)

    ttk.Label(
        frame,
        text="Model Passport",
        font=("TkDefaultFont", 13, "bold"),
        foreground="#0055aa",
    ).pack(anchor="w", pady=(0, 8))

    txt = tk.Text(frame, wrap="word", font=("Consolas", 9), padx=8, pady=8, background="#f8f9fa")
    txt.pack(fill="both", expand=True)

    # Format passport sections
    p_text = []
    p_text.append("=" * 64)
    p_text.append(f" MODEL NAME: {passport.get('model_name', 'N/A')}")
    p_text.append(f" CREATED AT: {passport.get('created_at', 'N/A')}")
    p_text.append(f" WEIGHTS:    {passport.get('weights_path', 'N/A')}")
    p_text.append("=" * 64)
    p_text.append("")
    p_text.append("[1] DATASET & SAMPLES:")
    p_text.append(f"  * Dataset Name: {passport.get('dataset', 'CIFAR-10')}")
    p_text.append(f"  * Dataset Source: {passport.get('dataset_source', 'N/A')}")
    p_text.append(f"  * Provenance: {passport.get('dataset_provenance', 'N/A')}")
    p_text.append(f"  * Sample Size:  {passport.get('sample_size', 'N/A')} samples")
    p_text.append(f"  * Train / Val / Test: {passport.get('train_samples', 'N/A')} / {passport.get('val_samples', 'N/A')} / {passport.get('test_samples', 'N/A')}")
    p_text.append(f"  * Random Seed: {passport.get('random_seed', 'N/A')}")
    p_text.append("")
    p_text.append("[2] CLASS NAMES (10 classes):")
    classes = passport.get("class_names", [])
    for idx, cname in enumerate(classes):
        p_text.append(f"  * Class {idx}: {cname}")
    p_text.append("")
    p_text.append("[3] PREPROCESSING INSTRUCTIONS:")
    prep = passport.get("preprocessing_instructions", {})
    for k, v in prep.items():
        p_text.append(f"  * {k}: {v}")
    p_text.append("")
    p_text.append("[4] ARCHITECTURE SUMMARY:")
    arch = passport.get("architecture", {})
    for k, v in arch.items():
        p_text.append(f"  * {k}: {v}")
    p_text.append("")
    p_text.append("[5] TRAINING SETUP:")
    p_text.append(f"  * Optimizer: {passport.get('optimizer', 'N/A')}")
    p_text.append(f"  * Learning Rate: {passport.get('learning_rate', 'N/A')}")
    p_text.append(f"  * Batch Size: {passport.get('batch_size', 'N/A')}")
    p_text.append(f"  * Epochs: {passport.get('epochs', 'N/A')}")
    p_text.append(f"  * Weights Filename: {passport.get('weights_filename', Path(str(passport.get('weights_path', ''))).name)}")
    p_text.append(f"  * Pretrained Base Weights: {passport.get('pretrained_base_weights', 'none')}")
    p_text.append("")
    p_text.append("[6] TRAINING HISTORY & METRICS:")
    metrics = passport.get("metrics", {})
    for k, v in metrics.items():
        if isinstance(v, float):
            p_text.append(f"  * {k}: {v:.4f}")
        else:
            p_text.append(f"  * {k}: {v}")

    history = passport.get("training_history", {})
    if "accuracy" in history and history["accuracy"]:
        p_text.append(f"  * Epochs Trained: {len(history['accuracy'])}")
        p_text.append(f"  * Initial Loss:   {history.get('loss', [0])[0]:.4f} -> Final Loss: {history.get('loss', [0])[-1]:.4f}")
        p_text.append(f"  * Initial Acc:    {history.get('accuracy', [0])[0] * 100:.2f}% -> Final Acc: {history.get('accuracy', [0])[-1] * 100:.2f}%")
        if "val_accuracy" in history and history["val_accuracy"]:
            p_text.append(f"  * Final Val Acc:  {history['val_accuracy'][-1] * 100:.2f}%")

    txt.insert("1.0", "\n".join(p_text))
    txt.configure(state="disabled")

    btn_frame = ttk.Frame(frame)
    btn_frame.pack(fill="x", pady=(10, 0))
    ttk.Button(btn_frame, text="Close", command=window.destroy).pack(side="right")


# ============================================================================
# Set 10: CLIP Zero-Shot Classification Dialog
# ============================================================================


def ask_clip_zero_shot_parameters(
    parent: tk.Misc | None,
    current_image_loaded: bool = False,
    default_model: str = "openai/clip-vit-base-patch32",
    default_template: str = "a photo of a {}.",
) -> dict[str, Any] | None:
    """Prompt for image source, CLIP model, prompt template, and arbitrary class/prompt pairs."""
    starter_classes: list[tuple[str, str]] = [
        ("cat", ""),
        ("dog", ""),
        ("coffee", ""),
        ("car", ""),
    ]
    if parent is None:
        return {
            "use_current_image": current_image_loaded,
            "custom_image_path": None,
            "model_name": default_model,
            "template": default_template,
            "classes": starter_classes,
        }

    dialog_result: dict[str, Any] = {"value": None}
    window = tk.Toplevel(parent)
    window.title("Set 10: CLIP Zero-Shot Classification")
    window.transient(parent)
    window.resizable(True, True)
    window.minsize(560, 520)

    frame = ttk.Frame(window, padding=14)
    frame.pack(fill="both", expand=True)

    ttk.Label(
        frame,
        text="CLIP Zero-Shot Image Classification",
        font=("TkDefaultFont", 11, "bold"),
        foreground="#0055aa",
    ).pack(anchor="w", pady=(0, 8))

    # Image source
    img_frame = ttk.LabelFrame(frame, text="Image Input", padding=8)
    img_frame.pack(fill="x", pady=(0, 8))

    source_var = tk.StringVar(value="current" if current_image_loaded else "file")
    rb_current = ttk.Radiobutton(
        img_frame,
        text="Use currently loaded image",
        variable=source_var,
        value="current",
        state="normal" if current_image_loaded else "disabled",
    )
    rb_current.pack(anchor="w")

    file_row = ttk.Frame(img_frame)
    file_row.pack(fill="x", pady=(4, 0))
    ttk.Radiobutton(
        file_row,
        text="Select image file:",
        variable=source_var,
        value="file",
    ).pack(side="left")
    img_path_var = tk.StringVar(value="")
    ttk.Entry(file_row, textvariable=img_path_var, width=36).pack(
        side="left", padx=6, fill="x", expand=True
    )

    def _browse_img() -> None:
        chosen = ask_image_file(window, title="Select image for CLIP classification")
        if chosen:
            img_path_var.set(chosen)
            source_var.set("file")

    ttk.Button(file_row, text="Browse...", command=_browse_img).pack(side="left")

    # Model & template
    model_frame = ttk.LabelFrame(frame, text="CLIP Model & Prompt Template", padding=8)
    model_frame.pack(fill="x", pady=(0, 8))
    model_frame.columnconfigure(1, weight=1)

    ttk.Label(model_frame, text="Model ID:").grid(row=0, column=0, sticky="w", pady=2)
    model_var = tk.StringVar(value=default_model)
    model_combo = ttk.Combobox(
        model_frame,
        textvariable=model_var,
        values=["openai/clip-vit-base-patch32", "openai/clip-vit-large-patch14"],
    )
    model_combo.grid(row=0, column=1, sticky="ew", padx=(6, 0), pady=2)

    ttk.Label(model_frame, text="Default template:").grid(row=1, column=0, sticky="w", pady=2)
    tmpl_var = tk.StringVar(value=default_template)
    ttk.Entry(model_frame, textvariable=tmpl_var).grid(row=1, column=1, sticky="ew", padx=(6, 0), pady=2)
    ttk.Label(
        model_frame,
        text="Used when a class has no custom text (e.g. 'a photo of a {}.').",
        font=("TkDefaultFont", 8),
        foreground="#555555",
    ).grid(row=2, column=1, sticky="w", padx=(6, 0))

    # Candidate classes table
    cls_frame = ttk.LabelFrame(frame, text="Candidate Classes & Optional Custom Text", padding=8)
    cls_frame.pack(fill="both", expand=True, pady=(0, 8))

    tree_wrap = ttk.Frame(cls_frame)
    tree_wrap.pack(fill="both", expand=True)
    tree = ttk.Treeview(
        tree_wrap,
        columns=("label", "prompt"),
        show="headings",
        height=6,
        selectmode="browse",
    )
    tree.heading("label", text="Class Label")
    tree.heading("prompt", text="Custom Prompt (blank = use template)")
    tree.column("label", width=160, stretch=True)
    tree.column("prompt", width=320, stretch=True)
    tree_scroll = ttk.Scrollbar(tree_wrap, orient="vertical", command=tree.yview)
    tree.configure(yscrollcommand=tree_scroll.set)
    tree.pack(side="left", fill="both", expand=True)
    tree_scroll.pack(side="right", fill="y")

    for lbl, prm in starter_classes:
        tree.insert("", "end", values=(lbl, prm))

    edit_row = ttk.Frame(cls_frame)
    edit_row.pack(fill="x", pady=(8, 4))
    ttk.Label(edit_row, text="Label:").pack(side="left")
    lbl_var = tk.StringVar()
    ttk.Entry(edit_row, textvariable=lbl_var, width=16).pack(side="left", padx=(4, 10))
    ttk.Label(edit_row, text="Custom text:").pack(side="left")
    prm_var = tk.StringVar()
    ttk.Entry(edit_row, textvariable=prm_var, width=28).pack(side="left", padx=4, fill="x", expand=True)

    def _on_select(_event: Any = None) -> None:
        sel = tree.selection()
        if not sel:
            return
        vals = tree.item(sel[0], "values")
        if len(vals) >= 2:
            lbl_var.set(vals[0])
            prm_var.set(vals[1])

    tree.bind("<<TreeviewSelect>>", _on_select)

    def _add_class() -> None:
        lbl = lbl_var.get().strip()
        prm = prm_var.get()
        if not lbl:
            messagebox.showerror("Empty Label", "Please enter a non-empty class label.", parent=window)
            return
        if prm != "" and not prm.strip():
            messagebox.showerror(
                "Blank Custom Prompt",
                "Custom prompt contains only whitespace. Leave it empty to use the template or enter text.",
                parent=window,
            )
            return
        existing = {str(tree.item(iid, "values")[0]).strip().casefold() for iid in tree.get_children()}
        if lbl.casefold() in existing:
            messagebox.showerror("Duplicate Label", f"Class label '{lbl}' is already in the list.", parent=window)
            return
        tree.insert("", "end", values=(lbl, prm.strip() if prm else ""))
        lbl_var.set("")
        prm_var.set("")

    def _update_selected() -> None:
        sel = tree.selection()
        if not sel:
            messagebox.showinfo("No Selection", "Select a class row to update.", parent=window)
            return
        lbl = lbl_var.get().strip()
        prm = prm_var.get()
        if not lbl:
            messagebox.showerror("Empty Label", "Class label cannot be empty.", parent=window)
            return
        if prm != "" and not prm.strip():
            messagebox.showerror(
                "Blank Custom Prompt",
                "Custom prompt contains only whitespace. Leave it empty to use the template or enter text.",
                parent=window,
            )
            return
        for iid in tree.get_children():
            if iid != sel[0] and str(tree.item(iid, "values")[0]).strip().casefold() == lbl.casefold():
                messagebox.showerror("Duplicate Label", f"Class label '{lbl}' already exists.", parent=window)
                return
        tree.item(sel[0], values=(lbl, prm.strip() if prm else ""))

    def _remove_selected() -> None:
        for iid in tree.selection():
            tree.delete(iid)

    def _clear_all() -> None:
        for iid in tree.get_children():
            tree.delete(iid)

    cls_btns = ttk.Frame(cls_frame)
    cls_btns.pack(fill="x")
    ttk.Button(cls_btns, text="Add", command=_add_class).pack(side="left")
    ttk.Button(cls_btns, text="Update Selected", command=_update_selected).pack(side="left", padx=(6, 0))
    ttk.Button(cls_btns, text="Remove Selected", command=_remove_selected).pack(side="left", padx=(6, 0))
    ttk.Button(cls_btns, text="Clear All", command=_clear_all).pack(side="left", padx=(6, 0))

    btn_frame = ttk.Frame(frame)
    btn_frame.pack(fill="x", pady=(6, 0))

    def on_ok() -> None:
        use_current = source_var.get() == "current" and current_image_loaded
        custom_path = img_path_var.get().strip() or None
        if not use_current and not custom_path:
            messagebox.showerror("Image Required", "Select an image file or load an image first.", parent=window)
            return
        if not model_var.get().strip():
            messagebox.showerror("Model Required", "Enter a CLIP model ID.", parent=window)
            return
        if not tmpl_var.get().strip():
            messagebox.showerror("Blank Template", "Default prompt template cannot be blank.", parent=window)
            return

        rows: list[tuple[str, str]] = []
        seen: set[str] = set()
        for iid in tree.get_children():
            vals = tree.item(iid, "values")
            lbl = str(vals[0]).strip() if len(vals) > 0 else ""
            prm = str(vals[1]) if len(vals) > 1 else ""
            if not lbl:
                messagebox.showerror("Empty Label", "All class labels must be non-empty.", parent=window)
                return
            if lbl.casefold() in seen:
                messagebox.showerror("Duplicate Label", f"Duplicate class label: '{lbl}'.", parent=window)
                return
            seen.add(lbl.casefold())
            rows.append((lbl, prm.strip() if prm else ""))

        if not rows:
            messagebox.showerror("Classes Required", "Add at least one class label.", parent=window)
            return

        dialog_result["value"] = {
            "use_current_image": use_current,
            "custom_image_path": custom_path if not use_current else None,
            "model_name": model_var.get().strip(),
            "template": tmpl_var.get().strip(),
            "classes": rows,
        }
        window.destroy()

    def on_cancel() -> None:
        window.destroy()

    ttk.Button(btn_frame, text="Cancel", command=on_cancel).pack(side="right", padx=(6, 0))
    ttk.Button(btn_frame, text="Classify", command=on_ok).pack(side="right")

    window.protocol("WM_DELETE_WINDOW", on_cancel)
    window.grab_set()
    window.wait_window()
    return dialog_result["value"]


# ============================================================================
# Set 11: U-Net Segmentation Dialogs
# ============================================================================


def ask_unet_training_parameters(
    parent: tk.Misc | None,
    default_dataset_dir: str = "",
    default_checkpoint_path: str = "",
    default_epochs: int = 10,
    default_batch_size: int = 16,
    default_lr: float = 1e-3,
    default_height: int = 128,
    default_width: int = 208,
) -> dict[str, Any] | None:
    """Configuration dialog for Set 11.1: U-Net training on TrayDataset."""
    if parent is None:
        return {
            "dataset_dir": default_dataset_dir,
            "checkpoint_path": default_checkpoint_path,
            "epochs": default_epochs,
            "batch_size": default_batch_size,
            "learning_rate": default_lr,
            "val_fraction": 0.15,
            "input_height": default_height,
            "input_width": default_width,
        }

    dialog_result: dict[str, Any] = {"value": None}
    window = tk.Toplevel(parent)
    window.title("Set 11: Train U-Net Food Segmentation")
    window.transient(parent)
    window.resizable(False, False)

    frame = ttk.Frame(window, padding=16)
    frame.pack(fill="both", expand=True)

    ttk.Label(
        frame,
        text="Train U-Net (Food vs. Not-Food on TrayDataset)",
        font=("TkDefaultFont", 11, "bold"),
        foreground="#0055aa",
    ).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 10))

    ttk.Label(frame, text="TrayDataset folder:").grid(row=1, column=0, sticky="w", pady=4)
    ds_var = tk.StringVar(value=default_dataset_dir)
    ttk.Entry(frame, textvariable=ds_var, width=36).grid(row=1, column=1, sticky="ew", padx=6, pady=4)

    def _browse_ds() -> None:
        folder = ask_image_folder(window, title="Select TrayDataset folder")
        if folder:
            ds_var.set(folder)

    ttk.Button(frame, text="Browse...", command=_browse_ds).grid(row=1, column=2, pady=4)

    ttk.Label(frame, text="Save checkpoint to:").grid(row=2, column=0, sticky="w", pady=4)
    ckpt_var = tk.StringVar(value=default_checkpoint_path)
    ttk.Entry(frame, textvariable=ckpt_var, width=36).grid(row=2, column=1, sticky="ew", padx=6, pady=4)

    def _browse_ckpt() -> None:
        chosen = filedialog.asksaveasfilename(
            parent=window,
            title="Save U-Net checkpoint",
            defaultextension=".pt",
            initialfile=Path(ckpt_var.get() or "unet_tray_food.pt").name,
            filetypes=[("PyTorch checkpoint", "*.pt *.pth"), ("All files", "*.*")],
        )
        if chosen:
            ckpt_var.set(chosen)

    ttk.Button(frame, text="Browse...", command=_browse_ckpt).grid(row=2, column=2, pady=4)

    ttk.Label(frame, text="Epochs:").grid(row=3, column=0, sticky="w", pady=4)
    epochs_var = tk.IntVar(value=default_epochs)
    ttk.Entry(frame, textvariable=epochs_var, width=12).grid(row=3, column=1, sticky="w", padx=6, pady=4)

    ttk.Label(frame, text="Batch size:").grid(row=4, column=0, sticky="w", pady=4)
    batch_var = tk.IntVar(value=default_batch_size)
    ttk.Entry(frame, textvariable=batch_var, width=12).grid(row=4, column=1, sticky="w", padx=6, pady=4)

    ttk.Label(frame, text="Learning rate:").grid(row=5, column=0, sticky="w", pady=4)
    lr_var = tk.DoubleVar(value=default_lr)
    ttk.Entry(frame, textvariable=lr_var, width=12).grid(row=5, column=1, sticky="w", padx=6, pady=4)

    ttk.Label(frame, text="Val split fraction:").grid(row=6, column=0, sticky="w", pady=4)
    val_var = tk.DoubleVar(value=0.15)
    ttk.Entry(frame, textvariable=val_var, width=12).grid(row=6, column=1, sticky="w", padx=6, pady=4)

    ttk.Label(frame, text="Input size (H x W):").grid(row=7, column=0, sticky="w", pady=4)
    size_row = ttk.Frame(frame)
    size_row.grid(row=7, column=1, sticky="w", padx=6, pady=4)
    h_var = tk.IntVar(value=default_height)
    w_var = tk.IntVar(value=default_width)
    ttk.Entry(size_row, textvariable=h_var, width=6).pack(side="left")
    ttk.Label(size_row, text=" x ").pack(side="left")
    ttk.Entry(size_row, textvariable=w_var, width=6).pack(side="left")
    ttk.Label(frame, text="Divisible by 8", font=("TkDefaultFont", 8), foreground="#555555").grid(
        row=7, column=2, sticky="w"
    )

    btn_frame = ttk.Frame(frame)
    btn_frame.grid(row=8, column=0, columnspan=3, sticky="e", pady=(14, 0))

    def on_ok() -> None:
        ds = ds_var.get().strip()
        ckpt = ckpt_var.get().strip()
        if not ds or not os.path.isdir(ds):
            messagebox.showerror("Invalid Dataset", "Please select an existing TrayDataset directory.", parent=window)
            return
        if not ckpt:
            messagebox.showerror("Checkpoint Required", "Please specify a checkpoint save path.", parent=window)
            return
        try:
            ep = int(epochs_var.get())
            bs = int(batch_var.get())
            lr = float(lr_var.get())
            vf = float(val_var.get())
            ih = int(h_var.get())
            iw = int(w_var.get())
        except (TypeError, ValueError):
            messagebox.showerror("Invalid Parameters", "Numeric training parameters must be valid numbers.", parent=window)
            return

        if ep < 1 or bs < 1 or lr <= 0:
            messagebox.showerror("Invalid Parameters", "Epochs, batch size, and learning rate must be positive.", parent=window)
            return
        if not (0.0 < vf < 0.5):
            messagebox.showerror("Invalid Split", "Validation split fraction must be between 0 and 0.5.", parent=window)
            return
        if ih < 16 or iw < 16 or ih % 8 != 0 or iw % 8 != 0:
            messagebox.showerror("Invalid Size", "Input height and width must be >= 16 and divisible by 8.", parent=window)
            return

        dialog_result["value"] = {
            "dataset_dir": ds,
            "checkpoint_path": ckpt,
            "epochs": ep,
            "batch_size": bs,
            "learning_rate": lr,
            "val_fraction": vf,
            "input_height": ih,
            "input_width": iw,
        }
        window.destroy()

    def on_cancel() -> None:
        window.destroy()

    ttk.Button(btn_frame, text="Cancel", command=on_cancel).pack(side="right", padx=(6, 0))
    ttk.Button(btn_frame, text="Start Training", command=on_ok).pack(side="right")

    window.protocol("WM_DELETE_WINDOW", on_cancel)
    window.grab_set()
    window.wait_window()
    return dialog_result["value"]


def ask_unet_segmentation_parameters(
    parent: tk.Misc | None,
    current_image_loaded: bool = False,
    default_checkpoint_path: str = "",
    default_dataset_dir: str = "",
    default_threshold: float = 0.5,
) -> dict[str, Any] | None:
    """Configuration dialog for Set 11.2: Load U-Net checkpoint and test/segment."""
    if parent is None:
        return {
            "checkpoint_path": default_checkpoint_path,
            "mode": "image",
            "use_current_image": current_image_loaded,
            "custom_image_path": None,
            "gt_mask_path": None,
            "dataset_dir": default_dataset_dir,
            "threshold": default_threshold,
        }

    dialog_result: dict[str, Any] = {"value": None}
    window = tk.Toplevel(parent)
    window.title("Set 11: Load U-Net & Segment")
    window.transient(parent)
    window.resizable(False, False)

    frame = ttk.Frame(window, padding=16)
    frame.pack(fill="both", expand=True)

    ttk.Label(
        frame,
        text="U-Net Food Segmentation & Evaluation",
        font=("TkDefaultFont", 11, "bold"),
        foreground="#0055aa",
    ).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 10))

    ttk.Label(frame, text="Checkpoint (.pt):").grid(row=1, column=0, sticky="w", pady=4)
    ckpt_var = tk.StringVar(value=default_checkpoint_path)
    ttk.Entry(frame, textvariable=ckpt_var, width=36).grid(row=1, column=1, sticky="ew", padx=6, pady=4)

    def _browse_ckpt() -> None:
        chosen = filedialog.askopenfilename(
            parent=window,
            title="Select U-Net checkpoint",
            filetypes=[("PyTorch checkpoint", "*.pt *.pth"), ("All files", "*.*")],
        )
        if chosen:
            ckpt_var.set(chosen)

    ttk.Button(frame, text="Browse...", command=_browse_ckpt).grid(row=1, column=2, pady=4)

    mode_var = tk.StringVar(value="image")
    mode_frame = ttk.LabelFrame(frame, text="Target Input", padding=8)
    mode_frame.grid(row=2, column=0, columnspan=3, sticky="ew", pady=8)

    ttk.Radiobutton(
        mode_frame,
        text="Segment a single image",
        variable=mode_var,
        value="image",
    ).pack(anchor="w")

    img_sub = ttk.Frame(mode_frame, padding=(16, 4, 0, 6))
    img_sub.pack(fill="x")
    img_src_var = tk.StringVar(value="current" if current_image_loaded else "file")
    ttk.Radiobutton(
        img_sub,
        text="Use currently loaded image",
        variable=img_src_var,
        value="current",
        state="normal" if current_image_loaded else "disabled",
    ).pack(anchor="w")

    file_row = ttk.Frame(img_sub)
    file_row.pack(fill="x", pady=2)
    ttk.Radiobutton(file_row, text="Image file:", variable=img_src_var, value="file").pack(side="left")
    img_var = tk.StringVar(value="")
    ttk.Entry(file_row, textvariable=img_var, width=28).pack(side="left", padx=6, fill="x", expand=True)

    def _browse_img() -> None:
        chosen = ask_image_file(window, title="Select image to segment")
        if chosen:
            img_var.set(chosen)
            img_src_var.set("file")
            mode_var.set("image")

    ttk.Button(file_row, text="Browse...", command=_browse_img).pack(side="left")

    gt_row = ttk.Frame(img_sub)
    gt_row.pack(fill="x", pady=2)
    ttk.Label(gt_row, text="Optional GT mask:").pack(side="left")
    gt_var = tk.StringVar(value="")
    ttk.Entry(gt_row, textvariable=gt_var, width=26).pack(side="left", padx=6, fill="x", expand=True)

    def _browse_gt() -> None:
        chosen = ask_image_file(window, title="Select ground-truth mask (optional)")
        if chosen:
            gt_var.set(chosen)

    ttk.Button(gt_row, text="Browse...", command=_browse_gt).pack(side="left")

    ttk.Radiobutton(
        mode_frame,
        text="Evaluate on TrayDataset test set (XTest / yTest)",
        variable=mode_var,
        value="test_set",
    ).pack(anchor="w", pady=(6, 0))

    ds_row = ttk.Frame(mode_frame, padding=(16, 4, 0, 0))
    ds_row.pack(fill="x")
    ttk.Label(ds_row, text="TrayDataset:").pack(side="left")
    ds_var = tk.StringVar(value=default_dataset_dir)
    ttk.Entry(ds_row, textvariable=ds_var, width=28).pack(side="left", padx=6, fill="x", expand=True)

    def _browse_ds() -> None:
        folder = ask_image_folder(window, title="Select TrayDataset folder")
        if folder:
            ds_var.set(folder)
            mode_var.set("test_set")

    ttk.Button(ds_row, text="Browse...", command=_browse_ds).pack(side="left")

    ttk.Label(frame, text="Mask threshold:").grid(row=3, column=0, sticky="w", pady=4)
    thr_var = tk.DoubleVar(value=default_threshold)
    ttk.Entry(frame, textvariable=thr_var, width=10).grid(row=3, column=1, sticky="w", padx=6, pady=4)

    btn_frame = ttk.Frame(frame)
    btn_frame.grid(row=4, column=0, columnspan=3, sticky="e", pady=(12, 0))

    def on_ok() -> None:
        ckpt = ckpt_var.get().strip()
        if not ckpt:
            messagebox.showerror("Checkpoint Required", "Please select a U-Net checkpoint file.", parent=window)
            return
        try:
            thr = float(thr_var.get())
        except (TypeError, ValueError):
            messagebox.showerror("Invalid Threshold", "Threshold must be a number between 0 and 1.", parent=window)
            return
        if not (0.0 < thr < 1.0):
            messagebox.showerror("Invalid Threshold", "Threshold must be strictly between 0 and 1.", parent=window)
            return

        mode = mode_var.get()
        if mode == "image":
            use_cur = img_src_var.get() == "current" and current_image_loaded
            custom_img = img_var.get().strip() or None
            if not use_cur and not custom_img:
                messagebox.showerror("Image Required", "Select an image file or load an image first.", parent=window)
                return
            dialog_result["value"] = {
                "checkpoint_path": ckpt,
                "mode": "image",
                "use_current_image": use_cur,
                "custom_image_path": custom_img if not use_cur else None,
                "gt_mask_path": gt_var.get().strip() or None,
                "dataset_dir": ds_var.get().strip(),
                "threshold": thr,
            }
        else:
            ds = ds_var.get().strip()
            if not ds or not os.path.isdir(ds):
                messagebox.showerror("Dataset Required", "Select an existing TrayDataset directory.", parent=window)
                return
            dialog_result["value"] = {
                "checkpoint_path": ckpt,
                "mode": "test_set",
                "use_current_image": False,
                "custom_image_path": None,
                "gt_mask_path": None,
                "dataset_dir": ds,
                "threshold": thr,
            }
        window.destroy()

    def on_cancel() -> None:
        window.destroy()

    ttk.Button(btn_frame, text="Cancel", command=on_cancel).pack(side="right", padx=(6, 0))
    ttk.Button(btn_frame, text="Run Segmentation", command=on_ok).pack(side="right")

    window.protocol("WM_DELETE_WINDOW", on_cancel)
    window.grab_set()
    window.wait_window()
    return dialog_result["value"]
