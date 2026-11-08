"""Small Tk dialogs used by parameterized tools."""

from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk

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


