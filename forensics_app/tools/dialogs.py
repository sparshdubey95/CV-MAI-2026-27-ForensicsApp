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
    ("NumPy compressed archive", "*.npz"),
    ("NumPy binary file", "*.npy"),
    ("All files", "*.*"),
]


def ask_npz_file(parent: tk.Misc | None, title: str = "Select Eigenface model (.npz)") -> str | None:
    """Ask user to select an existing model .npz file."""
    filename = filedialog.askopenfilename(
        parent=parent,
        title=title,
        filetypes=MODEL_FILETYPES,
    )
    return filename or None


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
    """Dialog displaying detected categories and allowing configuration of K, method, and test split."""
    c_count = len(class_names)
    recommended_k = 10 * c_count  # Coursework requirement: K = 10 * number of classes (K=30 for 3 classes)

    def _parse(k_str: str, method_val: str, split_str: str) -> tuple[int, str, float]:
        k = int(k_str.strip())
        if k < 2:
            raise ValueError("K (visual words) must be at least 2.")
        split = float(split_str.strip())
        if not (0.05 <= split <= 0.5):
            raise ValueError("Test split ratio must be between 0.05 and 0.5 (e.g. 0.25).")
        return k, method_val, split

    if parent is None:
        return recommended_k, "SIFT", default_split

    result: dict[str, tuple[int, str, float] | None] = {"value": None}
    window = tk.Toplevel(parent)
    window.title("BoVW Codebook Configuration")
    window.transient(parent)
    window.resizable(False, False)

    frame = ttk.Frame(window, padding=16)
    frame.pack(fill="both", expand=True)

    ttk.Label(
        frame,
        text=f"Detected {c_count} Classes: {', '.join(class_names)}",
        font=("TkDefaultFont", 10, "bold"),
        foreground="#0055aa",
    ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 10))

    k_var = tk.StringVar(value=str(recommended_k))
    method_var = tk.StringVar(value="SIFT")
    split_var = tk.StringVar(value=str(default_split))

    ttk.Label(frame, text=f"Visual Words (K = 10 × {c_count}):").grid(
        row=1, column=0, sticky="w", pady=(0, 6)
    )
    ttk.Entry(frame, textvariable=k_var, width=14).grid(
        row=1, column=1, sticky="e", padx=(12, 0), pady=(0, 6)
    )

    ttk.Label(frame, text="Descriptor Method:").grid(
        row=2, column=0, sticky="w", pady=(0, 6)
    )
    combo_method = ttk.Combobox(
        frame,
        textvariable=method_var,
        values=["SIFT", "ORB"],
        state="readonly",
        width=12,
    )
    combo_method.grid(row=2, column=1, sticky="e", padx=(12, 0), pady=(0, 6))

    ttk.Label(frame, text="Test Set Ratio (e.g. 0.25 = 25%):").grid(
        row=3, column=0, sticky="w", pady=(0, 6)
    )
    ttk.Entry(frame, textvariable=split_var, width=14).grid(
        row=3, column=1, sticky="e", padx=(12, 0), pady=(0, 6)
    )

    error_label = ttk.Label(frame, text="", foreground="red", wraplength=320)
    error_label.grid(row=4, column=0, columnspan=2, sticky="w", pady=(8, 0))

    buttons = ttk.Frame(frame)
    buttons.grid(row=5, column=0, columnspan=2, sticky="e", pady=(12, 0))

    def confirm() -> None:
        try:
            parsed = _parse(k_var.get(), method_var.get(), split_var.get())
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


