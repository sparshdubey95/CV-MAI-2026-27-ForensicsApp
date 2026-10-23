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
