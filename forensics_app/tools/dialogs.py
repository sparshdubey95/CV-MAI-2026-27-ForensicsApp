"""Small Tk dialogs used by parameterized tools."""

from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk

IMAGE_FILETYPES = [
    ("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"),
    ("All files", "*.*"),
]


def ask_image_file(parent: tk.Misc, title: str = "Select image") -> str | None:
    """Ask the user to select an image file; return the path or None if cancelled."""
    filename = filedialog.askopenfilename(
        parent=parent,
        title=title,
        filetypes=IMAGE_FILETYPES,
    )
    return filename or None


def show_error(parent: tk.Misc, title: str, message: str) -> None:
    """Display an error message dialog to the user."""
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
