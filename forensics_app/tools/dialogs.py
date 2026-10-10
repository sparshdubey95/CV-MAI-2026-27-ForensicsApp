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
    parent: tk.Misc,
    title: str,
    prompt: str,
    options: list[tuple[str, str]],
    initial: str | None = None,
) -> str | None:
    """Return the selected option value, or ``None`` if the user cancels.

    ``options`` is a list of ``(value, label)`` pairs.
    """
    chosen: dict[str, str | None] = {"value": None}
    window = tk.Toplevel(parent)
    window.title(title)
    window.transient(parent)
    window.resizable(False, False)

    frame = ttk.Frame(window, padding=12)
    frame.pack(fill="both", expand=True)
    ttk.Label(frame, text=prompt, wraplength=320).pack(anchor="w", pady=(0, 8))

    default = initial if initial is not None else options[0][0]
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
    parent: tk.Misc,
    title: str,
    prompt: str,
    initial: float = 1.0,
    min_val: float | None = None,
    max_val: float | None = None,
) -> float | None:
    """Ask for a floating point number; return None if cancelled."""
    kwargs: dict[str, object] = {"parent": parent, "initialvalue": initial}
    if min_val is not None:
        kwargs["minvalue"] = min_val
    if max_val is not None:
        kwargs["maxvalue"] = max_val
    return simpledialog.askfloat(title, prompt, **kwargs)


def ask_int(
    parent: tk.Misc,
    title: str,
    prompt: str,
    initial: int = 3,
    min_val: int | None = None,
    max_val: int | None = None,
) -> int | None:
    """Ask for an integer number; return None if cancelled."""
    kwargs: dict[str, object] = {"parent": parent, "initialvalue": initial}
    if min_val is not None:
        kwargs["minvalue"] = min_val
    if max_val is not None:
        kwargs["maxvalue"] = max_val
    return simpledialog.askinteger(title, prompt, **kwargs)
