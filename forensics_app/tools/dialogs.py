"""Small Tk dialogs used by parameterized tools."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk


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
