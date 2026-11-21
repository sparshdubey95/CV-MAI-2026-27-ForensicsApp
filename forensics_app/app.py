"""Application bootstrap.

Keeping startup here makes it possible to import and test the rest of the project
without creating a Tk window as a side effect.
"""

from __future__ import annotations

import tkinter as tk
import os
import warnings

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore", category=UserWarning)

from forensics_app.tools import build_tool_registry
from forensics_app.ui.main_window import MainWindow


def main() -> None:
    root = tk.Tk()
    MainWindow(root, build_tool_registry())
    root.mainloop()
