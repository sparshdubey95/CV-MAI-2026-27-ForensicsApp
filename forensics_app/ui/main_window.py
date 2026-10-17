"""The application's main window and event coordination."""

from __future__ import annotations

from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import UnidentifiedImageError

from forensics_app.core import ImageDocument
from forensics_app.tools.base import ForensicsTool
from forensics_app.tools.registry import ToolRegistry
from .image_view import ImageView


OPEN_TYPES = [
    ("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"),
    ("All files", "*.*"),
]
SAVE_TYPES = [("PNG image", "*.png"), ("JPEG image", "*.jpg"), ("TIFF image", "*.tiff")]


class MainWindow:
    def __init__(self, root: tk.Tk, registry: ToolRegistry) -> None:
        self.root = root
        self.registry = registry
        self.document = ImageDocument()
        self.status = tk.StringVar(value="Ready. Open an image to begin.")

        self._configure_window()
        self._build_menu()
        self._build_layout()
        self._bind_shortcuts()
        self._refresh()

    def _configure_window(self) -> None:
        self.root.title("ForensicsApp")
        self.root.geometry("1180x720")
        self.root.minsize(820, 520)
        style = ttk.Style(self.root)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure("Sidebar.TFrame", background="#eef1f5")
        style.configure("Category.TLabel", background="#eef1f5", font=("TkDefaultFont", 10, "bold"))
        style.configure("Tool.TButton", anchor="w", padding=(10, 7))
        style.configure("Title.TLabel", font=("TkDefaultFont", 16, "bold"))

    def _build_menu(self) -> None:
        menu = tk.Menu(self.root)
        file_menu = tk.Menu(menu, tearoff=False)
        file_menu.add_command(label="Open image…", command=self.open_image, accelerator="Ctrl+O")
        file_menu.add_command(label="Save result as…", command=self.save_image, accelerator="Ctrl+Shift+S")
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.destroy)
        menu.add_cascade(label="File", menu=file_menu)

        edit_menu = tk.Menu(menu, tearoff=False)
        edit_menu.add_command(label="Undo", command=self.undo, accelerator="Ctrl+Z")
        edit_menu.add_command(label="Redo", command=self.redo, accelerator="Ctrl+Y")
        edit_menu.add_command(label="Reset to original", command=self.reset)
        menu.add_cascade(label="Edit", menu=edit_menu)

        help_menu = tk.Menu(menu, tearoff=False)
        help_menu.add_command(label="About", command=self.show_about)
        menu.add_cascade(label="Help", menu=help_menu)
        self.root.configure(menu=menu)

    def _build_layout(self) -> None:
        container = ttk.Frame(self.root)
        container.pack(fill="both", expand=True)

        toolbar = ttk.Frame(container, padding=(10, 8))
        toolbar.pack(fill="x")
        ttk.Button(toolbar, text="Open image", command=self.open_image).pack(side="left")
        ttk.Button(toolbar, text="Save result", command=self.save_image).pack(side="left", padx=(6, 0))
        ttk.Separator(toolbar, orient="vertical").pack(side="left", fill="y", padx=10)
        self.undo_button = ttk.Button(toolbar, text="Undo", command=self.undo)
        self.undo_button.pack(side="left")
        self.redo_button = ttk.Button(toolbar, text="Redo", command=self.redo)
        self.redo_button.pack(side="left", padx=(6, 0))
        ttk.Button(toolbar, text="Reset", command=self.reset).pack(side="left", padx=(6, 0))

        body = ttk.Panedwindow(container, orient="horizontal")
        body.pack(fill="both", expand=True)

        # Scrollable Sidebar for tools (allowing smooth access to Set 3 & Set 4)
        sidebar_container = ttk.Frame(body, style="Sidebar.TFrame", width=255)
        sidebar_container.pack_propagate(False)
        body.add(sidebar_container, weight=0)

        canvas = tk.Canvas(
            sidebar_container,
            bg="#eef1f5",
            highlightthickness=0,
            bd=0,
        )
        sidebar_scrollbar = ttk.Scrollbar(
            sidebar_container,
            orient="vertical",
            command=canvas.yview,
        )
        scrollable_frame = ttk.Frame(canvas, style="Sidebar.TFrame", padding=(12, 12, 6, 12))
        frame_window_id = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")

        def _on_frame_configure(_event: tk.Event) -> None:
            canvas.configure(scrollregion=canvas.bbox("all"))

        def _on_canvas_configure(event: tk.Event) -> None:
            canvas.itemconfig(frame_window_id, width=event.width)

        scrollable_frame.bind("<Configure>", _on_frame_configure)
        canvas.bind("<Configure>", _on_canvas_configure)
        canvas.configure(yscrollcommand=sidebar_scrollbar.set)

        # Mouse-wheel scrolling support
        def _on_mousewheel(event: tk.Event) -> None:
            if event.delta:
                canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
            elif event.num == 4:
                canvas.yview_scroll(-1, "units")
            elif event.num == 5:
                canvas.yview_scroll(1, "units")

        def _bind_mousewheel(_event: tk.Event | None = None) -> None:
            canvas.bind_all("<MouseWheel>", _on_mousewheel)
            canvas.bind_all("<Button-4>", _on_mousewheel)
            canvas.bind_all("<Button-5>", _on_mousewheel)

        def _unbind_mousewheel(_event: tk.Event | None = None) -> None:
            try:
                x, y = sidebar_container.winfo_pointerxy()
                bx = sidebar_container.winfo_rootx()
                by = sidebar_container.winfo_rooty()
                bw = sidebar_container.winfo_width()
                bh = sidebar_container.winfo_height()
                if bx <= x <= bx + bw and by <= y <= by + bh:
                    return
            except Exception:
                pass
            canvas.unbind_all("<MouseWheel>")
            canvas.unbind_all("<Button-4>")
            canvas.unbind_all("<Button-5>")

        sidebar_container.bind("<Enter>", _bind_mousewheel)
        sidebar_container.bind("<Leave>", _unbind_mousewheel)
        canvas.bind("<Enter>", _bind_mousewheel)
        canvas.bind("<Leave>", _unbind_mousewheel)
        scrollable_frame.bind("<Enter>", _bind_mousewheel)
        scrollable_frame.bind("<Leave>", _unbind_mousewheel)

        sidebar_scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        ttk.Label(scrollable_frame, text="Forensics tools", style="Title.TLabel", background="#eef1f5").pack(
            anchor="w", pady=(0, 12)
        )
        for category, tools in self.registry.categories():
            cat_label = ttk.Label(scrollable_frame, text=category, style="Category.TLabel")
            cat_label.pack(anchor="w", pady=(9, 4))
            cat_label.bind("<Enter>", _bind_mousewheel)
            for tool in tools:
                button = ttk.Button(
                    scrollable_frame,
                    text=tool.title,
                    style="Tool.TButton",
                    command=lambda selected=tool: self.run_tool(selected),
                )
                button.pack(fill="x", pady=2)
                button.bind(
                    "<Enter>",
                    lambda _event, selected=tool: (self.status.set(selected.description), _bind_mousewheel()),
                )
                button.bind(
                    "<Leave>",
                    lambda _event: (self.status.set("Ready."), _unbind_mousewheel()),
                )

        self.image_view = ImageView(body)
        body.add(self.image_view, weight=1)

        inspector = ttk.Frame(body, padding=12, width=250)
        inspector.pack_propagate(False)
        body.add(inspector, weight=0)
        ttk.Label(inspector, text="Results", style="Title.TLabel").pack(anchor="w", pady=(0, 10))

        results_frame = ttk.Frame(inspector)
        results_frame.pack(fill="both", expand=True)
        results_scroll = ttk.Scrollbar(results_frame, orient="vertical")
        self.results = ttk.Treeview(
            results_frame,
            columns=("value",),
            show="tree headings",
            height=15,
            yscrollcommand=results_scroll.set,
        )
        results_scroll.configure(command=self.results.yview)
        results_scroll.pack(side="right", fill="y")
        self.results.heading("#0", text="Property")
        self.results.heading("value", text="Value")
        self.results.column("#0", width=95, stretch=True)
        self.results.column("value", width=120, stretch=True)
        self.results.pack(side="left", fill="both", expand=True)

        ttk.Label(container, textvariable=self.status, anchor="w", padding=(10, 6), relief="sunken").pack(fill="x")

    def _bind_shortcuts(self) -> None:
        self.root.bind_all("<Control-o>", lambda _event: self.open_image())
        self.root.bind_all("<Control-Shift-S>", lambda _event: self.save_image())
        self.root.bind_all("<Control-z>", lambda _event: self.undo())
        self.root.bind_all("<Control-y>", lambda _event: self.redo())

    def open_image(self) -> None:
        filename = filedialog.askopenfilename(title="Open evidence image", filetypes=OPEN_TYPES)
        if not filename:
            return
        try:
            self.document.load(filename)
        except (OSError, UnidentifiedImageError) as error:
            messagebox.showerror("Could not open image", str(error), parent=self.root)
            return
        self.status.set(f"Opened {Path(filename).name}")
        self._show_default_details()
        self._refresh()

    def save_image(self) -> None:
        if not self._require_image():
            return
        source = self.document.path
        initial = f"{source.stem}_result.png" if source else "result.png"
        filename = filedialog.asksaveasfilename(
            title="Save processed image",
            defaultextension=".png",
            initialfile=initial,
            filetypes=SAVE_TYPES,
        )
        if not filename:
            return
        try:
            self.document.save(filename)
        except OSError as error:
            messagebox.showerror("Could not save image", str(error), parent=self.root)
            return
        self.status.set(f"Saved result as {Path(filename).name}")

    def run_tool(self, tool: ForensicsTool) -> None:
        if tool.requires_image and not self._require_image():
            return
        try:
            result = tool.run(self.root, self.document)
        except Exception as error:  # keep one student feature from crashing the shell
            messagebox.showerror(f"{tool.title} failed", str(error), parent=self.root)
            self.status.set(f"Error in {tool.title}.")
            return
        if result is None:
            self.status.set(f"Cancelled {tool.title}.")
            return
        if result.image is not None:
            self.document.apply(result.image)
        self._show_details(result.details)
        self.status.set(result.message)
        self._refresh()

    def undo(self) -> None:
        if self.document.undo():
            self.status.set("Undid the last image operation.")
            self._show_default_details()
            self._refresh()

    def redo(self) -> None:
        if self.document.redo():
            self.status.set("Redid the image operation.")
            self._show_default_details()
            self._refresh()

    def reset(self) -> None:
        if self.document.reset():
            self.status.set("Restored the original image.")
            self._show_default_details()
            self._refresh()

    def _require_image(self) -> bool:
        if self.document.is_loaded:
            return True
        messagebox.showinfo("No image loaded", "Open an image first.", parent=self.root)
        return False

    def _refresh(self) -> None:
        self.image_view.show(self.document.current)
        self.undo_button.configure(state="normal" if self.document.can_undo else "disabled")
        self.redo_button.configure(state="normal" if self.document.can_redo else "disabled")
        title = self.document.path.name if self.document.path else "No image"
        self.root.title(f"ForensicsApp — {title}")

    def _show_default_details(self) -> None:
        image = self.document.current
        if image is None:
            self._show_details({})
            return
        self._show_details(
            {
                "File": self.document.path.name if self.document.path else "—",
                "Size": f"{image.width} × {image.height}",
                "Mode": image.mode,
            }
        )

    def _show_details(self, details: dict[str, object]) -> None:
        for item in self.results.get_children():
            self.results.delete(item)
        for name, value in details.items():
            self.results.insert("", "end", text=str(name), values=(str(value),))

    def show_about(self) -> None:
        messagebox.showinfo(
            "About ForensicsApp",
            "A modular image-forensics application for the Computer Vision course.\n\n"
            "Add each weekly feature as a tool in forensics_app/tools/.",
            parent=self.root,
        )
