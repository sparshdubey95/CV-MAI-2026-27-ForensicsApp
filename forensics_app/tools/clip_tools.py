"""Tkinter tool for Set 10: CLIP Zero-Shot Classification."""

from __future__ import annotations

from pathlib import Path
import threading
import tkinter as tk
from tkinter import ttk
from typing import Any

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
from .clip_core import (
    DEFAULT_CLIP_MODEL,
    DEFAULT_CLIP_TEMPLATE,
    classify_image_zero_shot,
    clip_dependencies_available,
    load_rgb_image,
    render_clip_zero_shot_plot,
)
from .dialogs import ask_clip_zero_shot_parameters, show_error


def _run_clip_with_progress(
    parent: tk.Misc | None,
    title: str,
    subtitle: str,
    worker: Any,
) -> tuple[Any, BaseException | None]:
    """Run a CLIP classification worker in a background thread with a progress dialog."""
    state: dict[str, Any] = {"result": None, "error": None}
    prog_win: tk.Toplevel | None = None
    status_var: tk.StringVar | None = None

    if parent is not None:
        prog_win = tk.Toplevel(parent)
        prog_win.title(title)
        prog_win.transient(parent)
        prog_win.resizable(False, False)
        prog_win.geometry("480x140")

        frame = ttk.Frame(prog_win, padding=16)
        frame.pack(fill="both", expand=True)
        ttk.Label(
            frame,
            text=subtitle,
            font=("TkDefaultFont", 10, "bold"),
            foreground="#0055aa",
        ).pack(anchor="w", pady=(0, 8))
        pbar = ttk.Progressbar(frame, mode="indeterminate")
        pbar.pack(fill="x", pady=(0, 8))
        pbar.start(15)
        status_var = tk.StringVar(value="Starting...")
        ttk.Label(
            frame,
            textvariable=status_var,
            font=("TkDefaultFont", 9, "italic"),
            wraplength=440,
        ).pack(anchor="w")

    def _on_progress(message: str) -> None:
        if status_var is not None and prog_win is not None:
            try:
                prog_win.after(0, lambda m=message: status_var.set(m))
            except Exception:
                pass

    def _thread_main() -> None:
        try:
            state["result"] = worker(_on_progress)
        except Exception as exc:
            state["error"] = exc
        finally:
            if prog_win is not None:
                try:
                    prog_win.after(60, prog_win.destroy)
                except Exception:
                    pass

    thread = threading.Thread(target=_thread_main, daemon=True)
    thread.start()
    if prog_win is not None:
        prog_win.grab_set()
        prog_win.wait_window()
    else:
        thread.join()

    return state["result"], state["error"]


class CLIPZeroShotTool(ForensicsTool):
    """Set 10: Zero-shot image classification using OpenAI CLIP."""

    tool_id = "clip_zero_shot"
    title = "CLIP zero-shot classifier"
    category = "Set 10: CLIP Zero-Shot Classification"
    description = "Classify an image against arbitrary class labels and optional custom prompts using CLIP."
    requires_image = False

    def run(self, parent: tk.Misc | None, document: ImageDocument) -> ToolResult | None:
        if not clip_dependencies_available():
            if parent is not None:
                show_error(
                    parent,
                    "CLIP Dependencies Required",
                    "PyTorch and 'transformers' are required for CLIP zero-shot classification.",
                )
            return None

        params = ask_clip_zero_shot_parameters(
            parent,
            current_image_loaded=document.is_loaded,
            default_model=DEFAULT_CLIP_MODEL,
            default_template=DEFAULT_CLIP_TEMPLATE,
        )
        if params is None:
            return None

        image_source: Any
        if params.get("custom_image_path"):
            image_source = params["custom_image_path"]
        elif document.is_loaded:
            image_source = document.current
        else:
            if parent is not None:
                show_error(parent, "No Image Selected", "Please load an image or select an image file.")
            return None

        try:
            rgb_image, source_name = load_rgb_image(image_source)
        except Exception as exc:
            if parent is not None:
                show_error(parent, "Invalid Image", str(exc))
            return None

        if params.get("use_current_image") and document.path is not None:
            source_name = document.path.name

        model_name = params.get("model_name") or DEFAULT_CLIP_MODEL
        template = params.get("template") or DEFAULT_CLIP_TEMPLATE
        class_entries = params.get("classes", [])

        def _worker(progress_cb: Any) -> dict[str, Any]:
            result = classify_image_zero_shot(
                image_input=rgb_image,
                class_entries=class_entries,
                model_name=model_name,
                template=template,
                progress_callback=progress_cb,
            )
            result["image_source"] = source_name
            progress_cb("Rendering zero-shot score chart...")
            result["rendered_image"] = render_clip_zero_shot_plot(rgb_image, result)
            return result

        output, error = _run_clip_with_progress(
            parent,
            title="CLIP Zero-Shot Classification",
            subtitle=f"Running {model_name}",
            worker=_worker,
        )
        if error is not None:
            if parent is not None:
                show_error(parent, "CLIP Classification Failed", str(error))
            return None

        ranked = output["results"]
        details: dict[str, Any] = {
            "Model": output["model_name"],
            "Device": output["device"],
            "Image": source_name,
            "Top Class": output["top_label"],
            "Top Probability": f"{output['top_probability'] * 100:.2f}%",
            "Top Cosine Sim": f"{output['top_cosine_similarity']:.4f}",
            "Scoring Note": output["note"],
        }
        for item in ranked:
            details[f"#{item['rank']} {item['label']}"] = (
                f"{item['probability'] * 100:.2f}% | cos={item['cosine_similarity']:.4f} | \"{item['prompt']}\""
            )

        message = (
            f"CLIP top prediction for {source_name}: '{output['top_label']}' "
            f"({output['top_probability'] * 100:.1f}%, cos={output['top_cosine_similarity']:.3f})."
        )
        return ToolResult(
            message=message,
            image=output["rendered_image"],
            details=details,
        )
