"""Core implementation for Set 10: CLIP Zero-Shot Classification.

Provides:
  - Device selection and in-memory model/processor caching for Hugging Face CLIP.
  - Validation of class labels, custom prompts, prompt templates, and image inputs.
  - Real zero-shot image-text similarity and softmax probability scoring.
  - Matplotlib visualization pairing the input image with ranked zero-shot scores.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Callable, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, UnidentifiedImageError

TORCH_AVAILABLE = False
TRANSFORMERS_AVAILABLE = False

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore
    F = None  # type: ignore

try:
    from transformers import CLIPModel, CLIPProcessor
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    CLIPModel = None  # type: ignore
    CLIPProcessor = None  # type: ignore


DEFAULT_CLIP_MODEL = "openai/clip-vit-base-patch32"
DEFAULT_CLIP_TEMPLATE = "a photo of a {}."
CLIP_CANDIDATE_MODELS: tuple[str, ...] = (
    "openai/clip-vit-base-patch32",
    "openai/clip-vit-large-patch14",
)
SUPPORTED_IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
)

# In-memory cache keyed by (model_name, device_str) -> (model, processor)
_CLIP_MODEL_CACHE: dict[tuple[str, str], tuple[Any, Any]] = {}


class CLIPModelLoadError(RuntimeError):
    """Raised when a CLIP model or processor cannot be downloaded or loaded."""


def clip_dependencies_available() -> bool:
    """Return True when both PyTorch and transformers CLIP classes are importable."""
    return TORCH_AVAILABLE and TRANSFORMERS_AVAILABLE


def get_clip_device(preferred: str | None = None) -> str:
    """Select the compute device following torch availability (cuda -> mps -> cpu)."""
    if preferred:
        return preferred
    if not TORCH_AVAILABLE:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return "mps"
    return "cpu"


def get_cached_clip_model(model_name: str, device: str | None = None) -> tuple[Any, Any] | None:
    """Return cached (model, processor) for (model_name, device) if already loaded."""
    dev = get_clip_device(device)
    return _CLIP_MODEL_CACHE.get((model_name, dev))


def clear_clip_cache() -> None:
    """Clear the in-memory CLIP model cache."""
    _CLIP_MODEL_CACHE.clear()


def load_clip_model(
    model_name: str = DEFAULT_CLIP_MODEL,
    device: str | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> tuple[Any, Any, str]:
    """Load and cache a CLIP model in evaluation mode along with its processor."""
    if not TORCH_AVAILABLE:
        raise CLIPModelLoadError("PyTorch is required for CLIP classification.")
    if not TRANSFORMERS_AVAILABLE:
        raise CLIPModelLoadError("The 'transformers' package is required for CLIP classification.")

    clean_name = (model_name or "").strip()
    if not clean_name:
        raise ValueError("CLIP model name cannot be empty.")

    dev = get_clip_device(device)
    cached = _CLIP_MODEL_CACHE.get((clean_name, dev))
    if cached is not None:
        if progress_callback:
            progress_callback(f"Using cached CLIP model '{clean_name}' on {dev}.")
        return cached[0], cached[1], dev

    if progress_callback:
        progress_callback(
            f"Loading CLIP model '{clean_name}' on {dev} (downloading if not cached)..."
        )

    try:
        processor = CLIPProcessor.from_pretrained(clean_name)
        model = CLIPModel.from_pretrained(clean_name)
        model = model.to(dev).eval()
    except Exception as exc:
        raise CLIPModelLoadError(
            f"Could not load CLIP model '{clean_name}': {exc}\n\n"
            "If you are offline, ensure the model is already present in your local Hugging Face cache "
            "or connect to the internet to download it."
        ) from exc

    _CLIP_MODEL_CACHE[(clean_name, dev)] = (model, processor)
    return model, processor, dev


def load_rgb_image(image_input: Image.Image | str | Path) -> tuple[Image.Image, str]:
    """Validate and convert an image path or PIL Image into an RGB PIL Image."""
    if isinstance(image_input, Image.Image):
        return image_input.convert("RGB"), "current_image"

    path = Path(image_input)
    if not path.is_file():
        raise ValueError(f"Image file not found: {path}")
    if path.suffix and path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
        raise ValueError(
            f"Unsupported image file extension '{path.suffix}'. "
            f"Supported formats: {', '.join(sorted(SUPPORTED_IMAGE_EXTENSIONS))}."
        )
    try:
        with Image.open(path) as img:
            rgb = img.convert("RGB")
        return rgb, path.name
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError(f"Could not open image file '{path.name}': {exc}") from exc


def validate_class_entries(
    entries: Sequence[str | tuple[str, str | None] | dict[str, Any]],
    template: str = DEFAULT_CLIP_TEMPLATE,
) -> list[tuple[str, str]]:
    """Validate class labels and optional custom prompts, returning (label, prompt) pairs."""
    if not entries:
        raise ValueError("Please provide at least one class label.")

    if template is None or not str(template).strip():
        raise ValueError("Default prompt template cannot be blank.")
    clean_template = str(template).strip()

    validated: list[tuple[str, str]] = []
    seen_labels: set[str] = set()

    for idx, item in enumerate(entries, start=1):
        if isinstance(item, str):
            raw_label, raw_prompt = item, None
        elif isinstance(item, dict):
            raw_label = item.get("label", "")
            raw_prompt = item.get("prompt")
        elif isinstance(item, (tuple, list)) and len(item) == 2:
            raw_label, raw_prompt = item[0], item[1]
        else:
            raise ValueError(f"Invalid class entry at position {idx}.")

        label = str(raw_label).strip() if raw_label is not None else ""
        if not label:
            raise ValueError(f"Class label #{idx} cannot be empty.")

        norm_key = label.casefold()
        if norm_key in seen_labels:
            raise ValueError(f"Duplicate class label: '{label}'.")
        seen_labels.add(norm_key)

        if raw_prompt is None or raw_prompt == "":
            if "{}" in clean_template:
                prompt = clean_template.format(label)
            else:
                prompt = f"{clean_template} {label}"
        else:
            prompt_str = str(raw_prompt).strip()
            if not prompt_str:
                raise ValueError(
                    f"Custom prompt for '{label}' is blank. "
                    "Leave it empty to use the default template, or enter prompt text."
                )
            prompt = prompt_str

        validated.append((label, prompt))

    return validated


def _as_feature_tensor(output: Any) -> Any:
    """Extract a tensor from either a raw tensor or a Hugging Face model output."""
    if TORCH_AVAILABLE and torch.is_tensor(output):
        return output
    if hasattr(output, "pooler_output") and output.pooler_output is not None:
        return output.pooler_output
    if hasattr(output, "image_embeds") and output.image_embeds is not None:
        return output.image_embeds
    if hasattr(output, "text_embeds") and output.text_embeds is not None:
        return output.text_embeds
    raise TypeError(f"Unexpected CLIP feature output type: {type(output).__name__}")


def classify_image_zero_shot(
    image_input: Image.Image | str | Path,
    class_entries: Sequence[str | tuple[str, str | None] | dict[str, Any]],
    model_name: str = DEFAULT_CLIP_MODEL,
    template: str = DEFAULT_CLIP_TEMPLATE,
    device: str | None = None,
    progress_callback: Callable[[str], None] | None = None,
    model_bundle: tuple[Any, Any, str] | None = None,
) -> dict[str, Any]:
    """Run CLIP zero-shot classification and return ranked similarity and probability scores."""
    rgb_image, source_name = load_rgb_image(image_input)
    pairs = validate_class_entries(class_entries, template=template)

    if model_bundle is not None:
        model, processor, dev = model_bundle
        model = model.to(dev).eval()
    else:
        model, processor, dev = load_clip_model(
            model_name=model_name,
            device=device,
            progress_callback=progress_callback,
        )

    if progress_callback:
        progress_callback(f"Encoding image and {len(pairs)} class prompt(s)...")

    labels = [label for label, _ in pairs]
    prompts = [prompt for _, prompt in pairs]

    with torch.no_grad():
        img_inputs = processor(images=[rgb_image], return_tensors="pt")
        img_inputs = {k: v.to(dev) for k, v in img_inputs.items()}
        img_emb = _as_feature_tensor(model.get_image_features(**img_inputs))
        img_emb = F.normalize(img_emb, dim=-1)

        txt_inputs = processor(text=prompts, return_tensors="pt", padding=True)
        txt_inputs = {k: v.to(dev) for k, v in txt_inputs.items()}
        txt_emb = _as_feature_tensor(model.get_text_features(**txt_inputs))
        txt_emb = F.normalize(txt_emb, dim=-1)

        cosine_sims = (img_emb @ txt_emb.T)[0]
        logit_scale = model.logit_scale.exp() if hasattr(model, "logit_scale") else 100.0
        logits = logit_scale * cosine_sims
        probs = logits.softmax(dim=-1)

    cos_np = cosine_sims.detach().cpu().numpy().astype(float)
    logit_np = logits.detach().cpu().numpy().astype(float)
    prob_np = probs.detach().cpu().numpy().astype(float)

    order = sorted(
        range(len(labels)),
        key=lambda i: (prob_np[i], cos_np[i]),
        reverse=True,
    )

    ranked_results = [
        {
            "rank": rank,
            "label": labels[i],
            "prompt": prompts[i],
            "probability": float(prob_np[i]),
            "cosine_similarity": float(cos_np[i]),
            "logit": float(logit_np[i]),
        }
        for rank, i in enumerate(order, start=1)
    ]

    return {
        "model_name": model_name,
        "device": dev,
        "image_source": source_name,
        "template": template,
        "results": ranked_results,
        "top_label": ranked_results[0]["label"],
        "top_probability": ranked_results[0]["probability"],
        "top_cosine_similarity": ranked_results[0]["cosine_similarity"],
        "note": (
            "Zero-shot probabilities are softmax-normalized over the provided candidate classes; "
            "the ranking depends on the candidate set."
        ),
    }


def render_clip_zero_shot_plot(
    image: Image.Image,
    classification_result: dict[str, Any],
    title: str | None = None,
) -> Image.Image:
    """Render the input image alongside a horizontal bar chart of ranked CLIP scores."""
    ranked = classification_result["results"]
    # Plot highest score at the top of the horizontal bar chart
    ordered_for_plot = list(reversed(ranked))
    labels = [item["label"] for item in ordered_for_plot]
    probs = [item["probability"] for item in ordered_for_plot]
    cosines = [item["cosine_similarity"] for item in ordered_for_plot]

    fig_height = max(3.6, 0.48 * len(labels) + 1.8)
    fig, (ax_img, ax_bar) = plt.subplots(
        1,
        2,
        figsize=(11.0, fig_height),
        gridspec_kw={"width_ratios": [1.0, 1.45]},
    )

    ax_img.imshow(image.convert("RGB"))
    ax_img.axis("off")
    ax_img.set_title(f"Input: {classification_result.get('image_source', 'image')}", fontsize=10)

    bars = ax_bar.barh(range(len(labels)), probs, color="#3b6ea8", height=0.65)
    ax_bar.set_yticks(range(len(labels)))
    ax_bar.set_yticklabels(labels, fontsize=9)
    ax_bar.set_xlim(0.0, 1.18)
    ax_bar.set_xlabel("CLIP Zero-Shot Probability (Softmax)", fontsize=9)
    ax_bar.set_title(
        title or f"CLIP Zero-Shot Scores ({classification_result.get('model_name', DEFAULT_CLIP_MODEL)})",
        fontsize=10,
        fontweight="bold",
    )
    ax_bar.grid(axis="x", linestyle="--", alpha=0.35)

    for bar, p, cos in zip(bars, probs, cosines):
        ax_bar.text(
            p + 0.015,
            bar.get_y() + bar.get_height() / 2.0,
            f"{p:.2f} (cos={cos:.3f})",
            va="center",
            fontsize=8.5,
        )

    fig.text(
        0.5,
        0.01,
        classification_result.get(
            "note",
            "Probabilities sum to 1 across the candidate classes provided; ranking depends on the candidate set.",
        ),
        ha="center",
        fontsize=8,
        color="#444444",
    )
    plt.tight_layout(rect=(0, 0.05, 1, 1))

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    with Image.open(buf) as rendered:
        return rendered.convert("RGB")
