"""Core implementation for Set 11: U-Net Binary Food Segmentation.

Implements:
  - TrayDataset discovery (XTrain, yTrain, XTest, yTest) and stem-based pairing.
  - Binary food-vs-not-food mask mapping (classes 0-4 not food, >=5 food) with
    nearest-neighbor mask resizing.
  - PyTorch U-Net architecture, BCE + Soft Dice loss, and Hard Dice evaluation.
  - Training with validation split, progress/cancellation, and checkpoint saving.
  - Checkpoint loading with strict validation, test-set evaluation, and overlays.
"""

from __future__ import annotations

import copy
from datetime import datetime
import io
from pathlib import Path
import random
import threading
import time
from typing import Any, Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, UnidentifiedImageError

TORCH_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore
    DataLoader = None  # type: ignore
    Dataset = object  # type: ignore


FIRST_FOOD_CLASS = 5
DEFAULT_INPUT_HEIGHT = 128
DEFAULT_INPUT_WIDTH = 208
DEFAULT_UNET_BASE = 16
DEFAULT_THRESHOLD = 0.5
SUPPORTED_IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
)

_ACTIVE_UNET_CHECKPOINT_PATH: str | None = None


class UNetTrainingCancelled(RuntimeError):
    """Raised when the user cancels U-Net training."""


class UNetCheckpointError(ValueError):
    """Raised when a U-Net checkpoint is missing, corrupted, or incompatible."""


def unet_dependencies_available() -> bool:
    """Return True when PyTorch is available."""
    return TORCH_AVAILABLE


def get_unet_device(preferred: str | None = None) -> str:
    """Select compute device (cuda -> mps -> cpu)."""
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


def get_default_unet_checkpoint_path() -> Path:
    """Return the default user-visible checkpoint path under forensics_app/data/unet/."""
    data_dir = Path(__file__).resolve().parents[1] / "data" / "unet"
    return data_dir / "unet_tray_food.pt"


def get_active_unet_checkpoint_path() -> str | None:
    """Return the active U-Net checkpoint path if it exists on disk."""
    global _ACTIVE_UNET_CHECKPOINT_PATH
    if _ACTIVE_UNET_CHECKPOINT_PATH and Path(_ACTIVE_UNET_CHECKPOINT_PATH).is_file():
        return _ACTIVE_UNET_CHECKPOINT_PATH
    default_path = get_default_unet_checkpoint_path()
    if default_path.is_file():
        _ACTIVE_UNET_CHECKPOINT_PATH = str(default_path)
        return _ACTIVE_UNET_CHECKPOINT_PATH
    return None


def set_active_unet_checkpoint_path(path: str | Path | None) -> None:
    """Cache the active U-Net checkpoint path in session memory."""
    global _ACTIVE_UNET_CHECKPOINT_PATH
    _ACTIVE_UNET_CHECKPOINT_PATH = str(path) if path else None


def find_default_tray_dataset() -> Path | None:
    """Locate the bundled TrayDataset folder if present in the workspace."""
    root = Path(__file__).resolve().parents[2]
    candidates = [
        root / "lab10" / "TrayDataset" / "TrayDataset",
        root / "lab10" / "TrayDataset",
        root / "TrayDataset",
        Path.cwd() / "lab10" / "TrayDataset" / "TrayDataset",
        Path.cwd() / "lab10" / "TrayDataset",
        Path.cwd() / "TrayDataset",
    ]
    for cand in candidates:
        if cand.is_dir():
            try:
                dirs = find_tray_dataset_dirs(cand, require_test=False)
                return dirs["XTrain"].parent
            except FileNotFoundError:
                continue
    return None


def _find_subfolder(root: Path, name: str) -> Path:
    """Find a subfolder named `name` inside `root`, ignoring __MACOSX metadata."""
    direct = root / name
    if direct.is_dir():
        return direct
    for candidate in sorted(root.rglob(name)):
        if candidate.is_dir() and "__MACOSX" not in candidate.parts:
            return candidate
    raise FileNotFoundError(f"Could not find required folder '{name}' inside '{root}'.")


def find_tray_dataset_dirs(
    root: str | Path,
    require_train: bool = True,
    require_test: bool = True,
) -> dict[str, Path]:
    """Locate XTrain, yTrain, XTest, and yTest folders inside a TrayDataset root."""
    root_path = Path(root)
    if not root_path.is_dir():
        raise FileNotFoundError(f"Dataset folder does not exist: {root_path}")

    found: dict[str, Path] = {}
    if require_train:
        found["XTrain"] = _find_subfolder(root_path, "XTrain")
        found["yTrain"] = _find_subfolder(root_path, "yTrain")
    if require_test:
        found["XTest"] = _find_subfolder(root_path, "XTest")
        found["yTest"] = _find_subfolder(root_path, "yTest")
    else:
        for opt_name in ("XTest", "yTest"):
            try:
                found[opt_name] = _find_subfolder(root_path, opt_name)
            except FileNotFoundError:
                pass
    return found


def list_image_files(folder: Path) -> list[Path]:
    """List supported image files in a directory sorted by filename."""
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    )


def pair_images_and_masks(
    img_dir: Path,
    mask_dir: Path,
) -> tuple[list[tuple[Path, Path]], dict[str, Any]]:
    """Match image and mask files strictly by filename stem and report unmatched files."""
    imgs = list_image_files(img_dir)
    masks = list_image_files(mask_dir)

    mask_by_stem = {p.stem: p for p in masks}
    img_by_stem = {p.stem: p for p in imgs}

    pairs = [(img_path, mask_by_stem[img_path.stem]) for img_path in imgs if img_path.stem in mask_by_stem]
    unmatched_images = [p.name for p in imgs if p.stem not in mask_by_stem]
    unmatched_masks = [p.name for p in masks if p.stem not in img_by_stem]

    if not pairs:
        raise ValueError(
            f"No matching image/mask filename stems found between '{img_dir.name}' "
            f"({len(imgs)} files) and '{mask_dir.name}' ({len(masks)} files)."
        )

    warning = ""
    if unmatched_images or unmatched_masks:
        warning = (
            f"Skipped {len(unmatched_images)} unmatched image(s) and "
            f"{len(unmatched_masks)} unmatched mask(s) in {img_dir.name}/{mask_dir.name}."
        )

    info = {
        "matched_count": len(pairs),
        "unmatched_images": unmatched_images,
        "unmatched_masks": unmatched_masks,
        "warning": warning,
    }
    return pairs, info


def read_raw_mask(path: str | Path) -> np.ndarray:
    """Read a segmentation mask image as a 2D integer array."""
    with Image.open(path) as img:
        arr = np.array(img)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr


def binarize_food_mask(
    mask: np.ndarray,
    first_food_class: int = FIRST_FOOD_CLASS,
) -> np.ndarray:
    """Convert raw TrayDataset class indices (0-4 non-food, >=5 food) to binary {0, 1}."""
    return (mask >= first_food_class).astype(np.uint8)


def load_resized_pairs(
    pairs: list[tuple[Path, Path]],
    height: int = DEFAULT_INPUT_HEIGHT,
    width: int = DEFAULT_INPUT_WIDTH,
    first_food_class: int = FIRST_FOOD_CLASS,
) -> tuple[np.ndarray, np.ndarray]:
    """Load and resize image/mask pairs using BILINEAR for images and NEAREST for masks."""
    n = len(pairs)
    X = np.zeros((n, height, width, 3), dtype=np.uint8)
    Y = np.zeros((n, height, width), dtype=np.uint8)
    for i, (img_path, mask_path) in enumerate(pairs):
        with Image.open(img_path) as img:
            resized_img = img.convert("RGB").resize((width, height), Image.BILINEAR)
            X[i] = np.array(resized_img, dtype=np.uint8)
        raw_mask = read_raw_mask(mask_path)
        resized_mask = Image.fromarray(raw_mask).resize((width, height), Image.NEAREST)
        Y[i] = binarize_food_mask(np.array(resized_mask), first_food_class=first_food_class)
    return X, Y


if TORCH_AVAILABLE:
    def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    class UNet(nn.Module):
        """Compact 3-level U-Net for binary food segmentation."""

        def __init__(self, in_ch: int = 3, out_ch: int = 1, base: int = DEFAULT_UNET_BASE) -> None:
            super().__init__()
            c1, c2, c3, c4 = base, base * 2, base * 4, base * 8
            self.enc1 = _conv_block(in_ch, c1)
            self.enc2 = _conv_block(c1, c2)
            self.enc3 = _conv_block(c2, c3)
            self.pool = nn.MaxPool2d(2)
            self.bottleneck = _conv_block(c3, c4)

            self.up3 = nn.ConvTranspose2d(c4, c3, 2, stride=2)
            self.dec3 = _conv_block(c3 * 2, c3)
            self.up2 = nn.ConvTranspose2d(c3, c2, 2, stride=2)
            self.dec2 = _conv_block(c2 * 2, c2)
            self.up1 = nn.ConvTranspose2d(c2, c1, 2, stride=2)
            self.dec1 = _conv_block(c1 * 2, c1)
            self.head = nn.Conv2d(c1, out_ch, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            e1 = self.enc1(x)
            e2 = self.enc2(self.pool(e1))
            e3 = self.enc3(self.pool(e2))
            b = self.bottleneck(self.pool(e3))
            d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
            d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
            d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
            return self.head(d1)

    class _TrayTorchDataset(Dataset):
        def __init__(self, X: np.ndarray, Y: np.ndarray, augment: bool = False) -> None:
            self.X = X
            self.Y = Y
            self.augment = augment

        def __len__(self) -> int:
            return len(self.X)

        def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
            x = torch.from_numpy(self.X[idx]).permute(2, 0, 1).float() / 255.0
            y = torch.from_numpy(self.Y[idx]).float().unsqueeze(0)
            if self.augment and random.random() < 0.5:
                x, y = x.flip(-1), y.flip(-1)
            return x, y

    def dice_coef_batch(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """Compute per-sample Dice coefficient for tensors of shape (B, 1, H, W)."""
        inter = (pred * target).sum(dim=(1, 2, 3))
        total = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
        return (2.0 * inter + eps) / (total + eps)

    def bce_dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Combined BCE-with-logits and soft Dice loss."""
        probs = torch.sigmoid(logits)
        bce = F.binary_cross_entropy_with_logits(logits, target)
        soft_dice = dice_coef_batch(probs, target).mean()
        return bce + (1.0 - soft_dice)

    @torch.no_grad()
    def evaluate_dice_loader(
        model: nn.Module,
        loader: DataLoader,
        device: str,
        threshold: float = DEFAULT_THRESHOLD,
    ) -> np.ndarray:
        """Evaluate hard Dice scores across a DataLoader."""
        model.eval()
        dices: list[torch.Tensor] = []
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = (torch.sigmoid(model(x)) > threshold).float()
            dices.append(dice_coef_batch(pred, y).cpu())
        return torch.cat(dices).numpy() if dices else np.array([], dtype=np.float32)
else:
    UNet = None  # type: ignore


def compute_binary_dice(pred_mask: np.ndarray, gt_mask: np.ndarray, eps: float = 1e-6) -> float:
    """Compute hard Dice score between two 2D binary masks."""
    p = (pred_mask > 0).astype(np.float64)
    g = (gt_mask > 0).astype(np.float64)
    inter = float((p * g).sum())
    total = float(p.sum() + g.sum())
    return (2.0 * inter + eps) / (total + eps)


def train_unet_model(
    dataset_root: str | Path,
    checkpoint_path: str | Path | None = None,
    epochs: int = 10,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    val_fraction: float = 0.15,
    input_height: int = DEFAULT_INPUT_HEIGHT,
    input_width: int = DEFAULT_INPUT_WIDTH,
    base_channels: int = DEFAULT_UNET_BASE,
    threshold: float = DEFAULT_THRESHOLD,
    seed: int = 42,
    device: str | None = None,
    progress_callback: Callable[[str], None] | None = None,
    cancel_event: threading.Event | None = None,
) -> dict[str, Any]:
    """Train a binary food-vs-not-food U-Net on TrayDataset and save a reloadable checkpoint."""
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required to train the U-Net model.")
    if input_height % 8 != 0 or input_width % 8 != 0:
        raise ValueError("Input height and width must be divisible by 8.")
    if epochs < 1:
        raise ValueError("Epochs must be at least 1.")
    if batch_size < 1:
        raise ValueError("Batch size must be at least 1.")
    if not (0.0 < val_fraction < 0.5):
        raise ValueError("Validation split fraction must be between 0 and 0.5.")

    def _emit(msg: str) -> None:
        if progress_callback:
            progress_callback(msg)

    def _check_cancel() -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise UNetTrainingCancelled("Training cancelled by user.")

    dirs = find_tray_dataset_dirs(dataset_root, require_train=True, require_test=False)
    train_pairs, pairing_info = pair_images_and_masks(dirs["XTrain"], dirs["yTrain"])
    if len(train_pairs) < 2:
        raise ValueError("At least 2 matched training pairs are required to create a validation split.")

    _check_cancel()
    _emit(f"Loading {len(train_pairs)} matched training pairs ({input_height}x{input_width})...")
    X_all, Y_all = load_resized_pairs(
        train_pairs,
        height=input_height,
        width=input_width,
        first_food_class=FIRST_FOOD_CLASS,
    )

    _check_cancel()
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(X_all))
    n_val = max(1, int(round(val_fraction * len(perm))))
    if n_val >= len(perm):
        n_val = len(perm) - 1
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    train_ds = _TrayTorchDataset(X_all[train_idx], Y_all[train_idx], augment=True)
    val_ds = _TrayTorchDataset(X_all[val_idx], Y_all[val_idx], augment=False)
    train_dl = DataLoader(train_ds, batch_size=min(batch_size, len(train_ds)), shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=min(batch_size, len(val_ds)), shuffle=False)

    dev = get_unet_device(device)
    model = UNet(in_ch=3, out_ch=1, base=base_channels).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    history: dict[str, list[float]] = {"train_loss": [], "val_dice": []}
    best_val_dice = -1.0
    best_state: dict[str, Any] | None = None

    for epoch in range(1, epochs + 1):
        _check_cancel()
        t0 = time.time()
        model.train()
        running_loss = 0.0

        for x, y in train_dl:
            _check_cancel()
            x, y = x.to(dev), y.to(dev)
            optimizer.zero_grad()
            loss = bce_dice_loss(model(x), y)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * len(x)

        train_loss = running_loss / len(train_ds)
        val_dice = float(evaluate_dice_loader(model, val_dl, dev, threshold=threshold).mean())
        history["train_loss"].append(train_loss)
        history["val_dice"].append(val_dice)

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            best_state = copy.deepcopy(model.state_dict())

        elapsed = time.time() - t0
        _emit(
            f"Epoch {epoch}/{epochs} | loss={train_loss:.4f} | "
            f"val Dice={val_dice:.4f} (best={best_val_dice:.4f}) | {elapsed:.1f}s"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    save_path = Path(checkpoint_path) if checkpoint_path else get_default_unet_checkpoint_path()
    save_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "format": "forensics_app_unet_v1",
        "model_state_dict": model.state_dict(),
        "config": {
            "in_ch": 3,
            "out_ch": 1,
            "base": int(base_channels),
            "input_height": int(input_height),
            "input_width": int(input_width),
            "first_food_class": int(FIRST_FOOD_CLASS),
            "threshold": float(threshold),
        },
        "metadata": {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "best_val_dice": float(best_val_dice),
            "final_train_loss": float(history["train_loss"][-1]),
            "epochs_trained": len(history["train_loss"]),
            "train_samples": int(len(train_idx)),
            "val_samples": int(len(val_idx)),
            "learning_rate": float(learning_rate),
            "batch_size": int(batch_size),
            "seed": int(seed),
            "dataset_root": str(Path(dataset_root).resolve()),
            "unmatched_train_images": pairing_info["unmatched_images"],
            "unmatched_train_masks": pairing_info["unmatched_masks"],
            "history": history,
        },
    }
    torch.save(checkpoint, save_path)
    set_active_unet_checkpoint_path(save_path)

    return {
        "checkpoint_path": str(save_path),
        "device": dev,
        "best_val_dice": float(best_val_dice),
        "final_train_loss": float(history["train_loss"][-1]),
        "epochs_trained": len(history["train_loss"]),
        "train_samples": int(len(train_idx)),
        "val_samples": int(len(val_idx)),
        "pairing_info": pairing_info,
        "history": history,
        "config": checkpoint["config"],
    }


def load_unet_checkpoint(
    checkpoint_path: str | Path,
    device: str | None = None,
) -> tuple[Any, dict[str, Any], dict[str, Any], str]:
    """Load and validate a saved U-Net checkpoint, raising UNetCheckpointError on failure."""
    if not TORCH_AVAILABLE:
        raise UNetCheckpointError("PyTorch is required to load a U-Net checkpoint.")

    path = Path(checkpoint_path)
    if not path.is_file():
        raise UNetCheckpointError(f"Checkpoint file not found: {path}")

    dev = get_unet_device(device)
    try:
        ckpt = torch.load(path, map_location=dev)
    except Exception as exc:
        raise UNetCheckpointError(
            f"Could not read checkpoint '{path.name}' (file may be corrupted or not a PyTorch checkpoint): {exc}"
        ) from exc

    if not isinstance(ckpt, dict):
        raise UNetCheckpointError(
            f"Incompatible checkpoint '{path.name}': expected a checkpoint dictionary."
        )

    state_dict = ckpt.get("model_state_dict")
    config = ckpt.get("config")
    if not isinstance(state_dict, dict) or not isinstance(config, dict):
        raise UNetCheckpointError(
            f"Incompatible checkpoint '{path.name}': missing 'model_state_dict' or 'config'."
        )

    required_keys = ("in_ch", "out_ch", "base", "input_height", "input_width")
    missing_keys = [k for k in required_keys if k not in config]
    if missing_keys:
        raise UNetCheckpointError(
            f"Incompatible checkpoint '{path.name}': config missing {missing_keys}."
        )

    try:
        model = UNet(
            in_ch=int(config["in_ch"]),
            out_ch=int(config["out_ch"]),
            base=int(config["base"]),
        ).to(dev)
        model.load_state_dict(state_dict, strict=True)
        model.eval()
    except Exception as exc:
        raise UNetCheckpointError(
            f"Incompatible checkpoint '{path.name}': weights do not match U-Net architecture ({exc})."
        ) from exc

    metadata = ckpt.get("metadata") if isinstance(ckpt.get("metadata"), dict) else {}
    set_active_unet_checkpoint_path(path)
    return model, config, metadata, dev


@torch.no_grad() if TORCH_AVAILABLE else (lambda fn: fn)
def segment_single_image(
    model: Any,
    image_input: Image.Image | str | Path,
    config: dict[str, Any],
    device: str,
    gt_mask_path: str | Path | None = None,
    threshold: float | None = None,
) -> dict[str, Any]:
    """Run U-Net segmentation on a single image, optionally comparing to a ground-truth mask."""
    if isinstance(image_input, Image.Image):
        pil_img = image_input.convert("RGB")
        source_name = "current_image"
    else:
        img_path = Path(image_input)
        if not img_path.is_file():
            raise ValueError(f"Image file not found: {img_path}")
        try:
            with Image.open(img_path) as opened:
                pil_img = opened.convert("RGB")
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise ValueError(f"Could not open image '{img_path.name}': {exc}") from exc
        source_name = img_path.name

    h = int(config["input_height"])
    w = int(config["input_width"])
    thr = float(threshold if threshold is not None else config.get("threshold", DEFAULT_THRESHOLD))
    first_food = int(config.get("first_food_class", FIRST_FOOD_CLASS))

    resized_rgb = np.array(pil_img.resize((w, h), Image.BILINEAR), dtype=np.uint8)
    tensor = torch.from_numpy(resized_rgb).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(device)

    model.eval()
    probs = torch.sigmoid(model(tensor))[0, 0].cpu().numpy()
    pred_mask = (probs > thr).astype(np.uint8)

    gt_binary: np.ndarray | None = None
    dice_score: float | None = None
    if gt_mask_path is not None and str(gt_mask_path).strip():
        gt_path = Path(gt_mask_path)
        if not gt_path.is_file():
            raise ValueError(f"Ground-truth mask file not found: {gt_path}")
        raw_gt = read_raw_mask(gt_path)
        resized_gt = Image.fromarray(raw_gt).resize((w, h), Image.NEAREST)
        gt_binary = binarize_food_mask(np.array(resized_gt), first_food_class=first_food)
        dice_score = compute_binary_dice(pred_mask, gt_binary)

    return {
        "image_source": source_name,
        "resized_rgb": resized_rgb,
        "prob_map": probs,
        "pred_mask": pred_mask,
        "gt_mask": gt_binary,
        "dice_score": dice_score,
        "food_fraction": float(pred_mask.mean()),
        "threshold": thr,
    }


def evaluate_unet_on_test_set(
    model: Any,
    config: dict[str, Any],
    device: str,
    dataset_root: str | Path,
    threshold: float | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Evaluate a loaded U-Net model on TrayDataset XTest/yTest pairs."""
    dirs = find_tray_dataset_dirs(dataset_root, require_train=False, require_test=True)
    test_pairs, pairing_info = pair_images_and_masks(dirs["XTest"], dirs["yTest"])

    if progress_callback:
        progress_callback(f"Evaluating on {len(test_pairs)} test pair(s)...")

    samples: list[dict[str, Any]] = []
    dices: list[float] = []
    for img_path, mask_path in test_pairs:
        res = segment_single_image(
            model=model,
            image_input=img_path,
            config=config,
            device=device,
            gt_mask_path=mask_path,
            threshold=threshold,
        )
        samples.append(res)
        if res["dice_score"] is not None:
            dices.append(float(res["dice_score"]))

    dice_arr = np.array(dices, dtype=np.float64)
    return {
        "samples": samples,
        "pairing_info": pairing_info,
        "n_test": len(samples),
        "mean_dice": float(dice_arr.mean()),
        "std_dice": float(dice_arr.std()),
        "min_dice": float(dice_arr.min()),
        "max_dice": float(dice_arr.max()),
    }


def create_mask_overlay(rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend a bright green food mask overlay onto an RGB uint8 image."""
    overlay = rgb.astype(np.float32).copy()
    food = mask > 0
    tint = np.array([40.0, 220.0, 80.0], dtype=np.float32)
    overlay[food] = (1.0 - alpha) * overlay[food] + alpha * tint
    return np.clip(overlay, 0, 255).astype(np.uint8)


def render_segmentation_visualization(
    seg_result: dict[str, Any],
    title: str | None = None,
) -> Image.Image:
    """Render Original, (optional Ground Truth), Predicted Mask, and Overlay panels."""
    rgb = seg_result["resized_rgb"]
    pred = seg_result["pred_mask"]
    gt = seg_result.get("gt_mask")
    dice = seg_result.get("dice_score")
    overlay = create_mask_overlay(rgb, pred)

    has_gt = gt is not None and dice is not None
    ncols = 4 if has_gt else 3
    fig, axes = plt.subplots(1, ncols, figsize=(3.6 * ncols, 3.6))

    axes[0].imshow(rgb)
    axes[0].set_title("Original Image", fontsize=10)

    if has_gt:
        axes[1].imshow(gt, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        axes[1].set_title("Ground Truth (Food)", fontsize=10)
        axes[2].imshow(pred, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        axes[2].set_title(f"Prediction (Dice = {dice:.3f})", fontsize=10)
        axes[3].imshow(overlay)
        axes[3].set_title("Food Overlay", fontsize=10)
    else:
        axes[1].imshow(pred, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        axes[1].set_title("Predicted Food Mask", fontsize=10)
        axes[2].imshow(overlay)
        axes[2].set_title("Food Overlay", fontsize=10)

    for ax in axes:
        ax.axis("off")

    if title:
        fig.suptitle(title, fontsize=11, fontweight="bold")
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    with Image.open(buf) as rendered:
        return rendered.convert("RGB")


def render_test_set_grid(test_eval: dict[str, Any], max_rows: int = 4) -> Image.Image:
    """Render a multi-row grid of test-set predictions alongside ground truth and overlays."""
    samples = test_eval["samples"][:max_rows]
    nrows = len(samples)
    fig, axes = plt.subplots(nrows, 4, figsize=(13.5, 3.0 * nrows), squeeze=False)

    for r, item in enumerate(samples):
        overlay = create_mask_overlay(item["resized_rgb"], item["pred_mask"])
        axes[r, 0].imshow(item["resized_rgb"])
        axes[r, 0].set_title(f"Image: {item['image_source']}", fontsize=9)
        axes[r, 1].imshow(item["gt_mask"], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        axes[r, 1].set_title("Ground Truth (Food)", fontsize=9)
        axes[r, 2].imshow(item["pred_mask"], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        axes[r, 2].set_title(f"Prediction (Dice = {item['dice_score']:.3f})", fontsize=9)
        axes[r, 3].imshow(overlay)
        axes[r, 3].set_title("Food Overlay", fontsize=9)
        for ax in axes[r]:
            ax.axis("off")

    fig.suptitle(
        f"Test Set Evaluation — Mean Dice: {test_eval['mean_dice']:.3f} ± "
        f"{test_eval['std_dice']:.3f} (n = {test_eval['n_test']})",
        fontsize=11,
        fontweight="bold",
    )
    plt.tight_layout(rect=(0, 0, 1, 0.96))

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    with Image.open(buf) as rendered:
        return rendered.convert("RGB")


def render_unet_training_report(training_summary: dict[str, Any]) -> Image.Image:
    """Render training loss and validation Dice curves."""
    history = training_summary["history"]
    epochs = list(range(1, len(history["train_loss"]) + 1))

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10.5, 4.0))
    ax0.plot(epochs, history["train_loss"], marker="o", color="#c0392b")
    ax0.set_title("Training Loss (BCE + Soft Dice)", fontsize=10, fontweight="bold")
    ax0.set_xlabel("Epoch")
    ax0.set_ylabel("Loss")
    ax0.grid(True, linestyle="--", alpha=0.35)

    ax1.plot(epochs, history["val_dice"], marker="o", color="#2980b9")
    ax1.set_title(
        f"Validation Dice (Best = {training_summary['best_val_dice']:.3f})",
        fontsize=10,
        fontweight="bold",
    )
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Dice")
    ax1.set_ylim(0.0, 1.05)
    ax1.grid(True, linestyle="--", alpha=0.35)

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    with Image.open(buf) as rendered:
        return rendered.convert("RGB")
