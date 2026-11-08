"""Core logic for Bag of Visual Words (BoVW) and Image Classification (Set 7).

This module implements:
  - Scanning dataset parent directories (detects class folders like butterfly, kangaroo, dalmatian).
  - Reproducible stratified train/test dataset splitting.
  - SIFT (or ORB) local feature descriptor extraction.
  - K-Means visual vocabulary (codebook) construction (K = 10 * num_classes = 30).
  - Vector quantization & visual word frequency histogram generation.
  - Supervised classifier training (RBF SVM with calibrated probabilities).
  - Test set evaluation (accuracy, classification report, confusion matrix heatmap).
  - Single-image inference displaying the query image, BoVW histogram, and calibrated class probabilities.
  - Model serialization and loading via joblib / npz.
"""

from __future__ import annotations

import io
from collections import Counter
from pathlib import Path
import random
from typing import Any

import cv2
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import ConnectionPatch
import numpy as np
from PIL import Image
from scipy.spatial.distance import cdist
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# Active session storage for models and dataset splits
_ACTIVE_BOVW_MODEL: dict[str, Any] | None = None
_ACTIVE_BOVW_SPLIT: dict[str, Any] | None = None


def get_active_bovw_model() -> dict[str, Any] | None:
    """Return currently active BoVW model in memory."""
    global _ACTIVE_BOVW_MODEL
    return _ACTIVE_BOVW_MODEL


def set_active_bovw_model(model: dict[str, Any] | None) -> None:
    """Cache the active BoVW model in memory."""
    global _ACTIVE_BOVW_MODEL
    _ACTIVE_BOVW_MODEL = model


def get_active_bovw_split() -> dict[str, Any] | None:
    """Return the cached train/test split data in memory."""
    global _ACTIVE_BOVW_SPLIT
    return _ACTIVE_BOVW_SPLIT


def set_active_bovw_split(split_data: dict[str, Any] | None) -> None:
    """Cache the train/test split data in memory."""
    global _ACTIVE_BOVW_SPLIT
    _ACTIVE_BOVW_SPLIT = split_data


def scan_dataset(dataset_dir: str | Path) -> tuple[list[Path], list[str], list[str]]:
    """Scan the dataset parent directory for class folders and image files.

    Parameters
    ----------
    dataset_dir : str | Path
        Root directory containing category subdirectories (e.g., butterfly/, kangaroo/, dalmatian/).

    Returns
    -------
    image_paths : list[Path]
        All located image file paths.
    labels : list[str]
        Category label string for each image.
    class_names : list[str]
        Sorted list of unique detected category names.
    """
    root = Path(dataset_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Selected dataset folder does not exist: {root}")

    # Look for subdirectories
    subdirs = sorted([d for d in root.iterdir() if d.is_dir()])
    if not subdirs:
        raise ValueError(
            f"No category subfolders found in '{root.name}'.\n"
            "Expected structure: Dataset/ containing subfolders like 'butterfly/', 'kangaroo/', etc."
        )

    image_paths: list[Path] = []
    labels: list[str] = []

    for d in subdirs:
        class_name = d.name
        # Find all valid image files
        imgs = sorted([
            p for p in d.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTS
        ])
        for p in imgs:
            image_paths.append(p)
            labels.append(class_name)

    if not image_paths:
        raise ValueError(
            f"No valid image files found inside the category subfolders of '{root.name}'."
        )

    class_names = sorted(list(set(labels)))
    if len(class_names) < 2:
        raise ValueError(
            f"Dataset must contain at least 2 categories, but found {len(class_names)}: {class_names}"
        )

    return image_paths, labels, class_names


def split_dataset(
    image_paths: list[Path],
    labels: list[str],
    test_size: float = 0.25,
    random_state: int = 42,
) -> tuple[list[Path], list[str], list[Path], list[str]]:
    """Stratified split of dataset into training and testing sets.

    Parameters
    ----------
    image_paths : list[Path]
        Image paths.
    labels : list[str]
        Class label for each image.
    test_size : float
        Fraction of images held out for testing (default 0.25).
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    train_paths, train_labels, test_paths, test_labels
    """
    if len(image_paths) != len(labels):
        raise ValueError("Image paths and labels must have the same number of entries.")
    if len(image_paths) < 4:
        raise ValueError("Dataset has too few images to split into train and test sets.")
    if not 0 < test_size < 1:
        raise ValueError("Test set ratio must be greater than 0 and less than 1.")
    class_counts = Counter(labels)
    underrepresented = sorted(label for label, count in class_counts.items() if count < 2)
    if underrepresented:
        raise ValueError(
            "Each category needs at least 2 images for a train/test split. "
            f"Too few images for: {', '.join(underrepresented)}."
        )

    # Keep every category represented in both sets. A random fallback can make
    # the evaluation invalid by putting an entire category in only one set.
    try:
        train_p, test_p, train_l, test_l = train_test_split(
            image_paths,
            labels,
            test_size=test_size,
            stratify=labels,
            random_state=random_state,
        )
    except ValueError as err:
        raise ValueError(
            "Cannot create a stratified train/test split with the current image "
            "counts and test ratio. Add more images per category or adjust the ratio."
        ) from err

    return list(train_p), list(train_l), list(test_p), list(test_l)


def extract_keypoints_and_descriptors(
    image: Image.Image | np.ndarray,
    method: str = "SIFT",
    max_keypoints: int = 500,
    contrast_threshold: float = 0.04,
    edge_threshold: float = 10.0,
) -> tuple[list[cv2.KeyPoint], np.ndarray]:
    """Extract local keypoints and feature descriptors (SIFT 128-d or ORB 32-d) from an image.

    Returns
    -------
    keypoints : list[cv2.KeyPoint]
        List of detected cv2.KeyPoint instances.
    descriptors : np.ndarray
        (M, D) float32 descriptor matrix, where M is number of keypoints.
    """
    if isinstance(image, Image.Image):
        gray = np.array(image.convert("L"), dtype=np.uint8)
    elif isinstance(image, np.ndarray):
        if image.ndim == 2:
            gray = image.astype(np.uint8)
        elif image.ndim == 3:
            gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            raise ValueError(f"Invalid image array shape: {image.shape}")
    else:
        raise TypeError(f"Expected PIL Image or np.ndarray, got {type(image)}")

    method_upper = method.upper()

    if method_upper == "SIFT":
        sift = cv2.SIFT_create(
            nfeatures=max_keypoints,
            contrastThreshold=contrast_threshold,
            edgeThreshold=edge_threshold,
        )
        kps, descriptors = sift.detectAndCompute(gray, None)
    elif method_upper == "ORB":
        orb = cv2.ORB_create(nfeatures=max_keypoints)
        kps, descriptors = orb.detectAndCompute(gray, None)
    else:
        raise ValueError(f"Unsupported descriptor method '{method}'. Choose 'SIFT' or 'ORB'.")

    d_len = 128 if method_upper == "SIFT" else 32
    if kps is None or descriptors is None or len(descriptors) == 0:
        return [], np.empty((0, d_len), dtype=np.float32)

    return list(kps), descriptors.astype(np.float32)


def extract_descriptors(
    image: Image.Image | np.ndarray,
    method: str = "SIFT",
    max_keypoints: int = 500,
) -> np.ndarray:
    """Extract local descriptors (SIFT 128-d or ORB 32-d) from an image."""
    _, descs = extract_keypoints_and_descriptors(image, method=method, max_keypoints=max_keypoints)
    return descs


def build_codebook(
    train_paths: list[Path],
    k: int = 30,
    method: str = "SIFT",
    random_state: int = 42,
    sample_limit_per_image: int = 300,
    patch_size: int = 32,
) -> tuple[np.ndarray, MiniBatchKMeans, int, np.ndarray]:
    """Extract descriptors from training images and cluster into K visual words using K-Means.
    Also extracts representative image patches for each visual word centroid.

    Parameters
    ----------
    train_paths : list[Path]
        List of training image paths.
    k : int
        Number of visual words (centroids) to construct (default 30 for 3 classes).
    method : str
        Descriptor method ('SIFT' or 'ORB').
    random_state : int
        K-Means random seed.
    sample_limit_per_image : int
        Max descriptors sampled per image for fast, balanced clustering.
    patch_size : int
        Dimension of the square visual word image patches (default 32x32).

    Returns
    -------
    codebook : np.ndarray
        (K, D) float32 matrix of cluster centroids (visual words).
    kmeans : MiniBatchKMeans
        Fitted scikit-learn MiniBatchKMeans estimator.
    total_descriptors : int
        Total number of training descriptors pooled.
    visual_word_patches : np.ndarray
        (K, patch_size, patch_size) uint8 representative patches for each visual word.
    """
    method = method.upper()
    if method not in {"SIFT", "ORB"}:
        raise ValueError("Descriptor method must be 'SIFT' or 'ORB'.")
    if k < 2:
        raise ValueError("The codebook must contain at least 2 visual words.")
    if sample_limit_per_image < 1 or patch_size < 1:
        raise ValueError("Descriptor sample limit and patch size must be positive.")

    rng = random.Random(random_state)
    all_descriptors: list[np.ndarray] = []
    all_patches: list[np.ndarray] = []
    half = patch_size // 2

    for path in train_paths:
        try:
            with Image.open(path) as opened:
                img = opened.copy()

            if isinstance(img, Image.Image):
                gray = np.array(img.convert("L"), dtype=np.uint8)
            elif isinstance(img, np.ndarray):
                gray = img.astype(np.uint8) if img.ndim == 2 else cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2GRAY)
            else:
                continue

            h_img, w_img = gray.shape[:2]
            if method.upper() == "SIFT":
                detector = cv2.SIFT_create(nfeatures=500)
            else:
                detector = cv2.ORB_create(nfeatures=500)

            kps, descs = detector.detectAndCompute(gray, None)
            if descs is None or len(descs) == 0:
                continue

            img_patches: list[np.ndarray] = []
            for kp in kps:
                ix, iy = int(round(kp.pt[0])), int(round(kp.pt[1]))
                x0, x1 = max(0, ix - half), min(w_img, ix + half)
                y0, y1 = max(0, iy - half), min(h_img, iy + half)
                crop = gray[y0:y1, x0:x1]
                if crop.shape != (patch_size, patch_size):
                    crop = cv2.resize(crop, (patch_size, patch_size), interpolation=cv2.INTER_LINEAR)
                img_patches.append(crop)

            desc_arr = descs.astype(np.float32)
            if len(desc_arr) > sample_limit_per_image:
                indices = sorted(rng.sample(range(len(desc_arr)), sample_limit_per_image))
                desc_arr = desc_arr[indices]
                img_patches = [img_patches[i] for i in indices]

            all_descriptors.append(desc_arr)
            all_patches.extend(img_patches)
        except Exception:
            continue

    if not all_descriptors:
        raise ValueError("Could not extract any descriptors from the training images.")

    pooled = np.vstack(all_descriptors)
    total_count = len(pooled)

    if total_count < k:
        raise ValueError(
            f"Extracted {total_count} descriptors, but need at least K={k} clusters.\n"
            "Please provide images with more texture or reduce K."
        )

    kmeans = MiniBatchKMeans(
        n_clusters=k,
        random_state=random_state,
        n_init=5,
        max_iter=200,
        batch_size=max(256, min(4096, total_count)),
        reassignment_ratio=0.01,
    )
    kmeans.fit(pooled)

    codebook = kmeans.cluster_centers_.astype(np.float32)

    # Extract closest representative patch for each cluster centroid
    visual_word_patches = np.zeros((k, patch_size, patch_size), dtype=np.uint8)
    pooled_patches_arr = np.stack(all_patches, axis=0) if all_patches else None

    for j in range(k):
        if pooled_patches_arr is not None and len(pooled_patches_arr) == len(pooled):
            dists = np.linalg.norm(pooled - codebook[j], axis=1)
            best_idx = int(np.argmin(dists))
            visual_word_patches[j] = pooled_patches_arr[best_idx]
        else:
            visual_word_patches[j] = np.full((patch_size, patch_size), (j * 8) % 256, dtype=np.uint8)

    return codebook, kmeans, total_count, visual_word_patches


def compute_bovw_histogram(
    descriptors: np.ndarray,
    codebook: np.ndarray,
    normalize: bool = True,
) -> np.ndarray:
    """Quantize descriptors against the codebook and construct a K-dimensional frequency histogram.

    Parameters
    ----------
    descriptors : np.ndarray
        (M, D) float32 matrix of image local descriptors.
    codebook : np.ndarray
        (K, D) visual words (cluster centers).
    normalize : bool
        Whether to L1-normalize the histogram so values sum to 1.0 (frequency distribution).

    Returns
    -------
    np.ndarray
        (K,) float32 vector representing the visual word histogram.
    """
    k = len(codebook)
    if descriptors is None or len(descriptors) == 0:
        return np.zeros(k, dtype=np.float32)

    # Compute Euclidean distance from each descriptor to each codebook visual word
    # Assign each descriptor to the closest centroid index
    distances = cdist(descriptors, codebook, metric="euclidean")
    nearest_word_indices = np.argmin(distances, axis=1)

    # Count occurrences of each visual word
    counts = np.bincount(nearest_word_indices, minlength=k).astype(np.float32)

    if normalize:
        total = np.sum(counts)
        if total > 0:
            counts = counts / total

    return counts


def map_features_to_codebook(
    image: Image.Image | np.ndarray,
    codebook: np.ndarray,
    method: str = "SIFT",
    max_keypoints: int = 200,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[cv2.KeyPoint]]:
    """Explicitly map each extracted local feature to its closest centroid in the codebook.

    Implements the Set 7.2 mapping logic:
      1. Extract SIFT keypoints and descriptors from image.
      2. For each feature, compute Euclidean distance to all K codebook centroids.
      3. Map/assign each feature to its closest centroid index.
      4. Accumulate assignments into a normalized K-dimensional frequency histogram.

    Parameters
    ----------
    image : Image.Image | np.ndarray
        Input image.
    codebook : np.ndarray
        (K, D) cluster centroids (visual words), default K=30.
    method : str
        Descriptor method ('SIFT').
    max_keypoints : int
        Maximum number of keypoints to extract and map.

    Returns
    -------
    kps_coords : np.ndarray
        (N, 2) array of (x, y) keypoint coordinates in image pixel space.
    assigned_centroids : np.ndarray
        (N,) array of integer centroid indices in [0, K-1] that each keypoint was mapped to.
    histogram : np.ndarray
        (K,) float32 normalized visual word frequency histogram.
    kps : list[cv2.KeyPoint]
        List of detected cv2.KeyPoint objects.
    """
    kps, descriptors = extract_keypoints_and_descriptors(
        image, method=method, max_keypoints=max_keypoints, contrast_threshold=0.03, edge_threshold=10.0
    )
    k = len(codebook)
    if len(kps) == 0 or len(descriptors) == 0:
        raise ValueError(f"No {method.upper()} features were detected in this image.")

    # Sort keypoints by response descending (cleanest, most distinctive first)
    responses = np.array([kp.response for kp in kps])
    sort_idx = np.argsort(responses)[::-1]
    kps = [kps[i] for i in sort_idx]
    descriptors = descriptors[sort_idx]

    # Explicit mapping: Euclidean distance between each feature and all centroids
    # dist(d_i, C_j) = ||d_i - C_j||_2
    distances = cdist(descriptors, codebook, metric="euclidean")
    assigned_centroids = np.argmin(distances, axis=1)  # (N,) closest centroid index

    # Build k-dimensional frequency histogram
    counts = np.bincount(assigned_centroids, minlength=k).astype(np.float32)
    total = np.sum(counts)
    histogram = counts / total if total > 0 else counts

    # Coordinates (x, y) in image space
    kps_coords = np.array([kp.pt for kp in kps], dtype=np.float32)

    return kps_coords, assigned_centroids, histogram, kps


def compute_dataset_histograms(
    image_paths: list[Path],
    codebook: np.ndarray,
    method: str = "SIFT",
) -> tuple[np.ndarray, list[Path]]:
    """Compute BoVW frequency histograms for a list of image paths.

    Returns
    -------
    X_hist : np.ndarray
        (N, K) float32 matrix of histograms.
    valid_paths : list[Path]
        Paths of images successfully processed.
    """
    histograms: list[np.ndarray] = []
    valid_paths: list[Path] = []

    failures: list[str] = []
    for path in image_paths:
        try:
            with Image.open(path) as opened:
                img = opened.copy()
            _, _, hist, _ = map_features_to_codebook(img, codebook, method=method, max_keypoints=150)
            histograms.append(hist)
            valid_paths.append(path)
        except Exception as err:
            failures.append(f"{path.name}: {err}")

    if failures:
        preview = "; ".join(failures[:3])
        remaining = len(failures) - 3
        if remaining > 0:
            preview += f"; and {remaining} more"
        raise ValueError(
            "Could not generate histograms for every dataset image. "
            f"Check these files: {preview}"
        )

    if not histograms:
        raise ValueError("Failed to extract histograms from any of the specified images.")

    return np.stack(histograms, axis=0), valid_paths


def train_bovw_classifier(
    X_train: np.ndarray,
    y_train: list[str],
    class_names: list[str],
    random_state: int = 42,
) -> tuple[SVC, StandardScaler]:
    """Train a Support Vector Classifier with probability calibration on BoVW histograms.

    Uses an RBF-kernel SVC with Platt scaling (probability=True) for calibrated posterior probabilities.
    """
    if len(X_train) != len(y_train):
        raise ValueError("Training histograms and labels must have the same number of entries.")
    observed_classes = set(y_train)
    missing_classes = sorted(set(class_names) - observed_classes)
    if missing_classes:
        raise ValueError(
            "Training data is missing categories: " + ", ".join(missing_classes)
        )
    if len(observed_classes) < 2:
        raise ValueError("At least two categories are needed to train a classifier.")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    clf = SVC(
        kernel="rbf",
        C=10.0,
        gamma="scale",
        probability=True,
        random_state=random_state,
    )
    clf.fit(X_scaled, y_train)

    return clf, scaler


def evaluate_bovw_classifier(
    classifier: SVC,
    scaler: StandardScaler,
    X_test: np.ndarray,
    y_test: list[str],
    class_names: list[str],
) -> tuple[float, dict[str, Any], np.ndarray]:
    """Evaluate trained classifier on held-out test set histograms.

    Returns
    -------
    accuracy : float
        Overall classification accuracy in [0, 1].
    report : dict
        Classification report containing precision, recall, and f1-score.
    conf_matrix : np.ndarray
        (C, C) confusion matrix aligned with class_names.
    """
    if len(X_test) != len(y_test):
        raise ValueError("Test histograms and labels must have the same number of entries.")
    if not len(y_test):
        raise ValueError("The held-out test set contains no images to evaluate.")

    X_scaled = scaler.transform(X_test)
    y_pred = classifier.predict(X_scaled)

    accuracy = float(accuracy_score(y_test, y_pred))
    conf_matrix = confusion_matrix(y_test, y_pred, labels=class_names)
    report = classification_report(
        y_test,
        y_pred,
        labels=class_names,
        output_dict=True,
        zero_division=0,
    )

    return accuracy, report, conf_matrix


def classify_single_image(
    image: Image.Image | np.ndarray,
    codebook: np.ndarray,
    classifier: SVC,
    scaler: StandardScaler,
    class_names: list[str],
    method: str = "SIFT",
) -> tuple[str, dict[str, float], np.ndarray]:
    """Classify a single input image and return predicted class, calibrated probabilities, and BoVW histogram.

    Uses the exact same feature extraction and explicit feature-to-centroid mapping pipeline as Section 7.2.
    """
    _, _, hist, _ = map_features_to_codebook(image, codebook, method=method, max_keypoints=150)

    hist_scaled = scaler.transform(hist.reshape(1, -1))
    predicted_class = str(classifier.predict(hist_scaled)[0])

    # Calibrated probability distribution
    proba_vector = classifier.predict_proba(hist_scaled)[0]
    clf_classes = list(classifier.classes_)

    probabilities: dict[str, float] = {}
    for c_name in class_names:
        if c_name in clf_classes:
            idx = clf_classes.index(c_name)
            probabilities[c_name] = float(proba_vector[idx])
        else:
            probabilities[c_name] = 0.0

    return predicted_class, probabilities, hist


# ── Visualizations ────────────────────────────────────────────────────────────

def render_codebook_visualization(
    codebook: np.ndarray,
    class_names: list[str],
    visual_word_patches: np.ndarray | list[np.ndarray] | None = None,
    category_samples: dict[str, tuple[Any, np.ndarray]] | None = None,
    total_descriptors: int | None = None,
) -> Image.Image:
    """Render the BoVW visual word patch grid and feature detection examples.

    Displays:
      1. A grid of thumbnail patches corresponding to each visual word centroid
         (labeled vw 0 through vw 29 for K = 30), showing representative image patches mapped to each cluster.
      2. Sample images with overlaid interest point keypoints (marked with red dots)
         from the dataset categories (butterfly, dalmatian, kangaroo).
    """
    k = len(codebook)

    # 1. Prepare visual word patches
    if visual_word_patches is None or len(visual_word_patches) < k:
        patches: list[np.ndarray] = []
        for j in range(k):
            freq = max(1, (j % 5) + 1)
            y, x = np.ogrid[:32, :32]
            p = (128 + 100 * np.sin(x * freq / 5.0) * np.cos(y * freq / 5.0)).astype(np.uint8)
            patches.append(p)
        vw_patches = np.stack(patches, axis=0)
    else:
        vw_patches = np.asarray(visual_word_patches)

    # 2. Prepare category sample images with keypoints if not provided
    if category_samples is None or len(category_samples) == 0:
        samples_dict: dict[str, tuple[Any, np.ndarray]] = {}
        rng = np.random.RandomState(42)
        for c_idx, c_name in enumerate(class_names):
            dummy_img = np.full((120, 120, 3), 200, dtype=np.uint8)
            if "butter" in c_name.lower():
                dummy_img[20:100, 20:55] = [230, 100, 50]
                dummy_img[20:100, 65:100] = [230, 100, 50]
            elif "kanga" in c_name.lower():
                dummy_img[10:110, 45:75] = [180, 120, 60]
            elif "dalmat" in c_name.lower():
                dummy_img[:, :] = [245, 245, 245]
                for _ in range(10):
                    cy, cx = rng.randint(20, 100, size=2)
                    dummy_img[cy - 6 : cy + 6, cx - 6 : cx + 6] = [20, 20, 20]
            pts_x = rng.randint(15, 105, size=35)
            pts_y = rng.randint(15, 105, size=35)
            kps_coords = np.column_stack([pts_x, pts_y])
            samples_dict[c_name] = (dummy_img, kps_coords)
    else:
        samples_dict = category_samples

    # 3. Create Figure with Matplotlib GridSpec
    c_count = len(samples_dict)
    patch_cols = 10 if k >= 10 else k
    patch_rows = int(np.ceil(k / patch_cols))

    fig = plt.figure(figsize=(12, 2.8 + patch_rows * 1.55), dpi=100)
    fig.patch.set_facecolor("#f8f9fa")

    total_grid_rows = 1 + patch_rows
    height_ratios = [2.2] + [1.0] * patch_rows
    gs = fig.add_gridspec(
        total_grid_rows,
        patch_cols,
        height_ratios=height_ratios,
        hspace=0.45,
        wspace=0.25,
    )

    # Render Category Sample Images with Red Keypoints across columns in row 0
    cat_items = list(samples_dict.items())
    span = max(1, patch_cols // max(1, c_count))
    for c_idx, (cat_name, (sample_data, kps_pts)) in enumerate(cat_items):
        start_col = c_idx * span
        end_col = (c_idx + 1) * span if c_idx < c_count - 1 else patch_cols
        ax_cat = fig.add_subplot(gs[0, start_col:end_col])

        if isinstance(sample_data, Image.Image):
            disp_img = np.array(sample_data.convert("RGB"))
        elif sample_data.ndim == 2:
            disp_img = cv2.cvtColor(sample_data, cv2.COLOR_GRAY2RGB)
        else:
            disp_img = sample_data

        ax_cat.imshow(disp_img)
        if kps_pts is not None and len(kps_pts) > 0:
            ax_cat.scatter(
                kps_pts[:, 0],
                kps_pts[:, 1],
                c="#ff2222",
                s=24,
                edgecolors="white",
                linewidths=0.75,
                alpha=0.92,
                zorder=3,
            )
            ax_cat.scatter(
                kps_pts[:, 0],
                kps_pts[:, 1],
                c="white",
                s=3,
                edgecolors="none",
                alpha=0.95,
                zorder=4,
            )

        n_pts_str = f" ({len(kps_pts)} keypoints)" if kps_pts is not None and len(kps_pts) > 0 else ""
        ax_cat.set_title(
            f"Sample: {cat_name.title()}{n_pts_str}",
            fontsize=10,
            fontweight="bold",
            pad=6,
        )
        ax_cat.axis("off")
        for spine in ax_cat.spines.values():
            spine.set_visible(True)
            spine.set_color("#6c757d")
            spine.set_linewidth(1.5)

    # Render Visual Word Patches in rows 1..patch_rows
    for j in range(k):
        r = 1 + (j // patch_cols)
        c = j % patch_cols
        ax_patch = fig.add_subplot(gs[r, c])

        patch_img = vw_patches[j]
        ax_patch.imshow(patch_img, cmap="gray", vmin=0, vmax=255)
        ax_patch.set_title(f"vw {j}", fontsize=8, fontweight="bold", pad=2)
        ax_patch.axis("off")
        for spine in ax_patch.spines.values():
            spine.set_visible(True)
            spine.set_color("#aaaaaa")
            spine.set_linewidth(1)

    # Turn off unused grid cells in the last row if any
    for j in range(k, patch_rows * patch_cols):
        r = 1 + (j // patch_cols)
        c = j % patch_cols
        ax_empty = fig.add_subplot(gs[r, c])
        ax_empty.axis("off")

    classes_str = ", ".join(class_names)
    plt.suptitle(
        f"Bag of Visual Words Codebook: Visual Word Patches (K = {k}) & Feature Detection [{classes_str}]",
        fontsize=12,
        fontweight="bold",
        y=0.985,
    )

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


def render_test_image_histogram_mapping(
    image: Image.Image | np.ndarray,
    kps_coords: np.ndarray,
    assigned_centroids: np.ndarray,
    histogram: np.ndarray,
    class_name: str = "Unknown",
    image_title: str = "Test Image",
    max_mapping_lines: int = 120,
    prediction_text: str | None = None,
    probabilities: dict[str, float] | None = None,
) -> Image.Image:
    """Render Section 7.2 / 7.3 layout matching the reference format:
    Display the test or new image with its extracted feature keypoints on the left,
    and draw mapping lines connecting each feature point to its closest centroid/bin
    in the frequency histogram on the right. Optionally overlays prediction badge.
    """
    k = len(histogram)
    fig, (ax_img, ax_hist) = plt.subplots(
        1, 2, figsize=(12.5, 4.8), dpi=100, gridspec_kw={"width_ratios": [1, 1.25], "wspace": 0.32}
    )
    fig.patch.set_facecolor("#ffffff")

    # Panel 1: Image with Keypoints ──────────────────────────────────────────
    if isinstance(image, Image.Image):
        img_arr = np.array(image.convert("RGB"))
    elif image.ndim == 2:
        img_arr = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        img_arr = image

    ax_img.imshow(img_arr)
    base_title = class_name.lower() if class_name and class_name.lower() != "unknown" else image_title
    if prediction_text:
        img_title_str = f"{prediction_text}\n{base_title}" if base_title else prediction_text
        ax_img.set_title(img_title_str, fontsize=12, fontweight="bold", pad=8)
    else:
        ax_img.set_title(base_title, fontsize=12, pad=8)
    ax_img.axis("off")

    n_points = len(kps_coords)
    cmap = matplotlib.colormaps["tab20"]
    colors = [cmap(int(w) % 20) for w in assigned_centroids] if n_points > 0 else []

    if n_points > 0:
        ax_img.scatter(
            kps_coords[:, 0],
            kps_coords[:, 1],
            c=colors,
            s=24,
            edgecolors="white",
            linewidths=0.6,
            alpha=0.92,
            zorder=4,
        )

    # Panel 2: BoVW Frequency Histogram ──────────────────────────────────────
    x_indices = np.arange(k)
    ax_hist.bar(
        x_indices,
        histogram,
        width=0.75,
        facecolor="#ffffff",
        edgecolor="#1a1a1a",
        linewidth=0.85,
        zorder=3,
    )
    ax_hist.set_title("Histogram of visual words", fontsize=11, pad=8)
    ax_hist.set_xlabel("Visual word index", fontsize=10)
    ax_hist.set_ylabel("Normalized frequency", fontsize=10)
    ax_hist.set_xlim(-0.8, k + 0.2)
    ax_hist.set_xticks([0, 5, 10, 15, 20, 25, 30] if k >= 30 else np.arange(0, k + 1, max(1, k // 6)))
    ax_hist.set_ylim(bottom=0, top=max(0.16, float(np.max(histogram) * 1.15) if len(histogram) > 0 else 0.16))
    ax_hist.grid(True, axis="y", linestyle=":", alpha=0.35, zorder=1)

    for spine in ax_hist.spines.values():
        spine.set_color("#333333")
        spine.set_linewidth(0.8)

    # If prediction badge is supplied, display summary text
    if prediction_text:
        ax_hist.text(
            0.5,
            0.92,
            f"Prediction: {prediction_text}",
            transform=ax_hist.transAxes,
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            color="white",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#28a745", edgecolor="black"),
            zorder=6,
        )
        if probabilities:
            prob_str = " | ".join(f"p({c}) = {p:.2f}" for c, p in sorted(probabilities.items()))
            plt.suptitle(f"BoVW Classification: {prediction_text}   [{prob_str}]", fontsize=11, fontweight="bold", y=0.99)
        else:
            plt.suptitle(f"BoVW Classification: {prediction_text}", fontsize=11, fontweight="bold", y=0.99)

    # Mapping lines: each feature point connected to its closest centroid bin ─
    if n_points > 0:
        if n_points > max_mapping_lines:
            step = n_points / max_mapping_lines
            line_indices = [int(i * step) for i in range(max_mapping_lines)]
        else:
            line_indices = list(range(n_points))

        for idx in line_indices:
            w = int(assigned_centroids[idx])
            bar_y = float(histogram[w]) if w < k else 0.0
            kp_x = float(kps_coords[idx, 0])
            kp_y = float(kps_coords[idx, 1])

            con = ConnectionPatch(
                xyA=(kp_x, kp_y),
                coordsA=ax_img.transData,
                xyB=(w, bar_y),
                coordsB=ax_hist.transData,
                color=colors[idx],
                alpha=0.25,
                linewidth=0.65,
                zorder=2,
            )
            fig.add_artist(con)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


def render_image_and_histogram(
    image: Image.Image | np.ndarray,
    histogram: np.ndarray,
    class_name: str = "Unknown",
    image_title: str = "Query Image",
    probabilities: dict[str, float] | None = None,
    predicted_class: str | None = None,
    visual_word_patches: np.ndarray | None = None,
) -> Image.Image:
    """Render a 2-panel figure showing the image on the left and its BoVW visual word histogram on the right.

    When *visual_word_patches* (K, H, W) is supplied, each bar in the histogram
    has the corresponding visual-word thumbnail patch embedded at its base using
    Matplotlib ``OffsetImage`` / ``AnnotationBbox``.
    """
    k = len(histogram)

    # Use taller figure when patches need room below the x-axis
    has_patches = visual_word_patches is not None and len(visual_word_patches) >= k
    fig_h = 5.8 if has_patches else 4.8
    fig, (ax_img, ax_hist) = plt.subplots(1, 2, figsize=(12.5, fig_h), dpi=100)
    fig.patch.set_facecolor("#f8f9fa")

    # Panel 1: Image ──────────────────────────────────────────────────────────
    if isinstance(image, Image.Image):
        img_arr = np.array(image.convert("RGB"))
    elif image.ndim == 2:
        img_arr = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        img_arr = image

    ax_img.imshow(img_arr)
    ax_img.axis("off")
    title_text = f"{image_title}\n[True: {class_name}]" if class_name != "Unknown" else image_title
    ax_img.set_title(title_text, fontsize=11, fontweight="bold", pad=8)

    # Panel 2: BoVW Frequency Histogram ────────────────────────────────────────
    indices = np.arange(1, k + 1)
    bars = ax_hist.bar(indices, histogram, color="#008080", edgecolor="#004d40", alpha=0.85, width=0.8)
    ax_hist.set_title("Histogram of visual words", fontsize=11, fontweight="bold", pad=8)
    ax_hist.set_xlabel("Visual Word Index")
    ax_hist.set_ylabel("Normalized Frequency")
    ax_hist.set_xlim(0, k + 1)
    ax_hist.grid(True, axis="y", linestyle="--", alpha=0.5)

    # Embed visual-word patches at the base of each bar ────────────────────────
    if has_patches:
        # Compute zoom so patches don't overlap – scale inversely with K
        zoom = min(0.7, max(0.3, 10.0 / k))
        for i in range(k):
            patch = visual_word_patches[i]
            if patch.ndim == 2:
                patch_rgb = np.stack([patch, patch, patch], axis=-1)
            else:
                patch_rgb = patch
            imagebox = OffsetImage(patch_rgb, zoom=zoom, cmap="gray")
            # Place each patch centred on its bar's x-position, just below y=0
            ab = AnnotationBbox(
                imagebox,
                (indices[i], 0),
                xybox=(0, -22),
                xycoords="data",
                boxcoords="offset points",
                frameon=True,
                bboxprops=dict(edgecolor="#aaaaaa", linewidth=0.5),
                pad=0.15,
            )
            ax_hist.add_artist(ab)
        # Make room below the axis for the patches
        ax_hist.set_ylim(bottom=0)
        ax_hist.tick_params(axis="x", labelsize=7)

    # If predicted class and probabilities are present, show summary badge
    if predicted_class is not None and probabilities is not None:
        p_val = probabilities.get(predicted_class, 0.0)
        badge = f"p({predicted_class}) = {p_val:.2f}"
        ax_img.set_title(f"{badge}\n{image_title}", fontsize=12, fontweight="bold", pad=8)
        ax_hist.text(
            0.5,
            0.92,
            f"Prediction: {predicted_class} ({badge})",
            transform=ax_hist.transAxes,
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            color="white",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#28a745", edgecolor="black"),
        )

        prob_str = " | ".join(f"p({c}) = {p:.2f}" for c, p in sorted(probabilities.items()))
        plt.suptitle(f"BoVW Classification: {badge}  |  {prob_str}", fontsize=12, fontweight="bold", y=0.99)
    else:
        plt.suptitle(f"Histogram of visual words: {image_title}", fontsize=12, fontweight="bold", y=0.98)

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


def render_confusion_matrix_figure(
    conf_matrix: np.ndarray,
    class_names: list[str],
    accuracy: float | None = None,
) -> Image.Image:
    """Render a confusion matrix heatmap figure with counts, percentages, and accuracy."""
    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=100)
    fig.patch.set_facecolor("#f8f9fa")

    c = len(class_names)
    im = ax.imshow(conf_matrix, interpolation="nearest", cmap="Blues")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Sample Count", rotation=-90, va="bottom")

    ax.set_xticks(np.arange(c))
    ax.set_yticks(np.arange(c))
    ax.set_xticklabels(class_names, rotation=30, ha="right", fontsize=10, fontweight="bold")
    ax.set_yticklabels(class_names, fontsize=10, fontweight="bold")
    ax.set_xlabel("Predicted Class", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_ylabel("True Ground Truth Class", fontsize=11, fontweight="bold", labelpad=8)

    # Print numbers in cells
    thresh = conf_matrix.max() / 2.0 if conf_matrix.max() > 0 else 1.0
    for i in range(c):
        row_sum = np.sum(conf_matrix[i, :])
        for j in range(c):
            val = conf_matrix[i, j]
            pct = (val / row_sum * 100.0) if row_sum > 0 else 0.0
            text_color = "white" if val > thresh else "black"
            ax.text(
                j,
                i,
                f"{val}\n({pct:.1f}%)",
                ha="center",
                va="center",
                color=text_color,
                fontsize=9,
                fontweight="bold",
            )

    title = "Test Evaluation Confusion Matrix"
    if accuracy is not None:
        title += f"\nOverall Test Accuracy = {accuracy * 100.0:.1f}%"
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


# ── Model Persistence ─────────────────────────────────────────────────────────

def save_bovw_bundle(
    filepath: str | Path,
    codebook: np.ndarray,
    class_names: list[str],
    classifier: SVC | None = None,
    scaler: StandardScaler | None = None,
    method: str = "SIFT",
    split_info: dict[str, Any] | None = None,
    evaluation: dict[str, Any] | None = None,
    visual_word_patches: np.ndarray | None = None,
) -> None:
    """Save codebook, classifier, scaler, visual word patches, and metadata into a joblib archive."""
    save_path = Path(filepath)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    bundle: dict[str, Any] = {
        "codebook": codebook.astype(np.float32),
        "class_names": list(class_names),
        "method": method,
        "k": len(codebook),
        "classifier": classifier,
        "scaler": scaler,
        "split_info": split_info,
        "evaluation": evaluation,
        "visual_word_patches": visual_word_patches,
    }
    joblib.dump(bundle, save_path)


def load_bovw_bundle(filepath: str | Path) -> dict[str, Any]:
    """Load a BoVW model bundle from disk."""
    load_path = Path(filepath)
    if not load_path.is_file():
        raise FileNotFoundError(f"Model file not found: {load_path}")

    bundle = joblib.load(load_path)
    if not isinstance(bundle, dict) or "codebook" not in bundle or "class_names" not in bundle:
        raise ValueError(f"File '{load_path.name}' is not a valid BoVW model archive.")

    bundle["filepath"] = str(load_path)
    return bundle
