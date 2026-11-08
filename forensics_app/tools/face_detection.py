"""Face detection using OpenCV YuNet (Set 6.1).

This module implements:
  - Fast, accurate deep learning face detection using OpenCV's FaceDetectorYN (YuNet).
  - Bounding box localization, 5 facial landmark detection, and confidence scoring.
  - Zero, single, and multi-face handling.
  - Fallback and missing model weights handling.
"""

from __future__ import annotations

import os
from pathlib import Path
import tkinter as tk
from typing import Any

import cv2
import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
from .dialogs import ask_image_file, show_error


# Default model weights path
DEFAULT_MODEL_FILENAME = "face_detection_yunet_2023mar.onnx"


def find_yunet_model(custom_path: str | Path | None = None) -> Path | None:
    """Locate the YuNet ONNX model file across known project directories."""
    if custom_path:
        p = Path(custom_path)
        if p.is_file():
            return p
        return None

    candidates = [
        Path(__file__).resolve().parent.parent / DEFAULT_MODEL_FILENAME,
        Path(__file__).resolve().parent.parent.parent / DEFAULT_MODEL_FILENAME,
        Path.cwd() / "forensics_app" / DEFAULT_MODEL_FILENAME,
        Path.cwd() / DEFAULT_MODEL_FILENAME,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    return None


def detect_faces_yunet(
    image: Image.Image | np.ndarray,
    model_path: str | Path | None = None,
    score_threshold: float = 0.6,
    nms_threshold: float = 0.3,
    draw_landmarks: bool = True,
) -> tuple[Image.Image, list[dict[str, Any]], dict[str, Any]]:
    """Detect faces using OpenCV YuNet FaceDetectorYN.

    Parameters
    ----------
    image : Image.Image | np.ndarray
        Input image in RGB or BGR format.
    model_path : str | Path | None
        Path to the YuNet ONNX weights. If None, searched automatically.
    score_threshold : float
        Detection score threshold (default 0.6).
    nms_threshold : float
        Non-maximum suppression threshold (default 0.3).
    draw_landmarks : bool
        Whether to draw 5 facial landmarks (eyes, nose, mouth corners).

    Returns
    -------
    tuple[Image.Image, list[dict[str, Any]], dict[str, Any]]
        Annotated PIL Image, list of face dictionaries, and details dictionary.
    """
    model_file = find_yunet_model(model_path)
    if model_file is None or not model_file.is_file():
        raise FileNotFoundError(
            f"YuNet ONNX model weights not found ('{DEFAULT_MODEL_FILENAME}'). "
            "Please ensure the ONNX model file is placed in 'forensics_app/'."
        )

    # Convert input to numpy BGR
    if isinstance(image, Image.Image):
        rgb_arr = np.array(image.convert("RGB"))
        bgr = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)
    elif isinstance(image, np.ndarray):
        if image.size == 0 or image.ndim < 2:
            raise ValueError("Input image is empty or has invalid dimensions.")
        if image.ndim == 2:
            bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.ndim == 3:
            if image.shape[2] == 4:
                bgr = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
            elif image.shape[2] == 3:
                # Assume RGB if coming from PIL or standard loader
                bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            else:
                raise ValueError(f"Unsupported number of channels: {image.shape[2]}")
        else:
            raise ValueError(f"Invalid image array shape: {image.shape}")
    else:
        raise TypeError(f"Expected PIL.Image or numpy.ndarray, got {type(image)}")

    h, w = bgr.shape[:2]
    if h == 0 or w == 0:
        raise ValueError("Image height or width is zero.")

    # Create YuNet FaceDetectorYN
    detector = cv2.FaceDetectorYN.create(
        model=str(model_file),
        config="",
        input_size=(w, h),
        score_threshold=float(score_threshold),
        nms_threshold=float(nms_threshold),
        top_k=5000,
    )
    detector.setInputSize((w, h))

    retval, raw_faces = detector.detect(bgr)

    annotated_bgr = bgr.copy()
    faces_list: list[dict[str, Any]] = []
    details: dict[str, Any] = {
        "Detector": "OpenCV YuNet (FaceDetectorYN)",
        "Image size (WxH)": f"{w} × {h}",
        "Score threshold": f"{score_threshold:.2f}",
        "NMS threshold": f"{nms_threshold:.2f}",
    }

    if raw_faces is None or len(raw_faces) == 0:
        details["Detected faces"] = 0
        details["Status"] = "No faces detected"
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(annotated_rgb), faces_list, details

    num_faces = len(raw_faces)
    details["Detected faces"] = num_faces

    # Landmark colors (BGR): right eye (cyan), left eye (cyan), nose (yellow), right mouth (magenta), left mouth (magenta)
    landmark_colors = [
        (255, 255, 0),  # Right eye
        (255, 255, 0),  # Left eye
        (0, 255, 255),  # Nose tip
        (255, 0, 255),  # Right mouth corner
        (255, 0, 255),  # Left mouth corner
    ]

    for idx, face in enumerate(raw_faces):
        x = max(0, int(round(face[0])))
        y = max(0, int(round(face[1])))
        bw = max(1, int(round(face[2])))
        bh = max(1, int(round(face[3])))
        score = float(face[14])

        # Clamp right/bottom coordinates
        bw = min(bw, w - x)
        bh = min(bh, h - y)

        landmarks: list[tuple[int, int]] = []
        for lm_i in range(5):
            lx = int(round(face[4 + 2 * lm_i]))
            ly = int(round(face[5 + 2 * lm_i]))
            landmarks.append((lx, ly))

        face_entry = {
            "index": idx + 1,
            "bbox": (x, y, bw, bh),
            "score": score,
            "landmarks": landmarks,
        }
        faces_list.append(face_entry)

        details[f"Face {idx + 1}"] = f"Box: [{x}, {y}, {bw}, {bh}], Score: {score:.3f}"

        # Draw bounding box (vibrant green, 2px)
        cv2.rectangle(annotated_bgr, (x, y), (x + bw, y + bh), (0, 255, 0), 2)

        # Draw landmarks if requested
        if draw_landmarks:
            for (lx, ly), color in zip(landmarks, landmark_colors):
                if 0 <= lx < w and 0 <= ly < h:
                    cv2.circle(annotated_bgr, (lx, ly), 3, color, -1)
                    cv2.circle(annotated_bgr, (lx, ly), 3, (0, 0, 0), 1)

        # Draw label badge with background
        label = f"Face {idx + 1}: {score:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)

        label_y = max(y - 6, th + 4)
        cv2.rectangle(
            annotated_bgr,
            (x, label_y - th - 3),
            (x + tw + 6, label_y + baseline),
            (0, 180, 0),
            -1,
        )
        cv2.putText(
            annotated_bgr,
            label,
            (x + 3, label_y),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(annotated_rgb), faces_list, details


class FaceDetectionTool(ForensicsTool):
    """Set 6.1: Detect faces using OpenCV YuNet deep learning model."""

    tool_id = "face_detection"
    title = "Face detection (YuNet)"
    category = "Set 6: Face Detection & Recognition"
    description = "Detect faces and facial landmarks using OpenCV YuNet (FaceDetectorYN)."
    requires_image = False

    def run(self, parent: tk.Misc | None, document: ImageDocument) -> ToolResult | None:
        target_image: Image.Image | None = None
        source_name = "current image"

        if document.is_loaded and document.current is not None:
            target_image = document.current
            if document.path:
                source_name = document.path.name
        else:
            path = ask_image_file(parent, title="Select image for face detection")
            if not path:
                return None
            try:
                with Image.open(path) as opened:
                    target_image = opened.copy()
                source_name = Path(path).name
            except Exception as err:
                if parent is not None:
                    show_error(parent, "Open Error", f"Could not open image: {err}")
                return None

        # Check model file
        model_file = find_yunet_model()
        if model_file is None:
            if parent is not None:
                show_error(
                    parent,
                    "Model Weights Missing",
                    f"Could not find '{DEFAULT_MODEL_FILENAME}'.\n"
                    "Please ensure the YuNet ONNX file is located in the forensics_app folder.",
                )
            return None

        try:
            annotated, faces, details = detect_faces_yunet(
                target_image,
                model_path=model_file,
                score_threshold=0.6,
                nms_threshold=0.3,
                draw_landmarks=True,
            )
        except Exception as err:
            if parent is not None:
                show_error(parent, "Face Detection Error", str(err))
            return None

        count = len(faces)
        details["Image Source"] = source_name
        message = (
            f"Face detection complete: found {count} face(s) in {source_name}."
            if count > 0
            else f"Face detection complete: 0 faces found in {source_name}."
        )

        return ToolResult(
            image=annotated,
            message=message,
            details=details,
        )
