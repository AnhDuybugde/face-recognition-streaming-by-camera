"""Light identity-preserving variants for an already aligned face."""

from __future__ import annotations

import cv2
import numpy as np


def _scale_intensity(image: np.ndarray, scale: float) -> np.ndarray:
    """Scale pixel intensity while preserving uint8 image bounds."""
    return np.clip(image.astype(np.float32) * scale, 0, 255).astype(np.uint8)


def generate_face_variants(aligned_face: np.ndarray) -> list[np.ndarray]:
    """Return five mild variants of one 112x112 aligned BGR face."""
    if aligned_face.ndim != 3 or aligned_face.dtype != np.uint8:
        raise ValueError("aligned_face must be a uint8 color image")

    darker = _scale_intensity(aligned_face, 0.90)
    brighter = _scale_intensity(aligned_face, 1.10)
    contrast = cv2.convertScaleAbs(aligned_face, alpha=1.10, beta=-12)
    return [
        aligned_face.copy(),
        cv2.flip(aligned_face, 1),
        darker,
        brighter,
        contrast,
    ]
