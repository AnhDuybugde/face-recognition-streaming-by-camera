"""OpenCV SFace embedding encoder."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class FaceEncoder:
    """Encode an already aligned face into a floating-point embedding."""

    input_size = (112, 112)

    def __init__(self, model_path: str | Path) -> None:
        model = Path(model_path)
        if not model.is_file():
            raise FileNotFoundError(
                f"SFace model file was not found: {model}. "
                "Download the model before building the gallery."
            )

        recognizer_type = getattr(cv2, "FaceRecognizerSF", None)
        if recognizer_type is not None and hasattr(recognizer_type, "create"):
            create = recognizer_type.create
        else:
            create = getattr(cv2, "FaceRecognizerSF_create", None)
        if create is None:
            raise RuntimeError(
                "This OpenCV build does not provide FaceRecognizerSF. "
                "Install a compatible opencv-python version."
            )

        try:
            self._recognizer = create(str(model), "")
        except cv2.error as error:
            raise RuntimeError(f"Could not initialize SFace model: {model}") from error

        self.model_path = model
        self.model_size_bytes = model.stat().st_size
        self.embedding_dimension: int | None = None

    def encode(self, aligned_face: np.ndarray) -> np.ndarray:
        """Return the encoder output for a 112x112 aligned BGR face."""
        if aligned_face.shape[:2] != self.input_size:
            raise ValueError(
                f"Expected an aligned face of {self.input_size}, "
                f"got {aligned_face.shape[:2]}"
            )

        try:
            embedding = self._recognizer.feature(aligned_face)
        except cv2.error as error:
            raise RuntimeError("SFace embedding extraction failed") from error

        result = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if result.size == 0 or not np.isfinite(result).all():
            raise ValueError("SFace returned an empty or non-finite embedding")
        self.embedding_dimension = int(result.size)
        return result
