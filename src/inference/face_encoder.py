"""ArcFace R50 embedding encoder through ONNX Runtime."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


class FaceEncoder:
    """Encode an aligned face with a single ArcFace ONNX model."""

    input_size = (112, 112)

    def __init__(self, model_path: str | Path) -> None:
        model = Path(model_path)
        if not model.is_file():
            raise FileNotFoundError(
                f"ArcFace model file was not found: {model}. "
                "Download the ArcFace R50 ONNX model before building the gallery."
            )

        try:
            self._session = ort.InferenceSession(
                str(model), providers=["CPUExecutionProvider"]
            )
        except (RuntimeError, ValueError) as error:
            raise RuntimeError(f"Could not initialize ArcFace model: {model}") from error

        inputs = self._session.get_inputs()
        outputs = self._session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError("ArcFace model must have exactly one input and one output")
        input_shape = inputs[0].shape
        if input_shape[1:] != [3, 112, 112] and tuple(input_shape[1:]) != (3, 112, 112):
            raise ValueError(f"Expected ArcFace input shape [N, 3, 112, 112], got {input_shape}")

        self._input_name = inputs[0].name
        self._output_name = outputs[0].name
        self.model_path = model
        self.model_size_bytes = model.stat().st_size
        self.embedding_dimension = int(outputs[0].shape[-1])

    def encode(self, aligned_face: np.ndarray) -> np.ndarray:
        """Return the ArcFace embedding for a 112x112 aligned BGR face."""
        if aligned_face.shape[:2] != self.input_size:
            raise ValueError(
                f"Expected an aligned face of {self.input_size}, "
                f"got {aligned_face.shape[:2]}"
            )
        if aligned_face.ndim != 3 or aligned_face.shape[2] != 3:
            raise ValueError("Expected a 3-channel BGR aligned face")

        blob = cv2.dnn.blobFromImage(
            aligned_face,
            scalefactor=1.0 / 127.5,
            size=self.input_size,
            mean=(127.5, 127.5, 127.5),
            swapRB=True,
        )
        try:
            embedding = self._session.run([self._output_name], {self._input_name: blob})[0]
        except (RuntimeError, ValueError) as error:
            raise RuntimeError("ArcFace embedding extraction failed") from error

        result = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if result.size != self.embedding_dimension or not np.isfinite(result).all():
            raise ValueError("ArcFace returned an invalid embedding")
        return result
