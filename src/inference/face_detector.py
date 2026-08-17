"""YuNet face detection through OpenCV's FaceDetectorYN API."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2


@dataclass(frozen=True)
class FaceDetection:
    """A detected face with its box, confidence, and five landmarks."""

    bbox: tuple[int, int, int, int]
    confidence: float
    landmarks: tuple[tuple[int, int], ...]
    raw_detection: tuple[float, ...] = field(repr=False)


class FaceDetector:
    """Detect faces with the YuNet model bundled as an external ONNX file."""

    def __init__(
        self,
        model_path: str | Path,
        confidence_threshold: float = 0.9,
        nms_threshold: float = 0.3,
        top_k: int = 5000,
    ) -> None:
        model = Path(model_path)
        if not model.is_file():
            raise FileNotFoundError(
                f"YuNet model file was not found: {model}. "
                "Download the model and pass its path to FaceDetector."
            )

        create_detector = getattr(cv2, "FaceDetectorYN", None)
        if create_detector is not None and hasattr(create_detector, "create"):
            create = create_detector.create
        else:
            create = getattr(cv2, "FaceDetectorYN_create", None)

        if create is None:
            raise RuntimeError(
                "This OpenCV build does not provide FaceDetectorYN. "
                "Install a compatible opencv-python version."
            )

        try:
            self._detector = create(
                str(model),
                "",
                (320, 320),
                confidence_threshold,
                nms_threshold,
                top_k,
            )
        except cv2.error as error:
            raise RuntimeError(
                f"Could not initialize YuNet model: {model}. "
                f"OpenCV {cv2.__version__} may be too old; install "
                "opencv-python>=4.10.0."
            ) from error

    def detect(self, frame) -> list[FaceDetection]:
        """Detect faces in a BGR frame and return normalized detection records."""
        if frame is None or getattr(frame, "ndim", 0) != 3:
            raise ValueError("detect() expects a valid color image frame")

        height, width = frame.shape[:2]
        self._detector.setInputSize((width, height))
        try:
            _, detections = self._detector.detect(frame)
        except cv2.error as error:
            raise RuntimeError(
                "YuNet face detection failed. "
                f"OpenCV {cv2.__version__} may be incompatible; install "
                "opencv-python>=4.10.0."
            ) from error

        if detections is None:
            return []

        results: list[FaceDetection] = []
        for detection in detections:
            x, y, box_width, box_height = detection[:4]
            landmarks = tuple(
                (int(detection[index]), int(detection[index + 1]))
                for index in range(4, 14, 2)
            )
            results.append(
                FaceDetection(
                    bbox=(int(x), int(y), int(box_width), int(box_height)),
                    confidence=float(detection[14]),
                    landmarks=landmarks,
                    raw_detection=tuple(float(value) for value in detection),
                )
            )
        return results
