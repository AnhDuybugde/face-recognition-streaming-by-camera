"""Webcam frame acquisition using OpenCV."""

from __future__ import annotations

from typing import Any

import cv2


class VideoSource:
    """Open and read frames from a local video source."""

    def __init__(self, source: int = 0, max_consecutive_failures: int = 3) -> None:
        if max_consecutive_failures < 1:
            raise ValueError("max_consecutive_failures must be at least 1")

        self._capture = cv2.VideoCapture(source)
        self._max_consecutive_failures = max_consecutive_failures
        self._consecutive_failures = 0

        if not self._capture.isOpened():
            self._capture.release()
            raise RuntimeError(f"Could not open video source: {source}")

    @property
    def width(self) -> float:
        """Return the width reported by the capture device."""
        return self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)

    @property
    def height(self) -> float:
        """Return the height reported by the capture device."""
        return self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT)

    @property
    def fps(self) -> float:
        """Return the FPS reported by the capture device."""
        return self._capture.get(cv2.CAP_PROP_FPS)

    def read(self) -> Any:
        """Read and return the next frame, or raise after repeated failures."""
        success, frame = self._capture.read()
        if success:
            self._consecutive_failures = 0
            return frame

        self._consecutive_failures += 1
        if self._consecutive_failures >= self._max_consecutive_failures:
            raise RuntimeError(
                "Failed to read frames from the video source "
                f"{self._consecutive_failures} times consecutively"
            )

        return None

    def release(self) -> None:
        """Release the underlying video capture resource."""
        self._capture.release()
