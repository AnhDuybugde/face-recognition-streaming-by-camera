"""Display YuNet face detections from the local webcam."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

from src.capture.video_source import VideoSource
from src.inference.face_detector import FaceDetector


def draw_detections(frame, detections) -> None:
    """Draw face boxes, confidence values, and five-point landmarks."""
    for detection in detections:
        x, y, width, height = detection.bbox
        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"face {detection.confidence:.2f}",
            (x, max(y - 8, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
        for landmark_x, landmark_y in detection.landmarks:
            cv2.circle(frame, (landmark_x, landmark_y), 2, (0, 0, 255), -1)


def main() -> None:
    """Run face detection until the user presses ``q``."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/face_detection_yunet_2023mar.onnx"),
        help="Path to the external YuNet ONNX model",
    )
    args = parser.parse_args()

    detector = FaceDetector(args.model)
    source = None
    frame_count = 0
    started_at = time.perf_counter()
    try:
        source = VideoSource(0)
        print("detector: YuNet")
        print(f"frame size: {source.width:.0f}x{source.height:.0f}")

        while True:
            frame = source.read()
            if frame is None:
                continue

            detection_started_at = time.perf_counter()
            detections = detector.detect(frame)
            detection_latency_ms = (time.perf_counter() - detection_started_at) * 1000
            draw_detections(frame, detections)
            cv2.imshow("Face Detection", frame)
            frame_count += 1

            if frame_count % 30 == 0:
                elapsed = time.perf_counter() - started_at
                fps = frame_count / elapsed if elapsed else 0.0
                print(
                    f"detected faces: {len(detections)} | "
                    f"detection latency: {detection_latency_ms:.2f} ms | "
                    f"approximate FPS: {fps:.2f}"
                )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        if source is not None:
            source.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
