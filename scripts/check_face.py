"""Check YuNet detections for one arbitrary image."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from src.inference.face_alignment import align_face
from src.inference.face_detector import FaceDetector, resize_for_detection


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect YuNet face detections in one image")
    parser.add_argument("image", type=Path, help="Path to an image")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/face_detection_yunet_2023mar.onnx"),
        help="Path to the YuNet ONNX model",
    )
    parser.add_argument("--save", type=Path, help="Optional path for an annotated image")
    parser.add_argument(
        "--save-aligned",
        type=Path,
        help="Optional path for the aligned 112x112 face when exactly one face is found",
    )
    args = parser.parse_args()

    image = cv2.imread(str(args.image))
    if image is None:
        print(f"ERROR: could not read image: {args.image}")
        return 1

    detection_image = resize_for_detection(image, max_side=320)
    original_height, original_width = image.shape[:2]
    height, width = detection_image.shape[:2]
    detector = FaceDetector(args.model)
    detector.set_input_size(detection_image)
    detections = detector.detect(detection_image)

    print(f"image: {args.image}")
    print(f"original size: {original_width}x{original_height}")
    print(f"detection input: {width}x{height}")
    print(f"faces detected: {len(detections)}")

    for index, detection in enumerate(detections, start=1):
        print(f"face {index}:")
        print(f"  confidence: {detection.confidence:.4f}")
        print(f"  bbox: {detection.bbox}")
        print(f"  landmarks: {detection.landmarks}")
        x, y, box_width, box_height = detection.bbox
        cv2.rectangle(
            detection_image,
            (x, y),
            (x + box_width, y + box_height),
            (0, 255, 0),
            2,
        )
        for landmark_x, landmark_y in detection.landmarks:
            cv2.circle(detection_image, (landmark_x, landmark_y), 2, (0, 0, 255), -1)

    if args.save:
        if not cv2.imwrite(str(args.save), detection_image):
            print(f"ERROR: could not save annotated image: {args.save}")
            return 1
        print(f"annotated image: {args.save}")

    if args.save_aligned:
        if len(detections) != 1:
            print("ERROR: aligned output requires exactly one detected face")
            return 2
        aligned = align_face(detection_image, detections[0].landmarks)
        if not cv2.imwrite(str(args.save_aligned), aligned):
            print(f"ERROR: could not save aligned image: {args.save_aligned}")
            return 1
        print(f"aligned image: {args.save_aligned}")

    return 0 if detections else 2


if __name__ == "__main__":
    raise SystemExit(main())
