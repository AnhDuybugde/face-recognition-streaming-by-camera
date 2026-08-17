"""Run deterministic face recognition on one image for local/container tests."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from src.inference.face_alignment import align_face
from src.inference.face_detector import FaceDetector
from src.inference.face_encoder import FaceEncoder
from src.inference.face_matcher import FaceMatcher


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--gallery", type=Path, default=Path("data/gallery/embeddings.npz"))
    parser.add_argument("--detector-model", type=Path, default=Path("models/face_detection_yunet_2023mar.onnx"))
    parser.add_argument("--encoder-model", type=Path, default=Path("models/face_recognition_arcface_r50.onnx"))
    args = parser.parse_args()
    if args.iterations < 1:
        raise ValueError("--iterations must be at least 1")

    image = cv2.imread(str(args.input))
    if image is None:
        raise FileNotFoundError(f"Input image could not be read: {args.input}")
    detector = FaceDetector(args.detector_model)
    encoder = FaceEncoder(args.encoder_model)
    matcher = FaceMatcher(args.gallery)
    if encoder.embedding_dimension != matcher.embedding_dimension:
        raise ValueError("Encoder and gallery embedding dimensions do not match")

    detection_times: list[float] = []
    encoding_times: list[float] = []
    end_to_end_times: list[float] = []
    last_match = None
    for _ in range(args.iterations):
        started = time.perf_counter()
        detection_started = time.perf_counter()
        detections = detector.detect(image)
        detection_times.append((time.perf_counter() - detection_started) * 1000)
        if len(detections) != 1:
            raise RuntimeError(f"Expected exactly one face, detected {len(detections)}")
        encoding_started = time.perf_counter()
        aligned = align_face(image, detections[0].landmarks)
        embedding = encoder.encode(aligned)
        encoding_times.append((time.perf_counter() - encoding_started) * 1000)
        last_match = matcher.match(embedding)
        end_to_end_times.append((time.perf_counter() - started) * 1000)

    print(f"input: {args.input} ({image.shape[1]}x{image.shape[0]})")
    print(f"iterations: {args.iterations}")
    print(f"match: {last_match.top1_identity} {last_match.top1_similarity:.4f}")
    print(f"average detection latency: {np.mean(detection_times):.2f} ms")
    print(f"average encoding latency: {np.mean(encoding_times):.2f} ms")
    average_end_to_end = float(np.mean(end_to_end_times))
    print(f"average end-to-end latency: {average_end_to_end:.2f} ms")
    print(f"approximate pipeline FPS: {1000.0 / average_end_to_end:.2f}")


if __name__ == "__main__":
    main()
