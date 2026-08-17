"""Run the live webcam recognition baseline and print observation metrics."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np

from src.capture.video_source import VideoSource
from src.inference.face_alignment import align_face
from src.inference.face_detector import FaceDetector
from src.inference.face_encoder import FaceEncoder
from src.inference.face_matcher import FaceMatcher, MatchResult


def draw_match(frame: np.ndarray, detection, result: MatchResult) -> None:
    """Draw a minimal face box and top-2 match summary on the frame."""
    x, y, width, height = detection.bbox
    cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 220, 0), 2)
    lines = (
        f"Top1: {result.top1_identity} {result.top1_similarity:.2f}",
        f"Top2: {result.top2_identity} {result.top2_similarity:.2f}",
        f"Margin: {result.margin:.2f}",
    )
    for offset, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (x, max(20, y - 45 + offset * 16)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 220, 0),
            1,
            cv2.LINE_AA,
        )


def write_observations(path: Path, observations: list[dict[str, object]]) -> None:
    """Write only numeric/identity observations; raw face images are not stored."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=observations[0].keys())
        writer.writeheader()
        writer.writerows(observations)


def print_observation_summary(observations: list[dict[str, object]], expected: str) -> None:
    """Print genuine identity and margin statistics for an observation session."""
    if not observations:
        print("observation samples: 0")
        return
    top1 = np.asarray([row["top1_similarity"] for row in observations], dtype=np.float32)
    margins = np.asarray([row["margin"] for row in observations], dtype=np.float32)
    correct = sum(row["top1_identity"] == expected for row in observations)
    print(f"observation samples: {len(observations)}")
    print(f"top-1 expected matches: {correct}/{len(observations)}")
    print(f"top-1 similarity mean/min/max: {top1.mean():.4f}/{top1.min():.4f}/{top1.max():.4f}")
    print(f"margin mean/min: {margins.mean():.4f}/{margins.min():.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=int, default=0)
    parser.add_argument("--gallery", type=Path, default=Path("data/gallery/embeddings.npz"))
    parser.add_argument(
        "--detector-model",
        type=Path,
        default=Path("models/face_detection_yunet_2023mar.onnx"),
    )
    parser.add_argument(
        "--encoder-model",
        type=Path,
        default=Path("models/face_recognition_sface_2021dec.onnx"),
    )
    parser.add_argument("--expected-identity", help="Enable genuine observation statistics")
    parser.add_argument("--observation-output", type=Path)
    parser.add_argument("--max-frames", type=int, help="Stop after this many frames")
    parser.add_argument("--no-display", action="store_true", help="Run without an OpenCV window")
    args = parser.parse_args()

    matcher = FaceMatcher(args.gallery)
    detector = FaceDetector(args.detector_model)
    encoder = FaceEncoder(args.encoder_model)
    if encoder.embedding_dimension not in (None, matcher.embedding_dimension):
        raise ValueError("Encoder and gallery embedding dimensions do not match")

    source = None
    observations: list[dict[str, object]] = []
    frame_count = 0
    total_frame_ms = 0.0
    detection_times: list[float] = []
    encoding_times: list[float] = []
    matching_times: list[float] = []
    try:
        source = VideoSource(args.source)
        print(f"Webcam: {source.width:.0f}x{source.height:.0f}, reported FPS={source.fps:.2f}")
        print(f"Gallery: {len(matcher.identities)} identities x {matcher.embedding_dimension} dimensions")
        print("Press q to stop observation.")
        while True:
            started = time.perf_counter()
            frame = source.read()
            if frame is None:
                continue

            detection_started = time.perf_counter()
            detections = detector.detect(frame)
            detection_ms = (time.perf_counter() - detection_started) * 1000
            detection_times.append(detection_ms)

            for detection in detections:
                encoding_started = time.perf_counter()
                aligned = align_face(frame, detection.landmarks)
                embedding = encoder.encode(aligned)
                encoding_times.append((time.perf_counter() - encoding_started) * 1000)

                matching_started = time.perf_counter()
                result = matcher.match(embedding)
                matching_times.append((time.perf_counter() - matching_started) * 1000)
                draw_match(frame, detection, result)
                if args.expected_identity:
                    observations.append(
                        {
                            "expected_identity": args.expected_identity,
                            "top1_identity": result.top1_identity,
                            "top1_similarity": result.top1_similarity,
                            "top2_identity": result.top2_identity,
                            "top2_similarity": result.top2_similarity,
                            "margin": result.margin,
                        }
                    )

            frame_ms = (time.perf_counter() - started) * 1000
            total_frame_ms += frame_ms
            frame_count += 1
            if not args.no_display:
                cv2.putText(
                    frame,
                    f"FPS: {1000.0 / frame_ms:.1f} | det: {detection_ms:.1f} ms",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 220, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow("Face Recognition Baseline", frame)
            if args.max_frames and frame_count >= args.max_frames:
                break
            if not args.no_display and cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        if source is not None:
            source.release()
        cv2.destroyAllWindows()

    print(f"processed frames: {frame_count}")
    if frame_count:
        print(f"average end-to-end FPS: {1000.0 / (total_frame_ms / frame_count):.2f}")
    if detection_times:
        print(f"average detection latency: {np.mean(detection_times):.2f} ms/frame")
    if encoding_times:
        print(f"average encoding latency: {np.mean(encoding_times):.2f} ms/face")
    if matching_times:
        print(f"average matching latency: {np.mean(matching_times):.4f} ms/face")
    if args.expected_identity:
        print_observation_summary(observations, args.expected_identity)
        if args.observation_output and observations:
            write_observations(args.observation_output, observations)
            print(f"observations: {args.observation_output}")


if __name__ == "__main__":
    main()
