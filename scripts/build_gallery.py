"""Build a normalized SFace embedding gallery from enrollment images."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np

from src.inference.face_alignment import align_face
from src.inference.face_detector import FaceDetector, resize_for_detection
from src.inference.face_encoder import FaceEncoder


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass
class EnrollmentResult:
    identity: str
    embedding: np.ndarray | None
    reason: str | None = None
    detection_ms: float = 0.0
    encoding_ms: float = 0.0


def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    """Return an L2-normalized finite embedding."""
    norm = float(np.linalg.norm(embedding))
    if not np.isfinite(norm) or norm == 0.0:
        raise ValueError("embedding has an invalid L2 norm")
    return (embedding / norm).astype(np.float32)


def process_image(
    image_path: Path,
    detector: FaceDetector,
    encoder: FaceEncoder,
) -> EnrollmentResult:
    """Process one enrollment image and return an explicit result."""
    identity = image_path.stem
    image = cv2.imread(str(image_path))
    if image is None:
        return EnrollmentResult(identity, None, "unreadable image")

    detection_image = resize_for_detection(image, max_side=320)
    detection_started = time.perf_counter()
    try:
        detector.set_input_size(detection_image)
        faces = detector.detect(detection_image)
    except (RuntimeError, ValueError) as error:
        return EnrollmentResult(identity, None, f"detection failure: {error}")
    detection_ms = (time.perf_counter() - detection_started) * 1000

    if not faces:
        return EnrollmentResult(identity, None, "no face detected", detection_ms)
    if len(faces) > 1:
        return EnrollmentResult(identity, None, "multiple faces detected", detection_ms)

    encoding_started = time.perf_counter()
    try:
        aligned = align_face(detection_image, faces[0].landmarks)
        embedding = normalize_embedding(encoder.encode(aligned))
    except (RuntimeError, ValueError) as error:
        return EnrollmentResult(
            identity,
            None,
            f"encoding/alignment failure: {error}",
            detection_ms,
            (time.perf_counter() - encoding_started) * 1000,
        )

    return EnrollmentResult(
        identity,
        embedding,
        detection_ms=detection_ms,
        encoding_ms=(time.perf_counter() - encoding_started) * 1000,
    )


def validate_gallery(identities: list[str], embeddings: np.ndarray) -> None:
    """Validate gallery shape, finiteness, dimensions, and L2 norms."""
    if embeddings.ndim != 2 or embeddings.shape[0] != len(identities):
        raise ValueError("gallery identities and embeddings have incompatible shapes")
    if embeddings.shape[0] == 0 or not np.isfinite(embeddings).all():
        raise ValueError("gallery is empty or contains non-finite values")
    norms = np.linalg.norm(embeddings, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-5):
        raise ValueError("gallery embeddings are not L2-normalized")


def similarity_report(identities: list[str], embeddings: np.ndarray) -> str:
    """Return inter-identity cosine similarity statistics, excluding the diagonal."""
    pairs = []
    for left, right in combinations(range(len(identities)), 2):
        pairs.append((float(np.dot(embeddings[left], embeddings[right])), left, right))
    if not pairs:
        return "inter-identity similarity: unavailable (fewer than two identities)"

    similarities = np.array([pair[0] for pair in pairs], dtype=np.float32)
    maximum = max(pairs)
    return (
        "inter-identity cosine similarity: "
        f"mean={similarities.mean():.4f}, "
        f"median={np.median(similarities):.4f}, "
        f"min={similarities.min():.4f}, "
        f"max={similarities.max():.4f} "
        f"({identities[maximum[1]]} <-> {identities[maximum[2]]})"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/input"))
    parser.add_argument("--output", type=Path, default=Path("data/gallery/embeddings.npz"))
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
    args = parser.parse_args()

    image_paths = sorted(
        path for path in args.input.iterdir() if path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not image_paths:
        raise RuntimeError(f"No supported enrollment images found in {args.input}")

    detector = FaceDetector(args.detector_model)
    encoder = FaceEncoder(args.encoder_model)
    processing_started = time.perf_counter()
    results = [process_image(path, detector, encoder) for path in image_paths]
    processing_ms = (time.perf_counter() - processing_started) * 1000
    successful = [result for result in results if result.embedding is not None]
    failed = [result for result in results if result.embedding is None]

    for result in failed:
        print(f"FAIL {result.identity}: {result.reason}")
    for result in successful:
        print(
            f"OK {result.identity}: detection={result.detection_ms:.2f} ms, "
            f"encoding={result.encoding_ms:.2f} ms"
        )

    if not successful:
        raise RuntimeError("No valid enrollments; refusing to write an empty gallery")

    identities = [result.identity for result in successful]
    embeddings = np.vstack([result.embedding for result in successful]).astype(np.float32)
    validate_gallery(identities, embeddings)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, identities=np.asarray(identities), embeddings=embeddings)

    detection_times = [result.detection_ms for result in successful]
    encoding_times = [result.encoding_ms for result in successful]
    print(f"total images: {len(results)}")
    print(f"successful enrollments: {len(successful)}")
    print(f"failed enrollments: {len(failed)}")
    print(f"gallery: {args.output}")
    print(f"embedding dimension: {embeddings.shape[1]}")
    print(f"model size: {encoder.model_size_bytes / (1024 * 1024):.2f} MiB")
    print(f"average YuNet detection time: {np.mean(detection_times):.2f} ms")
    print(f"average encoding latency: {np.mean(encoding_times):.2f} ms")
    print(f"total enrollment processing time: {processing_ms:.2f} ms")
    print(similarity_report(identities, embeddings))


if __name__ == "__main__":
    main()
