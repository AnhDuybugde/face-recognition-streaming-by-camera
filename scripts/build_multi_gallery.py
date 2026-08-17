"""Build and analyze a light-augmentation multi-embedding gallery."""

from __future__ import annotations

import argparse
import shutil
import time
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np

from scripts.build_gallery import SUPPORTED_EXTENSIONS, normalize_embedding
from src.inference.face_alignment import align_face
from src.inference.face_detector import FaceDetector, resize_for_detection
from src.inference.face_encoder import FaceEncoder
from src.inference.face_matcher import FaceMatcher, MultiGalleryMatcher
from src.preprocessing.face_augmentation import generate_face_variants


def build_multi_gallery(
    image_paths: list[Path], detector: FaceDetector, encoder: FaceEncoder
) -> tuple[list[str], np.ndarray, float, float]:
    """Encode five variants for every valid enrollment image."""
    identities: list[str] = []
    all_embeddings: list[np.ndarray] = []
    started = time.perf_counter()
    encoding_total_ms = 0.0
    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"Could not read enrollment image: {image_path}")
        detection_image = resize_for_detection(image, max_side=320)
        detector.set_input_size(detection_image)
        faces = detector.detect(detection_image)
        if len(faces) != 1:
            raise RuntimeError(f"Expected exactly one face in {image_path}, found {len(faces)}")

        aligned = align_face(detection_image, faces[0].landmarks)
        variants = generate_face_variants(aligned)
        identity_embeddings = []
        for variant in variants:
            encoding_started = time.perf_counter()
            identity_embeddings.append(normalize_embedding(encoder.encode(variant)))
            encoding_total_ms += (time.perf_counter() - encoding_started) * 1000
        identities.append(image_path.stem)
        all_embeddings.append(np.vstack(identity_embeddings))

    return identities, np.stack(all_embeddings), (time.perf_counter() - started) * 1000, encoding_total_ms


def validate_multi_gallery(identities: list[str], embeddings: np.ndarray) -> None:
    """Validate the explicit identity/variant/embedding dimensions."""
    if embeddings.ndim != 3 or embeddings.shape[0] != len(identities):
        raise ValueError("Multi-gallery identities and embeddings have incompatible shapes")
    if embeddings.shape[1] != 5 or embeddings.shape[2] == 0:
        raise ValueError(f"Expected multi-gallery shape (N, 5, D), got {embeddings.shape}")
    if not np.isfinite(embeddings).all():
        raise ValueError("Multi-gallery contains non-finite values")
    if not np.allclose(np.linalg.norm(embeddings, axis=2), 1.0, atol=1e-5):
        raise ValueError("Multi-gallery embeddings are not L2-normalized")


def inter_identity_report(matcher: MultiGalleryMatcher, strategy: str) -> str:
    """Report conservative cross-identity scores for each aggregation strategy."""
    pair_scores: list[tuple[float, int, int]] = []
    for left, right in combinations(range(len(matcher.identities)), 2):
        left_queries = matcher.embeddings[left]
        target = matcher.embeddings[right]
        similarities = left_queries @ target.T
        if strategy == "max":
            scores = similarities.max(axis=1)
        else:
            scores = np.sort(similarities, axis=1)[:, -2:].mean(axis=1)
        pair_scores.append((float(scores.max()), left, right))

    values = np.asarray([score[0] for score in pair_scores], dtype=np.float32)
    maximum = max(pair_scores)
    return (
        f"{strategy} inter-identity: mean={values.mean():.4f}, "
        f"median={np.median(values):.4f}, min={values.min():.4f}, "
        f"max={values.max():.4f} "
        f"({matcher.identities[maximum[1]]} <-> {matcher.identities[maximum[2]]})"
    )


def benchmark_matching(single: FaceMatcher, multi: MultiGalleryMatcher) -> dict[str, float]:
    """Measure matching latency for the same enrolled query set."""
    results: dict[str, float] = {}
    for name, operation in (
        ("single", lambda query: single.match(query)),
        ("multi_max", lambda query: multi.match(query, "max")),
        ("multi_top2_mean", lambda query: multi.match(query, "top2_mean")),
    ):
        started = time.perf_counter()
        for query in single.embeddings:
            operation(query)
        results[name] = (time.perf_counter() - started) * 1000 / len(single.embeddings)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/input"))
    parser.add_argument("--single-gallery", type=Path, default=Path("data/gallery/embeddings.npz"))
    parser.add_argument("--single-copy", type=Path, default=Path("data/gallery/embeddings_single.npz"))
    parser.add_argument("--multi-gallery", type=Path, default=Path("data/gallery/embeddings_multi.npz"))
    parser.add_argument("--detector-model", type=Path, default=Path("models/face_detection_yunet_2023mar.onnx"))
    parser.add_argument("--encoder-model", type=Path, default=Path("models/face_recognition_arcface_r50.onnx"))
    args = parser.parse_args()

    image_paths = sorted(path for path in args.input.iterdir() if path.suffix.lower() in SUPPORTED_EXTENSIONS)
    if not image_paths:
        raise RuntimeError(f"No supported enrollment images found in {args.input}")
    if not args.single_gallery.is_file():
        raise FileNotFoundError(f"Baseline gallery was not found: {args.single_gallery}")

    args.single_copy.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.single_gallery, args.single_copy)
    detector = FaceDetector(args.detector_model)
    encoder = FaceEncoder(args.encoder_model)
    identities, embeddings, build_ms, encoding_ms = build_multi_gallery(image_paths, detector, encoder)
    validate_multi_gallery(identities, embeddings)
    args.multi_gallery.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.multi_gallery, identities=np.asarray(identities), embeddings=embeddings)

    single = FaceMatcher(args.single_gallery)
    multi = MultiGalleryMatcher(args.multi_gallery)
    matching_ms = benchmark_matching(single, multi)
    print(f"identities: {len(identities)}")
    print(f"variants per identity: {embeddings.shape[1]}")
    print(f"multi embeddings: {embeddings.shape[0]} x {embeddings.shape[1]} x {embeddings.shape[2]}")
    print(f"single gallery size: {args.single_gallery.stat().st_size} bytes")
    print(f"multi gallery size: {args.multi_gallery.stat().st_size} bytes")
    print(f"multi build time: {build_ms:.2f} ms")
    print(f"average encoding time: {encoding_ms / embeddings.size * embeddings.shape[2]:.2f} ms/embedding")
    for strategy in ("max", "top2_mean"):
        print(inter_identity_report(multi, strategy))
    print(f"matching latency single: {matching_ms['single']:.4f} ms/query")
    print(f"matching latency multi MAX: {matching_ms['multi_max']:.4f} ms/query")
    print(f"matching latency multi TOP-2 MEAN: {matching_ms['multi_top2_mean']:.4f} ms/query")
    print("live end-to-end FPS: measured by run_face_recognition.py when webcam frames are available")


if __name__ == "__main__":
    main()
