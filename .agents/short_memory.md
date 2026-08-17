# Short Memory

## Current Session Goal

Implement Phase 3 one-shot enrollment and embedding gallery.

## Current State

- Phase 0 scaffold created.
- Phase 1 webcam capture implementation remains the shared input layer.
- Phase 2 YuNet detector and webcam demo added.
- Python syntax compilation succeeded.
- OpenCV runtime and real webcam testing were not completed because the Windows command runner returned `CreateProcessAsUserW` error 1312.
- YuNet model asset is not downloaded into `models/` yet.
- User runtime test confirmed webcam opens, but YuNet inference fails with OpenCV 4.6.0 (`Layer with requested id=-1`).
- Phase 3 pipeline ran successfully with OpenCV 5.0.0 on the local enrollment data.
- Local data contains 51 images, not the expected 52.
- Initial 49 enrollments succeeded; `DE190384` and `DE190692` had no face detected.
- Resolution diagnostic showed both failures recover after aspect-ratio-preserving resize to longest side 320.
- Production enrollment now resizes only oversized images to max side 320 before YuNet detection and aligns on that resized image.
- Rebuilt gallery validation passed: 51 identities, 128 dimensions, finite values, L2 norms approximately 1.

## Current Task

Oversized-image enrollment fix is complete; stop before implementing webcam recognition or identity matching.

## Next Actions

1. Confirm whether the missing 52nd enrollment image should be added to `data/input`.
2. Begin the later webcam recognition phase only after choosing a matching threshold from validation data.

## Open Questions

None currently.

## Last Session Summary

Added `src/inference/face_alignment.py`, `src/inference/face_encoder.py`, and `scripts/build_gallery.py`. Selected OpenCV SFace/MobileFaceNet, 36.90 MiB model, 128-D embeddings. Added reusable max-side-320 preprocessing for oversized enrollment images and explicit width-height YuNet input sizing. Rebuilt and validated `data/gallery/embeddings.npz` with 51 normalized embeddings; both prior failures now succeed. New metrics: average detection 1.86 ms, encoding 6.85 ms, total enrollment 500.76 ms. Inter-identity cosine similarity: mean 0.5569, median 0.5656, min 0.1082, max 0.8152 for `DE200258 <-> DE200437`. No webcam recognition, matching, tracking, Docker, or optimization was added.
