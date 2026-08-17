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
- 49 enrollments succeeded; `DE190384` and `DE190692` had no face detected.
- Gallery validation passed: 49 identities, 128 dimensions, finite values, L2 norms approximately 1.

## Current Task

Phase 3 is complete; stop before implementing webcam recognition or identity matching.

## Next Actions

1. Investigate the two enrollment images with no detected face if a full 52-identity gallery is required.
2. Begin the later webcam recognition phase only after choosing a matching threshold from validation data.

## Open Questions

None currently.

## Last Session Summary

Added `src/inference/face_alignment.py`, `src/inference/face_encoder.py`, and `scripts/build_gallery.py`. Selected OpenCV SFace/MobileFaceNet, 36.90 MiB model, 128-D embeddings. Generated and validated `data/gallery/embeddings.npz` with 49 normalized embeddings. Measured average YuNet detection 2.61 ms, average encoding 7.06 ms, and total enrollment processing 734.72 ms. Inter-identity cosine similarity: mean 0.5631, median 0.5701, min 0.1420, max 0.8152 for `DE200258 <-> DE200437`. No webcam recognition, matching, tracking, Docker, or optimization was added.
