# Short Memory

## Current Session Goal

Implement Phase 4 live webcam recognition baseline.

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
- Fixed five-point alignment landmark ordering after aligned output was observed rotated; rebuilt gallery with corrected upright alignment.

## Current Task

Phase 4 live recognition baseline is implemented; stop before threshold calibration or attendance logic.

## Utility Added

- `scripts/check_face.py` checks one arbitrary image with YuNet, reports face count/confidence/bbox/landmarks, and optionally saves annotated/aligned output.
- Tested on `DE190384.jpg`: detection input 213x320, one face, confidence 0.9318.
- Corrected aligned output is upright in `data/output/aligned_fixed.jpg`.
- Phase 4 live recognition baseline added in `scripts/run_face_recognition.py`.
- Gallery matching is implemented in `src/inference/face_matcher.py` using a NumPy dot product over 51 normalized 128-D embeddings.
- Top-1, Top-2, and margin are displayed without a KNOWN/UNKNOWN threshold.
- Optional `--expected-identity` observation statistics and ignored CSV output are supported.
- Gallery validation and a self-match test passed: 51 identities, 128 dimensions, Top-1 similarity 1.0000.
- Webcam opened at 640x480/30 FPS, but Windows MSMF failed to read frames three times consecutively (`-1072875772`), so no live similarity or FPS observations were collected.

## Next Actions

1. Retry webcam capture after resolving the Windows MSMF frame-read issue.
2. Collect genuine and impostor live similarity/margin observations.
3. Calibrate a KNOWN/UNKNOWN policy only after live observations exist.

## Open Questions

None currently.

## Last Session Summary

Added `src/inference/face_alignment.py`, `src/inference/face_encoder.py`, and `scripts/build_gallery.py`. Selected OpenCV SFace/MobileFaceNet, 36.90 MiB model, 128-D embeddings. Added reusable max-side-320 preprocessing for oversized enrollment images and explicit width-height YuNet input sizing. Corrected landmark ordering so aligned faces remain upright, then rebuilt and validated `data/gallery/embeddings.npz` with 51 normalized embeddings. Latest enrollment metrics: average detection 1.72 ms, encoding 6.61 ms, total 476.32 ms. Inter-identity cosine similarity: mean 0.1659, median 0.1648, min -0.1610, max 0.4812 for `DE190523 <-> SE200086`. Added and tested `scripts/check_face.py` for single-image YuNet inspection. Phase 4 added normalized NumPy Top-2 matching and live visualization with optional observation statistics. Webcam validation was blocked by Windows MSMF frame-read error `-1072875772`; no live similarity/FPS values were claimed.
