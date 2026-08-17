# Short Memory

## Current Session Goal

Compare single-embedding and light-augmentation multi-embedding galleries.

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

Multi-embedding gallery experiment is complete; stop before threshold calibration or attendance logic.

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
- Added five in-memory aligned-face variants: original, horizontal flip, 0.90 intensity, 1.10 intensity, and mild contrast variation.
- Generated `data/gallery/embeddings_multi.npz` with shape `(51, 5, 128)` and preserved the single baseline as `data/gallery/embeddings_single.npz`.
- Multi-gallery matching supports MAX and TOP-2 MEAN aggregation; runtime query encoding remains one SFace call.
- Multi-gallery file size is 121790 bytes versus 24883 bytes for the single gallery.
- Multi build time was 2206.71 ms; average SFace encoding was 7.01 ms per embedding.
- Offline matching latency was 0.0649 ms/query single, 0.0272 ms/query multi MAX, and 0.0403 ms/query multi TOP-2 MEAN.
- Offline inter-identity statistics: MAX mean/median/min/max = 0.1903/0.1865/-0.1377/0.4914; TOP-2 MEAN = 0.1858/0.1831/-0.1384/0.4909. Both maximum pairs were DE190523 <-> SE200086.
- A 10-frame webcam run succeeded at approximately 20.06 FPS with 10.32 ms detection/frame, but no face was present, so no live Top-1/Top-2/margin or query encoding latency was collected.

## Next Actions

1. Run the webcam comparison with a visible enrolled face and collect genuine/impostor Top-1, Top-2, and margin observations.
2. Compare multi-gallery behavior on labeled live samples before choosing a gallery strategy.
3. Calibrate a KNOWN/UNKNOWN policy only after live observations exist.

## Open Questions

None currently.

## Last Session Summary

Added `src/inference/face_alignment.py`, `src/inference/face_encoder.py`, and `scripts/build_gallery.py`. Selected OpenCV SFace/MobileFaceNet, 36.90 MiB model, 128-D embeddings. Added reusable max-side-320 preprocessing for oversized enrollment images and explicit width-height YuNet input sizing. Corrected landmark ordering so aligned faces remain upright, then rebuilt and validated `data/gallery/embeddings.npz` with 51 normalized embeddings. Latest enrollment metrics: average detection 1.72 ms, encoding 6.61 ms, total 476.32 ms. Inter-identity cosine similarity: mean 0.1659, median 0.1648, min -0.1610, max 0.4812 for `DE190523 <-> SE200086`. Added and tested `scripts/check_face.py` for single-image YuNet inspection. Phase 4 added normalized NumPy Top-2 matching and live visualization with optional observation statistics. The multi-embedding experiment added five mild post-alignment variants and MAX/TOP-2 MEAN aggregation. A 10-frame webcam smoke test reached 20.06 FPS and 10.32 ms detection/frame but detected no face, so live similarity behavior remains unmeasured.
