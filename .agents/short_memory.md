# Short Memory

## Current Session Goal

Replace SFace with a single ArcFace R50 production encoder.

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

ArcFace R50 production gallery rebuild is complete; stop before threshold calibration or attendance logic.

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
- User selected one larger ArcFace R50 encoder instead of multi-encoder fusion or multi-embedding production.
- Added `models/face_recognition_arcface_r50.onnx` and replaced the production encoder implementation with ONNX Runtime.
- ArcFace input is 112x112 BGR converted with `swapRB=True`, mean/std 127.5; model output is 512 dimensions.
- Rebuilt `data/gallery/embeddings.npz`: 51 successful identities, shape `(51, 512)`, finite and L2-normalized.
- ArcFace enrollment metrics: average detection 2.27 ms/image, average encoding 59.64 ms/image, total 3220.17 ms.
- ArcFace inter-identity cosine statistics: mean 0.1001, median 0.0943, min -0.1709, max 0.4274 for `DE200022 <-> DE200206`.
- ArcFace runtime smoke test processed 10 webcam frames at 18.97 FPS with 8.72 ms detection/frame; no face was visible, so live encoding/matching similarity remains unmeasured.

## Next Actions

1. Run webcam recognition with a visible enrolled face and collect genuine/impostor Top-1, Top-2, and margin observations.
2. Benchmark ArcFace live query latency/FPS against the previous SFace baseline.
3. Calibrate a KNOWN/UNKNOWN policy only after live observations exist.

## Open Questions

None currently.

## Last Session Summary

Replaced the production SFace/MobileFaceNet encoder with ArcFace R50 through ONNX Runtime. Rebuilt and validated the production gallery with 51 normalized 512-D embeddings. ArcFace enrollment measured 2.27 ms detection/image, 59.64 ms encoding/image, and 3220.17 ms total; inter-identity cosine mean 0.1001 and max 0.4274 for `DE200022 <-> DE200206`. The earlier multi-embedding experiment remains experimental and is not the production gallery. Live ArcFace query similarity and FPS with a visible face remain unmeasured.
