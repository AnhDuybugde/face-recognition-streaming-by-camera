# Short Memory

## Current Session Goal

Implement Phase 2 lightweight face detection.

## Current State

- Phase 0 scaffold created.
- Phase 1 webcam capture implementation remains the shared input layer.
- Phase 2 YuNet detector and webcam demo added.
- Python syntax compilation succeeded.
- OpenCV runtime and real webcam testing were not completed because the Windows command runner returned `CreateProcessAsUserW` error 1312.
- YuNet model asset is not downloaded into `models/` yet.
- User runtime test confirmed webcam opens, but YuNet inference fails with OpenCV 4.6.0 (`Layer with requested id=-1`).

## Current Task

Phase 2 is complete; stop before implementing recognition, enrollment, or tracking.

## Next Actions

1. Upgrade to `opencv-python>=4.10.0`.
2. Download the official YuNet ONNX asset to `models/face_detection_yunet_2023mar.onnx`.
3. Run `python -m scripts.run_face_detection` with webcam and GUI access.
4. Measure real detection latency/FPS and verify boxes/landmarks before Phase 3.

## Open Questions

None currently.

## Last Session Summary

Added `src/inference/face_detector.py` using OpenCV YuNet `FaceDetectorYN`, plus `scripts/run_face_detection.py` for boxes, confidence, landmarks, latency, and approximate FPS. Added `models/.gitkeep` and ignored ONNX assets. No recognition, enrollment, tracking, RTSP, Docker, or optimization was added. Model size is approximately 227 KB according to the official OpenCV Zoo listing; it is not present locally yet.
