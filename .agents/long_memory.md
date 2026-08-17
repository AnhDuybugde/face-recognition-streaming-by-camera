# Long Memory

## Project Goal

Build a small production-oriented real-time Computer Vision pipeline for AI Developer internship preparation.

Target progression:

camera/video
→ frame capture
→ detection
→ tracking
→ business logic
→ runtime benchmarking
→ Docker
→ optional GPU optimization

## Permanent Constraints

- Keep the repository small and understandable.
- Prefer simple Python architecture over unnecessary frameworks.
- Accuracy is not the only objective.
- Always consider latency, FPS, memory, GPU usage, and cost.
- Do not add Kubernetes, Kafka, Redis, Triton, Airflow, or similar infrastructure unless the project reaches a real need.
- Local development uses the user's own computer and camera.
- Docker containers use hardware resources from the host machine.
- Company/private data must never be assumed safe to upload to third-party services.
- Never commit datasets, secrets, API keys, credentials, or private camera URLs.

## Architecture Principles

- Capture, inference, tracking, and business logic must remain separable.
- Build the simplest working pipeline before optimizing.
- Do not prematurely introduce async, multiprocessing, queues, or microservices.
- Measure bottlenecks before optimizing them.
- Prefer configuration over hard-coded runtime values where it improves clarity.

## Current High-Level Roadmap

1. Webcam capture.
2. Video file capture.
3. Object detection.
4. FPS and latency measurement.
5. Tracking.
6. Simple event logic.
7. RTSP input.
8. Dockerize.
9. GPU execution from Docker.
10. Optimization experiments.

## Durable Decisions

Add only decisions that should survive across sessions.

Format:

### YYYY-MM-DD — Decision title
- Decision:
- Reason:
- Consequence:

### 2026-08-17 — Use YuNet for initial face detection
- Decision: Use OpenCV YuNet through the `FaceDetectorYN` API for Phase 2.
- Reason: It is a lightweight face-specific detector that provides bounding boxes, confidence scores, and five facial landmarks without adding a separate inference framework.
- Consequence: The ONNX model remains an external, gitignored asset and runtime compatibility must be checked before inference.
