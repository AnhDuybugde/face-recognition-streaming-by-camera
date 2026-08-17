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

### 2026-08-17 — Store one-shot enrollment as normalized NPZ embeddings
- Decision: Use OpenCV SFace/MobileFaceNet to produce 128-D L2-normalized embeddings and store the gallery in `data/gallery/embeddings.npz`.
- Reason: The gallery is small, offline, and does not justify a database or vector index; OpenCV keeps CPU deployment simple.
- Consequence: Enrollment images, model files, and generated biometric embeddings remain external runtime data and are excluded from Git.

### 2026-08-17 — Normalize oversized enrollment images before YuNet
- Decision: Before enrollment detection, downscale only images whose longest side exceeds 320 pixels, preserving aspect ratio; keep smaller images unchanged.
- Reason: Controlled diagnostics showed YuNet failed on two large source images but succeeded after resizing to 320, without changing thresholds or adding retries.
- Consequence: Detection, landmarks, alignment, and embedding for enrollment operate on the resized image; no coordinate remapping is needed for the current offline gallery.

### 2026-08-17 — Keep five-point face alignment upright
- Decision: Map YuNet's five landmark coordinates directly to the SFace upright template without an additional eye/mouth order swap.
- Reason: The previous extra swap produced visibly rotated aligned faces and inconsistent gallery embeddings.
- Consequence: The gallery must be rebuilt whenever this alignment convention changes; the current gallery uses the corrected upright alignment.

### 2026-08-17 — Match the small gallery with normalized NumPy dot products
- Decision: Keep the 51-entry gallery in memory and compute cosine similarity as a matrix-vector dot product.
- Reason: L2-normalized 128-D embeddings make this exact operation simple and far below the cost of face inference at this gallery size.
- Consequence: The live baseline reports Top-1, Top-2, and margin for later threshold calibration without introducing FAISS or a database.
