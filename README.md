# Streaming Face Recognition

CPU-first face detection and recognition pipeline for a local webcam or a deterministic image/video source.

## Architecture

```text
camera / video / image
    -> YuNet detection
    -> five-point alignment
    -> ArcFace R50 encoder
    -> normalized embedding
    -> gallery cosine similarity
    -> Top-1 / Top-2 / margin
```

The production gallery contains one 512-dimensional ArcFace embedding per identity. Recognition currently processes every detected face in every frame; it does not add tracking or temporal smoothing.

## Requirements

- Native: Python 3.10, OpenCV 4.10.0.84, ONNX Runtime 1.18.1.
- Docker: Docker Engine/Desktop with Compose.
- A CPU is sufficient. GPU execution is not implemented in this phase.

`opencv-python` is used for the native webcam window. The Docker image uses `opencv-python-headless` because it runs without GUI display.

## Models and private data

Place these external runtime files in `models/`:

- `face_detection_yunet_2023mar.onnx`
- `face_recognition_arcface_r50.onnx`

Place the private gallery at `data/gallery/embeddings.npz`. Enrollment images belong in `data/input/` only when rebuilding the gallery. Models, photos, embeddings, videos, and debug outputs are ignored by Git and are not copied into the Docker image.

## Native setup and run

```powershell
python -m pip install -r requirements.txt
python -m scripts.run_face_recognition
```

The source can be a webcam index or an OpenCV-readable path/URL:

```powershell
python -m scripts.run_face_recognition --source 0
python -m scripts.run_face_recognition --source path\to\video.mp4 --no-display
```

Rebuild the local gallery when enrollment data changes:

```powershell
python -m scripts.build_gallery
```

All model, gallery, and input paths are also configurable with command-line arguments; defaults are relative to the project root.

## Deterministic benchmark

Use one private image containing one enrolled face and repeat the same pipeline:

```powershell
python -m scripts.run_image_recognition --input data\input\DE190469.jpg --iterations 10
```

The command reports detector latency, ArcFace encoding latency, end-to-end latency, approximate FPS, and the final gallery match. It does not save face crops.

## Docker CPU

Build the image:

```powershell
docker build -t streaming-face-recognition:cpu .
```

Run the deterministic test with runtime files mounted from the host. Replace the example input with a local private image:

```powershell
docker run --rm -v "${PWD}\models:/app/models:ro" -v "${PWD}\data:/app/data:ro" streaming-face-recognition:cpu python -m scripts.run_image_recognition --input /app/data/input/DE190469.jpg --iterations 10
```

The image contains code and dependencies only. `/app/models` and `/app/data` are external mount points.

## Docker Compose

`compose.yaml` provides one CPU service and the same two read-only mounts:

```powershell
docker compose build
docker compose run --rm face-recognition python -m scripts.run_image_recognition --input /app/data/input/DE190469.jpg --iterations 10
```

The default Compose command expects `/app/data/test.jpg`; override it as shown when using another file.

## GPU status

Not implemented. The current encoder uses ONNX Runtime with `CPUExecutionProvider`, and YuNet uses OpenCV's CPU detector. A future GPU phase must add and verify the CUDA provider and compatible host/container runtime; GPU visibility alone is not sufficient evidence.

## Known limitations

- Webcam passthrough inside Docker Desktop on Windows is not assumed or required; native webcam mode remains available.
- The gallery and evaluation set are small and private.
- There is no attendance database, API, authentication, or production unknown-person threshold.
- Container validation should use a mounted image/video because physical camera device mapping is platform-specific.

## Security notes

Do not commit enrollment photos, biometric embeddings, private videos, secrets, credentials, or model files. Check `.gitignore` and `.dockerignore` before sharing the repository or building an image.
