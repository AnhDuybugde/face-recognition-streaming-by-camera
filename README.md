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

## Newcomer review: đọc pipeline từ đầu

Phần này dành cho người mới nhận repository và muốn hiểu hệ thống trước khi sửa code.

### 1. Bài toán đang giải quyết

Pipeline nhận một frame từ webcam hoặc video, tìm tất cả khuôn mặt, biến mỗi khuôn mặt thành một vector số, rồi so sánh vector đó với gallery đã tạo từ trước.

Đây là face recognition dạng gallery matching, không phải model classifier được train riêng cho 51 người. ArcFace tạo feature vector; `FaceMatcher` dùng cosine similarity để tìm identity gần nhất.

### 2. Cấu trúc cần đọc

```text
src/
  capture/video_source.py          đọc frame từ webcam/video
  inference/face_detector.py       YuNet: bbox, confidence, 5 landmarks
  inference/face_alignment.py      căn chỉnh mặt về ảnh 112x112
  inference/face_encoder.py        ArcFace R50 ONNX -> vector 512 chiều
  inference/face_matcher.py        cosine matching với gallery

scripts/
  build_gallery.py                 enrollment offline
  run_face_recognition.py          recognition realtime theo từng frame
  run_image_recognition.py         benchmark deterministic bằng một ảnh
  check_face.py                    kiểm tra detection/alignment của một ảnh

data/                              runtime/private data, không đưa vào Git
  input/                            ảnh enrollment hoặc ảnh test
  gallery/embeddings.npz           gallery đã build
  output/                          output debug nếu có

models/                             model ONNX, không đưa vào Git
Dockerfile                          image CPU, không chứa data/models
compose.yaml                        một service CPU và hai volume mount
```

Nên đọc theo thứ tự: `face_detector.py` → `face_alignment.py` → `face_encoder.py` → `face_matcher.py` → `build_gallery.py` → `run_face_recognition.py`.

### 3. Enrollment: gallery được tạo như thế nào?

Chạy:

```powershell
python -m scripts.build_gallery
```

Với mỗi ảnh trong `data/input/`, pipeline thực hiện:

```text
ảnh enrollment
  -> đọc bằng cv2.imread
  -> nếu cạnh dài > 320: resize giữ nguyên aspect ratio
  -> YuNet detect
  -> yêu cầu đúng một khuôn mặt
  -> lấy 5 landmarks của YuNet
  -> alignment về 112x112
  -> ArcFace R50 encode thành vector 512 chiều
  -> L2 normalize vector
  -> lưu identity + embedding vào embeddings.npz
```

Ảnh quá lớn được resize trước detection vì thực nghiệm cho thấy YuNet có thể không detect được khuôn mặt khi input quá lớn. Ảnh nhỏ không bị upscale. Detection, landmarks và alignment đều dùng cùng ảnh đã resize; pipeline hiện không remap tọa độ về ảnh gốc.

File `data/gallery/embeddings.npz` chứa hai array:

- `identities`: tên identity lấy từ filename, ví dụ `DE190469`.
- `embeddings`: ma trận `(N, 512)`, mỗi dòng là một vector đã L2-normalize.

Gallery hiện tại có 51 identity. Nếu thay model encoder, thay alignment convention, hoặc thay dữ liệu enrollment, phải rebuild gallery; query encoder và gallery encoder phải dùng cùng preprocessing/model.

### 4. Recognition realtime: chuyện gì xảy ra trong mỗi frame?

Chạy:

```powershell
python -m scripts.run_face_recognition
```

Luồng hiện tại là:

```text
frame 1
  -> YuNet detect tất cả mặt
  -> với từng mặt: alignment -> ArcFace -> cosine match
  -> vẽ Top-1, Top-2, margin

frame 2
  -> lặp lại toàn bộ quy trình
```

Điều quan trọng: hiện chưa có tracking, cache embedding, temporal smoothing, hoặc detect mỗi N frame. Vì vậy một khuôn mặt xuất hiện trong 30 frame sẽ được detect mỗi frame và thường được ArcFace encode lại mỗi frame. Đây là thiết kế baseline dễ hiểu, nhưng ArcFace là bottleneck chính.

Kết quả hiển thị:

- `Top1`: identity có cosine similarity cao nhất.
- `Top2`: identity đứng thứ hai.
- `Margin`: `Top1 similarity - Top2 similarity`.

Hiện chưa có ngưỡng `KNOWN/UNKNOWN`. Vì vậy Top-1 chỉ có nghĩa là “ứng viên gần nhất”, chưa đủ để kết luận người đó chắc chắn thuộc gallery.

### 5. Vai trò của từng thư viện và model

- OpenCV: đọc webcam/ảnh/video, resize, alignment warp, YuNet API, vẽ kết quả.
- YuNet ONNX: face detector nhẹ; output gồm bounding box, confidence và 5 landmarks.
- ONNX Runtime: chạy ArcFace R50 ONNX.
- ArcFace R50: face encoder; input aligned BGR 112x112, chuẩn hóa theo mean/std 127.5, output 512 chiều.
- NumPy: L2 normalization, lưu/đọc NPZ, dot product matching.

ArcFace không tự trả về tên người. Tên đến từ thứ tự identity trong `embeddings.npz`; `FaceMatcher` chỉ tính similarity giữa query embedding và các embedding trong gallery.

### 6. Cách debug khi kết quả sai

Nếu không detect được mặt:

```powershell
python -m scripts.check_face --input data\input\DE190384.jpg
```

Kiểm tra model YuNet, phiên bản OpenCV, kích thước ảnh và confidence. Không nên vội hạ threshold; trước hết xác định lỗi nằm ở input scale, model hoặc landmarks.

Nếu ảnh aligned bị xoay hoặc sai hướng, kiểm tra `src/inference/face_alignment.py`. Landmark order của YuNet phải được map trực tiếp đúng với template; không tự ý swap thêm mắt/trán/miệng.

Nếu gallery load lỗi, kiểm tra:

- file có tồn tại ở `data/gallery/embeddings.npz`;
- identities và embeddings có cùng số dòng;
- embedding có đúng 512 chiều;
- mọi vector có finite value và norm gần 1;
- encoder model dùng để query có cùng dimension với gallery.

Nếu match luôn sai nhưng detection đúng, kiểm tra alignment và preprocessing ArcFace trước khi chỉnh matcher. Gallery phải được rebuild sau thay đổi encoder hoặc alignment.

### 7. Cách benchmark đúng

Dùng cùng một ảnh, cùng model và cùng gallery:

```powershell
python -m scripts.run_image_recognition --input data\input\DE190469.jpg --iterations 10
```

Script đo riêng detection, encoding, end-to-end latency và approximate FPS. Đây là benchmark deterministic; nó không đại diện đầy đủ cho webcam nhiều mặt hoặc video chuyển động.

Docker dùng đúng script đó với data/model mount ngoài:

```powershell
docker run --rm -v "${PWD}\models:/app/models:ro" -v "${PWD}\data:/app/data:ro" streaming-face-recognition:cpu python -m scripts.run_image_recognition --input /app/data/input/DE190469.jpg --iterations 10
```

Benchmark đã đo trên cùng ảnh `DE190469.jpg`:

| Môi trường | Detection | ArcFace encoding | End-to-end | FPS xấp xỉ |
|---|---:|---:|---:|---:|
| Native | 8.27 ms | 80.22 ms | 88.88 ms | 11.25 |
| Docker CPU | 9.55 ms | 132.22 ms | 144.98 ms | 6.90 |

Kết quả này cho thấy container chạy đúng, nhưng ArcFace CPU trong Docker chậm hơn trên môi trường native hiện tại. Không nên suy ra GPU đã được dùng; encoder hiện cấu hình `CPUExecutionProvider`.

### 8. Những gì pipeline chưa làm

- Chưa có tracking hoặc temporal smoothing.
- Chưa có threshold UNKNOWN được calibrate bằng dữ liệu genuine/impostor thực tế.
- Chưa có attendance database, API, authentication hoặc event layer.
- Chưa có GPU inference; Docker GPU chưa được triển khai.
- Webcam passthrough trong Docker Desktop/Windows chưa phải đường chạy mặc định.
- Gallery nhỏ, dữ liệu đánh giá hạn chế; benchmark không phải accuracy evaluation đầy đủ.

### 9. Lộ trình đọc và cải tiến an toàn

Người mới nên xác nhận theo thứ tự:

1. Chạy `check_face.py` trên một ảnh và quan sát detection/alignment.
2. Chạy `build_gallery.py` và kiểm tra shape gallery.
3. Chạy `run_image_recognition.py` để xác nhận encoder + matcher.
4. Chạy webcam native với `--max-frames` để kiểm tra capture.
5. Chỉ sau khi có genuine/impostor observations mới calibrate UNKNOWN threshold.
6. Chỉ tối ưu tracking hoặc giảm tần suất detection sau khi đo bottleneck thực tế.

Mỗi thay đổi model, alignment hoặc preprocessing cần rebuild gallery và chạy lại benchmark. Không đưa ảnh, gallery embedding, model hoặc video private vào Git hay Docker image.
