Read the following files before making any changes:

- `.agents/long_memory.md`
- `.agents/short_memory.md`
- `.agents/rule_base.md`

We are starting Phase 1: Local Webcam Capture.

Project context:
- This is a lightweight real-time face recognition practice project.
- The final system will later include face detection, alignment, embedding, recognition, runtime benchmarking, and Docker.
- However, this phase must focus ONLY on reliable webcam frame acquisition.
- Do not implement any AI model yet.

Goal:
Build the smallest correct webcam capture pipeline using OpenCV.

Requirements:

1. Inspect the current repository structure before editing anything.

2. Add only the minimum dependency required for webcam capture.
   - Prefer `opencv-python`.
   - Do not add PyTorch, Ultralytics, InsightFace, ONNX Runtime, FastAPI, or other unrelated dependencies.

3. Implement webcam capture logic in:
   `src/capture/video_source.py`

4. Implement a thin runnable script in:
   `scripts/run_webcam.py`

5. The program must:
   - open the default webcam using source index `0`
   - verify that the camera opened successfully
   - continuously read frames
   - display the live frame using OpenCV
   - exit when the user presses `q`
   - always release the camera resource
   - always destroy OpenCV windows cleanly

6. Print basic webcam information after opening:
   - reported frame width
   - reported frame height
   - reported camera FPS

7. Keep responsibilities separated:

   `video_source.py`
   should handle:
   - opening the source
   - reading frames
   - exposing basic source information
   - releasing the source

   `run_webcam.py`
   should handle:
   - application loop
   - displaying frames
   - keyboard exit
   - user-facing output

8. Keep the interface simple.

A reasonable usage style is:

    source = VideoSource(0)
    frame = source.read()
    source.release()

Do not introduce unnecessary abstractions such as:
- CameraManager
- CaptureFactory
- StreamStrategy
- abstract base classes
- dependency injection
- async
- threading
- multiprocessing
- queues

unless there is a demonstrated need.

9. Error handling:
   - fail clearly if webcam cannot be opened
   - fail clearly if frame reading repeatedly fails
   - do not silently ignore errors
   - use cleanup logic such as `try/finally` where appropriate

10. Do not implement:
    - face detection
    - face recognition
    - image enrollment
    - YOLO
    - tracking
    - frame skipping
    - RTSP
    - Docker
    - GPU support
    - APIs
    - database
    - configuration frameworks
    - optimization

11. Do not over-engineer the project.

12. Testing:
    - run the script if the current environment has webcam access
    - if webcam access is unavailable, verify everything that can be verified statically/runtime without claiming that webcam testing succeeded
    - never fabricate test results

13. After implementation, briefly explain:
    - what `cv2.VideoCapture(0)` represents
    - what `ret, frame = cap.read()` means
    - what shape/type a frame normally has
    - why camera resources must be released
    - why reported camera FPS is not necessarily equal to actual application FPS

14. Update `.agents/short_memory.md` with:
    - what was implemented
    - files changed
    - whether real webcam testing succeeded
    - any blocker
    - next recommended step

15. Update `.agents/long_memory.md` only if a durable architectural decision was made.

16. Stop after Phase 1.

Before writing code, provide a short plan containing:
- files to modify
- dependency to add
- expected behavior

Then execute the plan.