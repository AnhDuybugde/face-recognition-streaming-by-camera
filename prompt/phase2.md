Read first:

- `.agents/long_memory.md`
- `.agents/short_memory.md`
- `.agents/rule_base.md`

We have successfully completed Phase 1: local webcam capture.

We are now starting Phase 2: Lightweight Face Detection.

Project goal:
Build a lightweight real-time face recognition system for approximately 52 enrolled identities, with only one enrollment image per person initially.

This phase must implement ONLY face detection on the webcam stream.

Goal:

webcam
→ frame
→ lightweight face detector
→ face bounding box + facial landmarks
→ visualization

Requirements:

1. Inspect the current Phase 1 implementation before modifying anything.

2. Reuse the existing webcam capture abstraction.
   Do not duplicate camera capture code.

3. Use a lightweight face-specific detector suitable for low-cost real-time inference.
   Prefer OpenCV YuNet if it is practical with the installed OpenCV version.

4. Before depending on a specific OpenCV API:
   - check the installed OpenCV version
   - verify that the required API actually exists
   - do not assume API signatures
   - do not fabricate model download paths or behavior

5. Keep detector logic separate from the application loop.

Suggested location:

src/
└── inference/
    └── face_detector.py

The detector should conceptually expose something simple such as:

    faces = detector.detect(frame)

Do not create unnecessary abstract classes or generic detector frameworks.

6. For each detected face, return or expose at least:
   - bounding box
   - confidence score
   - facial landmarks if supported by the detector

7. Add a thin webcam demo script, for example:

    scripts/run_face_detection.py

Reuse `VideoSource` from Phase 1.

8. The script should:
   - open webcam
   - read frames
   - run face detection
   - draw face bounding boxes
   - optionally draw landmarks
   - display detection confidence
   - press `q` to quit
   - cleanly release all resources

9. Measure basic runtime performance:
   - face detection latency per processed frame
   - approximate end-to-end FPS

Keep the measurement simple.
Do not introduce a benchmarking framework yet.

10. Print useful runtime information such as:

    detector: ...
    frame size: ...
    detected faces: ...
    detection latency: ... ms
    approximate FPS: ...

Avoid printing excessive logs every frame if that causes console spam.
A periodic update is acceptable.

11. Do NOT implement:
    - face recognition
    - embeddings
    - enrollment processing
    - cosine similarity
    - identity labels
    - training
    - fine-tuning
    - tracking
    - Docker
    - RTSP
    - async
    - multiprocessing
    - FastAPI
    - database
    - Kubernetes

12. Model assets:
    - keep model files separate from source code
    - do not embed binary model data into Python
    - do not commit unnecessarily large model files
    - ensure `.gitignore` rules remain appropriate

13. Error handling:
    - report clearly if the model file is missing
    - report clearly if model initialization fails
    - do not silently fall back to another detector
    - do not hide inference failures

14. After implementation, test with the real webcam if accessible.

Verify at least:
    - face is detected when looking at camera
    - bounding box approximately follows the face
    - program exits cleanly
    - measured latency/FPS are reported

Do not claim success for tests that were not actually executed.

15. After testing, explain briefly:
    - why face detection is needed even if enrollment images are already face photos
    - why landmarks will be useful in the next phase
    - difference between face detection and face recognition
    - measured detection latency and FPS

16. Update `.agents/short_memory.md` with:
    - files changed
    - detector selected
    - model size if known
    - measured latency/FPS
    - real webcam test result
    - observed problems
    - next step

17. Update `.agents/long_memory.md` only for durable architectural decisions.

18. Stop after Phase 2.

Before coding:
- provide a short implementation plan
- state which detector you intend to use and why
- state any new dependency/model asset required

Then execute the plan.