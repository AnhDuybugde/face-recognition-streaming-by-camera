Read first:

- `.agents/long_memory.md`
- `.agents/short_memory.md`
- `.agents/rule_base.md`

Phase 1 is complete:
- Local webcam capture works.

Phase 2 is complete:
- Detector: YuNet
- Webcam frame size: 640x480
- YuNet detection latency: approximately 8–9 ms/frame
- Application FPS stabilized around 29–30 FPS
- Face detection and alignment are working.

We are now starting Phase 3:
One-shot Face Enrollment and Embedding Gallery.

==================================================
PROJECT CONTEXT
==================================================

Final project goal:

Build a lightweight real-time face recognition / attendance system suitable for:
- small organizations / classrooms
- low-cost hardware
- small enrollment datasets
- real-time webcam inference
- future Docker deployment

Current enrollment dataset:

- 52 identities
- exactly 1 enrollment image per identity
- filename stem is the identity/student ID

Example:

    data/input/DE190045.jpg
    ->
    identity = "DE190045"

The images are approximately 146x111 pixels.

Important:

These 52 images are NOT considered a neural-network training dataset.

They are a one-shot enrollment gallery.

We are NOT training or fine-tuning a neural network in this phase.

==================================================
PHASE 3 GOAL
==================================================

Convert:

    52 enrollment images

into:

    student_id -> normalized face embedding

and save the generated gallery locally as:

    data/gallery/embeddings.npz

Conceptual pipeline:

    enrollment image
          ↓
    YuNet face detection
          ↓
    facial landmarks
          ↓
    face alignment
          ↓
    lightweight pretrained face encoder
          ↓
    embedding vector
          ↓
    L2 normalization
          ↓
    embeddings.npz

This is an OFFLINE enrollment process.

Webcam recognition is NOT part of Phase 3.

==================================================
1. INSPECT EXISTING CODE FIRST
==================================================

Before modifying anything:

1. Inspect the repository.
2. Reuse the existing YuNet detector from Phase 2.
3. Reuse the existing face alignment implementation if appropriate.
4. Do not duplicate working Phase 2 logic.
5. Do not refactor unrelated working code unless required.

Keep the project small.

==================================================
2. SELECT A LIGHTWEIGHT FACE ENCODER
==================================================

Select a pretrained lightweight face recognition encoder suitable for:

- CPU inference
- low latency
- small model size
- aligned face input
- embedding-based face recognition
- future low-cost deployment

Prefer an ONNX/OpenCV-friendly model if practical.

Avoid adding PyTorch, TensorFlow, InsightFace, or another large framework unless there is a strong technical reason.

Before implementation, report:

- chosen encoder
- why it was selected
- model file size if known
- expected input resolution
- embedding dimension
- inference backend
- new dependency required, if any

Do not automatically choose the largest or most accurate model.

The project priorities are:

1. sufficient recognition quality
2. low latency
3. low compute cost
4. small model
5. simple deployment

==================================================
3. FACE ENCODER MODULE
==================================================

Implement the encoder separately.

Suggested location:

    src/inference/face_encoder.py

Its responsibility must be only:

    aligned face
        ↓
    face encoder
        ↓
    embedding

It must NOT:

- know student IDs
- scan enrollment directories
- access webcam
- perform classification
- compare identities
- implement attendance logic

Return a clean NumPy embedding.

==================================================
4. ENROLLMENT PIPELINE
==================================================

Create a thin enrollment pipeline/script.

Suggested script:

    scripts/build_gallery.py

Input:

    data/input/

Output:

    data/gallery/embeddings.npz

For each supported image:

1. Load image.
2. Extract identity from filename stem.
3. Detect face using existing YuNet implementation.
4. Require exactly one usable face.
5. Obtain landmarks.
6. Align the face.
7. Pass aligned face to encoder.
8. Obtain embedding.
9. Validate embedding.
10. L2-normalize embedding.
11. Associate embedding with student ID.

Example:

    DE190045.jpg
          ↓
    identity = DE190045
          ↓
    detect
          ↓
    align
          ↓
    encode
          ↓
    normalized embedding

==================================================
5. ENROLLMENT VALIDATION
==================================================

Do not silently accept invalid images.

Each image must produce one explicit result:

- success
- unreadable image
- no face detected
- multiple faces detected
- invalid landmarks/alignment
- encoding failure

For enrollment:

If multiple faces exist, do NOT arbitrarily choose one.

Mark the image as failed and report it.

==================================================
6. SAVE EMBEDDING GALLERY
==================================================

Save successful enrollments into:

    data/gallery/embeddings.npz

Use a simple NumPy format.

Conceptually:

    identities.shape
        = (N,)

    embeddings.shape
        = (N, D)

where:

    N = number of successfully enrolled identities
    D = embedding dimension

Example:

    identities:
    [
        "DE190045",
        "DE190089",
        ...
    ]

    embeddings:
    [
        [...],
        [...],
        ...
    ]

Do NOT introduce:

- SQLite
- PostgreSQL
- MongoDB
- Redis
- FAISS
- Milvus
- vector databases

52 identities do not justify database infrastructure.

==================================================
7. DATA STORAGE RULE
==================================================

`embeddings.npz` is runtime data, NOT application code.

Do not embed enrollment data into source code.

Ensure:

    data/gallery/

and generated embeddings remain ignored by Git.

Treat face embeddings as sensitive data.

Do not commit:

- enrollment images
- aligned face images
- embeddings
- student biometric data

Future Docker deployment will mount runtime data externally.

Do NOT implement Docker in this phase.

==================================================
8. EMBEDDING VALIDATION
==================================================

After building the gallery, verify:

1. Number of embeddings equals number of successful identities.
2. All embeddings have the same dimension.
3. Embeddings contain only finite values.
4. No NaN.
5. No Inf.
6. L2 norm of each stored embedding is approximately 1.

If validation fails:

Do not save an invalid gallery silently.

==================================================
9. INTER-IDENTITY SIMILARITY ANALYSIS
==================================================

After generating the gallery, compute pairwise cosine similarity between all DIFFERENT identities.

Exclude self-similarity.

Do not report the diagonal similarity of 1.0.

Report:

- mean inter-identity cosine similarity
- median inter-identity cosine similarity
- maximum inter-identity cosine similarity
- minimum inter-identity cosine similarity
- identity pair with maximum similarity

Example only:

    highest inter-identity similarity:
    DE190045 <-> DE190089
    cosine similarity = ...

Use actual measured results.

Purpose:

Understand how well the pretrained embedding space separates the 52 enrolled identities.

IMPORTANT:

Do NOT call this "accuracy".

There is only one enrollment image per identity, so this analysis cannot measure recognition accuracy.

==================================================
10. PERFORMANCE MEASUREMENT
==================================================

Measure actual Phase 3 performance.

At minimum report:

- average YuNet detection/alignment time per image
- average face encoding latency per image
- total enrollment processing time
- encoder model size
- embedding dimension

If useful, also report approximate p50/p95 encoder latency, but do not build a complex benchmark framework.

Do not fabricate measurements.

==================================================
11. ALIGNED FACE DEBUGGING
==================================================

Optionally support saving aligned faces for manual inspection.

For example:

    data/debug/aligned/

But:

- disabled by default
- do not automatically save every face unless requested
- directory must be ignored by Git
- never overwrite original enrollment images

This feature should remain simple.

==================================================
12. DO NOT IMPLEMENT YET
==================================================

Do NOT implement:

- webcam face recognition
- cosine matching of webcam faces
- known/unknown threshold
- attendance system
- classifier training
- kNN classifier
- SVM classifier
- neural network training
- fine-tuning
- data augmentation
- YOLO
- tracking
- RTSP
- Docker
- GPU optimization
- FastAPI
- database
- vector database
- Kubernetes

These belong to later phases only if needed.

==================================================
13. IMPORTANT ENGINEERING PRINCIPLES
==================================================

Keep the implementation minimal.

Do not create unnecessary:

- factories
- managers
- repository patterns
- abstract base classes
- dependency injection
- configuration frameworks
- generic ML frameworks

Current architecture should remain approximately:

    YuNet detector
          +
    face alignment
          +
    lightweight encoder
          +
    simple gallery builder
          +
    NumPy storage

Before adding any dependency or abstraction, explain why it is necessary.

==================================================
14. TESTING
==================================================

Run the complete enrollment pipeline against the actual local dataset if accessible.

Expected dataset size:

    52 images

Report actual:

    total images
    successful enrollments
    failed enrollments

For every failure, report:

    identity
    reason

Never fabricate successful execution.

==================================================
15. AFTER IMPLEMENTATION EXPLAIN
==================================================

Briefly explain:

1. Why the 52 images are enrollment data rather than a neural-network training dataset.

2. Why pretrained face embeddings are preferable to training a CNN from scratch with one image per identity.

3. Why face alignment happens before encoding.

4. Why embeddings are L2-normalized.

5. Why cosine similarity will later be useful.

6. Why `.npz` is sufficient for 52 identities.

7. Why a database is unnecessary at the current scale.

8. Why face embeddings should still be treated as sensitive data.

Keep the explanation concise and engineering-oriented.

==================================================
16. MEMORY UPDATE
==================================================

After completion:

Update:

    .agents/short_memory.md

Include:

- encoder selected
- model size
- embedding dimension
- enrollment success/failure count
- measured encoder latency
- generated gallery path
- inter-identity similarity summary
- blockers/issues
- next recommended step

Update:

    .agents/long_memory.md

ONLY if durable architectural decisions were made.

Examples of durable decisions:

- selected face encoder
- NPZ gallery format
- embedding normalization policy

==================================================
17. STOP CONDITION
==================================================

Phase 3 is complete only when:

    52 raw enrollment images
              ↓
    detection/alignment
              ↓
    lightweight encoder
              ↓
    normalized embeddings
              ↓
    validated embeddings.npz
đ
has been tested as far as the local environment permits.

STOP after Phase 3.

Do NOT proceed automatically to webcam recognition.

Before writing code:

1. inspect existing implementation
2. provide a short implementation plan
3. identify files that will change
4. identify the selected encoder and dependency impact
5. then execute