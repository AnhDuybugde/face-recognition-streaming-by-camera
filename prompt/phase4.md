Read first:

- `.agents/long_memory.md`
- `.agents/short_memory.md`
- `.agents/rule_base.md`

Phase 3 is complete.

Current verified state:

- 51 enrollment images
- 51 successful identities
- Gallery:
  `data/gallery/embeddings.npz`
- Gallery shape:
  51 x 128
- Embeddings are finite and L2-normalized
- Detector:
  YuNet
- Encoder:
  OpenCV SFace / MobileFaceNet
- Enrollment preprocessing:
  if longest image side > 320, downscale to longest side = 320
  while preserving aspect ratio
- No upscaling
- No retry/fallback
- YuNet setInputSize((w, h)) is updated before detection

Measured enrollment baseline after preprocessing fix:

- average detection latency: 1.86 ms/image
- average encoding latency: 6.85 ms/image
- total enrollment time: 500.76 ms
- embedding dimension: 128

Inter-identity cosine similarity:

- mean: 0.5569
- median: 0.5656
- min: 0.1082
- max: 0.8152
- max pair:
  DE200258 <-> DE200437

We are now starting Phase 4:
Live Webcam Recognition Baseline.

==================================================
GOAL
==================================================

Build the smallest correct live recognition pipeline:

webcam
→ face detection
→ alignment
→ embedding
→ compare against 51 enrolled embeddings
→ report top matches

This phase is for OBSERVATION and BASELINE MEASUREMENT.

Do NOT implement attendance logic yet.

==================================================
1. REUSE EXISTING COMPONENTS
==================================================

Reuse:

- existing webcam capture
- existing YuNet detector
- existing alignment logic
- existing SFace encoder
- existing gallery file

Do not duplicate working code.

==================================================
2. LOAD GALLERY
==================================================

Load:

`data/gallery/embeddings.npz`

Validate:

- identities count matches embedding count
- embedding dimension is consistent
- embeddings are finite
- embeddings are L2 normalized

Fail clearly if gallery is invalid.

==================================================
3. LIVE PIPELINE
==================================================

For each webcam frame:

1. acquire frame
2. detect faces using YuNet
3. for each detected face:
   - align face
   - encode face
   - L2 normalize query embedding
   - compute cosine similarity against all 51 gallery embeddings

Because embeddings are normalized:

cosine similarity can be computed efficiently using dot product.

Do not add FAISS or a vector database.

==================================================
4. TOP-K MATCHING
==================================================

For each query face, compute at minimum:

- Top-1 identity
- Top-1 similarity
- Top-2 identity
- Top-2 similarity
- margin = Top-1 similarity - Top-2 similarity

Example display:

Top1: DE190xxx 0.81
Top2: DE20xxxx 0.59
Margin: 0.22

Do NOT assign final KNOWN/UNKNOWN yet using a hardcoded threshold.

This phase is for collecting real similarity behavior first.

==================================================
5. VISUALIZATION
==================================================

Display on the webcam frame:

- face bounding box
- Top-1 student ID
- Top-1 similarity
- Top-2 similarity
- margin

Keep UI minimal.

Do not create a GUI framework.

==================================================
6. TARGET TEST CASE
==================================================

The user currently has one practical live positive test identity:
the user themself.

The user's student ID is already present in the gallery.

During testing, observe repeated webcam samples under small natural variations:

- centered face
- slightly left
- slightly right
- slightly closer
- slightly farther
- normal expression

Do not artificially augment webcam frames.

==================================================
7. COLLECT SIMILARITY STATISTICS
==================================================

Add a lightweight optional observation mode.

For each processed query of the user's face, record:

- expected identity
- Top-1 identity
- Top-1 similarity
- Top-2 identity
- Top-2 similarity
- margin

Keep this data in memory or a local ignored file if necessary.

Do not collect raw face images unless explicitly requested.

At the end of an observation session, report:

- number of query samples
- number where Top-1 identity equals expected identity
- mean Top-1 genuine similarity
- min genuine similarity
- max genuine similarity
- mean Top-1/Top-2 margin
- minimum margin

If expected identity is not provided, run visualization only.

==================================================
8. UNKNOWN / IMPOSTOR OBSERVATION
==================================================

Do not implement a final unknown threshold yet.

However, the system should allow observation of faces not present in the gallery.

For an unknown person, report:

- Top-1 enrolled identity
- Top-1 similarity
- Top-2 similarity
- margin

This will later help calibrate the KNOWN/UNKNOWN threshold.

Do not store unknown people's face images.

==================================================
9. PERFORMANCE
==================================================

Measure:

- detection latency
- encoding latency per face
- matching latency
- end-to-end application FPS

Matching 51 x 128 embeddings should remain simple NumPy operations.

Do not optimize unless a measured bottleneck appears.

==================================================
10. IMPORTANT MATCHING RULE
==================================================

Do not implement:

"highest similarity always wins"

as the final decision policy.

Current inter-identity maximum similarity is already 0.8152.

Therefore final identity acceptance will later require calibration using:

- absolute Top-1 similarity
- Top-1 / Top-2 margin
- genuine vs impostor observations

Phase 4 must expose these values but not finalize the policy.

==================================================
11. DO NOT IMPLEMENT
==================================================

Do not implement:

- final KNOWN/UNKNOWN threshold
- attendance records
- database
- tracking
- temporal smoothing
- cooldown
- RTSP
- Docker
- GPU optimization
- model fine-tuning
- classifier training
- SVM
- kNN classifier
- YOLO
- API
- Kubernetes

==================================================
12. TESTING
==================================================

Run the live webcam pipeline if webcam access is available.

Never fabricate results.

Report actual:

- Top-1 identity behavior
- similarity range
- margin behavior
- detection latency
- encoding latency
- matching latency
- approximate FPS

If webcam access is unavailable in the execution environment, say so clearly.

==================================================
13. AFTER IMPLEMENTATION EXPLAIN
==================================================

Briefly explain:

1. Why cosine matching against 51 embeddings is cheap.
2. Why Top-2 matters.
3. Why margin matters.
4. Why the max inter-identity similarity of 0.8152 means we should not blindly use a generic threshold.
5. Why Phase 4 collects live statistics before implementing KNOWN/UNKNOWN.

==================================================
14. MEMORY
==================================================

Update `.agents/short_memory.md` with:

- live recognition implementation status
- actual measured latency
- actual similarity observations
- observed margin values
- blockers
- next recommended step

Update long memory only for durable architectural decisions.

==================================================
15. STOP CONDITION
==================================================

Stop after live recognition baseline and observation metrics are working.

Do NOT proceed to final threshold calibration or attendance logic automatically.

Before coding:

1. inspect repository
2. provide a short plan
3. list files to modify
4. explain how gallery matching will work
5. then execute