Read first:

- `.agents/long_memory.md`
- `.agents/short_memory.md`
- `.agents/rule_base.md`

Current verified pipeline:

YuNet
→ face alignment to 112x112
→ OpenCV SFace / MobileFaceNet
→ 128D embedding
→ L2 normalization
→ cosine similarity matching
→ Top-1 / Top-2 / margin

Current gallery:
- 51 identities
- 1 enrollment image per identity
- gallery file:
  `data/gallery/embeddings.npz`
- current embedding shape:
  `(51, 128)`

Current constraints:
- Do NOT collect additional real images.
- Do NOT train or fine-tune any model.
- Do NOT change YuNet.
- Do NOT change SFace.
- Do NOT increase webcam resolution.
- Do NOT change face alignment.
- Do NOT introduce a classifier.
- Do NOT add a second encoder.
- Do NOT implement Docker/database/RTSP/tracking.

We are starting a new experiment:

==================================================
MULTI-EMBEDDING PER IDENTITY USING LIGHT AUGMENTATION
==================================================

Goal:

Keep the runtime encoder unchanged, but make the enrollment gallery more robust by creating several identity-preserving embedding variants from the same aligned enrollment face.

Concept:

raw enrollment image
        ↓
YuNet
        ↓
alignment 112x112
        ↓
aligned face
        │
        ├── original
        ├── horizontal flip
        ├── mild brightness decrease
        ├── mild brightness increase
        ├── mild contrast variation
        └── optional very mild degradation
                ↓
          same SFace encoder
                ↓
       multiple normalized embeddings
                ↓
         multi-embedding gallery

IMPORTANT:

The runtime webcam query should still use:

webcam
→ YuNet
→ alignment
→ SFace ONCE
→ one query embedding

Do NOT run multiple encoder models at runtime.

==================================================
1. PRESERVE CURRENT BASELINE
==================================================

Do not overwrite or destroy the current single-embedding baseline.

Keep the existing gallery available for comparison.

For example:

`data/gallery/embeddings_single.npz`

and create a new experimental gallery such as:

`data/gallery/embeddings_multi.npz`

If the current file is named differently, preserve it cleanly without unnecessary refactoring.

==================================================
2. AUGMENTATION POLICY
==================================================

Apply augmentation AFTER face alignment.

Do NOT augment the raw full enrollment image before YuNet detection.

Reason:
this experiment is intended to improve recognition robustness, not detection.

Use only mild identity-preserving transforms.

Start with approximately 5 variants per identity:

V0:
- original aligned face

V1:
- horizontal flip

V2:
- mild darker version

V3:
- mild brighter version

V4:
- mild contrast variation

Optionally V5:
- extremely mild blur/compression-like degradation

Do not add V5 unless implementation remains simple and the effect is realistic.

Do NOT use aggressive transforms such as:

- large rotations
- perspective warp
- heavy crop
- random occlusion
- artificial glasses
- large color shift
- heavy blur
- synthetic profile generation
- geometric deformation
- dozens of variants

The purpose is NOT to manufacture fake data.

==================================================
3. AUGMENTATION IMPLEMENTATION
==================================================

Keep augmentation logic small and isolated.

Suggested location:

`src/preprocessing/face_augmentation.py`

or another existing preprocessing location if the repository already has one.

Do not create a large augmentation framework.

The function should conceptually accept:

aligned_face

and return:

list of aligned face variants

All variants must preserve:

- same dimensions
- valid image dtype/range
- identity information

==================================================
4. MULTI-EMBEDDING GALLERY
==================================================

For every identity:

1. load enrollment image
2. YuNet detection
3. align face
4. generate augmentation variants
5. encode each variant using the SAME SFace model
6. L2-normalize every embedding
7. store all embeddings for that identity

Target conceptual shape:

    embeddings.shape = (N, V, D)

where:

N = number of identities
V = variants per identity
D = 128

For example:

    (51, 5, 128)

Also store:

    identities.shape = (51,)

If a simpler flattened representation fits the current code better, it is acceptable, but the identity-to-variant relationship must remain explicit.

Prefer semantic clarity over clever storage.

==================================================
5. VALIDATION
==================================================

Validate the generated multi-gallery:

- identity count is correct
- same number of variants per identity if fixed-size design is used
- embedding dimension = 128
- all values finite
- no NaN
- no Inf
- every embedding has L2 norm approximately 1

Fail clearly if invalid.

==================================================
6. MATCHING STRATEGIES
==================================================

Do NOT immediately assume MAX similarity is best.

Implement and compare at least these per-identity aggregation methods:

A. MAX

For identity A:

score_A = max(sim(query, A_variant_i))

B. TOP-2 MEAN

For identity A:

score_A =
mean(two highest similarities among A variants)

If V < 2, handle safely.

Optionally compare:

C. MEAN over all variants

but only if implementation remains small.

For each aggregation strategy:

- compute one final score per identity
- then compute:
  - Top-1 identity
  - Top-1 score
  - Top-2 identity
  - Top-2 score
  - margin = Top1 - Top2

==================================================
7. IMPORTANT SAFETY AGAINST FALSE MATCHES
==================================================

Remember:

adding more embeddings per identity can increase the probability that one synthetic variant accidentally becomes very similar to an incorrect query.

Therefore do NOT judge improvement only by:

Top-1 score increasing.

Always also inspect:

- Top-2 score
- Top1-Top2 margin
- incorrect identity scores
- inter-identity similarity behavior

==================================================
8. OFFLINE GALLERY ANALYSIS
==================================================

After generating the multi-embedding gallery, perform an offline analysis.

Compare:

Baseline:
1 embedding/person

vs

Multi:
V embeddings/person

Report:

- number of identities
- variants per identity
- total embeddings
- gallery file size
- build time
- average encoder time per embedding

For multi-gallery inter-identity analysis:

Compare identities using the SAME aggregation rule used for recognition.

Do not compare embeddings from the same identity as impostor pairs.

Report for each matching strategy:

- mean inter-identity score
- median inter-identity score
- minimum inter-identity score
- maximum inter-identity score
- identity pair with maximum score

This helps reveal whether augmentation accidentally increases impostor similarity.

==================================================
9. LIVE RECOGNITION COMPARISON
==================================================

Reuse the current webcam recognition pipeline.

Do not change YuNet/SFace/runtime resolution.

Add a way to compare:

A. single-embedding gallery
B. multi-embedding gallery with MAX
C. multi-embedding gallery with TOP-2 MEAN

Prefer a simple runtime option or separate thin script.

Do not build a configuration framework.

For the same live query, report:

Baseline:
- Top1
- Top1 similarity
- Top2
- margin

Multi MAX:
- Top1
- Top1 score
- Top2
- margin

Multi TOP-2 MEAN:
- Top1
- Top1 score
- Top2
- margin

==================================================
10. PERFORMANCE BENCHMARK
==================================================

Measure actual additional runtime cost.

At minimum report:

- SFace query encoding latency
- single-gallery matching latency
- multi-gallery matching latency using MAX
- multi-gallery matching latency using TOP-2 MEAN
- end-to-end FPS

Important:

The query encoder must still run once.

The expected extra runtime cost should come mainly from additional cosine comparisons.

Do not optimize unless measured matching latency becomes meaningful.

==================================================
11. DO NOT CHANGE THRESHOLD YET
==================================================

Do NOT implement a final KNOWN/UNKNOWN threshold.

Do NOT tune threshold to make the experiment look better.

This experiment is about comparing representation/gallery strategies.

Keep the decision layer observational:

- Top1
- Top2
- margin

==================================================
12. DO NOT IMPLEMENT
==================================================

Do NOT implement:

- model training
- fine-tuning
- ArcFace training
- triplet loss
- contrastive loss
- Logistic Regression
- SVM
- kNN classifier
- second encoder
- ensemble models
- new face detector
- new webcam resolution
- temporal aggregation
- tracking
- RTSP
- Docker
- database
- API
- Kubernetes

==================================================
13. REQUIRED COMPARISON
==================================================

At the end, produce a compact comparison table:

System:
1. Single embedding/person
2. Multi embedding/person + MAX
3. Multi embedding/person + TOP-2 MEAN

Compare:

- gallery embeddings count
- gallery size
- matching latency
- end-to-end FPS
- live Top-1 score
- live Top-2 score
- live margin
- offline max inter-identity similarity

Do not declare a winner based on one metric only.

Prefer the strategy that improves genuine separation without causing a large increase in impostor similarity or runtime cost.

==================================================
14. INTERPRETATION RULES
==================================================

Do not claim:

"augmentation improved accuracy"

unless there is actual labeled evaluation data supporting accuracy.

Instead report facts such as:

- genuine Top-1 score increased/decreased
- margin increased/decreased
- impostor similarity increased/decreased
- matching latency changed
- FPS changed

Clearly distinguish observation from conclusion.

==================================================
15. SECURITY
==================================================

Do not save or commit augmented face images by default.

Generate augmentations in memory.

If debug saving is explicitly enabled:

- save only locally
- use ignored directory
- never overwrite enrollment image
- do not commit biometric data

Generated multi-gallery embeddings must remain ignored by Git.

==================================================
16. MEMORY UPDATE
==================================================

After completion update:

`.agents/short_memory.md`

Include:

- augmentation variants used
- number of embeddings/person
- multi-gallery shape
- gallery file size
- matching aggregation methods
- benchmark results
- comparison with single baseline
- blockers/issues
- recommended next experiment

Update `.agents/long_memory.md` only if a durable architecture decision is accepted.

Do not record an experimental strategy as permanent architecture before results support it.

==================================================
17. STOP CONDITION
==================================================

Stop after:

- multi-embedding gallery is generated
- gallery is validated
- MAX and TOP-2 MEAN matching are implemented
- baseline vs multi-gallery is benchmarked
- live comparison works as far as local environment permits

Do NOT proceed to:
- model replacement
- multi-model ensemble
- training
- threshold calibration

Before coding:

1. inspect the current repository
2. report which files will change
3. describe the exact augmentation variants
4. describe gallery storage format
5. describe MAX vs TOP-2 MEAN
6. explain expected runtime cost
7. then implement