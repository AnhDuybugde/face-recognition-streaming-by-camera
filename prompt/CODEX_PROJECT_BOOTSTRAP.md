# CODEX PROJECT BOOTSTRAP — Real-time Computer Vision Internship Practice

## 0. Purpose

This repository is a **small, production-oriented Computer Vision practice project** for preparing for an AI Intern / AI Developer role.

The project must teach and demonstrate:

- Webcam / video / RTSP input handling.
- Real-time Computer Vision inference.
- Object detection first; tracking later.
- Business logic on top of model outputs.
- FPS, latency, throughput, GPU/CPU usage awareness.
- Safe handling of data and secrets.
- Clean project organization.
- Dockerized execution.
- GPU usage from the host machine.
- Minimal complexity: do not add infrastructure unless it solves a real problem.

This is **not** a research repository and **not** a framework-building exercise.

Primary rule:

> Build the smallest correct system first. Add complexity only when a measured problem requires it.

---

# 1. Codex Operating Instructions

Before modifying the repository:

1. Read this file completely.
2. Read `.agents/long_memory.md`.
3. Read `.agents/short_memory.md`.
4. Read `.agents/rule_base.md`.
5. Inspect the existing repository before creating files.
6. Reuse existing code when possible.
7. Do not introduce new libraries, services, abstractions, or folders without a concrete reason.

After every meaningful work session:

1. Update `.agents/short_memory.md`.
2. Update `.agents/long_memory.md` only if a durable decision, architecture choice, constraint, or completed milestone should be remembered across sessions.
3. Do not fill memory files with verbose logs.
4. Keep memory concise and actionable.

---

# 2. Initial Repository Structure

Create the following structure if the repository is empty:

```text
project-root/
│
├── .agents/
│   ├── long_memory.md
│   ├── short_memory.md
│   └── rule_base.md
│
├── src/
│   ├── __init__.py
│   ├── app.py
│   │
│   ├── capture/
│   │   ├── __init__.py
│   │   └── video_source.py
│   │
│   ├── inference/
│   │   ├── __init__.py
│   │   └── detector.py
│   │
│   ├── tracking/
│   │   ├── __init__.py
│   │   └── tracker.py
│   │
│   ├── logic/
│   │   ├── __init__.py
│   │   └── events.py
│   │
│   └── utils/
│       ├── __init__.py
│       ├── metrics.py
│       └── logging_utils.py
│
├── configs/
│   └── default.yaml
│
├── scripts/
│   ├── run_webcam.py
│   ├── run_video.py
│   └── benchmark.py
│
├── tests/
│
├── data/
│   ├── input/
│   └── output/
│
├── .gitignore
├── .dockerignore
├── requirements.txt
├── Dockerfile
├── README.md
└── CODEX_PROJECT_BOOTSTRAP.md
```

Do **not** create all implementation files with large placeholder code.

Create only lightweight scaffolding first.

Folders may remain empty until needed.

---

# 3. Folder Responsibilities

## `src/capture/`

Responsible only for acquiring frames.

Expected future inputs:

- webcam
- `.mp4`
- RTSP stream

It must not contain model logic.

Conceptual interface:

```python
frame = source.read()
```

---

## `src/inference/`

Responsible only for AI inference.

Initial goal:

```text
frame
  ↓
detector
  ↓
detections
```

Detection output should eventually use a simple internal representation similar to:

```python
{
    "class_id": 0,
    "class_name": "person",
    "confidence": 0.91,
    "bbox": [x1, y1, x2, y2],
}
```

Do not couple the rest of the project directly to a vendor-specific result object unless necessary.

---

## `src/tracking/`

Tracking comes **after detection works correctly**.

Expected responsibility:

```text
detections at frame t
        ↓
tracker
        ↓
persistent object IDs
```

Do not implement tracking during Phase 1.

---

## `src/logic/`

Business logic.

Examples:

- line crossing
- enter / exit counting
- region intrusion
- event creation

Model inference must not directly contain business rules.

---

## `src/utils/metrics.py`

Runtime engineering measurements:

- FPS
- per-frame latency
- moving average latency
- processed frame count
- dropped/skipped frame count later if needed

Measurements should be simple and interpretable.

---

## `configs/`

Runtime configuration only.

Examples:

```yaml
source: 0
confidence_threshold: 0.5
device: auto
display: true
```

Do not put secrets into config files committed to Git.

---

## `scripts/`

Thin entry points.

Scripts should call reusable code from `src/`.

Avoid implementing the whole application inside one script.

---

## `data/`

Only local testing data.

Rules:

- never commit private/company data
- never commit large videos
- never commit camera recordings unless explicitly safe
- add appropriate `.gitignore` rules

---

# 4. `.agents/long_memory.md`

Create this file with the following purpose:

```markdown
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
```

---

# 5. `.agents/short_memory.md`

Create this file for **current-session context only**.

Initial content:

```markdown
# Short Memory

## Current Session Goal

Initialize the repository and prepare the smallest valid structure.

## Current State

- Repository newly initialized.
- No Computer Vision pipeline implemented yet.

## Current Task

Create project scaffolding without adding unnecessary implementation.

## Next Actions

1. Create the minimal folder structure.
2. Create memory files.
3. Create rule base.
4. Create a minimal README.
5. Stop before implementing the model unless explicitly asked.

## Open Questions

None currently.

## Last Session Summary

No previous session.
```

Rules for updating Short Memory:

- Keep under roughly 100 lines.
- Replace obsolete temporary information instead of endlessly appending.
- Record:
  - current task
  - files being changed
  - current blockers
  - immediate next steps
  - important temporary reasoning
- Do not treat it as a chat transcript.

---

# 6. `.agents/rule_base.md`

Create it with the following content.

```markdown
# Rule Base

These rules govern all AI-assisted work in this repository.

## 1. Anti-Hallucination

- Never claim a library, API, class, method, model capability, parameter, or configuration exists unless reasonably verified.
- If uncertain, explicitly say what is uncertain.
- Prefer checking installed versions, official documentation, or existing source code before making version-sensitive changes.
- Never fabricate benchmark results.
- Never fabricate successful execution.
- Never say code was tested unless it was actually executed.
- Distinguish clearly between:
  - observed fact
  - inference
  - recommendation

## 2. Explain Important Decisions

For meaningful architectural or implementation choices, briefly explain:

- what was chosen
- why it was chosen
- what simpler alternative existed
- why additional complexity is or is not justified

Do not over-explain trivial syntax.

## 3. Keep the Project Small

Before creating a new module, dependency, abstraction, service, or configuration layer, ask:

> Does the current project have a real problem that requires this?

If no, do not add it.

Avoid:

- unnecessary design patterns
- wrapper-on-wrapper abstractions
- generic framework code
- premature interfaces
- unnecessary inheritance
- excessive configuration
- duplicated utilities

## 4. Prefer Incremental Changes

Make small, reviewable changes.

Good:

```text
capture webcam
→ verify
→ add detector
→ verify
→ add metrics
```

Bad:

```text
implement full architecture + tracker + API + Docker + queues in one step
```

## 5. Do Not Hide Errors

- Do not use broad `except Exception: pass`.
- Preserve useful error context.
- Fail clearly when configuration is invalid.
- Camera disconnect handling may be added when RTSP work begins.

## 6. Security and Data Protection

Never place these in Git:

- company/private datasets
- API keys
- tokens
- passwords
- private RTSP URLs
- credentials
- `.env`
- customer images
- model artifacts containing confidential material

Do not automatically upload project data to:

- public GitHub
- Kaggle
- Colab
- personal cloud storage
- third-party AI APIs

Treat unknown company data as restricted by default.

Use `.gitignore` and `.dockerignore`.

Before adding logging, consider whether logs may expose:

- file paths
- identities
- image content
- camera URLs
- credentials
- request payloads

## 7. Docker Rules

Docker images should contain:

- application code
- dependencies
- runtime configuration defaults when safe

Docker images should normally not contain:

- datasets
- private videos
- secrets
- credentials
- local caches
- debug outputs

Remember:

> The container uses the CPU/GPU resources of the host on which it runs.

Do not introduce Kubernetes during the initial learning phases.

## 8. Performance Mindset

Whenever inference is implemented, evaluate at least:

- correctness
- latency
- FPS / throughput
- CPU usage when relevant
- GPU usage when relevant
- VRAM when relevant

Never optimize only for accuracy.

Do not optimize before obtaining a baseline.

## 9. Model Selection

Choose models according to:

- task
- latency target
- hardware
- deployment constraints
- acceptable accuracy
- dataset characteristics

Do not select GRU/LSTM merely because the dataset is small.

Temporal models should be used only when temporal information is needed.

## 10. Dependency Discipline

Before adding a dependency:

1. Check whether Python standard library or an existing dependency is enough.
2. Explain why the new dependency is needed.
3. Add only the required package.
4. Avoid adding large frameworks for small tasks.

## 11. Code Quality

Prefer:

- explicit names
- short functions
- clear data flow
- type hints when useful
- docstrings for non-obvious public functions

Avoid:

- giant files
- giant classes
- hidden global state
- magic constants
- duplicated logic

## 12. Testing

Test the smallest meaningful unit available.

Examples:

- camera opens
- one frame is read
- detector returns expected structure
- metrics compute correctly

Do not build a large testing framework before there is code worth testing.

## 13. Memory Discipline

At the start of a session:

1. read `long_memory.md`
2. read `short_memory.md`
3. read `rule_base.md`

At the end of a meaningful session:

1. update `short_memory.md`
2. update `long_memory.md` only for durable knowledge

Never overwrite long-term decisions accidentally.

## 14. Communication with the User

When proposing meaningful changes, report:

- files to change
- reason
- expected effect
- risk or tradeoff if any

When the user asks for learning support, explain the engineering concept behind the code rather than only producing code.

## 15. Stop Conditions

Stop and ask/report instead of escalating complexity when:

- requirements are ambiguous enough to materially change architecture
- a change would introduce sensitive-data risk
- required hardware is unavailable
- an external service or credential is required
- measured performance does not justify the proposed optimization
```

---

# 7. `.gitignore`

Create a practical initial `.gitignore`.

At minimum ignore:

```gitignore
# Python
__pycache__/
*.py[cod]
.venv/
venv/

# Environment / secrets
.env
.env.*
*.pem
*.key

# IDE
.vscode/
.idea/

# Local data
data/input/*
data/output/*
!data/input/.gitkeep
!data/output/.gitkeep

# Models / large artifacts
*.pt
*.pth
*.onnx
*.engine

# Logs
*.log
logs/

# OS
.DS_Store
Thumbs.db
```

Add `.gitkeep` files only if necessary to retain empty directories.

---

# 8. `.dockerignore`

Create:

```text
.git
.gitignore
.agents
data
tests
__pycache__
*.pyc
.venv
venv
.env
.env.*
*.log
.vscode
.idea
```

Important:

Do not blindly use:

```dockerfile
COPY . .
```

without checking `.dockerignore`.

---

# 9. Initial README

Create a small README only.

It should contain:

```markdown
# Real-time Computer Vision Practice

Small production-oriented Computer Vision project for AI Developer internship preparation.

## Planned Pipeline

camera / video / RTSP
→ capture
→ detection
→ tracking
→ business logic
→ metrics
→ Docker

## Engineering Priorities

- correctness
- latency
- FPS / throughput
- hardware usage
- maintainability
- data safety

## Status

Project initialization.
```

Do not write a long README before the project exists.

---

# 10. Learning / Implementation Roadmap

Follow these phases strictly unless the user changes priorities.

---

## Phase 0 — Repository Setup

Goal:

Create the clean repository structure and agent memory system.

Deliverables:

- `.agents/*`
- `.gitignore`
- `.dockerignore`
- minimal `README.md`
- minimal source structure

No model yet.

Completion condition:

The repository is clean, understandable, and ready to develop.

---

## Phase 1 — Webcam Capture

Goal:

Read frames from the user's local webcam.

Target flow:

```text
webcam
  ↓
OpenCV
  ↓
frame
  ↓
display
```

Learn:

- `cv2.VideoCapture`
- frame loop
- FPS meaning
- frame dimensions
- clean shutdown

Do not add:

- YOLO
- tracking
- async
- Docker

Completion condition:

Camera opens and displays reliably.

---

## Phase 2 — Video File Input

Goal:

Use the same capture abstraction for `.mp4`.

Learn:

- source abstraction
- native video FPS
- end-of-file behavior

Completion condition:

Switching from webcam to video requires minimal code changes.

---

## Phase 3 — Object Detection

Goal:

Run a pretrained detector on frames.

Initial target:

person detection is sufficient.

Pipeline:

```text
frame
  ↓
preprocess
  ↓
detector
  ↓
detections
  ↓
visualization
```

Priorities:

1. correct output
2. readable code
3. baseline speed

Do not fine-tune yet unless a real need appears.

---

## Phase 4 — Runtime Metrics

Goal:

Measure engineering performance.

Track:

- preprocessing latency
- inference latency
- postprocessing latency
- end-to-end latency
- FPS

Learn difference between:

```text
model inference FPS
```

and:

```text
end-to-end application FPS
```

Completion condition:

Performance numbers are measured rather than guessed.

---

## Phase 5 — Tracking

Goal:

Maintain persistent IDs across frames.

Pipeline:

```text
frame
  ↓
detector
  ↓
detections
  ↓
tracker
  ↓
track IDs
```

Only add tracking after detection baseline is stable.

---

## Phase 6 — Business Logic

Initial example:

person line crossing.

Pipeline:

```text
track trajectory
      ↓
crossing rule
      ↓
event
```

Example event shape:

```python
{
    "event": "person_enter",
    "track_id": 17,
    "timestamp": "...",
}
```

No API required yet.

---

## Phase 7 — RTSP

Goal:

Replace local input with an IP camera / simulated RTSP source.

Learn:

- network stream behavior
- buffering
- reconnect
- stale frames
- latency

Do not implement complex reconnect logic until an actual failure case is observed.

---

## Phase 8 — Docker

Goal:

Run the working application in a container.

Learn:

- image
- container
- volumes
- runtime config
- device access
- dependency reproducibility

First run CPU if simpler.

Then enable host GPU if available.

Remember:

```text
Docker container
     ↓
uses GPU from host
```

Kubernetes is not required.

---

## Phase 9 — Performance Optimization

Only begin after measurements exist.

Possible experiments:

- smaller model
- lower input resolution
- confidence threshold
- frame skipping
- detect every N frames
- tracking between detection frames
- FP16
- ONNX
- TensorRT
- batching if appropriate

Every optimization must compare:

```text
before
vs
after
```

using measured numbers.

---

# 11. Things Explicitly Out of Scope Initially

Do not add these until the project truly needs them:

- Kubernetes
- Kafka
- Redis
- Celery
- Airflow
- Triton Inference Server
- Prometheus
- Grafana
- cloud deployment
- microservices
- database
- authentication
- distributed inference
- training pipeline
- MLOps platform
- frontend

Knowing they exist is sufficient for now.

---

# 12. Development Decision Framework

For each requested feature, reason in this order:

```text
1. What problem are we solving?
2. Is it required now?
3. What is the simplest correct implementation?
4. How will we verify it?
5. What performance/security risk exists?
6. Only then write code.
```

---

# 13. Definition of a Good Solution

A solution is good when it is:

```text
correct
+ understandable
+ measurable
+ secure enough for its context
+ fast enough for the requirement
+ cheap enough for the requirement
```

Not when it merely has the highest accuracy.

---

# 14. First Codex Task

After reading this file, Codex should perform only the following bootstrap task:

1. Inspect the repository.
2. If empty, create the structure defined above.
3. Create:
   - `.agents/long_memory.md`
   - `.agents/short_memory.md`
   - `.agents/rule_base.md`
   - `.gitignore`
   - `.dockerignore`
   - minimal `README.md`
4. Create minimal Python package directories.
5. Do not install dependencies yet.
6. Do not implement YOLO.
7. Do not implement tracking.
8. Do not implement Dockerfile content beyond an empty/minimal placeholder unless explicitly requested.
9. Report exactly what was created and why.
10. Update `.agents/short_memory.md`.

Then stop.

---

# 15. User Learning Preference

The project is also a learning exercise.

When implementing future phases:

- explain the engineering reason behind important code
- prefer concise explanations
- distinguish academic practice from production concerns
- show tradeoffs
- avoid turning the project into a huge codebase
- prioritize concepts that are likely useful for an AI Intern / AI Developer
