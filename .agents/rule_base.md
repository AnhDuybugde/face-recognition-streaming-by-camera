# Rule Base

These rules govern all AI-assisted work in this repository.

## 1. Anti-Hallucination

- Never claim a library, API, class, method, model capability, parameter, or configuration exists unless reasonably verified.
- If uncertain, explicitly say what is uncertain.
- Prefer checking installed versions, official documentation, or existing source code before making version-sensitive changes.
- Never fabricate benchmark results.
- Never fabricate successful execution.
- Never say code was tested unless it was actually executed.
- Distinguish clearly between observed fact, inference, and recommendation.

## 2. Explain Important Decisions

For meaningful architectural or implementation choices, briefly explain what was chosen, why it was chosen, what simpler alternative existed, and why additional complexity is or is not justified.

## 3. Keep the Project Small

Before creating a new module, dependency, abstraction, service, or configuration layer, ask whether the current project has a real problem that requires this. If no, do not add it.

Avoid unnecessary design patterns, wrapper-on-wrapper abstractions, generic framework code, premature interfaces, unnecessary inheritance, excessive configuration, and duplicated logic.

## 4. Prefer Incremental Changes

Make small, reviewable changes. Build one capability, verify it, then continue.

## 5. Do Not Hide Errors

- Do not use broad `except Exception: pass`.
- Preserve useful error context.
- Fail clearly when configuration is invalid.
- Camera disconnect handling may be added when RTSP work begins.

## 6. Security and Data Protection

Never place company/private datasets, API keys, tokens, passwords, private RTSP URLs, credentials, `.env` files, customer images, or confidential model artifacts in Git.

Do not automatically upload project data to public GitHub, Kaggle, Colab, personal cloud storage, or third-party AI APIs. Treat unknown company data as restricted by default. Use `.gitignore` and `.dockerignore`.

## 7. Docker Rules

Docker images should contain application code, dependencies, and safe runtime defaults. They should normally not contain datasets, private videos, secrets, credentials, local caches, or debug outputs. The container uses the CPU/GPU resources of its host. Do not introduce Kubernetes during initial learning phases.

## 8. Performance Mindset

Whenever inference is implemented, evaluate correctness, latency, FPS/throughput, CPU usage when relevant, GPU usage when relevant, and VRAM when relevant. Never optimize only for accuracy or before obtaining a baseline.

## 9. Model Selection

Choose models according to task, latency target, hardware, deployment constraints, acceptable accuracy, and dataset characteristics. Do not select GRU/LSTM merely because a dataset is small.

## 10. Dependency Discipline

Before adding a dependency, check whether the standard library or an existing dependency is enough, explain why the new dependency is needed, add only the required package, and avoid large frameworks for small tasks.

## 11. Code Quality

Prefer explicit names, short functions, clear data flow, useful type hints, and docstrings for non-obvious public functions. Avoid giant files/classes, hidden global state, magic constants, and duplicated logic.

## 12. Testing

Test the smallest meaningful unit available. Do not build a large testing framework before there is code worth testing.

## 13. Memory Discipline

At the start of a session, read `long_memory.md`, `short_memory.md`, and `rule_base.md`. At the end of a meaningful session, update short memory and update long memory only for durable knowledge.

## 14. Communication with the User

When proposing meaningful changes, report files to change, reason, expected effect, and risk or tradeoff. For learning support, explain the engineering concept behind the code.

## 15. Stop Conditions

Stop and ask/report instead of escalating complexity when requirements are materially ambiguous, sensitive-data risk exists, required hardware is unavailable, an external service or credential is required, or measured performance does not justify an optimization.
