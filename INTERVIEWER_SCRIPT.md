# 30-Minute Interviewer Script — Proctoring System Walkthrough

Total time: 30 minutes. Use this script to guide an interviewer or to practice presenting the project.

0–2 min — Quick intro
- One-line elevator pitch: "A real-time online proctoring system built with FastAPI, MediaPipe and OpenCV that detects face presence, gaze, liveness, deepfakes and tab switches, and delivers live alerts to invigilators."
- Quick personal role and contribution summary (30–45s).

2–10 min — Architecture overview (8 minutes)
- Show `INTERVIEW_SUMMARY.md` slide or describe components:
  - FastAPI app (`backend/main.py`) — REST + WS + static
  - Detector (`backend/detector.py`) orchestrates MediaPipe + sub-modules
  - Liveness (`backend/liveness_detector.py`) maintains per-session state
  - Deepfake (`backend/deepfake_detector.py`) heuristic analyses
  - Face verification (`backend/face_verifier.py`) HOG + cosine similarity
  - SQLite + SQLAlchemy for persistence
- Interviewer prompt: "Which part would you like to dive deeper into — scaling, privacy, or detection?"

10–20 min — Deep dive (pick one) (10 minutes)
- Option A: Detection pipeline details (recommended)
  - Walk through `analyse_frame` flow: detection → mesh → gaze → liveness → deepfake → verification
  - Explain gaze calculation (iris ratio + nose vertical) and thresholds
  - Explain liveness (EAR, blink history, head pose, challenges) including `LivenessState`
  - Explain deepfake heuristics (FFT, noise/MRF, DCT blocks, blending analysis)
  - Discuss limitations and mitigation: aggregation across frames, threshold tuning, human-in-loop
- Option B: Scalability & production readiness
  - Talk about moving detectors to worker pool, Redis for state, PG for DB, socket routing, containerization
  - Discuss latency budgets and batching strategies

20–25 min — Demo / Example requests (5 minutes)
- If local demo available: start backend (`uvicorn main:app`) and open `/docs` and `/invigilator`.
- If not, walk through sample requests:
  - Register face: POST `/api/students/{id}/register-face` with image
  - Create session: POST `/api/sessions`
  - Submit frame: POST `/api/sessions/{id}/frame` and show JSON detection result
  - Show WebSocket alerts sent to invigilator

25–28 min — Operational & security considerations (3 minutes)
- Biometric data handling (embedding storage, encryption, retention policies)
- Auth & access control (recommend JWT/OAuth for APIs and WS)
- Monitoring: metrics for false positives, processing latency, error rates

28–30 min — Q&A and closing (2 minutes)
- Ask interviewer if they want follow-up materials: slides, flashcards, code tour
- Offer to walk through any file live or run local checks

Appendix: Suggested interviewer questions to ask candidate
- How would you scale WebSocket alerts across multiple app instances?
- Explain how you would test the deepfake detector in CI.
- What privacy and compliance steps would you implement before production?
- How would you migrate the face verification to a neural embedding while minimizing privacy impact?
