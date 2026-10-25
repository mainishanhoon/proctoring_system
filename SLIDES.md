# Slide Deck — Proctoring System (bullet slides)

-- Slide 1: Title

Proctoring System — Real-time Monitoring

-- Slide 2: Problem & Value
- Problem: Cheating during remote exams is hard to detect.
- Value: Real-time alerts, scalable reporting, human-in-the-loop verification.

-- Slide 3: Architecture Overview
- FastAPI backend (REST + WebSocket)
- CV pipeline: MediaPipe + OpenCV
- DB: SQLite (SQLAlchemy)
- Frontend: Vanilla JS (webcam + WS)

-- Slide 4: Detection Pipeline
- Face detection → mesh landmarks → gaze
- Liveness (EAR, challenges)
- Deepfake heuristics (FFT, noise, blending)
- Face verification (HOG + cosine similarity)

-- Slide 5: Data Flow
- Student uploads frame → `analyse_frame` → detectors → DB → WebSocket alerts

-- Slide 6: Trade-offs
- Heuristics are fast, explainable, but brittle
- No heavy DL models, so lower infra cost

-- Slide 7: Scaling Plan
- Move detectors to worker pool (Celery/RQ)
- Use Redis for session state and WS routing
- Replace SQLite with PostgreSQL

-- Slide 8: Security & Privacy
- Store only embeddings, encryption at rest, access control, retention policies

-- Slide 9: Next Steps
- Add CI tests for detectors, instrument metrics, containerize and deploy
