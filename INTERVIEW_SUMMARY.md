# Interview Prep — Proctoring System (one-page)

## Purpose
- Real-time online proctoring app: detect face presence, multiple people, gaze, liveness, deepfakes, tab switches, and identity mismatches. Backend: FastAPI. CV: MediaPipe + OpenCV. Persistence: SQLite (SQLAlchemy).

## High-level Architecture
- FastAPI server (`backend/main.py`): REST endpoints, WebSockets, static file serving.
- Detection pipeline (`backend/detector.py`): face detection, mesh, gaze, delegates to liveness, deepfake, verification.
- Liveness (`backend/liveness_detector.py`): EAR-based blink detection, head-pose checks, active challenges.
- Deepfake (`backend/deepfake_detector.py`): FFT, noise, DCT and blending heuristics to score authenticity.
- Face verification (`backend/face_verifier.py`): HOG descriptor extraction, cosine similarity vs stored JSON encoding.
- Database (`backend/database.py`): Exam, Student (face_encoding), Session, Violation.
- Frontend: `frontend/student.html`, `frontend/invigilator.html`, `frontend/register.html` + JS for webcam + WebSocket.

## Data / Sequence Flow (summary)
1. Create exam / students / sessions via API.
2. Student registers a face -> image POST to `/api/students/{id}/register-face` → HOG encoding stored in DB.
3. Student uploads frames (`/api/sessions/{id}/frame`) periodically; server runs `analyse_frame`.
4. `analyse_frame` runs MediaPipe face detector/mesh, gaze calc, calls `analyse_liveness`, `analyse_deepfake`, `verify_face`.
5. Violations persisted to `violations` table; high-severity flags session; WebSocket alerts sent to invigilator(s).

## Key Endpoints
- `POST /api/exams`, `GET /api/exams`
- `POST /api/students`, `POST /api/students/{id}/register-face`
- `POST /api/sessions`, `PUT /api/sessions/{id}/end`
- `POST /api/sessions/{id}/frame` (frame upload)
- `POST /api/sessions/{id}/tab-switch`
- `GET /api/sessions/{id}/report`
- WebSockets: `/ws/student/{id}`, `/ws/invigilator/{id}`

## Runtime Notes & Tunables
- EAR_BLINK_THRESHOLD = 0.18 (in `liveness_detector.py`)
- MATCH_THRESHOLD = 0.70 (in `face_verifier.py`)
- Deepfake verdict thresholds: combined > 0.65 → `likely_deepfake`; > 0.45 → `suspicious`.
- MediaPipe and model objects initialized at import to avoid reinitialisation per-frame.

## How to run locally
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# If MediaPipe/protobuf issues: pip install protobuf==3.20.3 --force-reinstall
cd backend
uvicorn main:app --reload --port 8000
```

Open: http://127.0.0.1:8000 (student), /invigilator, /register and API docs: /docs

## Short talking points (interview-ready)
- Privacy: store minimal embedding (JSON HOG vector); discuss encryption/retention policies.
- Scalability: in-memory WebSocket manager → needs Redis/pubsub or sticky sessions for multi-instance.
- Accuracy: heuristics (FFT, EAR) are fast but brittle; discuss dataset-driven calibration and fallback human review.
- Extensibility: detectors are modular; can add NN models as microservices and offload heavy compute.

## Quick facts to memorize
- DB: `sqlite:///./proctoring.db`
- Liveness: passive + active challenges (blink/turn/smile/nod)
- Face verification: HOG descriptor + cosine similarity
- Violations persisted and surfaced via WebSocket to invigilator UI
