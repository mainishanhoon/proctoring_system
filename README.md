# Online Proctoring System

A real-time online proctoring system built with Python (FastAPI) and vanilla JavaScript. Monitors students during online exams using AI-powered detection for face presence, eye tracking, multiple person detection, liveness verification, deepfake detection, tab switching, and face identity verification — all using OpenCV and MediaPipe with no external model downloads required.

---

## Features

- **Face detection** — verifies student is present and visible throughout the exam
- **Multiple person detection** — flags if more than one person appears in frame
- **Eye/gaze tracking** — detects if student is looking away from the screen
- **Liveness detection** — passive blink analysis and active challenges to prevent photo spoofing
- **Deepfake detection** — frequency analysis to detect AI-generated or face-swapped video
- **Face registration and verification** — registers student face once using OpenCV HOG descriptors and verifies identity every frame. No external model download required.
- **Tab switch detection** — instant alert when student switches tabs or minimises window
- **Live invigilator dashboard** — real-time violation alerts via WebSocket for all active sessions
- **Session reports** — full violation breakdown per student per exam
- **Multi-student support** — invigilator can monitor all students in an exam simultaneously

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python, FastAPI, Uvicorn |
| Computer Vision | MediaPipe, OpenCV |
| Face Verification | OpenCV HOG descriptor (no external model needed) |
| Database | SQLite (via SQLAlchemy) |
| Real-time | WebSockets |
| Frontend | HTML, CSS, Vanilla JavaScript |

---

## Project Structure

```
proctoring-system/
├── backend/
│   ├── main.py                  # FastAPI app, all routes, WebSocket manager
│   ├── detector.py              # Main frame analysis orchestrator
│   ├── liveness_detector.py     # Blink detection and challenge system
│   ├── deepfake_detector.py     # FFT frequency analysis for deepfake detection
│   ├── face_verifier.py         # Face encoding and identity verification (OpenCV HOG)
│   ├── database.py              # SQLAlchemy models and database setup
│   └── models.py                # Pydantic schemas for request/response
├── frontend/
│   ├── student.html             # Student exam page with webcam
│   ├── invigilator.html         # Invigilator monitoring dashboard
│   ├── register.html            # Face registration page
│   └── js/
│       ├── student.js           # Webcam capture, tab detection
│       └── invigilator.js       # WebSocket listener, dashboard logic
├── violations/                  # Saved violation screenshots (not in repo)
├── requirements.txt             # Python dependencies
├── .gitignore
└── README.md
```

---

## Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.11 (recommended) |
| pip | Latest |
| Git | Any recent version |
| Browser | Chrome or Firefox (for WebRTC webcam access) |

---

## Installation

### Windows

**1 — Clone the repository**

```cmd
git clone https://github.com/your-username/proctoring-system.git
cd proctoring-system
```

**2 — Create and activate virtual environment**

```cmd
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**3 — Install dependencies**

```cmd
pip install -r requirements.txt
```

**4 — Fix protobuf version (required for MediaPipe compatibility)**

```cmd
pip uninstall protobuf
pip install protobuf==3.20.3
```

**5 — Create required folders**

```cmd
mkdir violations
```

**6 — Run the server**

```cmd
cd backend
uvicorn main:app --reload --port 8000
```

---

### macOS / Linux

**1 — Clone the repository**

```bash
git clone https://github.com/your-username/proctoring-system.git
cd proctoring-system
```

**2 — Create and activate virtual environment**

```bash
python3 -m venv venv
source venv/bin/activate
```

**3 — Install dependencies**

```bash
pip install -r requirements.txt
```

**4 — Fix protobuf version (required for MediaPipe compatibility)**

```bash
pip install protobuf==3.20.3 --force-reinstall
```

**5 — Create required folders**

```bash
mkdir -p violations
```

**6 — Run the server**

```bash
cd backend
uvicorn main:app --reload --port 8000
```

---

## Usage

Once the server is running, open your browser and go to:

| Page | URL | Who uses it |
|---|---|---|
| API documentation | http://127.0.0.1:8000/docs | Setup and testing |
| Face registration | http://127.0.0.1:8000/register | Student (once before exam) |
| Student exam page | http://127.0.0.1:8000/ | Student |
| Invigilator dashboard | http://127.0.0.1:8000/invigilator | Invigilator |

---

## Setting Up an Exam (Step by Step)

### 1 — Create an exam

Go to `http://127.0.0.1:8000/docs`, find **POST /api/exams** and click Try it out:

```json
{
  "title": "Computer Science Final",
  "description": "End of semester examination"
}
```

Note the `id` returned — this is your **exam ID**.

### 2 — Create students

Find **POST /api/students** and create each student:

```json
{
  "name": "Student Name",
  "email": "student@example.com"
}
```

Note each student's `id`.

### 3 — Create sessions

Find **POST /api/sessions** and create one session per student:

```json
{
  "exam_id": 1,
  "student_id": 1
}
```

Note each session's `id` — students will use this to join the exam.

### 4 — Register student faces

Each student goes to `http://127.0.0.1:8000/register` and registers their face using their student ID. This only needs to be done once — the face is stored and reused for all future exams.

### 5 — Start monitoring

Invigilator opens `http://127.0.0.1:8000/invigilator`, enters the exam ID, and clicks Load to see all students automatically.

### 6 — Students join

Each student opens `http://127.0.0.1:8000/`, enters their session ID and name, allows camera access, and begins the exam.

---

## Violation Types

| Violation | Severity | Description |
|---|---|---|
| `no_face` | Medium | Student not visible in camera |
| `multiple_faces` | High | More than one person detected |
| `suspicious_gaze` | Medium | Student looking away from screen |
| `tab_switch` | High | Student switched tabs or minimised window |
| `no_blink_detected` | High | No blinking detected — possible photo spoof |
| `low_blink_rate` | Medium | Unusually low blink rate |
| `no_head_movement` | Medium | No head movement — possible static image |
| `challenge_failed` | High | Failed liveness challenge |
| `deepfake_detected` | High | Deepfake or face-swap detected |
| `wrong_person` | High | Face does not match registered student |

---

## Environment Notes

### Windows specific

- Use `venv\Scripts\activate` to activate the virtual environment
- Use backslashes for paths in the terminal
- If MediaPipe has protobuf conflicts run:

```cmd
pip install protobuf==3.20.3 --force-reinstall
```

- If you see `AttributeError: module mediapipe has no attribute solutions`, your MediaPipe version is wrong. Fix with:

```cmd
pip install mediapipe==0.10.14
```

### macOS specific

- Use `source venv/bin/activate` to activate the virtual environment
- If OpenCV has camera permission issues go to System Preferences → Security and Privacy → Camera and allow Terminal
- On Apple Silicon (M1/M2) you may need to install MediaPipe differently:

```bash
pip install mediapipe-silicon
```

### Both platforms

- Always activate the virtual environment before running the server
- The browser must allow camera access — Chrome and Firefox both work
- Always run the server from inside the `backend/` folder
- The SQLite database (`proctoring.db`) is created automatically on first run

---

## Troubleshooting

**Server won't start — protobuf conflict**

```
pip install protobuf==3.20.3 --force-reinstall
```

**MediaPipe has no attribute solutions**

```
pip install mediapipe==0.10.14
pip install protobuf==3.20.3 --force-reinstall
```

**Camera not working in browser**

- Make sure you are on `http://127.0.0.1:8000` not a different address
- Chrome: click the camera icon in the address bar and allow access
- Firefox: click the lock icon in the address bar and allow camera

**Port already in use**

```
uvicorn main:app --reload --port 8001
```

Then update the `API` constant in all three frontend HTML files from `8000` to `8001`.

**Database errors after schema changes**

Windows:
```cmd
del backend\proctoring.db
```

macOS/Linux:
```bash
rm backend/proctoring.db
```

Then restart the server — it recreates the database automatically.

**Face registration fails — no face detected**

- Ensure you are in a well-lit room
- Look directly at the camera
- Make sure your face fills a reasonable portion of the frame
- Remove glasses or hats if possible

**Liveness challenges failing incorrectly**

The detection thresholds may need tuning for your specific camera and lighting. In `liveness_detector.py` adjust:

```python
EAR_BLINK_THRESHOLD = 0.21  # increase if blinks not detected
MATCH_THRESHOLD = 0.75  # decrease in face_verifier.py if too strict
```

---

## API Reference

Full interactive API documentation is available at:

```
http://127.0.0.1:8000/docs
```

Key endpoints:

| Method | Endpoint | Description |
|---|---|---|
| POST | /api/exams | Create an exam |
| GET | /api/exams | List all exams |
| POST | /api/students | Create a student |
| POST | /api/students/{id}/register-face | Register student face |
| GET | /api/students/{id}/face-status | Check if face is registered |
| POST | /api/sessions | Create a session |
| GET | /api/exams/{id}/sessions | Get all sessions for an exam |
| POST | /api/sessions/{id}/frame | Submit webcam frame for analysis |
| POST | /api/sessions/{id}/tab-switch | Log a tab switch event |
| PUT | /api/sessions/{id}/end | End a session |
| GET | /api/sessions/{id}/report | Get full session report |
| WS | /ws/student/{id} | Student WebSocket connection |
| WS | /ws/invigilator/{id} | Invigilator WebSocket connection |

---

## License

MIT License — free to use and modify.
