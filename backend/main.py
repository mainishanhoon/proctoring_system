import os
from datetime import datetime

from database import Exam, Student, Violation, create_tables, get_db
from database import Session as DbSession
from detector import analyse_frame
from face_verifier import encoding_to_str, extract_encoding
from fastapi import (
    Depends,
    FastAPI,
    File,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from models import (
    ExamCreate,
    ExamResponse,
    SessionCreate,
    SessionResponse,
    StudentCreate,
    StudentResponse,
    TabSwitchEvent,
)
from sqlalchemy.orm import Session

# ── App setup ─────────────────────────────────────────────

app = FastAPI(title="Proctoring System", version="1.0.0")

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# Serve frontend files
app.mount("/static", StaticFiles(directory="../frontend"), name="static")


# Create DB tables on startup
@app.on_event("startup")
def startup():
    create_tables()
    os.makedirs("../violations", exist_ok=True)
    print("Database ready")
    print("Proctoring system running")


# ── WebSocket connection manager ──────────────────────────


class ConnectionManager:
    def __init__(self):
        # Maps session_id → list of connected invigilator websockets
        self.invigilator_connections: dict = {}
        # Maps session_id → student websocket
        self.student_connections: dict = {}

    async def connect_invigilator(self, session_id: int, websocket: WebSocket):
        await websocket.accept()
        if session_id not in self.invigilator_connections:
            self.invigilator_connections[session_id] = []
        self.invigilator_connections[session_id].append(websocket)
        print(f"Invigilator connected to session {session_id}")

    async def connect_student(self, session_id: int, websocket: WebSocket):
        await websocket.accept()
        self.student_connections[session_id] = websocket
        print(f"Student connected to session {session_id}")

    def disconnect_invigilator(self, session_id: int, websocket: WebSocket):
        if session_id in self.invigilator_connections:
            self.invigilator_connections[session_id].remove(websocket)

    def disconnect_student(self, session_id: int):
        if session_id in self.student_connections:
            del self.student_connections[session_id]

    async def alert_invigilators(self, session_id: int, message: dict):
        """Push violation alert to all invigilators watching this session"""
        if session_id in self.invigilator_connections:
            dead = []
            for ws in self.invigilator_connections[session_id]:
                try:
                    await ws.send_json(message)
                except Exception:
                    dead.append(ws)
            for ws in dead:
                self.invigilator_connections[session_id].remove(ws)

    async def warn_student(self, session_id: int, message: dict):
        """Send warning banner to student"""
        if session_id in self.student_connections:
            try:
                await self.student_connections[session_id].send_json(message)
            except Exception:
                self.disconnect_student(session_id)


manager = ConnectionManager()


# ── Pages ─────────────────────────────────────────────────


@app.get("/")
def home():
    return FileResponse("../frontend/student.html")


@app.get("/invigilator")
def invigilator_page():
    return FileResponse("../frontend/invigilator.html")


@app.get("/register")
def register_page():
    return FileResponse("../frontend/register.html")


# ── Exam routes ───────────────────────────────────────────


@app.post("/api/exams", response_model=ExamResponse)
def create_exam(exam: ExamCreate, db: Session = Depends(get_db)):
    db_exam = Exam(title=exam.title, description=exam.description)
    db.add(db_exam)
    db.commit()
    db.refresh(db_exam)
    return db_exam


@app.get("/api/exams", response_model=list[ExamResponse])
def get_exams(db: Session = Depends(get_db)):
    return db.query(Exam).all()


# ── Student routes ────────────────────────────────────────


@app.post("/api/students", response_model=StudentResponse)
def create_student(student: StudentCreate, db: Session = Depends(get_db)):
    existing = db.query(Student).filter(Student.email == student.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    db_student = Student(name=student.name, email=student.email)
    db.add(db_student)
    db.commit()
    db.refresh(db_student)
    return db_student


@app.get("/api/students", response_model=list[StudentResponse])
def get_students(db: Session = Depends(get_db)):
    return db.query(Student).all()


@app.post("/api/students/{student_id}/register-face")
async def register_face(
    student_id: int, file: UploadFile = File(...), db: Session = Depends(get_db)
):
    student = db.query(Student).filter(Student.id == student_id).first()
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")

    image_bytes = await file.read()
    encoding = extract_encoding(image_bytes)

    if encoding is None:
        raise HTTPException(
            status_code=400,
            detail="No face detected in the photo. Please try again with a clearer selfie.",
        )

    student.face_encoding = encoding_to_str(encoding)
    db.commit()

    return {
        "success": True,
        "message": f"Face registered successfully for {student.name}",
        "student_id": student_id,
    }


@app.get("/api/students/{student_id}/face-status")
def face_registration_status(student_id: int, db: Session = Depends(get_db)):
    student = db.query(Student).filter(Student.id == student_id).first()
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")

    return {
        "student_id": student_id,
        "name": student.name,
        "face_registered": student.face_encoding is not None,
    }


# ── Session routes ────────────────────────────────────────


@app.post("/api/sessions", response_model=SessionResponse)
def create_session(session: SessionCreate, db: Session = Depends(get_db)):
    db_session = DbSession(exam_id=session.exam_id, student_id=session.student_id)
    db.add(db_session)
    db.commit()
    db.refresh(db_session)
    return db_session


@app.put("/api/sessions/{session_id}/end")
def end_session(session_id: int, db: Session = Depends(get_db)):
    session = db.query(DbSession).filter(DbSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    session.ended_at = datetime.utcnow()
    session.status = "completed"
    db.commit()
    return {"message": "Session ended"}


@app.get("/api/exams/{exam_id}/sessions")
def get_exam_sessions(exam_id: int, db: Session = Depends(get_db)):
    """Returns all sessions for a given exam with student info and violation count"""
    sessions = db.query(DbSession).filter(DbSession.exam_id == exam_id).all()

    if not sessions:
        raise HTTPException(status_code=404, detail="No sessions found for this exam")

    result = []
    for session in sessions:
        violation_count = (
            db.query(Violation).filter(Violation.session_id == session.id).count()
        )

        high_count = (
            db.query(Violation)
            .filter(Violation.session_id == session.id, Violation.severity == "high")
            .count()
        )

        result.append(
            {
                "session_id": session.id,
                "student_id": session.student_id,
                "student_name": session.student.name,
                "student_email": session.student.email,
                "status": session.status,
                "started_at": session.started_at,
                "ended_at": session.ended_at,
                "total_violations": violation_count,
                "high_violations": high_count,
            }
        )

    return {
        "exam_id": exam_id,
        "exam_title": sessions[0].exam.title,
        "total_students": len(result),
        "sessions": result,
    }


@app.get("/api/sessions/{session_id}/report")
def get_session_report(session_id: int, db: Session = Depends(get_db)):
    session = db.query(DbSession).filter(DbSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    violations = db.query(Violation).filter(Violation.session_id == session_id).all()
    breakdown = {}
    for v in violations:
        breakdown[v.violation_type] = breakdown.get(v.violation_type, 0) + 1

    return {
        "session_id": session_id,
        "student": session.student.name,
        "exam": session.exam.title,
        "started_at": session.started_at,
        "ended_at": session.ended_at,
        "status": session.status,
        "total_violations": len(violations),
        "violation_breakdown": breakdown,
        "violations": [
            {
                "type": v.violation_type,
                "severity": v.severity,
                "description": v.description,
                "timestamp": v.timestamp,
                "screenshot": v.screenshot,
            }
            for v in violations
        ],
    }


# ── Frame analysis route ──────────────────────────────────


@app.post("/api/sessions/{session_id}/frame")
async def process_frame(
    session_id: int, file: UploadFile = File(...), db: Session = Depends(get_db)
):
    """
    Receives a webcam frame from the student every 2 seconds.
    Runs detection and logs any violations found.
    """
    session = db.query(DbSession).filter(DbSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    image_bytes = await file.read()
    student = session.student
    stored_encoding = student.face_encoding
    detection = analyse_frame(image_bytes, session_id, stored_encoding)

    logged_violations = []

    for v in detection["violations"]:
        # Save to database
        db_violation = Violation(
            session_id=session_id,
            violation_type=v["type"],
            severity=v["severity"],
            description=v["description"],
        )
        db.add(db_violation)
        db.commit()
        db.refresh(db_violation)

        # Flag session if high severity
        if v["severity"] == "high" and session.status == "active":
            session.status = "flagged"
            db.commit()

        logged_violations.append(v)

        # Push live alert to invigilator
        await manager.alert_invigilators(
            session_id,
            {
                "type": "violation",
                "session_id": session_id,
                "violation": v,
                "timestamp": detection["timestamp"],
            },
        )

        # Warn the student
        await manager.warn_student(
            session_id, {"type": "warning", "message": get_warning_message(v["type"])}
        )

    return {
        "session_id": session_id,
        "detection": detection,
        "violations_logged": len(logged_violations),
    }


# ── Tab switch route ──────────────────────────────────────


@app.post("/api/sessions/{session_id}/tab-switch")
async def log_tab_switch(
    session_id: int, event: TabSwitchEvent, db: Session = Depends(get_db)
):
    db_violation = Violation(
        session_id=session_id,
        violation_type="tab_switch",
        severity="high",
        description=event.description,
    )
    db.add(db_violation)
    db.commit()

    # Alert invigilator immediately
    await manager.alert_invigilators(
        session_id,
        {
            "type": "violation",
            "session_id": session_id,
            "violation": {
                "type": "tab_switch",
                "severity": "high",
                "description": event.description,
            },
            "timestamp": event.timestamp,
        },
    )

    return {"message": "Tab switch logged"}


# ── WebSocket endpoints ───────────────────────────────────


@app.websocket("/ws/invigilator/{session_id}")
async def invigilator_ws(websocket: WebSocket, session_id: int):
    await manager.connect_invigilator(session_id, websocket)
    try:
        while True:
            # Keep connection alive — invigilator just listens
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect_invigilator(session_id, websocket)
        print(f"Invigilator disconnected from session {session_id}")


@app.websocket("/ws/student/{session_id}")
async def student_ws(websocket: WebSocket, session_id: int):
    await manager.connect_student(session_id, websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect_student(session_id)
        print(f"Student disconnected from session {session_id}")


# ── Helpers ───────────────────────────────────────────────


def get_warning_message(violation_type: str) -> str:
    messages = {
        "no_face": "Warning: Your face is not visible. Please stay in front of the camera.",
        "multiple_faces": "Warning: Multiple people detected. Only the student should be in frame.",
        "suspicious_gaze": "Warning: Please keep your eyes on the screen.",
        "tab_switch": "Warning: Tab switching has been detected and logged.",
    }
    return messages.get(violation_type, "Warning: A violation has been detected.")
