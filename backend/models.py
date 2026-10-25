from datetime import datetime
from typing import Optional

from pydantic import BaseModel

# ── Exam schemas ──────────────────────────────────────────


class ExamCreate(BaseModel):
    title: str
    description: Optional[str] = None


class ExamResponse(BaseModel):
    id: int
    title: str
    description: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True


# ── Student schemas ───────────────────────────────────────


class StudentCreate(BaseModel):
    name: str
    email: str


class StudentResponse(BaseModel):
    id: int
    name: str
    email: str
    created_at: datetime

    class Config:
        from_attributes = True


# ── Session schemas ───────────────────────────────────────


class SessionCreate(BaseModel):
    exam_id: int
    student_id: int


class SessionResponse(BaseModel):
    id: int
    exam_id: int
    student_id: int
    started_at: datetime
    ended_at: Optional[datetime]
    status: str

    class Config:
        from_attributes = True


# ── Violation schemas ─────────────────────────────────────


class ViolationCreate(BaseModel):
    session_id: int
    violation_type: str
    severity: str = "medium"
    description: Optional[str] = None
    screenshot: Optional[str] = None


class ViolationResponse(BaseModel):
    id: int
    session_id: int
    violation_type: str
    severity: str
    description: Optional[str]
    screenshot: Optional[str]
    timestamp: datetime

    class Config:
        from_attributes = True


# ── Tab switch event (sent from frontend) ─────────────────


class TabSwitchEvent(BaseModel):
    session_id: int
    timestamp: str
    description: str = "Student switched tabs or minimised window"


# ── Session report ────────────────────────────────────────


class SessionReport(BaseModel):
    session: SessionResponse
    student: StudentResponse
    exam: ExamResponse
    violations: list[ViolationResponse]
    total_violations: int
    violation_breakdown: dict
