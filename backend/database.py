from datetime import datetime

from sqlalchemy import (
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker

DATABASE_URL = "sqlite:///./proctoring.db"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},  # needed for SQLite + FastAPI
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# ── Tables ────────────────────────────────────────────────


class Exam(Base):
    __tablename__ = "exams"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    sessions = relationship("Session", back_populates="exam")


class Student(Base):
    __tablename__ = "students"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    email = Column(String, unique=True, nullable=False)
    face_encoding = Column(Text, nullable=True)  # ← add this
    created_at = Column(DateTime, default=datetime.utcnow)

    sessions = relationship("Session", back_populates="student")


class Session(Base):
    __tablename__ = "sessions"

    id = Column(Integer, primary_key=True, index=True)
    exam_id = Column(Integer, ForeignKey("exams.id"))
    student_id = Column(Integer, ForeignKey("students.id"))
    started_at = Column(DateTime, default=datetime.utcnow)
    ended_at = Column(DateTime, nullable=True)
    status = Column(String, default="active")  # active, completed, flagged

    exam = relationship("Exam", back_populates="sessions")
    student = relationship("Student", back_populates="sessions")
    violations = relationship("Violation", back_populates="session")


class Violation(Base):
    __tablename__ = "violations"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("sessions.id"))
    violation_type = Column(
        String, nullable=False
    )  # no_face, multiple_faces, gaze, tab_switch
    severity = Column(String, default="medium")  # low, medium, high
    description = Column(Text, nullable=True)
    screenshot = Column(String, nullable=True)  # path to saved screenshot
    timestamp = Column(DateTime, default=datetime.utcnow)

    session = relationship("Session", back_populates="violations")


# ── Helpers ───────────────────────────────────────────────


def create_tables():
    Base.metadata.create_all(bind=engine)


def get_db():
    """Dependency — gives a database session to each request"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
