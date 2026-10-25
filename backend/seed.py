from datetime import datetime, timedelta

from database import Base, Exam, Session, SessionLocal, Student, Violation, engine


def populate_database():
    print("🗑️ Dropping old tables to reset the database completely...")
    Base.metadata.drop_all(bind=engine)

    # 1. Ensure all database tables exist fresh in proctoring.db
    print("📁 Creating fresh database schema tables...")

    # 1. Ensure all database tables exist in proctoring.db
    Base.metadata.create_all(bind=engine)

    # 2. Open a session transaction
    db = SessionLocal()

    try:
        # Prevent duplicate entries if the script is run multiple times
        if db.query(Student).first() is not None:
            print("⚠️ Data already exists in proctoring.db. Skipping seeding process.")
            return

        print("🌱 Seeding sqlite database with test proctoring records...")

        # 3. Create Sample Students
        # (face_encoding strings simulate face vector data placeholders)
        student_1 = Student(
            name="Aarav Sharma",
            email="aarav.sharma@edu.in",
            face_encoding="[0.012,-0.45,0.78,0.05]",
        )
        student_2 = Student(
            name="Diya Patel",
            email="diya.patel@edu.in",
            face_encoding="[-0.09,0.31,0.54,-0.12]",
        )
        student_3 = Student(
            name="John Doe",
            email="john.doe@edu.in",
            face_encoding="[0.18,-0.22,0.62,0.09]",
        )

        db.add_all([student_1, student_2, student_3])
        db.flush()  # Generates primary keys (IDs) so we can link relations

        # 4. Create Sample Exams
        exam_1 = Exam(
            title="Data Structures & Algorithms",
            description="Midterm evaluation. Topics: Trees, Graphs, Sorting.",
        )
        exam_2 = Exam(
            title="Introduction to Machine Learning",
            description="Final assessment covering Regressions and Clustering.",
        )

        db.add_all([exam_1, exam_2])
        db.flush()

        # 5. Create Sample Testing Sessions
        # Completed Session (Clean)
        session_1 = Session(
            exam_id=exam_1.id,
            student_id=student_1.id,
            started_at=datetime.utcnow() - timedelta(hours=2),
            ended_at=datetime.utcnow() - timedelta(hours=1),
            status="completed",
        )
        # Active Flagged Session (With violations)
        session_2 = Session(
            exam_id=exam_1.id,
            student_id=student_2.id,
            started_at=datetime.utcnow() - timedelta(minutes=45),
            status="flagged",
        )
        # Active Normal Session
        session_3 = Session(
            exam_id=exam_2.id,
            student_id=student_3.id,
            started_at=datetime.utcnow() - timedelta(minutes=15),
            status="active",
        )

        # 💡 Add this under your Step 5 inside populate_database()
        # A scheduled exam session waiting for Aarav to click "Join"
        joinable_session = Session(
            exam_id=exam_2.id,  # Machine Learning Exam
            student_id=student_1.id,  # Aarav Sharma
            started_at=None,  # Empty because they haven't clicked join yet
            ended_at=None,
            status="scheduled",  # custom flag your app can look for
        )

        db.add(joinable_session)
        db.add_all([session_1, session_2, session_3])
        db.flush()

        # 6. Create Sample Violations linked to the Flagged Session
        v1 = Violation(
            session_id=session_2.id,
            violation_type="tab_switch",
            severity="medium",
            description="Student alt-tabbed out of the exam environment.",
            screenshot="/static/screenshots/viol_tab_1.png",
            timestamp=datetime.utcnow() - timedelta(minutes=30),
        )
        v2 = Violation(
            session_id=session_2.id,
            violation_type="multiple_faces",
            severity="high",
            description="Secondary face detected in the web camera background.",
            screenshot="/static/screenshots/viol_face_2.png",
            timestamp=datetime.utcnow() - timedelta(minutes=25),
        )
        v3 = Violation(
            session_id=session_2.id,
            violation_type="gaze",
            severity="low",
            description="Student repeatedly looked away from the monitor.",
            screenshot="/static/screenshots/viol_gaze_3.png",
            timestamp=datetime.utcnow() - timedelta(minutes=10),
        )

        db.add_all([v1, v2, v3])

        # Commit everything permanently to the SQLite file
        db.commit()
        print("✅ Database successfully populated with sample data!")

    except Exception as e:
        db.rollback()
        print(f"❌ Insertion failed. Database changes rolled back. Error: {e}")
    finally:
        db.close()


if __name__ == "__main__":
    populate_database()
