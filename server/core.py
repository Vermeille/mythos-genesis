from typing import Annotated
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, Request, status
import logging
from datetime import datetime, timedelta
import shutil
import os
import uuid
import json
from fastapi import (
    BackgroundTasks,
    Cookie,
    Depends,
    HTTPException,
    UploadFile,
    File,
    Body,
    Form,
)
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Float,
    DateTime,
    ForeignKey,
    JSON,
)
from sqlalchemy.orm import sessionmaker, relationship, Session, declarative_base
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates


# Database setup
DATABASE_URL = "sqlite:///./leaderboard.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

# Security
security = HTTPBearer()
optional_security = HTTPBearer(auto_error=False)

# Templates setup
templates = Jinja2Templates(directory="templates")


class SubmissionError(Exception):
    pass


# Models
class Student(Base):
    __tablename__ = "students"
    id = Column(Integer, primary_key=True)
    name = Column(String)
    token = Column(String, unique=True)

    training_submissions = relationship("TrainingSubmission", back_populates="student")
    test_submissions = relationship("TestSubmission", back_populates="student")

    @property
    def is_teacher(self):
        return self.name == "Teacher"


class TrainingSubmission(Base):
    __tablename__ = "training_submissions"
    id = Column(Integer, primary_key=True)
    student_id = Column(Integer, ForeignKey("students.id"))
    accuracy = Column(Float)
    loss = Column(Float)
    hyperparameters = Column(JSON)
    code_zip_path = Column(String)
    pid = Column(Integer)
    tag = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)

    student = relationship("Student", back_populates="training_submissions")


class TestSubmission(Base):
    __tablename__ = "test_submissions"
    id = Column(Integer, primary_key=True)
    student_id = Column(Integer, ForeignKey("students.id"))
    predictions_file_path = Column(String)
    accuracy = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)

    student = relationship("Student", back_populates="test_submissions")


Base.metadata.create_all(bind=engine)


# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Authentication dependency
def get_current_student(
    credentials: HTTPAuthorizationCredentials = Depends(optional_security),
    db: Session = Depends(get_db),
) -> Student | None:
    if credentials is None:
        return None
    token = credentials.credentials
    student = db.query(Student).filter(Student.token == token).first()
    return student


def ensure_current_student(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db),
) -> Student | None:
    token = credentials.credentials
    student = db.query(Student).filter(Student.token == token).first()
    if student is None:
        raise HTTPException(status_code=401, detail="Invalid authentication token")
    return student


def ensure_teacher(teacher: Student = Depends(ensure_current_student)):
    if teacher is None or not teacher.is_teacher:
        raise HTTPException(status_code=401, detail="Invalid authentication token")
    return teacher


def create_student(db: Session, name: str, token: str):
    student = db.query(Student).filter(Student.name == name).first()
    if student:
        return student, False

    student = Student(name=name, token=token)
    db.add(student)
    db.commit()
    db.refresh(student)
    return student, True


def create_app(grading_fn, after_submission=None):
    app = FastAPI()
    app.mount("/static", StaticFiles(directory="static"), name="static")
    app.templates = templates
    os.makedirs("static", exist_ok=True)

    # Endpoint to generate tokens (Teacher)
    @app.post("/generate_token")
    def generate_token(
        name: str = Body(..., embed=True),
        db: Session = Depends(get_db),
        student: Student | None = Depends(get_current_student),
    ):
        if name == "Teacher":
            raise HTTPException(status_code=400, detail="Nice try.")

        token = str(uuid.uuid4())
        new_student, created = create_student(db, name, token)
        if created or (student is not None and student.is_teacher):
            return {"name": new_student.name, "token": new_student.token}
        else:
            return {"name": new_student.name}

    @app.get("/generate_token.html")
    def view_generate_token(request: Request):
        return templates.TemplateResponse("generate_token.html", {"request": request})

    @app.get("/instructions.html")
    def view_instructions(request: Request):
        return templates.TemplateResponse("instructions.html", {"request": request})

    @app.get("/tokens")
    def tokens(
        db: Session = Depends(get_db), teacher: Student = Depends(ensure_teacher)
    ):
        students = db.query(Student).all()
        return [{"name": student.name, "token": student.token} for student in students]

    # Endpoint for students to submit training information
    @app.post("/train_submission")
    def submit_training(
        accuracy: float = Form(...),
        loss: float = Form(...),
        hyperparameters: str = Form(...),
        pid: int = Form(...),
        tag: str = Form(...),
        code_zip: UploadFile = File(...),
        student: Student = Depends(ensure_current_student),
        db: Session = Depends(get_db),
    ):
        # Save the uploaded code zip file
        try:
            hyperparameters = json.loads(hyperparameters)
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=400, detail="Invalid JSON format in hyperparameters"
            )
        submission_id = str(uuid.uuid4())
        code_dir = os.path.join("submissions", "training", str(student.name))
        os.makedirs(code_dir, exist_ok=True)
        code_zip_path = os.path.join(code_dir, f"{submission_id}.zip")
        with open(code_zip_path, "wb") as buffer:
            shutil.copyfileobj(code_zip.file, buffer)
        code_zip.file.close()

        # Create a new training submission record
        submission = TrainingSubmission(
            student_id=student.id,
            accuracy=accuracy,
            loss=loss,
            hyperparameters=json.dumps(hyperparameters),
            code_zip_path=code_zip_path,
            pid=pid,
            tag=tag,
        )
        db.add(submission)
        db.commit()
        db.refresh(submission)
        return {
            "message": "Training submission successful",
            "submission_id": submission.id,
        }

    # Endpoint for students to submit test predictions (limited to 1 per day)
    @app.post("/test_submission")
    def submit_test(
        after_tasks: BackgroundTasks,
        predictions: str = Form(...),
        student: Student = Depends(ensure_current_student),
        db: Session = Depends(get_db),
    ):
        try:
            predictions = json.loads(predictions)
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=400, detail="Invalid JSON format in predictions"
            )
        # Check for submission limit
        limit = datetime.utcnow() - timedelta(hours=12)
        recent_submission = (
            db.query(TestSubmission)
            .filter(
                TestSubmission.student_id == student.id,
                TestSubmission.timestamp >= limit,
            )
            .first()
        )
        if recent_submission and not student.is_teacher:
            raise HTTPException(
                status_code=400, detail="Test submission limit reached for today"
            )

        # Save the predictions file
        submission_id = str(uuid.uuid4())
        pred_dir = os.path.join("submissions", "test", str(student.name))
        os.makedirs(pred_dir, exist_ok=True)
        predictions_file_path = os.path.join(pred_dir, f"{submission_id}.json")
        with open(predictions_file_path, "w") as buffer:
            json.dump(predictions, buffer)

        try:
            accuracy = grading_fn(predictions)
        except SubmissionError as e:
            raise HTTPException(status_code=400, detail=str(e))

        # Create a new test submission record
        submission = TestSubmission(
            student_id=student.id,
            predictions_file_path=predictions_file_path,
            accuracy=accuracy,
        )
        db.add(submission)
        db.commit()
        db.refresh(submission)
        if after_submission is not None:
            after_tasks.add_task(after_submission, student, predictions)
        return {
            "message": "Test submission successful",
            "submission_id": submission.id,
            "accuracy": accuracy,
        }

    # Endpoint to view the leaderboard in JSON format
    @app.get("/leaderboard")
    def get_leaderboard(
        db: Session = Depends(get_db),
        student: Student | None = Depends(get_current_student),
    ):
        # Get training submissions
        if student is None:
            training_entries = []
        elif student.is_teacher:
            training_entries = (
                db.query(TrainingSubmission)
                .order_by(TrainingSubmission.accuracy.desc())
                .all()
            )
        else:
            training_entries = (
                db.query(TrainingSubmission)
                .filter(TrainingSubmission.student_id == student.id)
                .order_by(TrainingSubmission.accuracy.desc())
                .all()
            )
        training_leaderboard = [
            {
                "id": submission.id,
                "student_name": submission.student.name,
                "accuracy": submission.accuracy,
                "tag": submission.tag,
                "timestamp": submission.timestamp,
            }
            for submission in training_entries
        ]

        # Get test submissions, ordered by accuracy
        test_entries = (
            db.query(TestSubmission).order_by(TestSubmission.accuracy.desc()).all()
        )
        test_leaderboard = [
            {
                "student_name": submission.student.name,
                "accuracy": submission.accuracy,
                "submission_time": submission.timestamp,
            }
            for submission in test_entries
        ]

        return {
            "training_leaderboard": training_leaderboard,
            "test_leaderboard": test_leaderboard,
        }

    def cookie_auth(
        db: Session = Depends(get_db), token: Annotated[str | None, Cookie()] = None
    ):
        if token is None:
            return None
        return db.query(Student).filter(Student.token == token).first()

    # Root endpoint to view the leaderboard in HTML format
    @app.get("/", response_class=HTMLResponse)
    def read_leaderboard(
        request: Request,
        db: Session = Depends(get_db),
        student: Student | None = Depends(cookie_auth),
    ):
        # Get training submissions
        leaderboard = get_leaderboard(db, student)
        training_leaderboard = leaderboard["training_leaderboard"]
        test_leaderboard = leaderboard["test_leaderboard"]

        # Prepare data for the line plot
        # Group submissions by student
        student_submissions = {}
        submissions = db.query(TestSubmission).order_by(TestSubmission.timestamp).all()
        for submission in submissions:
            student_name = submission.student.name
            if student_name not in student_submissions:
                student_submissions[student_name] = []
            student_submissions[student_name].append(
                {
                    "timestamp": submission.timestamp.isoformat(),
                    "accuracy": submission.accuracy,
                }
            )

        return templates.TemplateResponse(
            "leaderboard.html",
            {
                "request": request,
                "student": student,
                "training_leaderboard": training_leaderboard,
                "test_leaderboard": test_leaderboard,
                "student_submissions": student_submissions,
            },
        )

    # Endpoint for teacher to download code submissions
    @app.get("/download_code/{submission_id}")
    def download_code(
        submission_id: int,
        db: Session = Depends(get_db),
        teacher: Student = Depends(cookie_auth),
    ):
        submission = (
            db.query(TrainingSubmission)
            .filter(TrainingSubmission.id == submission_id)
            .first()
        )
        if teacher.id != submission.student_id and teacher.name != "Teacher":
            raise HTTPException(status_code=401, detail="Invalid authentication token")

        if submission is None:
            raise HTTPException(status_code=404, detail="Submission not found")

        code_zip_path = submission.code_zip_path
        if not os.path.exists(code_zip_path):
            raise HTTPException(status_code=404, detail="Code file not found")

        return FileResponse(
            path=code_zip_path, filename=os.path.basename(code_zip_path)
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ):
        exc_str = f"{exc}".replace("\n", " ").replace("   ", " ")
        logging.error(f"{request}: {exc_str}")
        content = {"status_code": 10422, "message": exc_str, "data": None}
        return JSONResponse(
            content=content, status_code=status.HTTP_422_UNPROCESSABLE_ENTITY
        )

    return app


# Create directories if they don't exist
os.makedirs("submissions/training", exist_ok=True)
os.makedirs("submissions/test", exist_ok=True)
os.makedirs("templates", exist_ok=True)
create_student(SessionLocal(), "Teacher", os.environ["TEACHER_TOKEN"])
