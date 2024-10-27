from fastapi.responses import JSONResponse
from fastapi import FastAPI, Request, status
import logging
from datetime import datetime, timedelta
import shutil
import os
import uuid
import json
from fastapi import (
    FastAPI,
    Depends,
    HTTPException,
    UploadFile,
    File,
    Body,
    Request,
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
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.templating import Jinja2Templates

app = FastAPI()

# Database setup
DATABASE_URL = "sqlite:///./leaderboard.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

# Security
security = HTTPBearer()

# Load reference test labels
try:
    with open("reference_test.json", "r") as f:
        reference_test = json.load(f)
    reference_labels = {}
except FileNotFoundError:
    reference_labels = {}
    print("reference_test.json not found. Test submissions will not be evaluated.")

# Templates setup
templates = Jinja2Templates(directory="templates")


# Models
class Student(Base):
    __tablename__ = "students"
    id = Column(Integer, primary_key=True)
    name = Column(String)
    token = Column(String, unique=True)

    training_submissions = relationship(
        "TrainingSubmission", back_populates="student")
    test_submissions = relationship("TestSubmission", back_populates="student")


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
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db),
):
    token = credentials.credentials
    student = db.query(Student).filter(Student.token == token).first()
    if student is None:
        raise HTTPException(
            status_code=401, detail="Invalid authentication token")
    return student


# Endpoint to generate tokens (Teacher)
@app.post("/generate_token")
def generate_token(name: str = Body(..., embed=True), db: Session = Depends(get_db)):
    # check if the student already exists
    student = db.query(Student).filter(Student.name == name).first()
    if student:
        return {"name": student.name, "token": student.token}

    token = str(uuid.uuid4())
    student = Student(name=name, token=token)
    db.add(student)
    db.commit()
    db.refresh(student)
    return {"name": student.name, "token": student.token}


@app.get("/tokens")
def tokens(db: Session = Depends(get_db)):
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
    student: Student = Depends(get_current_student),
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
    code_dir = os.path.join("submissions", "training", str(student.id))
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
    return {"message": "Training submission successful", "submission_id": submission.id}


# Endpoint for students to submit test predictions (limited to 1 per day)
@app.post("/test_submission")
def submit_test(
    predictions: str = Form(...),
    student: Student = Depends(get_current_student),
    db: Session = Depends(get_db),
):
    try:
        predictions = json.loads(predictions)
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=400, detail="Invalid JSON format in predictions"
        )
    # Check for submission limit
    one_day_ago = datetime.utcnow() - timedelta(days=1)
    recent_submission = (
        db.query(TestSubmission)
        .filter(
            TestSubmission.student_id == student.id,
            TestSubmission.timestamp >= one_day_ago,
        )
        .first()
    )
    if recent_submission:
        raise HTTPException(
            status_code=400, detail="Test submission limit reached for today"
        )

    # Save the predictions file
    submission_id = str(uuid.uuid4())
    pred_dir = os.path.join("submissions", "test", str(student.id))
    os.makedirs(pred_dir, exist_ok=True)
    predictions_file_path = os.path.join(pred_dir, f"{submission_id}.json")
    with open(predictions_file_path, "w") as buffer:
        json.dump(predictions, buffer)

    # Compute accuracy
    correct = 0
    for file, true_label in reference_test.items():
        correct += int(true_label == predictions.get(file, [None])[0])
    accuracy = correct / len(reference_test)

    # Create a new test submission record
    submission = TestSubmission(
        student_id=student.id,
        predictions_file_path=predictions_file_path,
        accuracy=accuracy,
    )
    db.add(submission)
    db.commit()
    db.refresh(submission)
    return {
        "message": "Test submission successful",
        "submission_id": submission.id,
        "accuracy": accuracy,
    }


# Endpoint to view the leaderboard in JSON format
@app.get("/leaderboard")
def get_leaderboard(db: Session = Depends(get_db)):
    # Get training submissions
    training_entries = (
        db.query(TrainingSubmission).order_by(
            TrainingSubmission.accuracy.desc()).all()
    )
    training_leaderboard = [
        {
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


# Root endpoint to view the leaderboard in HTML format
@app.get("/", response_class=HTMLResponse)
def read_leaderboard(request: Request, db: Session = Depends(get_db)):
    # Get training submissions
    training_entries = (
        db.query(TrainingSubmission).order_by(
            TrainingSubmission.accuracy.desc()).all()
    )
    training_leaderboard = [
        {
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

    return templates.TemplateResponse(
        "leaderboard.html",
        {
            "request": request,
            "training_leaderboard": training_leaderboard,
            "test_leaderboard": test_leaderboard,
        },
    )


# Endpoint for teacher to download code submissions
@app.get("/download_code/{submission_id}")
def download_code(submission_id: int, db: Session = Depends(get_db)):
    submission = (
        db.query(TrainingSubmission)
        .filter(TrainingSubmission.id == submission_id)
        .first()
    )
    if submission is None:
        raise HTTPException(status_code=404, detail="Submission not found")

    code_zip_path = submission.code_zip_path
    if not os.path.exists(code_zip_path):
        raise HTTPException(status_code=404, detail="Code file not found")

    return FileResponse(path=code_zip_path, filename=os.path.basename(code_zip_path))


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    exc_str = f"{exc}".replace("\n", " ").replace("   ", " ")
    logging.error(f"{request}: {exc_str}")
    content = {"status_code": 10422, "message": exc_str, "data": None}
    return JSONResponse(
        content=content, status_code=status.HTTP_422_UNPROCESSABLE_ENTITY
    )


# Create directories if they don't exist
os.makedirs("submissions/training", exist_ok=True)
os.makedirs("submissions/test", exist_ok=True)
os.makedirs("templates", exist_ok=True)
