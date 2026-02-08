import os
import shutil
import uuid
import uvicorn
import numpy as np
from typing import List, Dict, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from sqlmodel import Session, select
from datetime import datetime, timedelta
from jose import JWTError, jwt
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
import bcrypt
from processor import BrainProcessor
from database import create_db_and_tables, get_session
from models import Scan, User, UserCreate, UserLogin

# Auth Configuration
SECRET_KEY = "neuroscan-secret-key-change-this-in-production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/auth/login")


def verify_password(plain_password, hashed_password):
    return bcrypt.checkpw(
        plain_password.encode("utf-8"), hashed_password.encode("utf-8")
    )


def get_password_hash(password):
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode("utf-8"), salt)
    return hashed.decode("utf-8")


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def get_current_user(
    token: str = Depends(oauth2_scheme), session: Session = Depends(get_session)
):
    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user = session.exec(select(User).where(User.username == username)).first()
    if user is None:
        raise credentials_exception
    return user


# Initialize models
processor = BrainProcessor()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Setup database
    print("Initializing database...")
    create_db_and_tables()

    # Seeding: Create a default test user if needed, but leave scans empty as requested
    with next(get_session()) as session:
        # Check if default user exists
        default_user = session.exec(
            select(User).where(User.username == "drrebecca")
        ).first()
        if not default_user:
            print("Creating default test user...")
            hashed_pass = get_password_hash("password123")
            user = User(
                username="drrebecca",
                email="r.torres@hospital.org",
                hashed_password=hashed_pass,
                fullName="Dr. Rebecca Torres",
                title="Neuroradiology",
                department="Neuroradiology",
                institution="Central Medical Centre",
            )
            session.add(user)
            session.commit()
            print("Default user 'drrebecca' created with password 'password123'")

    # Load AI models
    print("Loading AI models...")
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        detection_path = os.path.join(
            base_dir, "Detection", "ResNet50-Binary-Detection.keras"
        )
        classification_path = os.path.join(
            base_dir, "Classification", "Brain-Tumor-Classification-ResNet50.h5"
        )
        segmentation_path = os.path.join(
            base_dir, "Segmentation", "BraTS2020_nnU_Net_Segmentation.pth"
        )

        processor.load_models(
            detection_path=detection_path,
            classification_path=classification_path,
            segmentation_path=segmentation_path,
        )
        print("All models loaded successfully")
    except Exception as e:
        print(f"CRITICAL: Could not load models. Error: {e}")

    yield
    print("Shutting down...")


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# Auth Endpoints
@app.post("/api/auth/register")
async def register(user_data: UserCreate, session: Session = Depends(get_session)):
    # Check if user exists
    existing_user = session.exec(
        select(User).where(User.username == user_data.username)
    ).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Username already registered")

    new_user = User(
        username=user_data.username,
        email=user_data.email,
        hashed_password=get_password_hash(user_data.password),
        fullName=user_data.fullName,
        title=user_data.title,
        department=user_data.department,
        institution=user_data.institution,
    )
    session.add(new_user)
    session.commit()
    session.refresh(new_user)
    return {"message": "User created successfully"}


@app.post("/api/auth/login")
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    session: Session = Depends(get_session),
):
    user = session.exec(select(User).where(User.username == form_data.username)).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Incorrect username or password")

    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "id": user.id,
            "username": user.username,
            "fullName": user.fullName,
            "email": user.email,
            "title": user.title,
        },
    }


@app.get("/api/auth/me")
async def get_me(current_user: User = Depends(get_current_user)):
    return current_user


@app.get("/api/scans")
async def get_scans(
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    statement = select(Scan).order_by(Scan.createdAt.desc())
    results = session.exec(statement).all()
    return results


@app.get("/api/scans/{scan_id}")
async def get_scan(
    scan_id: str,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    scan = session.get(Scan, scan_id)
    if not scan:
        raise HTTPException(status_code=404, detail="Scan not found")
    return scan


@app.post("/api/process-mri")
async def process_mri(
    files: List[UploadFile] = File(...),
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    if processor.detection_model is None or processor.classification_model is None:
        raise HTTPException(status_code=503, detail="AI models not loaded.")

    temp_dir = os.path.join(os.getcwd(), f"temp_process_{uuid.uuid4().hex}")
    os.makedirs(temp_dir, exist_ok=True)

    saved_files = {}
    try:
        for file in files:
            file_path = os.path.join(temp_dir, file.filename)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            fname_upper = file.filename.upper()
            if "FLAIR" in fname_upper:
                saved_files["flair"] = file_path
            elif "T1CE" in fname_upper or "T1C" in fname_upper:
                saved_files["t1ce"] = file_path
            elif "T2" in fname_upper:
                saved_files["t2"] = file_path
            elif "T1" in fname_upper:
                saved_files["t1"] = file_path

        rep_file = next(
            (
                saved_files.get(m)
                for m in ["flair", "t1ce", "t2", "t1"]
                if saved_files.get(m)
            ),
            None,
        )
        if not rep_file:
            raise HTTPException(status_code=400, detail="No valid MRI modality found.")

        # Inference
        image_2d = processor.nifti_to_2d(rep_file)
        prob_tumor = processor.run_detection(image_2d)
        has_tumor = prob_tumor > 0.5
        cls_result = processor.run_classification(image_2d)

        seg_result = {"tumorVolume": 0, "wtVolume": 0, "tcVolume": 0, "etVolume": 0}
        if len(saved_files) >= 4:
            try:
                seg_result = processor.run_segmentation(saved_files)
            except Exception as seg_err:
                print(f"Segmentation error: {seg_err}")

        # Save to DB
        new_scan = Scan(
            id=f"scan-{uuid.uuid4().hex[:6]}",
            patientId=f"PT-2026-{uuid.uuid4().hex[:4].upper()}",
            patientName="Uploaded Scan",
            scanDate="2026-02-08",
            modalities=[m.upper() for m in saved_files.keys()],
            status="completed",
            progress=100,
            pipelineStep="complete",
            results={
                "detected": has_tumor,
                "classification": (
                    cls_result["class"] if has_tumor else "No Tumour Detected"
                ),
                "confidence": round(
                    float(cls_result["confidence"] if has_tumor else (1 - prob_tumor)),
                    4,
                ),
                "tumorVolume": round(seg_result["tumorVolume"], 2),
                "wtVolume": round(seg_result["wtVolume"], 2),
                "tcVolume": round(seg_result["tcVolume"], 2),
                "etVolume": round(seg_result["etVolume"], 2),
            },
        )
        session.add(new_scan)
        session.commit()
        session.refresh(new_scan)
        return new_scan

    except Exception as e:
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
