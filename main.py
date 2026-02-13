import os
import shutil
import uuid
import uvicorn
import cv2
import numpy as np
from typing import List, Dict, Optional
from fastapi import (
    FastAPI,
    UploadFile,
    File,
    Form,
    HTTPException,
    Depends,
    Body,
    Response,
    Query,
)
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from sqlmodel import Session, select
from datetime import datetime, timedelta
from jose import JWTError, jwt
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
import bcrypt
from processor import BrainProcessor
from database import create_db_and_tables, get_session
from models import Scan, User, UserCreate, UserLogin, ScanUpdate
import io

UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
SECRET_KEY = "neuroscan-secret-key-change-this-in-production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/auth/login", auto_error=False)


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
    token: Optional[str] = Depends(oauth2_scheme),
    token_query: Optional[str] = Query(None, alias="token"),
    session: Session = Depends(get_session),
):
    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    auth_token = token or token_query
    if not auth_token:
        raise credentials_exception
    try:
        payload = jwt.decode(auth_token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = session.exec(select(User).where(User.username == username)).first()
    if user is None:
        raise credentials_exception
    return user


processor = BrainProcessor()


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Initializing database...")
    create_db_and_tables()
    print("Loading AI models...")
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        detection_path = os.path.join(
            base_dir, "Detection", "ResNet50-Binary-Detection.keras"
        )
        classification_path = os.path.join(
            base_dir, "Classification", "Brain-Tumor-Classification-ResNet50.keras"
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
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)
from pydantic import BaseModel


class UserUpdate(BaseModel):
    fullName: Optional[str] = None
    email: Optional[str] = None
    title: Optional[str] = None
    department: Optional[str] = None
    institution: Optional[str] = None


@app.post("/api/auth/register")
async def register(user_data: UserCreate, session: Session = Depends(get_session)):
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

    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": new_user.username}, expires_delta=access_token_expires
    )

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "id": new_user.id,
            "username": new_user.username,
            "fullName": new_user.fullName,
            "email": new_user.email,
            "title": new_user.title,
        },
    }


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


@app.put("/api/auth/me")
async def update_me(
    user_update: UserUpdate,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    if user_update.fullName is not None:
        current_user.fullName = user_update.fullName
    if user_update.email is not None:
        existing = session.exec(
            select(User)
            .where(User.email == user_update.email)
            .where(User.id != current_user.id)
        ).first()
        if existing:
            raise HTTPException(status_code=400, detail="Email already in use")
        current_user.email = user_update.email
    if user_update.title is not None:
        current_user.title = user_update.title
    if user_update.department is not None:
        current_user.department = user_update.department
    if user_update.institution is not None:
        current_user.institution = user_update.institution
    session.add(current_user)
    session.commit()
    session.refresh(current_user)
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
    patientName: str = Form("Uploaded Scan"),
    patientId: str = Form(None),
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    if not patientId:
        patientId = f"PT-2026-{uuid.uuid4().hex[:4].upper()}"
    if processor.detection_model is None or processor.classification_model is None:
        raise HTTPException(status_code=503, detail="AI models not loaded.")
    scan_id = f"scan-{uuid.uuid4().hex[:6]}"
    scan_dir = os.path.join(UPLOAD_DIR, scan_id)
    os.makedirs(scan_dir, exist_ok=True)
    saved_files = {}
    try:
        for file in files:
            file_path = os.path.join(scan_dir, file.filename)
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
        image_2d = processor.nifti_to_2d(rep_file)
        prob_tumor = processor.run_detection(image_2d)
        has_tumor = prob_tumor > 0.5
        cls_result = processor.run_classification(image_2d)
        seg_result = {"tumorVolume": 0, "wtVolume": 0, "tcVolume": 0, "etVolume": 0}
        if len(saved_files) >= 4:
            try:
                seg_path = os.path.join(scan_dir, "segmentation.nii")
                seg_result = processor.run_segmentation(saved_files, save_path=seg_path)
            except Exception as seg_err:
                print(f"Segmentation error: {seg_err}")
        new_scan = Scan(
            id=scan_id,
            patientId=patientId,
            patientName=patientName,
            scanDate=datetime.now().strftime("%Y-%m-%d"),
            modalities=[m.upper() for m in saved_files.keys()],
            filePaths={
                m: os.path.relpath(p, UPLOAD_DIR) for (m, p) in saved_files.items()
            },
            status="completed",
            progress=100,
            pipelineStep="complete",
            userId=current_user.id,
            results={
                "detected": has_tumor,
                "classification": (
                    cls_result["class"] if has_tumor else "No Tumour Detected"
                ),
                "confidence": round(
                    float(cls_result["confidence"] if has_tumor else 1 - prob_tumor), 4
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
        if os.path.exists(scan_dir):
            shutil.rmtree(scan_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/scans/{scan_id}/slice/{slice_idx}")
async def get_scan_slice(
    scan_id: str,
    slice_idx: int,
    modality: str = "flair",
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    scan = session.get(Scan, scan_id)
    if not scan or not scan.filePaths:
        raise HTTPException(status_code=404, detail="Scan or files not found")
    modality = modality.lower()
    if modality not in scan.filePaths:
        modality = list(scan.filePaths.keys())[0]
    file_path = os.path.join(UPLOAD_DIR, scan.filePaths[modality])
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="MRI file not found on disk")
    image, total_slices = processor.get_slice_as_image(file_path, slice_idx)
    if image is None:
        raise HTTPException(status_code=500, detail="Failed to extract slice")
    _, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    return Response(
        content=buffer.tobytes(),
        media_type="image/jpeg",
        headers={"X-Total-Slices": str(total_slices)},
    )


@app.get("/api/scans/{scan_id}/segmentation/{slice_idx}")
async def get_segmentation_slice(
    scan_id: str,
    slice_idx: int,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    scan = session.get(Scan, scan_id)
    if not scan:
        raise HTTPException(status_code=404, detail="Scan not found")
    seg_path = os.path.join(UPLOAD_DIR, scan_id, "segmentation.nii")
    if not os.path.exists(seg_path):
        raise HTTPException(status_code=404, detail="Segmentation not found")
    mask = processor.get_segmentation_slice(seg_path, slice_idx)
    if mask is None:
        raise HTTPException(status_code=500, detail="Failed to extract mask slice")
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode(".png", mask_bgr)
    return Response(content=buffer.tobytes(), media_type="image/png")


@app.get("/api/scans/{scan_id}/download/{key}")
async def download_scan_file(
    scan_id: str,
    key: str,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    scan = session.get(Scan, scan_id)
    if not scan:
        raise HTTPException(status_code=404, detail="Scan not found")

    key = key.lower()
    file_path = None

    if key == "segmentation":
        file_path = os.path.join(UPLOAD_DIR, scan_id, "segmentation.nii")
    elif scan.filePaths and key in scan.filePaths:
        file_path = os.path.join(UPLOAD_DIR, scan.filePaths[key])
    else:
        # Try as direct filename within the scan folder for safety
        potential_path = os.path.join(UPLOAD_DIR, scan_id, key)
        # Prevent directory traversal
        if os.path.commonpath(
            [UPLOAD_DIR, potential_path]
        ) == UPLOAD_DIR and os.path.exists(potential_path):
            file_path = potential_path

    if not file_path or not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(
        file_path,
        media_type="application/octet-stream",
        filename=os.path.basename(file_path),
    )


@app.put("/api/scans/{scan_id}", response_model=Scan)
async def update_scan(
    scan_id: str,
    scan_update: ScanUpdate,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    scan = session.get(Scan, scan_id)
    if not scan:
        raise HTTPException(status_code=404, detail="Scan not found")
    scan_data = scan_update.dict(exclude_unset=True)
    for key, value in scan_data.items():
        setattr(scan, key, value)
    session.add(scan)
    session.commit()
    session.refresh(scan)
    return scan


@app.delete("/api/scans/{scan_id}")
async def delete_scan(
    scan_id: str,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    scan = session.get(Scan, scan_id)
    if not scan:
        raise HTTPException(status_code=404, detail="Scan not found")
    scan_dir = os.path.join(UPLOAD_DIR, scan_id)
    if os.path.exists(scan_dir):
        shutil.rmtree(scan_dir, ignore_errors=True)
    session.delete(scan)
    session.commit()
    return {"message": "Scan deleted successfully"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
