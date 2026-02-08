from sqlmodel import SQLModel, Field, JSON, Column
from typing import Optional, List, Dict
from datetime import datetime


class User(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    username: str = Field(index=True, unique=True)
    email: str = Field(index=True, unique=True)
    hashed_password: str
    fullName: str
    title: Optional[str] = None
    department: Optional[str] = None
    institution: Optional[str] = None
    createdAt: datetime = Field(default_factory=datetime.utcnow)


class UserCreate(SQLModel):
    username: str
    email: str
    password: str
    fullName: str
    title: Optional[str] = None
    department: Optional[str] = None
    institution: Optional[str] = None


class UserLogin(SQLModel):
    username: str
    password: str


class Scan(SQLModel, table=True):
    id: str = Field(primary_key=True)
    patientId: str
    patientName: str
    scanDate: str
    modalities: List[str] = Field(sa_column=Column(JSON))
    filePaths: Optional[Dict[str, str]] = Field(default=None, sa_column=Column(JSON))
    status: str
    progress: int = Field(default=0)
    pipelineStep: str = Field(default="queued")
    results: Optional[Dict] = Field(default=None, sa_column=Column(JSON))
    createdAt: datetime = Field(default_factory=datetime.utcnow)
    userId: Optional[int] = Field(default=None, foreign_key="user.id")


class ScanCreate(SQLModel):
    patientName: str = "Uploaded Scan"
    modalities: List[str]


class ScanUpdate(SQLModel):
    patientName: Optional[str] = None
    patientId: Optional[str] = None
