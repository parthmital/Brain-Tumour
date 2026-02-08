# Brain Tumor Analysis System

**Backend Implementation - Complete AI-Powered Medical Analysis Platform**

---

## 1. Project Overview

This project provides a **fully functional brain tumor analysis backend** intended for use by hospitals, doctors, and medical staff.  
The system performs **tumor detection, segmentation, classification, visualization, reporting, and follow-up analysis**, using **locally deployed AI models** trained primarily on **Kaggle datasets**.

The architecture follows a **Topaz Photo AI–style approach**:

- **Multiple specialized models**
- **Each model trained for a specific dataset or function**
- **Inference-time orchestration instead of a single "one-size-fits-all" model**

---

## 2. Current Implementation Status

### Completed Components

- **FastAPI Backend Server** - Complete REST API with authentication
- **User Management System** - Registration, login, profile management
- **Database Layer** - SQLite with SQLModel for users and scans
- **AI Model Integration** - All three model types implemented
- **File Processing Pipeline** - Multi-modality MRI upload and processing
- **Authentication & Security** - JWT-based auth with bcrypt password hashing
- **API Endpoints** - Full CRUD operations for users and scans

### Production-Ready Features

- **Real-time tumor detection** (binary classification)
- **Multi-class tumor classification** (glioma, meningioma, pituitary, no tumor)
- **3D tumor segmentation** with volume calculations
- **Multi-modality support** (T1, T1CE, T2, FLAIR)
- **Automatic patient ID generation**
- **Comprehensive scan metadata storage**
- **Error handling and logging**
- **CORS support for frontend integration**

---

## 3. API Architecture & Endpoints

### Core API Structure

**Base URL**: `http://localhost:8000`

#### Authentication Endpoints

- `POST /api/auth/register` - User registration
- `POST /api/auth/login` - User login (JWT token)
- `GET /api/auth/me` - Get current user profile
- `PUT /api/auth/me` - Update user profile

#### Scan Management Endpoints

- `GET /api/scans` - List all scans for authenticated user
- `GET /api/scans/{scan_id}` - Get specific scan details
- `POST /api/process-mri` - Upload and process MRI files

### Database Schema

**Users Table**

- id, username, email, hashed_password
- fullName, title, department, institution
- createdAt timestamps

**Scans Table**

- id, patientId, patientName, scanDate
- modalities (JSON array), status, progress
- pipelineStep, results (JSON with AI outputs)
- createdAt, userId (foreign key)

---

## 4. AI Model Implementation

### 4.1 Tumor Detection Model

**Model**: ResNet50 Binary Classification

- **Input**: 224x224 MRI slices
- **Output**: Tumor presence probability
- **File**: `Detection/ResNet50-Binary-Detection.keras`
- **Status**: Trained and deployed

### 4.2 Tumor Classification Model

**Model**: ResNet50 Multi-class Classification

- **Classes**: glioma, meningioma, notumor, pituitary
- **Input**: 224x224 MRI slices
- **Output**: Class probabilities
- **File**: `Classification/Brain-Tumor-Classification-ResNet50.h5`
- **Status**: Trained and deployed

### 4.3 Tumor Segmentation Model

**Model**: Custom nnU-Net 3D Architecture

- **Input**: 128x128x128 4-modality volumes
- **Output**: WT, TC, ET segmentation masks
- **File**: `Segmentation/BraTS2020_nnU_Net_Segmentation.pth`
- **Status**: Trained and deployed

### Model Orchestration Logic

1. **Detection First**: Binary tumor detection on representative slice
2. **Conditional Classification**: Only if tumor detected
3. **Full Segmentation**: Requires all 4 modalities (T1, T1CE, T2, FLAIR)
4. **Volume Calculation**: Automatic tumor volume metrics

---

## 5. File Processing Pipeline

### Supported Modalities

- **FLAIR** - Primary detection modality
- **T1CE** - Contrast-enhanced T1
- **T2** - T2-weighted
- **T1** - T1-weighted

### Processing Steps

1. **File Upload**: Multi-file upload with modality detection
2. **NIfTI Parsing**: Convert 3D volumes to 2D slices
3. **Preprocessing**: Brain cropping, normalization, resizing
4. **Inference**: Sequential model execution
5. **Result Storage**: Database persistence with metadata

### Error Handling

- Graceful fallback for missing modalities
- Comprehensive error logging
- Model loading validation on startup

---

## 6. Core Functional Capabilities

- Tumor presence detection
- Tumor type classification (glioma, meningioma, pituitary)
- Tumor segmentation (slice-level, pixel-accurate)
- Tumor metrics computation (area, volume approximation)
- Visualization with overlays
- Automated report generation
- Follow-up comparison across scans
- Optional prognosis / survival estimation

---

## 7. Technology Stack

### Backend Framework

- **FastAPI** - Modern, fast web framework for APIs
- **SQLModel** - SQL toolkit with Pydantic models
- **SQLite** - Local database storage
- **JWT** - Authentication tokens
- **bcrypt** - Password hashing

### AI/ML Libraries

- **TensorFlow/Keras** - Detection and classification models
- **PyTorch** - Segmentation model
- **OpenCV** - Image preprocessing
- **NiBabel** - Medical image format support
- **NumPy** - Numerical computations

### Development Tools

- **Uvicorn** - ASGI server
- **Python-multipart** - File upload support
- **CORS middleware** - Frontend integration

---

## 8. Usage Examples

### User Registration

```bash
curl -X POST "http://localhost:8000/api/auth/register" \
  -H "Content-Type: application/json" \
  -d '{
    "username": "doctor1",
    "email": "doctor@hospital.com",
    "password": "securepassword",
    "fullName": "Dr. John Smith",
    "title": "Neurologist",
    "department": "Neurology"
  }'
```

### MRI Processing

```bash
curl -X POST "http://localhost:8000/api/process-mri" \
  -H "Authorization: Bearer <jwt-token>" \
  -F "files=@flair.nii.gz" \
  -F "files=@t1ce.nii.gz" \
  -F "files=@t2.nii.gz" \
  -F "files=@t1.nii.gz" \
  -F "patientName=John Doe"
```

---

## 9. Future Enhancements

### Planned Features

- **DICOM support** - Direct medical format compatibility
- **3D visualization** - Interactive tumor visualization
- **PDF report generation** - Automated clinical reports
- **Batch processing** - Multiple scan processing
- **Model versioning** - A/B testing for model improvements
- **Export capabilities** - Results export to various formats

### Scalability Options

- **PostgreSQL migration** - For larger deployments
- **Redis caching** - For session management
- **Docker containerization** - For deployment
- **Load balancing** - For high-availability setups
