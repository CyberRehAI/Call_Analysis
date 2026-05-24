# Codebase Analysis and Findings

**Date:** December 2025  
**Project:** RealtimeCallSemanticAnal - AI-Powered Call Analysis System  
**Analysis Type:** Complete Codebase Review and Reorganization Plan

---

## Executive Summary

This document provides a comprehensive analysis of the RealtimeCallSemanticAnal codebase, including architecture, structure, dependencies, and recommendations for file organization. The project is a multimodal sentiment analysis system that combines audio and text features to predict sales probability in call center conversations.

**Overall Assessment:**
- **Architecture:** Well-structured with clear separation of frontend and backend
- **Code Quality:** Good modular design with proper class-based organization
- **Documentation:** Comprehensive documentation present
- **File Organization:** Needs reorganization - several utility scripts and documentation files are at root level
- **Completion Status:** ~35-40% (per REPORT.md assessment)

---

## 1. Project Structure Overview

### 1.1 Current Directory Structure

```
RealtimeCallSemanticAnal/
├── frontend/                    # Next.js 14 frontend application
│   ├── src/
│   │   ├── app/                # Next.js App Router pages
│   │   │   ├── page.tsx        # Main landing page
│   │   │   ├── layout.tsx      # Root layout
│   │   │   ├── about/          # About page
│   │   │   └── history/        # History page
│   │   └── components/         # React components
│   │       ├── AnalysisDashboard.tsx
│   │       ├── EmotionChart.tsx
│   │       ├── KeyPhrases.tsx
│   │       ├── Navbar.tsx
│   │       ├── SaleGauge.tsx
│   │       ├── SentimentChart.tsx
│   │       └── UploadForm.tsx
│   ├── package.json
│   ├── tailwind.config.js
│   └── tsconfig.json
│
├── backend/                     # Python Flask backend
│   ├── src/
│   │   └── call_analysis/      # Core analysis modules
│   │       ├── __init__.py
│   │       ├── preprocessing.py    # Audio/text preprocessing
│   │       ├── feature_extraction.py  # Feature engineering
│   │       ├── models.py           # ML models (sentiment, emotion, sale prediction)
│   │       ├── dashboard.py        # Dashboard generation
│   │       ├── web_app.py          # Flask API endpoints
│   │       └── demo.py             # Demo system
│   ├── run_full_analysis.py    # Full pipeline runner
│   ├── run_diarization_only.py # Diarization testing script
│   ├── run_web_app.py          # Web app launcher
│   ├── config.py               # Configuration management
│   ├── db_connection.py        # MongoDB connection
│   └── CONFIGURATION.md        # Configuration guide
│
├── Document/                    # Documentation directory (currently empty)
│
├── [ROOT LEVEL FILES - NEED REORGANIZATION]
│   ├── analyze_pyannote_output.py  # Utility script
│   ├── transcribe_audio.py         # Utility script
│   ├── transcription_output.txt    # Output file
│   ├── proj.txt                    # Project documentation
│   ├── REPORT.md                   # Progress report
│   ├── ARCHITECTURE_DIAGRAMS.md    # Architecture documentation
│   ├── README.md                   # Project README (should stay)
│   ├── requirements.txt            # Python dependencies
│   ├── pyproject.toml              # Python project config
│   └── uv.lock                     # Python lock file
```

---

## 2. Technology Stack Analysis

### 2.1 Frontend Stack

**Framework & Core:**
- **Next.js 14.0.3** - React framework with App Router
- **React 18.2.0** - UI library
- **TypeScript 5.3.3** - Type safety

**Styling & UI:**
- **Tailwind CSS 3.3.6** - Utility-first CSS framework
- **Lucide React** - Icon library
- **React Hot Toast** - Toast notifications

**Data Visualization:**
- **Recharts 2.8.0** - Chart library (used in components)

**HTTP Client:**
- **Axios 1.6.2** - API communication

**Utilities:**
- **clsx & tailwind-merge** - Conditional class utilities

### 2.2 Backend Stack

**Core Framework:**
- **Python 3.10+** - Programming language
- **Flask 2.3.0+** - Web framework
- **Flask-CORS 4.0.0+** - CORS handling

**Machine Learning & NLP:**
- **PyTorch 2.0.0+** - Deep learning framework
- **Transformers 4.30.0+** - Hugging Face transformers (BERT, DistilBERT)
- **scikit-learn 1.3.0+** - ML utilities
- **XGBoost 1.7.0+** - Gradient boosting for sale prediction

**Audio Processing:**
- **librosa 0.10.0+** - Audio analysis
- **openai-whisper 20230314+** - Speech-to-text transcription
- **whisperx 3.7.0+** - Fast Whisper with alignment
- **pyannote.audio 3.1.0+** - Speaker diarization
- **resemblyzer 0.1.1+** - Speaker embeddings (CPU-friendly alternative)
- **pydub 0.25.0+** - Audio manipulation
- **soundfile 0.12.0+** - Audio I/O

**Data Processing:**
- **pandas 2.0.0+** - Data manipulation
- **numpy 1.24.0+** - Numerical computing
- **scipy 1.10.0+** - Scientific computing (clustering)

**Visualization:**
- **matplotlib 3.7.0+** - Plotting
- **seaborn 0.12.0+** - Statistical visualization
- **plotly 5.15.0+** - Interactive charts

**Database:**
- **pymongo 4.5.0+** - MongoDB driver
- **PostgreSQL** (mentioned in config, not actively used)

**Utilities:**
- **python-dotenv 1.0.0+** - Environment variable management
- **spacy 3.7.0+** - NLP (PII masking)
- **jiwer 4.0.0+** - Word Error Rate calculation
- **joblib 1.3.0+** - Model serialization

**Project Management:**
- **pyproject.toml** - Modern Python project configuration
- **uv.lock** - Dependency lock file (UV package manager)

---

## 3. Architecture Analysis

### 3.1 System Architecture

The system follows a **client-server architecture** with clear separation:

1. **Frontend (Next.js)**
   - Client-side React application
   - Communicates with backend via REST API
   - Handles file uploads, displays analysis results
   - Real-time status polling for long-running analysis

2. **Backend (Flask)**
   - RESTful API endpoints
   - ML pipeline orchestration
   - Database operations
   - File management

3. **Storage**
   - MongoDB for call records and results
   - File system for audio uploads and outputs

### 3.2 Processing Pipeline

The analysis pipeline follows this flow:

```
Audio File → Validation → Whisper Transcription → Pyannote Diarization 
→ Text Processing (PII Masking) → Feature Extraction (Audio + Text) 
→ Sentiment Analysis (BERT) → Emotion Detection (CNN+LSTM) 
→ Sale Prediction (XGBoost) → Dashboard Generation → Storage
```

### 3.3 Key Modules

#### Backend Modules (`backend/src/call_analysis/`)

1. **preprocessing.py**
   - `AudioProcessor`: Handles audio transcription, diarization, feature extraction
   - `TextProcessor`: Text normalization, segmentation, PII masking
   - Supports both WhisperX (fast) and Pyannote (accurate) diarization

2. **feature_extraction.py**
   - `FeatureExtractor`: Combines audio and text features
   - BERT embeddings for text
   - MFCC, spectral, chroma features for audio
   - Temporal feature engineering

3. **models.py**
   - `SentimentAnalyzer`: Text sentiment analysis (currently keyword-based, BERT available)
   - `EmotionDetector`: Audio emotion recognition (CNN+LSTM architecture)
   - `SalePredictor`: Sale probability prediction (XGBoost/LSTM)
   - **Note:** Models currently use demo/synthetic data per REPORT.md

4. **dashboard.py**
   - `Dashboard`: Generates HTML dashboards with Plotly charts
   - Sentiment timelines, emotion distributions, key phrases

5. **web_app.py**
   - Flask application with REST endpoints:
     - `/api/upload` - File upload
     - `/api/analyze` - Start analysis
     - `/api/status/:id` - Check analysis status
     - `/api/results/:id` - Get results
     - `/api/history` - Get analysis history
     - `/api/export/:id/pdf` - Export PDF
     - `/api/export/:id/csv` - Export CSV

6. **demo.py**
   - Demo system with simulated conversations
   - Useful for testing without real audio processing

#### Frontend Components (`frontend/src/components/`)

1. **UploadForm.tsx** - File upload interface
2. **AnalysisDashboard.tsx** - Main results display
3. **SentimentChart.tsx** - Sentiment over time visualization
4. **EmotionChart.tsx** - Emotion distribution chart
5. **SaleGauge.tsx** - Sale probability gauge
6. **KeyPhrases.tsx** - Key phrase extraction display
7. **Navbar.tsx** - Navigation component

---

## 4. File Organization Issues

### 4.1 Root-Level Files That Need Reorganization

**Utility Scripts (should move to `backend/scripts/` or `backend/`):**
- `analyze_pyannote_output.py` - Analysis utility for diarization output
- `transcribe_audio.py` - Standalone transcription script

**Output Files (should move to `output/` or `backend/output/`):**
- `transcription_output.txt` - Generated output file

**Documentation (should move to `Document/`):**
- `proj.txt` - Project documentation
- `REPORT.md` - Progress report
- `ARCHITECTURE_DIAGRAMS.md` - Architecture documentation

**Python Project Files (should move to `backend/`):**
- `pyproject.toml` - Python project configuration and dependencies (primary)
- `uv.lock` - Dependency lock file for uv package manager
- `requirements.txt` - Kept for compatibility, but dependencies are managed via pyproject.toml

**Files That Should Stay at Root:**
- `README.md` - Standard practice to keep at root

### 4.2 Import Dependencies Analysis

**analyze_pyannote_output.py:**
- Imports: `json`, `sys`, `io`, `collections`, `datetime`
- **No backend imports** - Safe to move
- Reads from `output/` directory (hardcoded path needs update)

**transcribe_audio.py:**
- Imports: `sys`, `os`, `pathlib`, `dotenv`
- Imports from: `backend/src/call_analysis/preprocessing`
- **Path dependency:** `sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend', 'src'))`
- **Action needed:** Update path when moved to `backend/`

---

## 5. Configuration Analysis

### 5.1 Environment Variables

The system uses `.env` file (loaded via `python-dotenv`) with:

**Required:**
- `HF_TOKEN` - Hugging Face token (for Pyannote.audio models)

**Database:**
- `MONGODB_URI` - MongoDB connection string
- `MONGODB_DATABASE` - Database name

**Flask:**
- `FLASK_SECRET_KEY` - Session secret
- `FLASK_DEBUG` - Debug mode
- `FLASK_HOST` - Server host
- `FLASK_PORT` - Server port

**Models:**
- `WHISPER_MODEL_SIZE` - Whisper model size (tiny/base/small/medium/large)
- `BERT_MODEL_NAME` - BERT model identifier
- `PYANNOTE_AUDIO_MODEL` - Pyannote model identifier

**File Upload:**
- `UPLOAD_FOLDER` - Upload directory
- `MAX_CONTENT_LENGTH` - Max file size
- `ALLOWED_EXTENSIONS` - Allowed file types

**Features:**
- `DEMO_MODE` - Enable demo mode
- `USE_SIMULATED_AUDIO` - Use simulated data
- `PII_MASKING_ENABLED` - Enable PII masking

### 5.2 Configuration Files

- `backend/config.py` - Centralized configuration class
- `backend/CONFIGURATION.md` - Configuration guide
- `.env` - Environment variables (not in repo, should be created)

---

## 6. Dependencies and Package Management

### 6.1 Python Dependencies

**Package Managers:**
- **`uv`** - Primary package manager (fast, modern Python package installer)
- `pyproject.toml` - Project configuration and dependencies (PEP 518 standard)
- `uv.lock` - Lock file for reproducible builds
- `requirements.txt` - Kept for compatibility, but not primary (dependencies are in pyproject.toml)

**Package Management:** This project uses **uv** as the primary package manager. Dependencies are defined in `pyproject.toml` and locked in `uv.lock`. Install dependencies with `uv sync`.

### 6.2 Frontend Dependencies

Managed via `package.json` with npm/yarn. All dependencies are properly versioned.

---

## 7. Code Quality Observations

### 7.1 Strengths

1. **Modular Design:** Clear separation of concerns with dedicated modules
2. **Class-Based Architecture:** Well-organized classes with single responsibilities
3. **Error Handling:** Try-catch blocks and fallback mechanisms present
4. **Documentation:** Comprehensive docstrings and comments
5. **Configuration Management:** Centralized config with environment variables
6. **Type Hints:** Python type hints used in backend code
7. **Modern Frontend:** Next.js 14 with App Router, TypeScript, Tailwind

### 7.2 Areas for Improvement

1. **Model Implementation:** Currently uses demo/synthetic data (per REPORT.md)
2. **Path Hardcoding:** Some hardcoded paths (e.g., `output/` in analyze_pyannote_output.py)
3. **Database Connection:** Hardcoded to localhost (should use environment variables)
4. **Frontend-Backend Integration:** Not fully tested end-to-end
5. **Error Messages:** Could be more user-friendly
6. **Testing:** No visible test files or test infrastructure

---

## 8. Import Path Analysis

### 8.1 Current Import Patterns

**Backend Internal Imports:**
```python
from call_analysis.preprocessing import AudioProcessor
from call_analysis.feature_extraction import FeatureExtractor
from call_analysis.models import ConversationAnalyzer
```

**External Imports:**
```python
from dotenv import load_dotenv
import whisper
from pyannote.audio import Pipeline
```

**Path Manipulation:**
```python
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
```

### 8.2 Impact of File Moves

**Files Safe to Move (no import dependencies):**
- `analyze_pyannote_output.py` - Only uses standard library
- `proj.txt`, `REPORT.md`, `ARCHITECTURE_DIAGRAMS.md` - Documentation
- `transcription_output.txt` - Output file

**Files Requiring Path Updates:**
- `transcribe_audio.py` - Has path manipulation for backend imports
  - Current: `sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend', 'src'))`
  - After move to `backend/`: `sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))`

**Files Requiring Content Updates:**
- `analyze_pyannote_output.py` - Hardcoded paths to `output/` directory
  - Should use relative paths or config-based paths

---

## 9. Recommendations

### 9.1 Immediate Actions (File Reorganization)

1. **Move utility scripts to `backend/scripts/`:**
   - `analyze_pyannote_output.py`
   - `transcribe_audio.py` (update import paths)

2. **Move documentation to `Document/`:**
   - `proj.txt`
   - `REPORT.md`
   - `ARCHITECTURE_DIAGRAMS.md`

3. **Move Python project files to `backend/`:**
   - `requirements.txt`
   - `pyproject.toml`
   - `uv.lock`

4. **Move output files to `output/`:**
   - `transcription_output.txt`

5. **Keep at root:**
   - `README.md` (standard practice)

### 9.2 Code Improvements

1. **Update hardcoded paths** to use configuration or relative paths
2. **Package management standardized** - Using uv with pyproject.toml (primary), requirements.txt kept for compatibility
3. **Add path configuration** for output directories
4. **Update import paths** in moved files
5. **Create `.env.example`** template file

### 9.3 Testing & Integration

1. **Test import paths** after reorganization
2. **Verify frontend-backend API integration**
3. **Test file upload and analysis pipeline**
4. **Validate database connections**

---

## 10. Reorganization Plan

### Phase 1: Create Directory Structure
- Ensure `backend/scripts/` exists
- Ensure `output/` exists
- Ensure `Document/` exists (already present)

### Phase 2: Move Files
1. Move utility scripts → `backend/scripts/`
2. Move documentation → `Document/`
3. Move Python config → `backend/`
4. Move output files → `output/`

### Phase 3: Update Imports and Paths
1. Update `transcribe_audio.py` import paths
2. Update `analyze_pyannote_output.py` output paths
3. Verify all imports work

### Phase 4: Testing
1. Test script execution
2. Test import resolution
3. Verify file paths in code

---

## 11. Conclusion

The RealtimeCallSemanticAnal codebase is well-structured with a clear separation between frontend and backend. The main issue is file organization, with several utility scripts, documentation, and configuration files at the root level that should be moved to appropriate directories.

**Key Findings:**
- ✅ Good architecture and modular design
- ✅ Comprehensive documentation
- ✅ Modern tech stack
- ⚠️ File organization needs improvement
- ⚠️ Some hardcoded paths need configuration
- ⚠️ Import paths need updates after reorganization

**Next Steps:**
1. Execute file reorganization as outlined
2. Update import paths and hardcoded file paths
3. Test all functionality after reorganization
4. Update documentation with new file locations

---

**Document Prepared By:** AI Code Analysis System  
**Last Updated:** December 2025

