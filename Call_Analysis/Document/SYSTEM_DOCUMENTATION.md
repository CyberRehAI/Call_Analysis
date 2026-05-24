# Call Analysis System - Professional System Documentation

## 1. Executive Summary

The **Call Analysis System** is an AI-powered, multimodal sentiment analysis platform designed to evaluate call center recordings and predict the probability of a successful sale. By combining acoustic emotion recognition with natural language processing (NLP), the system provides a holistic view of conversational dynamics between agents and customers.

**Key Capabilities:**
- **Multimodal Analysis:** Fuses audio intelligence (emotion, tone) with text intelligence (sentiment, context).
- **Sale Probability Prediction:** Outputs a calibrated 0-100% chance of a successful sale based on the conversation's trajectory.
- **Advanced Processing Pipeline:** Incorporates speaker diarization, automatic speech recognition (ASR), and PII masking.
- **Real-Time Interactive Dashboard:** A Next.js frontend offering visualizations of sentiment curves, emotion distributions, and key phrases.
- **Enterprise-Ready Integrations:** Supports MongoDB Atlas for data persistence and provides standard export formats (PDF, CSV, JSON).

---

## 2. System Architecture

The architecture is built on a modern decoupling of the frontend and backend, ensuring scalability and ease of integration.

### 2.1 High-Level Flow
1. **Frontend (Next.js):** Provides a robust web interface for agents and analysts to upload audio and view insights.
2. **Backend API (FastAPI):** Exposes RESTful endpoints (`/api/upload`, `/api/analyze`, `/api/results`) for seamless asynchronous processing.
3. **Core Services:** The intelligent heart of the system, breaking down into Preprocessing, Feature Extraction, ML Models, and Dashboard generation.
4. **Storage (MongoDB):** Maintains a persistent record of call metadata, analysis results, and system logs.

### 2.2 Processing Pipeline
- **Ingestion & Validation:** Audio files (WAV, MP3, M4A) are validated and standardized.
- **ASR & Diarization:** Whisper translates speech to text, while Pyannote segments the audio by speaker (Agent vs. Customer).
- **PII Masking:** SpaCy Regex masks sensitive data (Personally Identifiable Information) natively before deep analysis.
- **Feature Extraction:** Acoustic features (MFCC, Spectral, Chroma) and Text features (BERT embeddings) are computed.
- **Machine Learning Inference:** Data is routed through three primary models: Sentiment, Emotion, and Sale Probability.
- **Result Aggregation:** Insights are aggregated into JSON documents and pushed to MongoDB for dashboard retrieval.

---

## 3. Core Modules & Technologies

### 3.1 Preprocessing Module (`preprocessing.py`)
Responsible for transforming raw audio into highly structured, diarized transcripts.
- **Technologies:** OpenAI Whisper (Transcription), Pyannote.audio (Diarization), SpaCy (PII Masking).
- **Features:** Speaker role identification and asynchronous batch-processing capability.

### 3.2 Feature Extraction (`feature_extraction.py`)
Translates structured data into mathematical vectors suitable for ML models.
- **Acoustic:** Utilizes Librosa for extracting MFCCs, pitch, and energy.
- **Textual:** Extracts BERT dense vector embeddings via Hugging Face.
- **Conversational Dynamics:** Computes real-world metrics like silence ratios, interruptions, filler word frequency ("um", "uh"), and talk/listen ratios.

### 3.3 Sentiment & Emotion Models (`models.py`)
The dual-engine intelligence system predicting human affect.
- **Text Sentiment:** Powered by DistilBERT (for general conversations) or FinBERT (specialized for financial contexts), supplemented by key phrase extraction using spaCy.
- **Acoustic Emotion:** Built on a hybrid CNN+LSTM deep learning network trained on the RAVDESS dataset, detecting nuanced vocal changes independent of words.

### 3.4 Sale Prediction Model
The final decision engine fusing multimodal signals.
- **Technology:** XGBoost classifier with Platt scaling for probability calibration.
- **Outputs:** Evaluates the fused feature vector to predict a conversion, alongside rigorous confidence intervals and feature importance metrics.

### 3.5 Visualization Dashboard
A React-based frontend designed for immediate visual comprehension.
- **Technologies:** Next.js 14, React 18, TypeScript, Tailwind CSS, Recharts/Plotly.
- **Views:** Time-series sentiment tracking, aggregate emotion pie charts, sale probability gauges, and conversational dynamic KPIs.

---

## 4. API Reference Overview

The API is fully documented via Swagger UI (`http://localhost:5000/docs`). Core endpoints include:

- `POST /api/upload`: Asynchronously uploads an audio file and creates a pending call record.
- `POST /api/analyze`: Triggers the heavy analytical pipeline for a specific `call_id`.
- `GET /api/status/{call_id}`: Allows the frontend to poll the current processing state.
- `GET /api/results/{call_id}`: Retrieves the final multi-modal analysis payload.
- `GET /api/history`: Returns paginated historical data of analyzed calls in the database.
- `GET /api/export/{call_id}/[pdf|csv|json]`: Downloads structured summaries.

---

## 5. Developer Guide & Setup

### 5.1 Prerequisites
- **Python 3.10+**
- **Node.js 18+**
- **MongoDB** (Local instance running on `27017` or an Atlas URI)
- **Hugging Face Token** (Required for accessing gated models like Pyannote)

### 5.2 Environment Configuration
Create a `.env` in the root:
```env
HF_TOKEN=your_huggingface_token_here
MONGODB_URI=mongodb://localhost:27017/
MONGODB_DATABASE=call_center_db
SENTIMENT_MODEL=distilbert # or finbert
```

### 5.3 Quick Installation

**1. Backend:**
```bash
cd backend
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync  # Uses pyproject.toml
python -m spacy download en_core_web_sm
python -m uvicorn src.call_analysis.web_app_fastapi:app --reload --port 5000
```

**2. Frontend:**
```bash
cd frontend
npm install
npm run dev
```

### 5.4 Model Management
Pre-trained models reside in `backend/models/`. If retuning is needed:
- **Emotion Model:** `python backend/scripts/train_emotion_model.py`
- **Sale Predictor:** `python backend/scripts/train_sale_predictor.py`
- **Validation:** `python backend/scripts/validate_trained_models.py`

---

## 6. Current State & Extensibility 

**Project Status (Production Ready MVP):**
- Fast API Backend: ✅ Fully functional with async task execution.
- ML Models: ✅ Trained and integrated (XGBoost, CNN+LSTM, DistilBERT/FinBERT).
- Frontend UI: ✅ Real-time Next.js application completed.
- Database Integration: ✅ MongoDB integrated for history and metadata persistence.

**Extensibility:** 
The modularity of `backend/src/call_analysis/` ensures models can be swapped (e.g., upgrading Whisper to Whisper-v3, or replacing the XGBoost model with a Deep Neural Network) without altering integration layers or frontend logic.
