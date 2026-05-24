# Complete System Flow Visualization

This document visualizes the entire process from model training to real-world usage.

---

## 📊 Complete System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         TRAINING PHASE                                   │
└─────────────────────────────────────────────────────────────────────────┘

┌──────────────────────┐         ┌──────────────────────┐
│  Emotion Model       │         │  Sale Predictor      │
│  Training            │         │  Training            │
├──────────────────────┤         ├──────────────────────┤
│ Input:               │         │ Input:               │
│ • RAVDESS Dataset    │         │ • CSV with fused     │
│   (1440 audio files) │         │   features          │
│                      │         │ • Columns:           │
│ Process:             │         │   - sentiment_mean   │
│ • Extract MFCC       │         │   - sentiment_var   │
│ • Extract Mel-Spec   │         │   - emotion_probs   │
│ • Train CNN+LSTM     │         │   - dynamics         │
│ • 5 emotion classes  │         │   - sale_outcome     │
│                      │         │                      │
│ Output:              │         │ Process:             │
│ • emotion_model.pth  │         │ • Train XGBoost      │
│ • emotion_dataset_   │         │ • Calibrate probs    │
│   stats.json         │         │ • Feature importance │
│ • training_history   │         │                      │
└──────────────────────┘         │ Output:              │
                                 │ • sale_model.pkl     │
                                 │ • sale_model_scaler  │
                                 │ • training_results   │
                                 └──────────────────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         MODEL STORAGE                                    │
└─────────────────────────────────────────────────────────────────────────┘

backend/models/
├── emotion_model.pth                    (2.7MB)  ← CNN+LSTM weights
├── emotion_dataset_stats.json           (normalization stats)
├── emotion_training_history.json        (training metrics)
├── sale_model.pkl                       (41KB)   ← XGBoost model
├── sale_model_scaler.pkl                (scaler for features)
└── sale_training_results.json           (training metrics)

                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      INFERENCE PHASE (Production)                       │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│  STEP 1: Audio Upload & Preprocessing                                   │
└─────────────────────────────────────────────────────────────────────────┘

User uploads audio file (.wav, .mp3, .m4a)
         │
         ▼
┌────────────────────┐
│ AudioProcessor     │
│ • Validate format  │
│ • Convert if needed│
└────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STEP 2: Speech Recognition & Diarization                               │
└─────────────────────────────────────────────────────────────────────────┘

┌────────────────────┐         ┌────────────────────┐
│ Whisper ASR        │         │ Pyannote.audio     │
│ • Transcribe audio │────────▶│ • Speaker          │
│ • Text segments    │         │   diarization      │
│ • Timestamps       │         │ • Customer/Agent   │
└────────────────────┘         │   labels           │
                               └────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STEP 3: Feature Extraction                                             │
└─────────────────────────────────────────────────────────────────────────┘

┌────────────────────┐         ┌────────────────────┐
│ Audio Features     │         │ Text Features      │
│ (librosa)          │         │ (spaCy + BERT)     │
├────────────────────┤         ├────────────────────┤
│ • Mel-Spectrogram  │         │ • PII Masking      │
│ • MFCC (40)        │         │ • Text segments    │
│ • Pitch (F0)       │         │ • BERT embeddings  │
│ • Energy           │         │ • Key phrases      │
│ • Formants         │         └────────────────────┘
└────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STEP 4: Model Inference (Where Models Are Used)                        │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│  Model 1: SentimentAnalyzer (DistilBERT)                                │
├─────────────────────────────────────────────────────────────────────────┤
│ Location: backend/src/call_analysis/models.py                           │
│                                                                          │
│ Input:  Text segments (from diarization)                                │
│ Process:                                                                 │
│   • Load DistilBERT (lazy loading - only when needed)                   │
│   • Analyze sentiment per segment                                       │
│   • Extract key phrases                                                 │
│   • Calculate sentiment drift                                           │
│                                                                          │
│ Output:                                                                  │
│   • Sentiment scores (positive/negative/neutral)                        │
│   • Confidence scores                                                    │
│   • Key phrases                                                         │
│   • Sentiment trend over time                                           │
└─────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Model 2: EmotionDetector (AcousticEmotionModel - CNN+LSTM)             │
├─────────────────────────────────────────────────────────────────────────┤
│ Location: backend/src/call_analysis/models.py                           │
│ Model File: backend/models/emotion_model.pth                            │
│                                                                          │
│ Input:  Audio features (Mel-Spectrogram + MFCC)                         │
│ Process:                                                                 │
│   • Load emotion_model.pth (trained CNN+LSTM)                            │
│   • Extract segment-level mel-spectrograms                              │
│   • Run inference per segment                                           │
│   • Apply normalization (from emotion_dataset_stats.json)               │
│                                                                          │
│ Output:                                                                  │
│   • Emotion per segment (neutral/happiness/anger/sadness/frustration)  │
│   • Emotion probabilities                                               │
│   • Confidence scores                                                   │
│   • Emotion transitions over time                                      │
└─────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Model 3: SalePredictor (XGBoost)                                       │
├─────────────────────────────────────────────────────────────────────────┤
│ Location: backend/src/call_analysis/models.py                            │
│ Model File: backend/models/sale_model.pkl                                │
│                                                                          │
│ Input:  Fused Feature Vector (11 features)                              │
│   • Sentiment mean/variance (2)                                         │
│   • Emotion probabilities (5)                                          │
│   • Conversational dynamics (4)                                        │
│                                                                          │
│ Process:                                                                 │
│   • Load sale_model.pkl (trained XGBoost)                               │
│   • Apply scaler (sale_model_scaler.pkl)                                │
│   • Run XGBoost prediction                                              │
│   • Calculate confidence intervals                                      │
│                                                                          │
│ Output:                                                                  │
│   • Sale probability (0-100%)                                           │
│   • Prediction (sale/no_sale)                                           │
│   • Confidence interval                                                  │
│   • Feature importance                                                  │
└─────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STEP 5: Results Assembly & Storage                                     │
└─────────────────────────────────────────────────────────────────────────┘

┌────────────────────┐
│ ConversationAnalyzer│
│ (Orchestrator)     │
│ • Combines all     │
│   model outputs    │
│ • Calculates       │
│   metrics          │
│ • Generates summary│
└────────────────────┘
         │
         ▼
┌────────────────────┐         ┌────────────────────┐
│ MongoDB Storage    │         │ Dashboard          │
│ • Save results     │────────▶│ Generation         │
│ • Store segments   │         │ • Sentiment chart  │
│ • Store predictions│         │ • Emotion chart    │
│                    │         │ • Sale gauge       │
│                    │         │ • Key phrases      │
└────────────────────┘         └────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STEP 6: Frontend Display                                               │
└─────────────────────────────────────────────────────────────────────────┘

Frontend (Next.js/React)
├── AnalysisDashboard.tsx
│   • SentimentChart (sentiment over time)
│   • EmotionChart (emotion distribution)
│   • SaleGauge (conversion probability)
│   • KeyPhrases (important phrases)
│
└── API Calls
    • GET /api/results/{call_id}
    • GET /api/analyze/{conversation_id}
    • GET /api/export/{call_id}/pdf

```

---

## 🔄 Complete Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    TRAINING WORKFLOW                                     │
└─────────────────────────────────────────────────────────────────────────┘

[RAVDESS Dataset] ──▶ [train_emotion_model.py]
                           │
                           ├─▶ Extract Features (librosa)
                           ├─▶ Train CNN+LSTM
                           └─▶ Save → backend/models/emotion_model.pth

[CSV with Features] ──▶ [train_sale_predictor.py]
                           │
                           ├─▶ Train XGBoost
                           ├─▶ Calibrate Probabilities
                           └─▶ Save → backend/models/sale_model.pkl


┌─────────────────────────────────────────────────────────────────────────┐
│                    INFERENCE WORKFLOW (Production)                       │
└─────────────────────────────────────────────────────────────────────────┘

[Audio File Upload]
        │
        ▼
[AudioProcessor]
        │
        ├─▶ Whisper ASR ──▶ [Text Segments]
        │
        ├─▶ Pyannote ──▶ [Speaker Labels]
        │
        └─▶ librosa ──▶ [Audio Features]
                           │
                           ├─▶ Mel-Spectrogram
                           └─▶ MFCC

        │
        ▼
[ConversationAnalyzer]
        │
        ├─▶ SentimentAnalyzer (DistilBERT)
        │   └─▶ Sentiment scores per segment
        │
        ├─▶ EmotionDetector (CNN+LSTM)
        │   └─▶ Load: emotion_model.pth
        │   └─▶ Emotion per segment
        │
        └─▶ SalePredictor (XGBoost)
            └─▶ Load: sale_model.pkl
            └─▶ Sale probability

        │
        ▼
[Results Assembly]
        │
        ├─▶ Sentiment trends
        ├─▶ Emotion distribution
        ├─▶ Sale probability
        └─▶ Key phrases

        │
        ▼
[Storage & Display]
        │
        ├─▶ MongoDB (persistent storage)
        └─▶ Frontend Dashboard (visualization)

```

---

## 📍 Where Each Model Is Used

### 1. **EmotionDetector (CNN+LSTM)**

**Training:**
- Script: `backend/scripts/train_emotion_model.py`
- Input: RAVDESS audio files
- Output: `backend/models/emotion_model.pth`

**Inference:**
- Location: `backend/src/call_analysis/models.py` → `EmotionDetector.detect_conversation_emotions()`
- Called by: `ConversationAnalyzer.analyze_conversation()`
- Used in: `backend/src/call_analysis/web_app_fastapi.py` → `/api/analyze` endpoint
- Frontend: `frontend/src/components/EmotionChart.tsx`

**Flow:**
```
Audio File → librosa → Mel-Spectrogram → emotion_model.pth → Emotion Labels
```

---

### 2. **SentimentAnalyzer (DistilBERT)**

**Training:**
- Pre-trained model (no training needed)
- Model: `distilbert-base-uncased-finetuned-sst-2-english`
- Loaded from: HuggingFace (lazy loading)

**Inference:**
- Location: `backend/src/call_analysis/models.py` → `SentimentAnalyzer.analyze_conversation_sentiment()`
- Called by: `ConversationAnalyzer.analyze_conversation()`
- Used in: `backend/src/call_analysis/web_app_fastapi.py` → `/api/analyze` endpoint
- Frontend: `frontend/src/components/SentimentChart.tsx`

**Flow:**
```
Text Segments → DistilBERT → Sentiment Scores → Sentiment Trends
```

---

### 3. **SalePredictor (XGBoost)**

**Training:**
- Script: `backend/scripts/train_sale_predictor.py`
- Input: CSV with fused features
- Output: `backend/models/sale_model.pkl`

**Inference:**
- Location: `backend/src/call_analysis/models.py` → `SalePredictor.predict_sale_probability()`
- Called by: `ConversationAnalyzer.analyze_conversation()`
- Used in: `backend/src/call_analysis/web_app_fastapi.py` → `/api/analyze` endpoint
- Frontend: `frontend/src/components/SaleGauge.tsx`

**Flow:**
```
Sentiment + Emotion + Dynamics → Fused Features → sale_model.pkl → Sale Probability
```

---

## 🎯 Real-World Usage Example

### Scenario: User uploads a call recording

```python
# 1. User uploads audio via frontend
POST /api/upload
→ Audio saved to uploads/
→ Call record created in MongoDB

# 2. User triggers analysis
POST /api/analyze/{call_id}
→ Background processing starts

# 3. Processing pipeline (web_app_fastapi.py)
audio_processor.transcribe_audio()      # Whisper ASR
audio_processor.perform_speaker_diarization()  # Pyannote
text_processor.segment_conversation()    # Text processing
audio_processor.extract_audio_features() # librosa features

# 4. Model inference (ConversationAnalyzer)
analyzer.sentiment_analyzer.analyze_conversation_sentiment()
  → Uses DistilBERT (loaded lazily)
  
analyzer.emotion_detector.detect_conversation_emotions()
  → Loads emotion_model.pth
  → Runs CNN+LSTM inference
  
analyzer.sale_predictor.predict_sale_probability()
  → Loads sale_model.pkl
  → Runs XGBoost prediction

# 5. Results stored
→ MongoDB: Full analysis results
→ Frontend: Real-time updates via polling

# 6. User views dashboard
GET /api/results/{call_id}
→ Frontend displays:
  • SentimentChart (from SentimentAnalyzer)
  • EmotionChart (from EmotionDetector)
  • SaleGauge (from SalePredictor)
  • KeyPhrases (from SentimentAnalyzer)
```

---

## 📂 File Locations Summary

### Training Scripts
- `backend/scripts/train_emotion_model.py` - Trains CNN+LSTM
- `backend/scripts/train_sale_predictor.py` - Trains XGBoost

### Model Files (Saved After Training)
- `backend/models/emotion_model.pth` - CNN+LSTM weights
- `backend/models/emotion_dataset_stats.json` - Normalization stats
- `backend/models/sale_model.pkl` - XGBoost model
- `backend/models/sale_model_scaler.pkl` - Feature scaler

### Model Classes (Used During Inference)
- `backend/src/call_analysis/models.py`
  - `AcousticEmotionModel` - CNN+LSTM architecture
  - `EmotionDetector` - Wrapper for emotion detection
  - `SentimentAnalyzer` - DistilBERT sentiment analysis
  - `SalePredictor` - XGBoost sale prediction
  - `ConversationAnalyzer` - Orchestrator combining all models

### API Endpoints (Where Models Are Called)
- `backend/src/call_analysis/web_app_fastapi.py`
  - `/api/upload` - Upload audio
  - `/api/analyze/{call_id}` - Run full analysis (uses all 3 models)
  - `/api/results/{call_id}` - Get analysis results

### Frontend Components (Display Model Outputs)
- `frontend/src/components/SentimentChart.tsx` - Shows sentiment trends
- `frontend/src/components/EmotionChart.tsx` - Shows emotion distribution
- `frontend/src/components/SaleGauge.tsx` - Shows sale probability
- `frontend/src/components/AnalysisDashboard.tsx` - Main dashboard

---

## 🔑 Key Points

1. **Models are trained ONCE** → Saved to `backend/models/`
2. **Models are loaded during inference** → Used by `ConversationAnalyzer`
3. **All models work together** → Combined by `ConversationAnalyzer`
4. **Results flow to frontend** → Displayed in React dashboard
5. **Everything is stored** → MongoDB for persistence

---

## 🎬 End-to-End Example

```
User Action: Upload call_recording.wav
    │
    ▼
[FastAPI Backend]
    │
    ├─▶ AudioProcessor.transcribe_audio()
    │   └─▶ Whisper ASR → Text
    │
    ├─▶ AudioProcessor.perform_speaker_diarization()
    │   └─▶ Pyannote → Customer/Agent segments
    │
    ├─▶ AudioProcessor.extract_audio_features()
    │   └─▶ librosa → Mel-Spectrogram + MFCC
    │
    └─▶ ConversationAnalyzer.analyze_conversation()
        │
        ├─▶ SentimentAnalyzer (DistilBERT)
        │   └─▶ Sentiment: positive/negative/neutral per segment
        │
        ├─▶ EmotionDetector (CNN+LSTM)
        │   └─▶ Loads: emotion_model.pth
        │   └─▶ Emotion: happiness/anger/sadness/etc per segment
        │
        └─▶ SalePredictor (XGBoost)
            └─▶ Loads: sale_model.pkl
            └─▶ Sale Probability: 75%
    │
    ▼
[Results]
    │
    ├─▶ Sentiment Analysis: 8 segments, avg score: 0.65
    ├─▶ Emotion Analysis: Dominant: happiness (60%)
    ├─▶ Sale Prediction: 75% probability
    └─▶ Key Phrases: "great service", "very helpful"
    │
    ▼
[MongoDB Storage]
    │
    ▼
[Frontend Dashboard]
    │
    ├─▶ SentimentChart: Shows trend over time
    ├─▶ EmotionChart: Shows distribution
    ├─▶ SaleGauge: Shows 75% probability
    └─▶ KeyPhrases: Lists important phrases
```

---

This is the complete flow from training to production usage! 🚀

