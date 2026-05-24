# FastAPI Web Application (`web_app_fastapi.py`)

This document explains the **FastAPI web application** that powers the Call Analysis backend. It exposes REST endpoints for upload, analysis, results, status, history, and exports, and runs the full analysis pipeline in a **background thread** so the API stays responsive.

---

## What This Module Does

1. **REST API** – Upload audio, start analysis, get status and results, list history, export JSON/PDF/CSV.
2. **CORS** – Allows the Next.js frontend (e.g. `localhost:3000`) to call the API.
3. **MongoDB** – Stores call metadata (call_id, status, progress, results) in the `calls` collection.
4. **Background analysis** – When the user clicks “Analyze”, the server starts a **thread** that runs transcription → diarization → segments → features → ML, and updates MongoDB with progress and final results.
5. **Demo endpoints** – Optional endpoints for demo conversations and dashboards.

The frontend **polls** `/api/status/{call_id}` until status is `completed` or `failed`, then fetches `/api/results/{call_id}`.

---

## API Endpoints Overview

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/` | API info and list of endpoints |
| GET | `/health` | Health check (and MongoDB status) |
| POST | `/api/upload` | Upload audio file → returns `call_id` |
| POST | `/api/analyze` | Start analysis for a `call_id` (runs in background) |
| GET | `/api/status/{call_id}` | Get status and progress (pending / processing / completed / failed) |
| GET | `/api/results/{call_id}` | Get analysis results (sentiment, emotions, sale probability, key phrases) |
| GET | `/api/history` | List all calls (for history UI) |
| GET | `/api/export/{call_id}` | Download results as JSON |
| GET | `/api/export/{call_id}/pdf` | Download PDF report |
| GET | `/api/export/{call_id}/csv` | Download CSV data |
| GET | `/api/conversations` | Demo: list demo conversations |
| GET | `/api/analyze/{id}` | Demo: analyze a demo conversation |
| GET | `/api/dashboard/{id}` | Demo: generate dashboard HTML |
| GET | `/api/insights` | Demo: agent insights |

---

## 1. App Creation and CORS

The app is created inside `create_app()`. CORS is configured so the Next.js dev server can call the API without browser blocking.

**Code: App and CORS setup**

```python
def create_app() -> FastAPI:
    setup_project_logging()
    progress_log = get_progress_logger()

    app = FastAPI(
        title="Call Analysis System API",
        description="FastAPI backend for call analysis, compatible with existing Next.js frontend.",
        version="1.0.0",
    )

    origins = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
```

**Explanation:**  
- **origins**: Only these origins can send requests; typically the Next.js app on port 3000.  
- **allow_credentials=True**: Cookies/auth headers are allowed if you add them later.  
- **allow_methods=["*"]** and **allow_headers=["*"]**: Any method and header are allowed so the frontend can use POST with JSON and multipart uploads.

---

## 2. Shared Components and MongoDB

Inside `create_app()`, the same instances are created once and reused for every request. MongoDB is optional; if it fails, the app still starts but DB-dependent endpoints will return 503 until the connection is fixed.

**Code: Initializing shared components and MongoDB**

```python
    demo_system = DemoSystem(hf_token=Config.HF_TOKEN)
    analyzer = ConversationAnalyzer()
    dashboard = Dashboard()

    mongo_client = None
    db = None
    try:
        if Config.MONGODB_URI.startswith('mongodb+srv://') or 'mongodb.net' in Config.MONGODB_URI:
            mongo_client = MongoClient(
                Config.MONGODB_URI,
                serverSelectionTimeoutMS=30000,
                connectTimeoutMS=30000,
                tls=True,
                ...
            )
        else:
            mongo_client = MongoClient(Config.MONGODB_URI, serverSelectionTimeoutMS=10000)
        mongo_client.admin.command('ping')
        db = mongo_client[Config.MONGODB_DATABASE]
        logger.info(f"Successfully connected to MongoDB: {Config.MONGODB_DATABASE}")
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        mongo_client = None
        db = None  # App still runs; DB endpoints will raise 503
```

**Explanation:**  
- **DemoSystem**: Holds AudioProcessor, TextProcessor, FeatureExtractor, ConversationAnalyzer; used for both demo and **real** upload analysis.  
- **analyzer**: ConversationAnalyzer (sentiment, emotion, sale prediction).  
- **dashboard**: Dashboard (PII masking, optional Plotly charts).  
- **db**: Database used for `calls`, `analyses`, `exports`, etc. If connection fails, `db` stays `None` and `check_mongodb_connection()` raises HTTP 503 for DB-dependent routes.

---

## 3. Upload Endpoint

Upload accepts a single audio file, saves it to disk, generates a unique `call_id`, and creates a MongoDB document with status `pending`. Analysis is **not** started here; the client must call `POST /api/analyze` with that `call_id`.

**Code: Upload handler (simplified)**

```python
@app.post("/api/upload")
async def upload_audio(
    file: UploadFile = File(None),
    audio: UploadFile = File(None)
) -> JSONResponse:
    upload_file = file if file else audio
    if not upload_file or not upload_file.filename:
        raise HTTPException(status_code=400, detail="No file selected")
    if not allowed_file(upload_file.filename):
        raise HTTPException(status_code=400, detail="Invalid file type. Use .wav, .mp3, or .m4a")

    content = await upload_file.read()
    file_size = len(content)
    if file_size > 100 * 1024 * 1024:  # 100MB
        raise HTTPException(status_code=400, detail="File size exceeds 100MB")
    if file_size == 0:
        raise HTTPException(status_code=400, detail="File is empty")

    os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    base_name, ext = os.path.splitext(os.path.basename(upload_file.filename))
    unique_filename = f"{base_name}_{timestamp}{ext}"
    audio_path = os.path.join(Config.UPLOAD_FOLDER, unique_filename)
    with open(audio_path, "wb") as f:
        f.write(content)

    call_id = f"upload_{timestamp}"

    if db is not None:
        db.calls.insert_one({
            "call_id": call_id,
            "filename": filename,
            "audio_path": audio_path,
            "timestamp": datetime.now(),
            "status": "pending",
            "progress": 0,
            "duration": 0,
            "participants": 0,
            "avg_sentiment": 0,
            "sale_probability": 0,
            "sentiment_scores": [],
            "emotions": {},
            "key_phrases": {"positive": [], "negative": []},
        })

    return JSONResponse({
        "status": "success",
        "message": "File uploaded successfully. Use /api/analyze to start analysis.",
        "call_id": call_id,
        "filename": filename,
        "size": file_size,
    })
```

**Explanation:**  
- Accepts either `file` or `audio` form field for compatibility.  
- Validates extension (.wav, .mp3, .m4a) and size (max 100MB).  
- Saves file with a timestamped name to avoid overwrites.  
- `call_id` is `upload_<timestamp>` and is used in all later endpoints.  
- The MongoDB document is the “call record”; it will be updated by the background analysis with status, progress, and results.

---

## 4. Start Analysis (Background Thread)

When the client calls `POST /api/analyze` with `{"call_id": "upload_..."}`, the server checks that the call exists and is not already processing or completed, then starts a **daemon thread** that runs the full pipeline. The endpoint returns immediately with `status: "processing"`.

**Code: Starting analysis**

```python
@app.post("/api/analyze")
async def start_analysis(request: Request) -> JSONResponse:
    body = await request.json()
    call_id = body.get("call_id") or body.get("callId")
    if not call_id or not isinstance(call_id, str):
        raise HTTPException(status_code=422, detail="Request body must include 'call_id' or 'callId'.")

    check_mongodb_connection()
    call_doc = db.calls.find_one({"call_id": call_id})
    if not call_doc:
        raise HTTPException(status_code=404, detail="Call not found")

    if call_doc.get("status") == "processing":
        return JSONResponse({"message": "Analysis already in progress", "call_id": call_id, "status": "processing"})
    if call_doc.get("status") == "completed":
        return JSONResponse({"message": "Analysis already completed", "call_id": call_id, "status": "completed"})

    db.calls.update_one(
        {"call_id": call_id},
        {"$set": {"status": "processing", "progress": 0, "updated_at": datetime.now()}},
    )

    audio_path = call_doc.get("audio_path")
    if not audio_path or not os.path.exists(audio_path):
        raise HTTPException(status_code=404, detail="Audio file not found")

    analysis_thread = threading.Thread(
        target=lambda: _run_analysis_background(call_id, audio_path, call_doc.get("filename", "unknown")),
        daemon=True,
    )
    analysis_thread.start()

    return JSONResponse({
        "message": "Analysis started",
        "call_id": call_id,
        "status": "processing",
    })
```

**Explanation:**  
- **call_id** can be sent as `call_id` or `callId`.  
- We avoid duplicate work: if already `processing` or `completed`, we return that and do not start another thread.  
- Status is set to `processing` and progress to 0 before starting the thread.  
- **Daemon thread** runs `_run_analysis_background` so the main process can exit without waiting for analysis to finish. The frontend learns when it’s done by polling `/api/status/{call_id}`.

---

## 5. Background Analysis Pipeline (_run_analysis_background)

This function runs in the background thread. It performs transcription, diarization, segment merging, feature extraction, and full ML analysis, then writes results and PII-masked fields back to MongoDB.

**Code: Steps inside the background thread (concept)**

```python
def _run_analysis_background(call_id: str, audio_path: str, filename: str):
    try:
        # Step 1: Transcription (0–20% progress)
        db.calls.update_one({"call_id": call_id}, {"$set": {"status": "processing", "progress": 10, ...}})
        transcription = demo_system.audio_processor.transcribe_audio(audio_path, call_id)

        # Step 2: Diarization (20%)
        db.calls.update_one({"call_id": call_id}, {"$set": {"progress": 20, ...}})
        segments = demo_system.audio_processor.perform_speaker_diarization(audio_path, call_id)

        # Step 3: Merge transcript + diarization → processed segments (40%)
        db.calls.update_one({"call_id": call_id}, {"$set": {"progress": 40, ...}})
        processed_segments = demo_system.text_processor.segment_conversation(
            transcription.get("text", ""), segments, call_id,
            transcription_segments=transcription.get("segments", []),
        )

        # Step 4: Audio features for emotion (60%)
        db.calls.update_one({"call_id": call_id}, {"$set": {"progress": 60, ...}})
        audio_features = demo_system.audio_processor.extract_audio_features(audio_path)

        # Step 5: Full ML analysis (80% → 100%)
        db.calls.update_one({"call_id": call_id}, {"$set": {"progress": 80, ...}})
        result = demo_system.analyzer.analyze_conversation(
            audio_path=audio_path,
            segments=processed_segments,
            audio_features=audio_features,
            call_id=call_id,
        )

        # Mask PII in result text
        for seg in result.get("sentiment_analysis", []):
            seg["text"] = dashboard.mask_pii(seg["text"])
        result["summary"] = dashboard.mask_pii(result.get("summary", ""))

        # Transform result into the shape the frontend expects (sentiment_scores, emotions, key_phrases, ...)
        # ...

        # Update MongoDB with final results
        db.calls.update_one(
            {"call_id": call_id},
            {"$set": {
                "status": "completed",
                "progress": 100,
                "duration": actual_duration,
                "participants": participants,
                "avg_sentiment": float(avg_sentiment),
                "sale_probability": float(sale_probability),
                "sentiment_scores": sentiment_scores_formatted,
                "emotions": emotions_formatted,
                "key_phrases": key_phrases_formatted,
                "result": result,
                "updated_at": datetime.now(),
            }},
        )
    except Exception as e:
        db.calls.update_one(
            {"call_id": call_id},
            {"$set": {"status": "failed", "progress": 0, "error": str(e), "updated_at": datetime.now()}},
        )
```

**Explanation:**  
- Progress is updated at each stage (10 → 20 → 40 → 60 → 80 → 100) so the frontend can show a progress bar.  
- **analyze_conversation** runs sentiment per segment, emotion per segment (using segment mel from feature_extraction), conversational dynamics, fused feature vector, and sale prediction.  
- Before storing, all user-facing text in the result is PII-masked.  
- The code then maps the internal result format to the frontend format (e.g. `sentiment_scores` with `timestamp`, `score`, `label`; `emotions` dict; `key_phrases.positive` / `key_phrases.negative`).  
- On success, the call document is updated with `status: "completed"` and all result fields; on exception, `status: "failed"` and `error` message are set.

---

## 6. Status and Results Endpoints

The frontend polls status until it’s no longer `pending` or `processing`, then fetches results.

**Code: Get status**

```python
@app.get("/api/status/{call_id}")
def get_status(call_id: str) -> JSONResponse:
    check_mongodb_connection()
    call_doc = db.calls.find_one({"call_id": call_id})
    if not call_doc:
        raise HTTPException(status_code=404, detail="Call not found")
    return JSONResponse({
        "call_id": call_id,
        "status": call_doc.get("status", "unknown"),
        "progress": call_doc.get("progress", 0),
    })
```

**Explanation:**  
Returns the current `status` and `progress` (0–100) so the UI can show “Analyzing…” and a progress bar.

**Code: Get results (shape for frontend)**

```python
@app.get("/api/results/{call_id}")
def get_results(call_id: str) -> JSONResponse:
    check_mongodb_connection()
    call_doc = db.calls.find_one({"call_id": call_id})
    if not call_doc:
        raise HTTPException(status_code=404, detail="Call not found")

    # Ensure formats match frontend expectations (sentiment_scores, emotions, key_phrases)
    sentiment_scores = call_doc.get("sentiment_scores", [])
    emotions = call_doc.get("emotions", {})
    key_phrases = call_doc.get("key_phrases", {"positive": [], "negative": []})
    # ... normalize keys (e.g. required emotion keys, positive/negative lists) ...

    return JSONResponse({
        "call_id": call_id,
        "sentiment_scores": sentiment_scores,
        "emotions": emotions,
        "sale_probability": float(call_doc.get("sale_probability", 0)),
        "key_phrases": key_phrases,
        "summary": {
            "avg_sentiment": float(call_doc.get("avg_sentiment", 0)),
            "total_duration": float(call_doc.get("duration", 0)),
            "participants": int(call_doc.get("participants", 2)),
        },
    })
```

**Explanation:**  
Results are read from the `calls` document. The handler normalizes structure (e.g. sentiment_scores with `timestamp`/`score`/`label`, all required emotion keys, key_phrases.positive/negative) so the frontend dashboard can consume them without extra logic.

---

## 7. History Endpoint

History returns a list of all calls (e.g. for the “History” section in the UI), sorted by timestamp descending.

**Code: Get history**

```python
@app.get("/api/history")
def get_history() -> JSONResponse:
    check_mongodb_connection()
    calls = list(
        db.calls.find(
            {},
            {
                "call_id": 1, "filename": 1, "timestamp": 1, "duration": 1,
                "avg_sentiment": 1, "sale_probability": 1, "participants": 1, "status": 1,
            },
        ).sort("timestamp", -1)
    )
    formatted_calls = []
    for call in calls:
        formatted_calls.append({
            "call_id": call.get("call_id", ""),
            "filename": call.get("filename", "unknown"),
            "timestamp": call.get("timestamp").isoformat() if isinstance(call.get("timestamp"), datetime) else str(call.get("timestamp")),
            "duration": float(call.get("duration", 0)),
            "avg_sentiment": float(call.get("avg_sentiment", 0)),
            "sale_probability": float(call.get("sale_probability", 0)),
            "participants": int(call.get("participants", 0)),
            "status": call.get("status", "unknown"),
        })
    return JSONResponse(formatted_calls)
```

**Explanation:**  
Only the fields needed for the history list are projected. Timestamps are converted to ISO strings for JSON. The frontend can use this to show a table or list and link to `/results/[callId]`.

---

## 8. Export Endpoints (JSON, PDF, CSV)

Exports read the same call document and return file downloads.

- **JSON**: Builds a dict with call_id, filename, timestamp, status, duration, participants, avg_sentiment, sale_probability, sentiment_scores, emotions, key_phrases, and full `result`; writes to a file and returns it with `FileResponse`.
- **PDF**: Uses ReportLab to build a report (title, call info table, summary table, emotion distribution table, key phrases). Saves to `exports/{call_id}_report.pdf` and returns it.
- **CSV**: Writes call info, summary, emotion distribution, sentiment scores over time, and key phrases to a single CSV in memory and returns it with `Response(content=..., media_type="text/csv")` and `Content-Disposition: attachment`.

**Code: PDF export (structure)**

```python
@app.get("/api/export/{call_id}/pdf")
def export_pdf(call_id: str) -> FileResponse:
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, ...
    check_mongodb_connection()
    call_doc = db.calls.find_one({"call_id": call_id})
    if not call_doc:
        raise HTTPException(status_code=404, detail="Call not found")

    pdf_path = os.path.join("exports", f"{call_id}_report.pdf")
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    story = []
    story.append(Paragraph("Call Analysis Report", title_style))
    # Call info table: Call ID, Filename, Date, Duration, Participants
    # Summary table: Average Sentiment, Sale Probability, Overall Sentiment
    # Emotion distribution table
    # Key phrases (positive / negative)
    doc.build(story)
    return FileResponse(pdf_path, media_type="application/pdf", filename=os.path.basename(pdf_path))
```

**Explanation:**  
ReportLab is used to build a simple report. All data comes from `call_doc`; no extra analysis is run. The file is written under `exports/` and sent as a downloadable PDF.

---

## 9. Health Check

The health endpoint reports API and MongoDB status. Useful for load balancers and monitoring.

**Code: Health check**

```python
@app.get("/health")
def health_check() -> JSONResponse:
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "mongodb": "connected" if db is not None else "disconnected"
    }
    if db is not None:
        try:
            db.command("ping")
        except Exception as e:
            health_status["mongodb"] = "error"
            health_status["mongodb_error"] = str(e)
    else:
        health_status["status"] = "degraded"
        health_status["mongodb"] = "not_configured"
    status_code = 200 if health_status["status"] == "healthy" else 503
    return JSONResponse(health_status, status_code=status_code)
```

**Explanation:**  
- If MongoDB is connected and `ping` succeeds, status is `healthy` and mongodb is `connected`.  
- If MongoDB is not configured or ping fails, status can be `degraded` and HTTP 503 is returned so monitors can alert.

---

## 10. Request Flow Summary

1. **Upload**: Client sends audio → server saves file and creates DB document with `call_id`, status `pending` → response includes `call_id`.
2. **Analyze**: Client sends `{ "call_id": "..." }` → server sets status to `processing` and starts background thread → returns immediately.
3. **Polling**: Client repeatedly calls `GET /api/status/{call_id}` (e.g. every 2 seconds) until `status` is `completed` or `failed`.
4. **Results**: When `completed`, client calls `GET /api/results/{call_id}` and displays the dashboard.
5. **History**: Client calls `GET /api/history` to list calls and link to `/results/[callId]`.
6. **Export**: Client calls `/api/export/{call_id}`, `/api/export/{call_id}/pdf`, or `/api/export/{call_id}/csv` to download files.

---

## Summary Table

| Topic | Detail |
|--------|--------|
| **Framework** | FastAPI; app created in `create_app()`. |
| **CORS** | localhost:3000 (and 127.0.0.1:3000) allowed for frontend. |
| **Storage** | MongoDB `calls` collection for metadata and results; uploads on disk. |
| **Upload** | POST /api/upload → save file, create call document, return call_id. |
| **Analysis** | POST /api/analyze → start daemon thread running full pipeline; progress/results written to DB. |
| **Status/Results** | GET /api/status and GET /api/results read from same call document. |
| **Exports** | JSON, PDF (ReportLab), CSV generated from call document and returned as files. |
| **Health** | GET /health returns API and MongoDB status; 503 if DB down. |

Together, these endpoints and the background pipeline form the backend that the Next.js frontend uses for the Call Analysis app.
