# How to Run the Call Analysis Project

This guide explains how to run the **full stack (frontend + backend)** and the **CLI pipeline**. Pipeline steps use **`[CALL_ANALYSIS]`**-prefixed logs so you can see progress in the backend terminal.

---

## Run the full stack (frontend + backend)

Use this when you want to **upload a file in the browser** and run the full pipeline (transcription → diarization → text processing → feature extraction → analysis → results) with results shown on the frontend.

### 1. Start the backend

Open a terminal and run:

```powershell
cd C:\Users\PMYLS\Desktop\fyp\Call_Analysis\backend
pip install -r requirements.txt
python run_web_app.py
```

- The API runs at **http://localhost:5000** (configurable via `FLASK_PORT` in `.env`).
- You should see: `Access the API at: http://0.0.0.0:5000`.
- Keep this terminal open; pipeline progress will appear here as `[CALL_ANALYSIS]` logs when you run an analysis.

### 2. Start the frontend

Open a **second** terminal and run:

```powershell
cd C:\Users\PMYLS\Desktop\fyp\Call_Analysis\frontend
npm install
npm run dev
```

- The app runs at **http://localhost:3000**.
- The frontend calls the backend at **http://localhost:5000** by default. To use a different URL, create `frontend/.env.local` with `NEXT_PUBLIC_API_URL=http://your-backend-host:5000`.

### 3. Use the app

1. Open **http://localhost:3000** in your browser.
2. In **Upload Audio File**, select a `.wav`, `.mp3`, or `.m4a` file (up to 100MB) and upload.
3. After upload, click **Run Analysis**. The backend will run the full pipeline in the background.
4. The page will poll for status; when analysis is complete, **Analysis Results** will show sentiment, emotions, sale probability, and key phrases.

Backend progress (transcription, diarization, features, ML analysis) will appear in the **backend** terminal with the `[CALL_ANALYSIS]` prefix.

### Prerequisites for full stack

- **Python 3.8+** and **Node.js** (for the frontend).
- **`.env`** in the project root or in `backend/` with:
  - `HF_TOKEN=<your_huggingface_token>` — required for speaker diarization. Get a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) and accept the Pyannote model terms.
  - `MONGODB_URI` — required for the web app to store uploads and results (e.g. `mongodb://localhost:27017/` for local MongoDB, or a MongoDB Atlas connection string). If MongoDB is not available, upload may still work but analysis results may not be stored or retrieved correctly.

**Troubleshooting: "The DNS query name does not exist" (MongoDB Atlas)**  
- The hostname in your URI (e.g. `cluster0.gh40us5.mongodb.net`) no longer resolves. In [MongoDB Atlas](https://cloud.mongodb.com): open your cluster → **Connect** → **Drivers** and copy the connection string again (the cluster ID in the hostname may have changed if the cluster was recreated). Put it in `.env` as `MONGODB_URI=...`. In **Network Access**, add your IP or `0.0.0.0/0` for testing. Restart the backend.

---

## Run the entire project (full pipeline) via CLI

The full pipeline runs: **audio → transcription → diarization → text processing → feature extraction → analysis → dashboard → results**.

### 1. From project root (recommended)

```powershell
cd C:\Users\PMYLS\Desktop\fyp\Call_Analysis
python backend/run_full_analysis.py real_audio.wav
```

- **Output** is written to `output/` (created if missing).
- **Call ID** is auto-generated (e.g. `call_20250220_143000`) unless you pass `--call-id`.
- Optional args:
  - `--call-id MY_CALL` — use a specific call ID.
  - `--output-dir my_output` — use a different output directory (relative to project root).

### 2. From the backend directory

```powershell
cd C:\Users\PMYLS\Desktop\fyp\Call_Analysis\backend
python run_full_analysis.py ../real_audio.wav
```

Use a path to your audio file relative to the current directory or absolute.

### Example with options

```powershell
python backend/run_full_analysis.py real_audio.wav --call-id demo_call --output-dir output
```

---

## Progress logging

Every step logs with the **`[CALL_ANALYSIS]`** prefix so you can:

- **See progress** while the pipeline runs (e.g. “Step 1/10”, “Step 2/10”, …).
- **Filter** only project messages (e.g. in PowerShell:  
  `python backend/run_full_analysis.py real_audio.wav 2>&1 | Select-String "CALL_ANALYSIS"`).

Example output:

```
[CALL_ANALYSIS] INFO | ======================================================================
[CALL_ANALYSIS] INFO | COMPLETE CALL ANALYSIS PIPELINE
[CALL_ANALYSIS] INFO | Step 1/10: Initializing components...
[CALL_ANALYSIS] INFO | Step 2/10: Transcribing audio with Whisper...
...
```

---

## Prerequisites

1. **Python** (3.8+). Create and use a virtual environment if you prefer:
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

2. **Dependencies** (from `backend/`):
   ```powershell
   cd backend
   pip install -r requirements.txt
   ```
   Or install the main packages: `openai-whisper`, `whisperx`, `pyannote.audio`, `pydub`, `python-dotenv`, etc., as in your project.

3. **`.env` in project root or `backend/`** with:
   - `HF_TOKEN=<your_huggingface_token>` — required for **speaker diarization** (Pyannote/WhisperX).  
     Get a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) and accept the Pyannote model terms.

4. **Audio file**: WAV, MP3, or M4A. For a quick test you can generate a 5‑minute diarization test file (see below).

---

## Optional scripts

| Goal | Command (from project root) |
|------|-----------------------------|
| **Transcription only** (no diarization/analysis) | `python backend/scripts/transcribe_audio.py real_audio.wav` |
| **Diarization only** (audio + existing transcription JSON) | `python backend/run_diarization_only.py real_audio.wav output/call_xxx_transcription.json --num-speakers 2` |
| **Generate 5‑min test audio** (male + female voices for diarization) | `python backend/scripts/generate_5min_diarization_test.py` |

All of these use the same **`[CALL_ANALYSIS]`** progress logging.

---

## Output files (full pipeline)

After `run_full_analysis.py` finishes, in `output/` you should see:

- `{call_id}_transcription.json` — full transcript and segments
- `{call_id}_diarization.json` — speaker segments and timings
- `{call_id}_results.json` — analysis summary and metrics
- `{call_id}_dashboard.html` — dashboard (if generation succeeded)

---

## Quick test (no existing audio)

1. Generate 5‑minute test audio (two speakers, male/female):
   ```powershell
   pip install edge-tts pydub
   python backend/scripts/generate_5min_diarization_test.py
   ```
   This creates `output/5min_diarization_test.wav` and related JSON files.

2. Run the full pipeline on it:
   ```powershell
   python backend/run_full_analysis.py output/5min_diarization_test.wav --call-id test5min
   ```

You should see **`[CALL_ANALYSIS]`** step-by-step progress in the console.
