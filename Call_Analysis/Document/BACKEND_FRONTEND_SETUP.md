# Backend and Frontend Setup Guide

This guide explains how to run the backend and frontend, and how they connect together.

---

## Table of Contents

1. [Backend Setup](#backend-setup)
2. [Frontend Setup](#frontend-setup)
3. [How They Connect](#how-they-connect)
4. [Running Both Together](#running-both-together)
5. [Troubleshooting](#troubleshooting)

**Note:** This project now uses **FastAPI** (served via **uvicorn**) as the primary backend framework, and **uv** as the Python package manager. For detailed uv usage, see [UV_PACKAGE_MANAGER.md](./UV_PACKAGE_MANAGER.md).

---

## Backend Setup

### Prerequisites

- Python 3.10 or higher
- **uv** package manager (recommended and used in this project)
- MongoDB (local or Atlas connection string)
- (Optional) Hugging Face token for Pyannote.audio models

### Step 1: Navigate to Backend Directory

```bash
cd backend
```

### Step 2: Install uv (if not already installed)

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or via pip (temporary, until uv is installed)
pip install uv
```

### Step 3: Create Virtual Environment and Install Dependencies

This project uses `uv` with `pyproject.toml` for dependency management.

```bash
# Create virtual environment and install all dependencies
uv sync

# This will:
# - Create a virtual environment (if needed)
# - Install all dependencies from pyproject.toml
# - Create/update uv.lock file
```

**Alternative: Manual virtual environment setup**
```bash
# Create virtual environment with uv
uv venv

# Activate virtual environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate

# Install dependencies
uv pip install -e .
```

### Step 4: Create Environment File

Create a `.env` file in the `backend/` directory:

```bash
# Copy from CONFIGURATION.md template or create new
touch .env
```

Minimum required `.env` configuration:

```bash
# Hugging Face Configuration (optional for basic functionality)
HF_TOKEN=your_huggingface_token_here

# Database Configuration
MONGODB_URI=mongodb://localhost:27017/
MONGODB_DATABASE=call_center_db

# Flask Configuration
FLASK_SECRET_KEY=your-secret-key-here
FLASK_DEBUG=True
FLASK_HOST=0.0.0.0
FLASK_PORT=5000

# File Upload Configuration
UPLOAD_FOLDER=uploads
MAX_CONTENT_LENGTH=104857600
ALLOWED_EXTENSIONS=wav,mp3,m4a,flac

# Demo Configuration
DEMO_MODE=True
USE_SIMULATED_AUDIO=True
```

**Note:** For local MongoDB, ensure MongoDB is running:
```bash
# macOS (if installed via Homebrew)
brew services start mongodb-community

# Linux
sudo systemctl start mongod

# Or use MongoDB Atlas connection string in MONGODB_URI
```

### Step 5: Run the Backend

```bash
# From backend directory
python run_web_app.py
```

Or directly:

```bash
python -m flask run --host=0.0.0.0 --port=5000
```

**Expected Output:**
```
Starting Call Analysis Web Application...
==================================================
Access the dashboard at: http://localhost:5000
Press Ctrl+C to stop the server
==================================================
 * Running on http://0.0.0.0:5000
```

### Step 6: Verify Backend is Running

Open your browser or use curl:

```bash
# Health check
curl http://localhost:5000/health

# API root
curl http://localhost:5000/
```

You should see JSON responses indicating the API is working.

---

## Frontend Setup

### Prerequisites

- Node.js 18+ and npm/yarn
- Backend must be running (see above)

### Step 1: Navigate to Frontend Directory

```bash
cd frontend
```

### Step 2: Install Dependencies

```bash
npm install
# OR
yarn install
```

### Step 3: Run the Frontend

```bash
npm run dev
# OR
yarn dev
```

**Expected Output:**
```
  ▲ Next.js 14.0.3
  - Local:        http://localhost:3000
  - Ready in 2.3s
```

### Step 4: Access the Frontend

Open your browser to:
```
http://localhost:3000
```

---

## How They Connect

### Architecture Overview

```
┌─────────────────┐         HTTP Requests          ┌─────────────────┐
│                 │ ──────────────────────────────> │                 │
│   Frontend      │                                │    Backend      │
│   (Next.js)     │ <──────────────────────────── │    (Flask)      │
│   Port: 3000    │      JSON Responses           │   Port: 5000    │
└─────────────────┘                                └─────────────────┘
```

### Connection Methods

#### Method 1: Next.js Rewrite (Development - Default)

The frontend uses Next.js rewrites to proxy API requests. This is configured in `frontend/next.config.js`:

```javascript
async rewrites() {
  return [
    {
      source: '/api/:path*',
      destination: 'http://localhost:5000/api/:path*',
    },
  ];
}
```

**How it works:**
- Frontend makes requests to `/api/upload` (relative URL)
- Next.js automatically rewrites to `http://localhost:5000/api/upload`
- No CORS issues in development
- Backend runs on port 5000, frontend on port 3000

#### Method 2: Direct API URL (Production)

For production, set environment variable:

```bash
# In frontend/.env.local
NEXT_PUBLIC_API_URL=http://your-backend-server:5000
```

The API service (`frontend/src/lib/api.ts`) will use this URL directly.

### API Endpoints

The frontend communicates with these backend endpoints:

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/upload` | POST | Upload audio file |
| `/api/analyze` | POST | Start analysis for a call |
| `/api/status/<call_id>` | GET | Get analysis status |
| `/api/results/<call_id>` | GET | Get analysis results |
| `/api/history` | GET | Get call history |
| `/api/export/<call_id>/pdf` | GET | Export PDF report |
| `/api/export/<call_id>/csv` | GET | Export CSV data |
| `/health` | GET | Health check |

### API Service

The frontend uses a centralized API service located at:
- `frontend/src/lib/api.ts`

This service:
- Uses axios for HTTP requests
- Handles errors consistently
- Provides typed methods for all API calls
- Automatically uses the correct base URL

**Example usage in components:**
```typescript
import { apiService } from '@/lib/api';

// Upload file
const response = await apiService.uploadAudio(file);

// Get results
const results = await apiService.getResults(callId);
```

---

## Running Both Together

### Quick Start Script

Create a script to run both (optional):

**`start-dev.sh` (macOS/Linux):**
```bash
#!/bin/bash

# Terminal 1: Start Backend
cd backend
source .venv/bin/activate  # uv creates .venv by default
python run_web_app.py &

# Terminal 2: Start Frontend
cd frontend
npm run dev
```

**`start-dev.bat` (Windows):**
```batch
@echo off
start cmd /k "cd backend && python run_web_app.py"
timeout /t 3
start cmd /k "cd frontend && npm run dev"
```

### Manual Steps

1. **Terminal 1 - Start Backend:**
   ```bash
   cd backend
   python run_web_app.py
   ```
   Wait for: `Running on http://0.0.0.0:5000`

2. **Terminal 2 - Start Frontend:**
   ```bash
   cd frontend
   npm run dev
   ```
   Wait for: `Ready in X.Xs`

3. **Open Browser:**
   ```
   http://localhost:3000
   ```

### Verification

1. **Backend Health Check:**
   ```bash
   curl http://localhost:5000/health
   ```
   Should return: `{"status": "healthy", ...}`

2. **Frontend API Test:**
   - Open browser DevTools (F12)
   - Go to Network tab
   - Upload a file in the frontend
   - Check that requests go to `localhost:5000/api/...`

---

## CORS Configuration

### Current Setup

The backend **does not currently have CORS enabled**. This works in development because:
- Next.js rewrites proxy requests (no CORS needed)
- Same-origin policy is satisfied

### If You Need CORS (Direct API Calls)

If you want to make direct API calls from the frontend (not using Next.js rewrites), add CORS to the backend:

**1. Install flask-cors:**
```bash
# Using uv
uv pip install flask-cors

# Or add to pyproject.toml and run:
uv sync
```

**2. Update `backend/src/call_analysis/web_app.py`:**
```python
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
# OR configure specific origins:
# CORS(app, resources={r"/api/*": {"origins": "http://localhost:3000"}})
```

**3. Restart backend**

---

## Troubleshooting

### Backend Issues

**Problem: Port 5000 already in use**
```bash
# Find process using port 5000
lsof -i :5000  # macOS/Linux
netstat -ano | findstr :5000  # Windows

# Kill process or change port in .env
FLASK_PORT=5001
```

**Problem: MongoDB connection failed**
```bash
# Check MongoDB is running
mongosh  # or mongo

# Or use MongoDB Atlas connection string
MONGODB_URI=
```

**Problem: Module not found errors**
```bash
# Ensure you're in the backend directory
cd backend

# Reinstall dependencies with uv
uv sync

# Or if using manual venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
uv pip install -e .
```

mongodb+srv:

//user:pass@cluster.mongodb.net/

### Frontend Issues

**Problem: Cannot connect to backend**
- Ensure backend is running on port 5000
- Check `next.config.js` has correct rewrite destination
- Verify no firewall blocking port 5000

**Problem: CORS errors**
- Use Next.js rewrites (default) instead of direct API calls
- Or enable CORS in backend (see above)

**Problem: API service not found**
- Ensure `frontend/src/lib/api.ts` exists
- Check imports use `@/lib/api` (Next.js path alias)

### Connection Issues

**Problem: Frontend shows "Unable to connect to server"**
1. Check backend is running: `curl http://localhost:5000/health`
2. Check backend logs for errors
3. Verify port 5000 is not blocked
4. Check `next.config.js` rewrite configuration

**Problem: 404 errors on API calls**
1. Verify backend routes match frontend expectations
2. Check API endpoint paths in `web_app.py`
3. Ensure Next.js rewrite is working (check Network tab)

---

## Environment Variables Summary

### Backend (`.env` in `backend/`)

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `HF_TOKEN` | No | `''` | Hugging Face token for models |
| `MONGODB_URI` | Yes | `mongodb://localhost:27017/` | MongoDB connection |
| `MONGODB_DATABASE` | Yes | `call_center_db` | Database name |
| `FLASK_PORT` | No | `5000` | Backend port |
| `FLASK_DEBUG` | No | `True` | Debug mode |
| `DEMO_MODE` | No | `True` | Use demo data |

### Frontend (`.env.local` in `frontend/`)

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `NEXT_PUBLIC_API_URL` | No | `http://localhost:5000` | Backend API URL |

---

## Production Deployment

### Backend

1. Set `FLASK_DEBUG=False` in `.env`
2. Use production WSGI server (gunicorn, uwsgi)
3. Configure proper MongoDB connection
4. Set secure `FLASK_SECRET_KEY`

```bash
# Example with gunicorn
uv pip install gunicorn
# Or add gunicorn to pyproject.toml and run: uv sync

gunicorn -w 4 -b 0.0.0.0:5000 "call_analysis.web_app:app"
```

### Frontend

1. Build the frontend:
   ```bash
   cd frontend
   npm run build
   ```

2. Set `NEXT_PUBLIC_API_URL` to production backend URL

3. Run production server:
   ```bash
   npm start
   ```

---

## Quick Reference

### Start Backend
```bash
cd backend
python run_web_app.py
```

### Start Frontend
```bash
cd frontend
npm run dev
```

### Test Backend
```bash
curl http://localhost:5000/health
```

### Test Frontend
```
Open http://localhost:3000
```

---

**Need Help?** Check the logs in both terminals for detailed error messages.

