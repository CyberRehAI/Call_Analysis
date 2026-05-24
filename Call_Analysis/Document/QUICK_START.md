# Quick Start Guide

## 🚀 Running the Application

### Prerequisites
- Python 3.10+
- **uv** package manager (install: `curl -LsSf https://astral.sh/uv/install.sh | sh`)
- Node.js 18+
- MongoDB (local or Atlas)

---

## Backend Setup (5 minutes)

### 1. Navigate to backend
```bash
cd backend
```

### 2. Install dependencies with uv
```bash
# This project uses uv with pyproject.toml
# uv sync will create venv and install all dependencies
uv sync
```

**Note:** If you don't have uv installed:
```bash
# Install uv first
curl -LsSf https://astral.sh/uv/install.sh | sh
# Then run: uv sync
```

### 4. Create `.env` file in `backend/` directory
```bash
# Minimum required configuration
MONGODB_URI=mongodb://localhost:27017/
MONGODB_DATABASE=call_center_db
FLASK_PORT=5000
FLASK_DEBUG=True
DEMO_MODE=True
```

### 5. Run backend
```bash
python run_web_app.py
```

**✅ Backend running on:** `http://localhost:5000`

---

## Frontend Setup (2 minutes)

### 1. Navigate to frontend
```bash
cd frontend
```

### 2. Install dependencies
```bash
npm install
```

### 3. Run frontend
```bash
npm run dev
```

**✅ Frontend running on:** `http://localhost:3000`

---

## How They Connect

### Architecture
```
Frontend (Next.js) :3000  ──>  Backend (Flask) :5000
```

### Connection Method
- **Development:** Next.js rewrites proxy API calls automatically
- **No CORS issues:** Backend has CORS enabled for localhost:3000
- **API Service:** Frontend uses `frontend/src/lib/api.ts` for all API calls

### API Endpoints Used
- `POST /api/upload` - Upload audio file
- `POST /api/analyze` - Start analysis
- `GET /api/status/<call_id>` - Check status
- `GET /api/results/<call_id>` - Get results
- `GET /api/history` - Get call history

---

## Testing the Connection

### 1. Test Backend
```bash
curl http://localhost:5000/health
```
Should return: `{"status": "healthy", ...}`

### 2. Test Frontend
- Open `http://localhost:3000`
- Upload an audio file
- Check browser DevTools Network tab for API calls

---

## Troubleshooting

### Backend won't start
- Check port 5000 is free: `lsof -i :5000`
- Verify MongoDB is running
- Check `.env` file exists in `backend/` directory

### Frontend can't connect
- Ensure backend is running on port 5000
- Check `next.config.js` has rewrite configuration
- Verify no firewall blocking port 5000

### CORS errors
- Backend has CORS enabled (already configured)
- If issues persist, check `web_app.py` CORS settings

---

## Full Documentation

For detailed setup instructions, see:
- **`Document/BACKEND_FRONTEND_SETUP.md`** - Complete setup guide
- **`Document/UV_PACKAGE_MANAGER.md`** - UV package manager guide
- **`backend/CONFIGURATION.md`** - Environment variables reference

---

## Quick Commands Reference

```bash
# Start Backend
cd backend && python run_web_app.py

# Start Frontend (in another terminal)
cd frontend && npm run dev

# Test Backend
curl http://localhost:5000/health

# Access Frontend
open http://localhost:3000
```

---

**That's it!** You should now have both backend and frontend running and connected. 🎉

