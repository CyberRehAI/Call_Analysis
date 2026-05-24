"""
FastAPI web application for call analysis system.

This is a FastAPI port of the original Flask-based `web_app.py`, exposing the
same REST API surface so the existing Next.js frontend can keep working:

- GET    /                    -> API root metadata
- GET    /health              -> health check
- GET    /api/conversations   -> demo conversations
- GET    /api/analyze/{id}    -> analyze demo conversation
- GET    /api/analyze-all     -> analyze all demo conversations
- GET    /api/dashboard/{id}  -> generate dashboard HTML
- GET    /api/insights        -> aggregate agent insights
- POST   /api/upload          -> upload & analyze real audio
- POST   /api/analyze         -> start analysis for uploaded call
- GET    /api/results/{id}    -> get analysis results for a call
- GET    /api/status/{id}     -> get analysis status
- GET    /api/history         -> call history
- GET   /api/export/{id}      -> export JSON
- GET   /api/export/{id}/pdf  -> export PDF
- GET   /api/export/{id}/csv  -> export CSV
"""

from __future__ import annotations

import warnings

# Globally silence known noisy but harmless third-party warnings so they don't
# clutter the FastAPI/Uvicorn logs.
warnings.filterwarnings("ignore", message=".*torchcodec is not installed correctly.*")
warnings.filterwarnings("ignore", message=".*libtorchcodec_core[0-9].*")
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated as an API.*")

import os
import ssl
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import (
    FastAPI,
    Request,
    UploadFile,
    File,
    Form,
    HTTPException,
    status,
    Depends,
    Body,
)
import threading
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, Response
from pymongo import MongoClient

from .demo import DemoSystem
from .models import ConversationAnalyzer
from .dashboard import Dashboard
from .progress_logger import setup_project_logging, get_progress_logger

# Import configuration (same pattern as Flask app)
import sys as _sys

_sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import Config  # type: ignore  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    setup_project_logging()
    progress_log = get_progress_logger()

    app = FastAPI(
        title="Call Analysis System API",
        description="FastAPI backend for call analysis, compatible with existing Next.js frontend.",
        version="1.0.0",
    )

    # CORS for frontend (Next.js on localhost:3000)
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

    # Initialize shared components (same as Flask version)
    demo_system = DemoSystem(hf_token=Config.HF_TOKEN)
    analyzer = ConversationAnalyzer()
    dashboard = Dashboard()
    
    # Initialize MongoDB client with proper connection options for Atlas
    mongo_client = None
    db = None
    
    # TLS 1.2-only default context (fixes TLSV1_ALERT_INTERNAL_ERROR with Atlas on some Windows/Python setups)
    def _make_tls12_context():
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        ctx.load_default_certs()
        if hasattr(ssl, "TLSVersion"):
            ctx.minimum_version = ssl.TLSVersion.TLSv1_2
            ctx.maximum_version = ssl.TLSVersion.TLSv1_2
        return ctx

    _orig_create_default_context = ssl.create_default_context
    ssl.create_default_context = lambda purpose=ssl.Purpose.SERVER_AUTH: _make_tls12_context()
    try:
        logger.info("Attempting to connect to MongoDB...")
        
        # For MongoDB Atlas, use connection options that handle SSL/TLS properly
        if Config.MONGODB_URI.startswith('mongodb+srv://') or 'mongodb.net' in Config.MONGODB_URI:
            # Atlas connection - use longer timeouts and retry options
            logger.info("Detected MongoDB Atlas connection string")
            mongo_client = MongoClient(
                Config.MONGODB_URI,
                serverSelectionTimeoutMS=30000,  # 30 second timeout (increased)
                connectTimeoutMS=30000,  # 30 second connection timeout
                socketTimeoutMS=30000,  # 30 second socket timeout
                retryWrites=True,
                retryReads=True,
                tls=True,  # Explicitly enable TLS for Atlas
                tlsAllowInvalidCertificates=False,  # Use proper SSL validation
            )
        else:
            # Local MongoDB connection
            logger.info("Detected local MongoDB connection")
            mongo_client = MongoClient(
                Config.MONGODB_URI,
                serverSelectionTimeoutMS=10000,
            )
        
        # Test connection immediately with better error reporting
        logger.info("Testing MongoDB connection...")
        mongo_client.admin.command('ping')
        logger.info(f"✅ Successfully connected to MongoDB: {Config.MONGODB_DATABASE}")
        db = mongo_client[Config.MONGODB_DATABASE]
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"❌ Failed to connect to MongoDB")
        logger.error(f"   Error: {error_msg}")
        
        # Provide helpful diagnostic messages
        if 'ServerSelectionTimeoutError' in error_msg or 'No replica set members found' in error_msg:
            logger.error("")
            logger.error("🔍 DIAGNOSTIC: This usually means:")
            logger.error("   1. Your IP address is not whitelisted in MongoDB Atlas")
            logger.error("      → Go to Atlas → Network Access → Add IP Address")
            logger.error("      → Add '0.0.0.0/0' for development (allows all IPs)")
            logger.error("   2. Firewall is blocking the connection")
            logger.error("   3. MongoDB Atlas cluster might be paused")
            logger.error("")
            logger.error("💡 Quick fix: Add your IP to Atlas Network Access:")
            logger.error("   https://cloud.mongodb.com/v2#/security/network/whitelist")
            logger.error("")
        elif 'authentication' in error_msg.lower() or 'auth' in error_msg.lower():
            logger.error("")
            logger.error("🔍 DIAGNOSTIC: Authentication failed")
            logger.error("   → Check your username and password in MONGODB_URI")
            logger.error("   → Verify database user has read/write permissions")
            logger.error("")
        elif 'SSL' in error_msg or 'TLS' in error_msg:
            logger.error("")
            logger.error("🔍 DIAGNOSTIC: SSL/TLS handshake failed")
            logger.error("   → Check your connection string format")
            logger.error("   → Ensure you're using 'mongodb+srv://' for Atlas")
            logger.error("")
        
        logger.error(f"   Connection URI format: {'mongodb+srv://' if Config.MONGODB_URI.startswith('mongodb+srv://') else 'mongodb://'}...")
        logger.error("")
        logger.error("⚠️  Backend will start but MongoDB-dependent features will fail.")
        logger.error("   Fix the connection and restart the server.")
        logger.error("")
        
        # Don't raise error - allow app to start but log the issue
        # This way the API can still serve non-database endpoints
        mongo_client = None
        db = None
    finally:
        ssl.create_default_context = _orig_create_default_context

    # --------- Helpers ----------

    def allowed_file(filename: str) -> bool:
        return "." in filename and filename.rsplit(".", 1)[1].lower() in Config.ALLOWED_EXTENSIONS
    
    def check_mongodb_connection() -> None:
        """Check if MongoDB is connected, raise HTTPException if not."""
        if db is None:
            raise HTTPException(
                status_code=503,
                detail="MongoDB connection not available. Please check your MONGODB_URI configuration and ensure MongoDB Atlas network access is configured."
            )

    # --------- Routes ----------

    @app.get("/")
    def index() -> Dict[str, Any]:
        """API root endpoint."""
        return {
            "message": "Call Analysis System API (FastAPI)",
            "version": "1.0.0",
            "endpoints": {
                "health": "/health",
                "upload": "/api/upload",
                "analyze": "/api/analyze",
                "results": "/api/results/{call_id}",
                "status": "/api/status/{call_id}",
                "history": "/api/history",
            },
        }

    @app.get("/api/conversations")
    def get_conversations() -> JSONResponse:
        """Get list of demo conversations."""
        try:
            conversations = [
                {"id": conv["id"], "title": conv["title"], "segments": len(conv["segments"])}
                for conv in demo_system.demo_conversations
            ]
            return JSONResponse(conversations)
        except Exception as e:  # pragma: no cover - defensive
            logger.error(f"Error fetching conversations: {e}")
            raise HTTPException(status_code=500, detail="Failed to load conversations")

    @app.get("/api/analyze/{conversation_id}")
    def analyze_conversation(conversation_id: str) -> JSONResponse:
        """Analyze a specific demo conversation."""
        try:
            result = demo_system.analyze_single_conversation(conversation_id)

            # Mask PII in results (same as Flask)
            for seg in result.get("sentiment_analysis", []):
                seg["text"] = dashboard.mask_pii(seg["text"])
            for seg in result.get("segments", []):
                seg["text"] = dashboard.mask_pii(seg["text"])
            result["summary"] = dashboard.mask_pii(result.get("summary", ""))

            # Save to MongoDB
            if db is not None:
                try:
                    check_mongodb_connection()
                    db["analyses"].insert_one(
                        {
                            "conversation_id": conversation_id,
                            "result": result,
                            "timestamp": datetime.now(),
                        }
                    )
                except Exception as e:
                    logger.warning(f"Failed to save to MongoDB (non-critical): {e}")
            return JSONResponse(result)
        except ValueError as e:
            logger.error(f"Conversation {conversation_id} not found: {e}")
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:  # pragma: no cover - defensive
            logger.error(f"Analysis failed for {conversation_id}: {e}")
            raise HTTPException(status_code=500, detail="Analysis failed")

    @app.get("/api/analyze-all")
    def analyze_all() -> JSONResponse:
        """Analyze all demo conversations."""
        try:
            results = demo_system.batch_analyze_all()
            # Mask PII
            for result in results:
                for seg in result.get("sentiment_analysis", []):
                    seg["text"] = dashboard.mask_pii(seg["text"])
                for seg in result.get("segments", []):
                    seg["text"] = dashboard.mask_pii(seg["text"])
                result["summary"] = dashboard.mask_pii(result.get("summary", ""))

            if db is not None:
                try:
                    check_mongodb_connection()
                    db["batch_analyses"].insert_one(
                        {
                            "results": results,
                            "timestamp": datetime.now(),
                        }
                    )
                except Exception as e:
                    logger.warning(f"Failed to save to MongoDB (non-critical): {e}")
            return JSONResponse(results)
        except Exception as e:  # pragma: no cover - defensive
            logger.error(f"Batch analysis failed: {e}")
            raise HTTPException(status_code=500, detail="Batch analysis failed")

    @app.get("/api/dashboard/{conversation_id}")
    def get_dashboard(conversation_id: str) -> FileResponse:
        """Generate and serve dashboard HTML for a conversation."""
        try:
            result = demo_system.analyze_single_conversation(conversation_id)
            dashboard_html = demo_system.create_dashboard(result)
            if not os.path.exists(dashboard_html):
                raise HTTPException(status_code=500, detail="Dashboard file not found")
            return FileResponse(
                dashboard_html,
                media_type="text/html",
                filename=os.path.basename(dashboard_html),
            )
        except HTTPException:
            raise
        except Exception as e:  # pragma: no cover - defensive
            logger.error(f"Dashboard generation failed for {conversation_id}: {e}")
            raise HTTPException(status_code=500, detail="Dashboard generation failed")

    @app.get("/api/insights")
    def get_insights() -> JSONResponse:
        """Get agent insights from all conversations."""
        try:
            results = demo_system.batch_analyze_all()
            insights = analyzer.get_agent_insights(results)

            if db is not None:
                try:
                    check_mongodb_connection()
                    db["insights"].insert_one(
                        {
                            "insights": insights,
                            "timestamp": datetime.now(),
                        }
                    )
                except Exception as e:
                    logger.warning(f"Failed to save to MongoDB (non-critical): {e}")
            return JSONResponse(insights)
        except Exception as e:  # pragma: no cover - defensive
            logger.error(f"Insights generation failed: {e}")
            raise HTTPException(status_code=500, detail="Insights generation failed")

    @app.post("/api/upload")
    async def upload_audio(
        file: UploadFile = File(None),
        audio: UploadFile = File(None)
    ) -> JSONResponse:
        """Upload audio file (async - only saves file, analysis triggered separately)."""
        try:
            # Accept both 'file' and 'audio' field names for compatibility
            upload_file = file if file else audio
            
            if not upload_file or not upload_file.filename:
                raise HTTPException(status_code=400, detail="No file selected")
            
            if not allowed_file(upload_file.filename):
                raise HTTPException(
                    status_code=400,
                    detail="Invalid file type. Use .wav, .mp3, or .m4a",
                )

            # Read file content to check size
            content = await upload_file.read()
            file_size = len(content)
            
            # Validate file size (100MB limit)
            max_size = 100 * 1024 * 1024  # 100MB
            if file_size > max_size:
                raise HTTPException(
                    status_code=400,
                    detail=f"File size ({file_size / (1024*1024):.2f}MB) exceeds maximum allowed size (100MB)",
                )
            
            if file_size == 0:
                raise HTTPException(
                    status_code=400,
                    detail="File is empty",
                )

            # Save file
            os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
            filename = os.path.basename(upload_file.filename)
            # Ensure unique filename to avoid overwrites
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            base_name, ext = os.path.splitext(filename)
            unique_filename = f"{base_name}_{timestamp}{ext}"
            audio_path = os.path.join(Config.UPLOAD_FOLDER, unique_filename)

            with open(audio_path, "wb") as f:
                f.write(content)

            # Generate call_id
            call_id = f"upload_{timestamp}"

            # Create MongoDB document with status 'pending' (analysis will be triggered separately)
            # Handle MongoDB connection gracefully - allow upload even if DB is unavailable
            if db is not None:
                try:
                    check_mongodb_connection()
                    call_doc = {
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
                    }
                    db.calls.insert_one(call_doc)
                except Exception as db_error:
                    logger.warning(f"MongoDB save failed (non-critical): {db_error}. File uploaded but not saved to database.")
            else:
                logger.warning("MongoDB not available. File uploaded but not saved to database.")

            return JSONResponse(
                {
                    "status": "success",
                    "message": "File uploaded successfully. Use /api/analyze to start analysis.",
                    "call_id": call_id,
                    "filename": filename,
                    "size": file_size,
                }
            )
        except HTTPException:
            raise
        except Exception as e:  # pragma: no cover - defensive
            logger.error(f"Audio upload failed: {e}")
            raise HTTPException(status_code=500, detail=f"Audio upload failed: {str(e)}")

    @app.get("/api/export/{call_id}")
    def export_results(call_id: str) -> FileResponse:
        """Export analysis results as JSON file."""
        try:
            check_mongodb_connection()
            call_doc = db.calls.find_one({"call_id": call_id})
            if not call_doc:
                # Fallback to demo conversation if not found in calls collection
                try:
                    result = demo_system.analyze_single_conversation(call_id)
                    # Mask PII
                    for seg in result.get("sentiment_analysis", []):
                        seg["text"] = dashboard.mask_pii(seg["text"])
                    for seg in result.get("segments", []):
                        seg["text"] = dashboard.mask_pii(seg["text"])
                    result["summary"] = dashboard.mask_pii(result.get("summary", ""))
                    
                    export_data = {
                        "conversation_id": call_id,
                        "timestamp": datetime.now().isoformat(),
                        "analysis": result,
                    }
                except ValueError:
                    raise HTTPException(status_code=404, detail="Call not found")
            else:
                # Export from real call data
                result = call_doc.get("result", {})
                export_data = {
                    "call_id": call_id,
                    "filename": call_doc.get("filename", "unknown"),
                    "timestamp": call_doc.get("timestamp", datetime.now()).isoformat() if isinstance(call_doc.get("timestamp"), datetime) else str(call_doc.get("timestamp", datetime.now())),
                    "status": call_doc.get("status", "unknown"),
                    "duration": call_doc.get("duration", 0),
                    "participants": call_doc.get("participants", 0),
                    "avg_sentiment": call_doc.get("avg_sentiment", 0),
                    "sale_probability": call_doc.get("sale_probability", 0),
                    "sentiment_scores": call_doc.get("sentiment_scores", []),
                    "emotions": call_doc.get("emotions", {}),
                    "key_phrases": call_doc.get("key_phrases", {"positive": [], "negative": []}),
                    "analysis": result,
                }

            output_dir = "output"
            os.makedirs(output_dir, exist_ok=True)
            export_path = os.path.join(output_dir, f"{call_id}_export.json")

            with open(export_path, "w", encoding="utf-8") as f:
                json.dump(export_data, f, indent=2, default=str)

            if db is not None:
                try:
                    check_mongodb_connection()
                    db["exports"].insert_one(
                        {
                            "call_id": call_id,
                            "export_path": export_path,
                            "data": export_data,
                            "timestamp": datetime.now(),
                        }
                    )
                except Exception as e:
                    logger.warning(f"Failed to save to MongoDB (non-critical): {e}")

            return FileResponse(
                export_path,
                media_type="application/json",
                filename=os.path.basename(export_path),
            )
        except HTTPException:
            raise
        except Exception as e:  # pragma: no cover - defensive
            logger.error(f"Export failed for {call_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")

    @app.get("/api/export/{call_id}/pdf")
    def export_pdf(call_id: str) -> FileResponse:
        """Export analysis results as PDF report."""
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch
            from reportlab.lib import colors
        except ImportError:  # pragma: no cover - optional dependency
            logger.error("ReportLab not installed. Install with: uv pip install reportlab")
            raise HTTPException(status_code=500, detail="PDF export requires reportlab package")

        try:
            check_mongodb_connection()
            call_doc = db.calls.find_one({"call_id": call_id})
            if not call_doc:
                raise HTTPException(status_code=404, detail="Call not found")

            export_dir = "exports"
            os.makedirs(export_dir, exist_ok=True)
            pdf_path = os.path.join(export_dir, f"{call_id}_report.pdf")

            doc = SimpleDocTemplate(pdf_path, pagesize=letter)
            styles = getSampleStyleSheet()
            story: List[Any] = []

            title_style = ParagraphStyle(
                "CustomTitle",
                parent=styles["Heading1"],
                fontSize=18,
                spaceAfter=30,
                alignment=1,
            )
            story.append(Paragraph("Call Analysis Report", title_style))
            story.append(Spacer(1, 12))

            # Call info
            story.append(Paragraph("Call Information", styles["Heading2"]))
            call_info = [
                ["Call ID:", call_id],
                ["Filename:", call_doc.get("filename", "N/A")],
                [
                    "Date:",
                    call_doc.get("timestamp", "N/A").strftime("%Y-%m-%d %H:%M:%S")
                    if call_doc.get("timestamp")
                    else "N/A",
                ],
                ["Duration:", f"{call_doc.get('duration', 0)} seconds"],
                ["Participants:", str(call_doc.get("participants", 2))],
            ]

            call_table = Table(call_info, colWidths=[2 * inch, 4 * inch])
            call_table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (0, -1), colors.grey),
                        ("TEXTCOLOR", (0, 0), (-1, -1), colors.black),
                        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                        ("FONTSIZE", (0, 0), (-1, -1), 10),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
                        ("BACKGROUND", (1, 0), (1, -1), colors.beige),
                        ("GRID", (0, 0), (-1, -1), 1, colors.black),
                    ]
                )
            )
            story.append(call_table)
            story.append(Spacer(1, 20))

            # Summary
            story.append(Paragraph("Analysis Results", styles["Heading2"]))
            story.append(Paragraph("Summary", styles["Heading3"]))
            summary_data = [
                ["Average Sentiment:", f"{call_doc.get('avg_sentiment', 0):.2f}"],
                ["Sale Probability:", f"{call_doc.get('sale_probability', 0) * 100:.1f}%"],
                [
                    "Overall Sentiment:",
                    "Positive"
                    if call_doc.get("avg_sentiment", 0) > 0.6
                    else "Neutral"
                    if call_doc.get("avg_sentiment", 0) > 0.4
                    else "Negative",
                ],
            ]

            summary_table = Table(summary_data, colWidths=[2 * inch, 4 * inch])
            summary_table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (0, -1), colors.lightblue),
                        ("TEXTCOLOR", (0, 0), (-1, -1), colors.black),
                        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                        ("FONTSIZE", (0, 0), (-1, -1), 10),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
                        ("GRID", (0, 0), (-1, -1), 1, colors.black),
                    ]
                )
            )
            story.append(summary_table)
            story.append(Spacer(1, 20))

            # Emotion distribution
            story.append(Paragraph("Emotion Distribution", styles["Heading3"]))
            emotions = call_doc.get("emotions", {})
            if emotions:
                emotion_data = [["Emotion", "Percentage"]]
                for emotion, value in emotions.items():
                    emotion_data.append([emotion.title(), f"{value * 100:.1f}%"])

                emotion_table = Table(emotion_data, colWidths=[2 * inch, 2 * inch])
                emotion_table.setStyle(
                    TableStyle(
                        [
                            ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                            ("FONTSIZE", (0, 0), (-1, 0), 12),
                            ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                            ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                            ("GRID", (0, 0), (-1, -1), 1, colors.black),
                        ]
                    )
                )
                story.append(emotion_table)

            story.append(Spacer(1, 20))

            # Key phrases
            story.append(Paragraph("Key Phrases", styles["Heading3"]))
            key_phrases = call_doc.get("key_phrases", {"positive": [], "negative": []})

            if key_phrases.get("positive"):
                story.append(Paragraph("Positive Phrases:", styles["Heading4"]))
                for phrase in key_phrases["positive"][:5]:
                    story.append(Paragraph(f"• {phrase}", styles["Normal"]))

            if key_phrases.get("negative"):
                story.append(Paragraph("Negative Phrases:", styles["Heading4"]))
                for phrase in key_phrases["negative"][:5]:
                    story.append(Paragraph(f"• {phrase}", styles["Normal"]))

            doc.build(story)

            return FileResponse(
                pdf_path,
                media_type="application/pdf",
                filename=os.path.basename(pdf_path),
            )
        except HTTPException:
            raise
        except Exception as e:  # pragma: no cover - defensive
            logger.error(f"PDF export failed for {call_id}: {e}")
            raise HTTPException(status_code=500, detail="PDF export failed")

    @app.get("/api/export/{call_id}/csv")
    def export_csv(call_id: str) -> Response:
        """Export analysis results as CSV data."""
        try:
            import csv
            import io
        except ImportError:  # pragma: no cover
            raise HTTPException(status_code=500, detail="CSV export requires csv and io modules")

        try:
            check_mongodb_connection()
            call_doc = db.calls.find_one({"call_id": call_id})
            if not call_doc:
                raise HTTPException(status_code=404, detail="Call not found")

            output = io.StringIO()
            writer = csv.writer(output)

            writer.writerow(["Call Analysis Data"])
            writer.writerow(["Call ID", call_id])
            writer.writerow(["Filename", call_doc.get("filename", "N/A")])
            writer.writerow(
                [
                    "Date",
                    call_doc.get("timestamp", "N/A").strftime("%Y-%m-%d %H:%M:%S")
                    if call_doc.get("timestamp")
                    else "N/A",
                ]
            )
            writer.writerow(["Duration (seconds)", call_doc.get("duration", 0)])
            writer.writerow(["Participants", call_doc.get("participants", 2)])
            writer.writerow([])

            writer.writerow(["SUMMARY"])
            writer.writerow(["Average Sentiment", f"{call_doc.get('avg_sentiment', 0):.2f}"])
            writer.writerow(["Sale Probability (%)", f"{call_doc.get('sale_probability', 0) * 100:.1f}"])
            writer.writerow(
                [
                    "Overall Sentiment",
                    "Positive"
                    if call_doc.get("avg_sentiment", 0) > 0.6
                    else "Neutral"
                    if call_doc.get("avg_sentiment", 0) > 0.4
                    else "Negative",
                ]
            )
            writer.writerow([])

            writer.writerow(["EMOTION DISTRIBUTION"])
            writer.writerow(["Emotion", "Percentage"])
            emotions = call_doc.get("emotions", {})
            for emotion, value in emotions.items():
                writer.writerow([emotion.title(), f"{value * 100:.1f}%"])
            writer.writerow([])

            writer.writerow(["SENTIMENT SCORES OVER TIME"])
            writer.writerow(["Timestamp", "Sentiment Score"])
            for i, score_data in enumerate(call_doc.get("sentiment_scores", [])):
                timestamp = score_data.get("timestamp", i)
                score = score_data.get("score", 0)
                writer.writerow([f"T{timestamp}", f"{score:.2f}"])
            writer.writerow([])

            writer.writerow(["KEY PHRASES"])
            key_phrases = call_doc.get("key_phrases", {"positive": [], "negative": []})

            writer.writerow(["Positive Phrases"])
            for phrase in key_phrases.get("positive", []):
                writer.writerow([phrase])

            writer.writerow(["Negative Phrases"])
            for phrase in key_phrases.get("negative", []):
                writer.writerow([phrase])

            csv_content = output.getvalue()
            output.close()

            return Response(
                content=csv_content,
                media_type="text/csv",
                headers={"Content-Disposition": f'attachment; filename="{call_id}_analysis.csv"'},
            )
        except HTTPException:
            raise
        except Exception as e:  # pragma: no cover
            logger.error(f"CSV export failed for {call_id}: {e}")
            raise HTTPException(status_code=500, detail="CSV export failed")

    def _run_analysis_background(call_id: str, audio_path: str, filename: str):
        """Run analysis in background thread with progress updates."""
        try:
            # Update progress: Transcription (0-20%)
            db.calls.update_one(
                {"call_id": call_id},
                {"$set": {"status": "processing", "progress": 10, "updated_at": datetime.now()}},
            )
            progress_log.info(f"[{call_id}] Step 1/5: Starting transcription...")
            transcription = demo_system.audio_processor.transcribe_audio(audio_path, call_id)

            db.calls.update_one(
                {"call_id": call_id},
                {"$set": {"progress": 20, "updated_at": datetime.now()}},
            )
            progress_log.info(f"[{call_id}] Step 2/5: Starting diarization...")
            segments = demo_system.audio_processor.perform_speaker_diarization(audio_path, call_id)

            db.calls.update_one(
                {"call_id": call_id},
                {"$set": {"progress": 40, "updated_at": datetime.now()}},
            )
            progress_log.info(f"[{call_id}] Step 3/5: Processing segments...")
            processed_segments = demo_system.text_processor.segment_conversation(
                transcription.get("text", ""),
                segments,
                call_id,
                transcription_segments=transcription.get("segments", []),
            )
            
            db.calls.update_one(
                {"call_id": call_id},
                {"$set": {"progress": 60, "updated_at": datetime.now()}},
            )
            progress_log.info(f"[{call_id}] Step 4/5: Extracting features...")
            audio_features = demo_system.audio_processor.extract_audio_features(audio_path)

            db.calls.update_one(
                {"call_id": call_id},
                {"$set": {"progress": 80, "updated_at": datetime.now()}},
            )
            progress_log.info(f"[{call_id}] Step 5/5: Running ML analysis...")
            result = demo_system.analyzer.analyze_conversation(
                audio_path=audio_path,
                segments=processed_segments,
                audio_features=audio_features,
                call_id=call_id,
            )

            # Mask PII
            for seg in result.get("sentiment_analysis", []):
                seg["text"] = dashboard.mask_pii(seg["text"])
            result["summary"] = dashboard.mask_pii(result.get("summary", ""))

            # Transform sentiment_scores to match frontend format
            sentiment_scores_formatted = []
            sentiment_analysis = result.get("sentiment_analysis", [])
            for seg in sentiment_analysis:
                sentiment_scores_formatted.append({
                    "timestamp": seg.get("start_time", 0),
                    "score": seg.get("score", 0),
                    "label": seg.get("sentiment", "neutral")
                })
            
            # Transform key_phrases from sentiment_analysis segments
            key_phrases_formatted = {"positive": [], "negative": []}
            all_phrases = []
            for seg in sentiment_analysis:
                # Extract phrases from each segment
                seg_phrases = seg.get("key_phrases", [])
                if isinstance(seg_phrases, list):
                    all_phrases.extend(seg_phrases)
                # Also check conversation_key_phrases
                conv_phrases = seg.get("conversation_key_phrases", [])
                if isinstance(conv_phrases, list):
                    all_phrases.extend(conv_phrases)
            
            # Sort and categorize phrases
            for phrase in all_phrases:
                if isinstance(phrase, dict):
                    score = phrase.get("sentiment_score", phrase.get("score", 0))
                    phrase_text = phrase.get("phrase", "")
                    if score > 0 and phrase_text:
                        key_phrases_formatted["positive"].append({
                            "phrase": phrase_text,
                            "score": abs(score)
                        })
                    elif score < 0 and phrase_text:
                        key_phrases_formatted["negative"].append({
                            "phrase": phrase_text,
                            "score": abs(score)
                        })
            
            # Remove duplicates and limit to top 10 each
            def dedupe_phrases(phrases):
                seen = set()
                unique = []
                for p in phrases:
                    key = p["phrase"].lower()
                    if key not in seen:
                        seen.add(key)
                        unique.append(p)
                return sorted(unique, key=lambda x: x["score"], reverse=True)[:10]
            
            key_phrases_formatted["positive"] = dedupe_phrases(key_phrases_formatted["positive"])
            key_phrases_formatted["negative"] = dedupe_phrases(key_phrases_formatted["negative"])
            
            # Transform emotions from emotion_analysis
            emotions_formatted = {"neutral": 0.0, "happiness": 0.0, "anger": 0.0, "sadness": 0.0, "frustration": 0.0}
            emotion_analysis = result.get("emotion_analysis", [])
            if emotion_analysis:
                # Aggregate emotion probabilities across all segments
                emotion_probs = {"neutral": [], "happiness": [], "anger": [], "sadness": [], "frustration": []}
                for emo_result in emotion_analysis:
                    probs = emo_result.get("probabilities", {})
                    if isinstance(probs, dict):
                        for emotion in emotion_probs.keys():
                            if emotion in probs:
                                emotion_probs[emotion].append(probs[emotion])
                
                # Average probabilities
                for emotion in emotions_formatted.keys():
                    if emotion_probs[emotion]:
                        emotions_formatted[emotion] = float(np.mean(emotion_probs[emotion]))
            
            # Get sale probability and avg sentiment
            sale_prediction = result.get("sale_prediction", {})
            sale_probability = sale_prediction.get("sale_probability", 0) if isinstance(sale_prediction, dict) else 0
            
            conversation_metrics = result.get("conversation_metrics", {})
            avg_sentiment = conversation_metrics.get("avg_sentiment", np.mean([s["score"] for s in sentiment_analysis]) if sentiment_analysis else 0)

            # Count unique participants
            participants = len(set([s.get("speaker", "Unknown") for s in processed_segments]))

            # Use actual audio file duration so reported length matches the real call (e.g. 3m23s), not just last segment end
            try:
                import librosa
                actual_duration = float(librosa.get_duration(path=audio_path))
            except Exception:
                actual_duration = float(result.get("duration", 0))
            result["duration"] = actual_duration

            # Update MongoDB with results
            db.calls.update_one(
                {"call_id": call_id},
                {
                    "$set": {
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
                    }
                },
            )

            # Also save to analyses collection for backward compatibility
            db.analyses.insert_one(
                {
                    "conversation_id": call_id,
                    "result": result,
                    "audio_path": audio_path,
                    "timestamp": datetime.now(),
                }
            )

            progress_log.info(f"[{call_id}] Analysis completed successfully")
        except Exception as e:
            progress_log.error(f"[{call_id}] Analysis failed: {e}")
            logger.exception(f"[{call_id}] Analysis failed")
            db.calls.update_one(
                {"call_id": call_id},
                {
                    "$set": {
                        "status": "failed",
                        "progress": 0,
                        "error": str(e),
                        "updated_at": datetime.now(),
                    }
                },
            )

    @app.post("/api/analyze")
    async def start_analysis(request: Request) -> JSONResponse:
        """Start analysis for uploaded file (runs in background). Accepts JSON body with call_id or callId."""
        try:
            try:
                body = await request.json()
            except Exception as e:
                logger.warning(f"/api/analyze received non-JSON body: {e}")
                raise HTTPException(
                    status_code=422,
                    detail="Request body must be JSON with 'call_id' or 'callId'.",
                ) from e
            if not isinstance(body, dict):
                raise HTTPException(status_code=422, detail="Request body must be a JSON object.")
            call_id = body.get("call_id") or body.get("callId")
            if not call_id or not isinstance(call_id, str):
                logger.warning(f"/api/analyze missing call_id/callId in body: {body}")
                raise HTTPException(
                    status_code=422,
                    detail="Request body must include 'call_id' or 'callId' (value from upload response).",
                )

            check_mongodb_connection()
            call_doc = db.calls.find_one({"call_id": call_id})
            if not call_doc:
                raise HTTPException(status_code=404, detail="Call not found")

            # Check if already processing or completed
            current_status = call_doc.get("status", "unknown")
            if current_status == "processing":
                return JSONResponse(
                    {
                        "message": "Analysis already in progress",
                        "call_id": call_id,
                        "status": "processing",
                    }
                )
            if current_status == "completed":
                return JSONResponse(
                    {
                        "message": "Analysis already completed",
                        "call_id": call_id,
                        "status": "completed",
                    }
                )

            # Update status to processing
            db.calls.update_one(
                {"call_id": call_id},
                {"$set": {"status": "processing", "progress": 0, "updated_at": datetime.now()}},
            )

            # Get audio path from call document
            audio_path = call_doc.get("audio_path")
            if not audio_path or not os.path.exists(audio_path):
                raise HTTPException(status_code=404, detail="Audio file not found")

            filename = call_doc.get("filename", "unknown")

            # Start background analysis thread
            analysis_thread = threading.Thread(
                target=lambda: _run_analysis_background(
                    call_id, audio_path, filename
                ),
                daemon=True,
            )
            analysis_thread.start()

            return JSONResponse(
                {
                    "message": "Analysis started",
                    "call_id": call_id,
                    "status": "processing",
                }
            )
        except HTTPException:
            raise
        except Exception as e:  # pragma: no cover
            logger.error(f"Analysis start failed: {e}")
            raise HTTPException(status_code=500, detail=f"Analysis start failed: {str(e)}")

    @app.get("/api/results/{call_id}")
    def get_results(call_id: str) -> JSONResponse:
        """Get analysis results for a call."""
        try:
            check_mongodb_connection()
            call_doc = db.calls.find_one({"call_id": call_id})
            if not call_doc:
                raise HTTPException(status_code=404, detail="Call not found")

            # Ensure sentiment_scores format matches frontend expectations
            sentiment_scores = call_doc.get("sentiment_scores", [])
            if sentiment_scores and isinstance(sentiment_scores, list):
                # Verify each item has required fields
                formatted_scores = []
                for score in sentiment_scores:
                    if isinstance(score, dict):
                        formatted_scores.append({
                            "timestamp": score.get("timestamp", score.get("start_time", 0)),
                            "score": score.get("score", 0),
                            "label": score.get("label", score.get("sentiment", "neutral"))
                        })
                sentiment_scores = formatted_scores

            # Ensure emotions format matches frontend expectations
            emotions = call_doc.get("emotions", {})
            if not isinstance(emotions, dict):
                emotions = {}
            # Ensure all required emotion keys exist
            required_emotions = ["neutral", "happiness", "anger", "sadness", "frustration"]
            for emotion in required_emotions:
                if emotion not in emotions:
                    emotions[emotion] = 0.0

            # Ensure key_phrases format matches frontend expectations
            key_phrases = call_doc.get("key_phrases", {"positive": [], "negative": []})
            if not isinstance(key_phrases, dict):
                key_phrases = {"positive": [], "negative": []}
            if "positive" not in key_phrases:
                key_phrases["positive"] = []
            if "negative" not in key_phrases:
                key_phrases["negative"] = []

            return JSONResponse(
                {
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
                }
            )
        except HTTPException:
            raise
        except Exception as e:  # pragma: no cover
            logger.error(f"Results retrieval failed for {call_id}: {e}")
            raise HTTPException(status_code=500, detail="Results retrieval failed")

    @app.get("/api/status/{call_id}")
    def get_status(call_id: str) -> JSONResponse:
        """Get analysis status for a call."""
        try:
            check_mongodb_connection()
            call_doc = db.calls.find_one({"call_id": call_id})
            if not call_doc:
                raise HTTPException(status_code=404, detail="Call not found")

            return JSONResponse(
                {
                    "call_id": call_id,
                    "status": call_doc.get("status", "unknown"),
                    "progress": call_doc.get("progress", 0),
                }
            )
        except HTTPException:
            raise
        except Exception as e:  # pragma: no cover
            logger.error(f"Status check failed for {call_id}: {e}")
            raise HTTPException(status_code=500, detail="Status check failed")

    @app.get("/api/history")
    def get_history() -> JSONResponse:
        """Get call history."""
        try:
            check_mongodb_connection()
            calls = list(
                db.calls.find(
                    {},
                    {
                        "call_id": 1,
                        "filename": 1,
                        "timestamp": 1,
                        "duration": 1,
                        "avg_sentiment": 1,
                        "sale_probability": 1,
                        "participants": 1,
                        "status": 1,
                    },
                ).sort("timestamp", -1)
            )

            # Format response to match CallHistoryItem interface
            formatted_calls = []
            for call in calls:
                # Convert timestamp to ISO string if it's a datetime object
                timestamp = call.get("timestamp")
                if isinstance(timestamp, datetime):
                    timestamp = timestamp.isoformat()
                elif timestamp is None:
                    timestamp = datetime.now().isoformat()
                else:
                    timestamp = str(timestamp)

                formatted_calls.append({
                    "call_id": call.get("call_id", ""),
                    "filename": call.get("filename", "unknown"),
                    "timestamp": timestamp,
                    "duration": float(call.get("duration", 0)),
                    "avg_sentiment": float(call.get("avg_sentiment", 0)),
                    "sale_probability": float(call.get("sale_probability", 0)),
                    "participants": int(call.get("participants", 0)),
                    "status": call.get("status", "unknown"),
                })

            return JSONResponse(formatted_calls)
        except Exception as e:  # pragma: no cover
            logger.error(f"History retrieval failed: {e}")
            raise HTTPException(status_code=500, detail="History retrieval failed")

    @app.get("/health")
    def health_check() -> JSONResponse:
        """Health check endpoint."""
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
                logger.error(f"MongoDB health check failed: {e}")
                health_status["mongodb"] = "error"
                health_status["mongodb_error"] = str(e)
        else:
            health_status["status"] = "degraded"
            health_status["mongodb"] = "not_configured"
        
        status_code = status.HTTP_200_OK if health_status["status"] == "healthy" else status.HTTP_503_SERVICE_UNAVAILABLE
        return JSONResponse(health_status, status_code=status_code)

    return app


app = create_app()


