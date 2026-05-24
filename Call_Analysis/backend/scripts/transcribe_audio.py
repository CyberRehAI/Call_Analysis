#!/usr/bin/env python3
"""
Transcribe audio file using Whisper
Simple script to extract text from audio file
"""

import sys
import os
from pathlib import Path

# Add src to path (now we're in backend/scripts/, so go up one level to backend/, then to src)
script_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(script_dir)
src_dir = os.path.join(backend_dir, 'src')
sys.path.insert(0, src_dir)

from dotenv import load_dotenv  # type: ignore[import]
from call_analysis.progress_logger import setup_project_logging, get_progress_logger
from call_analysis.preprocessing import AudioProcessor

# Load environment variables
load_dotenv()


def transcribe_audio_file(audio_path: str, model_size: str = "base"):
    """
    Transcribe audio file to text using Whisper.
    """
    setup_project_logging()
    log = get_progress_logger()

    log.info("=" * 60)
    log.info("WHISPER AUDIO TRANSCRIPTION")
    log.info("=" * 60)

    if not os.path.exists(audio_path):
        log.error(f"Audio file not found: {audio_path}")
        return

    file_size = os.path.getsize(audio_path) / (1024 * 1024)
    log.info(f"Audio file: {audio_path} ({file_size:.2f} MB)")

    log.info(f"Step 1/4: Initializing Whisper (model: {model_size})...")
    try:
        hf_token = os.getenv('HF_TOKEN', None)
        processor = AudioProcessor(model_size=model_size, hf_token=hf_token)
        log.info("Step 1/4: Whisper initialized.")
    except Exception as e:
        log.error(f"Failed to initialize Whisper: {e}")
        log.info("Troubleshooting: pip install openai-whisper; try model_size='tiny' or 'base'.")
        return

    log.info("Step 2/4: Starting transcription (may take several minutes)...")
    try:
        call_id = "transcription_001"
        transcription = processor.transcribe_audio(audio_path, call_id)
        text = transcription.get('text', '')
        language = transcription.get('language', 'unknown')
        duration = transcription.get('duration', 0)
        segments = transcription.get('segments', [])

        log.info(f"Step 2/4: Transcription completed. Language: {language}, Duration: {duration:.2f}s, Segments: {len(segments)}, Chars: {len(text)}.")
        log.info("Step 3/4: Preparing output...")

        # Save to file
        script_dir = os.path.dirname(os.path.abspath(__file__))
        backend_dir = os.path.dirname(script_dir)
        project_root = os.path.dirname(backend_dir)
        output_dir = os.path.join(project_root, 'output')
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "transcription_output.txt")

        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("=" * 60 + "\n")
                f.write("WHISPER TRANSCRIPTION OUTPUT\n")
                f.write("=" * 60 + "\n\n")
                f.write(f"Audio file: {audio_path}\n")
                f.write(f"Language: {language}\n")
                f.write(f"Duration: {duration:.2f} seconds\n")
                f.write(f"Number of segments: {len(segments)}\n")
                f.write(f"Text length: {len(text)} characters\n\n")
                f.write("=" * 60 + "\n")
                f.write("FULL TRANSCRIPT\n")
                f.write("=" * 60 + "\n\n")
                f.write(text)
                f.write("\n\n")
                f.write("=" * 60 + "\n")
                f.write("SEGMENTS WITH TIMESTAMPS\n")
                f.write("=" * 60 + "\n\n")
                for i, seg in enumerate(segments, 1):
                    start = seg.get('start', 0)
                    end = seg.get('end', 0)
                    seg_text = seg.get('text', '').strip()
                    f.write(f"{i}. [{start:.1f}s - {end:.1f}s]: {seg_text}\n")
            log.info(f"Step 3/4: Transcript saved: {output_file}")
        except Exception as e:
            log.warning(f"Could not save transcript to file: {e}")

        log.info("Step 4/4: Done.")
        if text:
            preview = (text[:200] + "...") if len(text) > 200 else text
            log.info(f"Transcript preview: {preview}")
        else:
            log.warning("No speech detected in audio file.")

    except Exception as e:
        log.error(f"Transcription failed: {e}")
        log.info("Common issues: unsupported format (use WAV/MP3/M4A), corrupted file, or try model_size='tiny'.")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    # Get project root (go up from backend/scripts/ to project root)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    backend_dir = os.path.dirname(script_dir)
    project_root = os.path.dirname(backend_dir)

    # Determine audio file:
    # 1) If a CLI argument is provided, use that (with smart resolution)
    # 2) Otherwise, fall back to the default real_audio.wav in the project root
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
        # If relative path, try resolving relative to project root first
        if not os.path.isabs(audio_file):
            test_path = os.path.join(project_root, audio_file)
            if os.path.exists(test_path):
                audio_file = test_path
            # Otherwise, leave as-is (relative to current working directory)
    else:
        # Default to real_audio.wav in project root
        audio_file = os.path.join(project_root, "real_audio.wav")
    
    # Optional: specify model size as second argument
    model_size = "base"
    if len(sys.argv) > 2:
        model_size = sys.argv[2]
    
    transcribe_audio_file(audio_file, model_size)

