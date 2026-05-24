#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone Diarization Script for Testing
Runs diarization on already transcribed audio without re-running transcription.
This allows you to test different diarization parameters quickly.
"""

import sys
import os
import json
import argparse
from pathlib import Path
from datetime import datetime

# Fix Windows console encoding for emojis
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from dotenv import load_dotenv
from call_analysis.progress_logger import setup_project_logging, get_progress_logger
from call_analysis.preprocessing import AudioProcessor, TextProcessor
from config import Config

# Load environment variables
load_dotenv()


def run_diarization_only(
    audio_path: str,
    transcription_json: str,
    output_dir: str = None,
    max_speakers: int = None,
    min_speakers: int = None,
    num_speakers: int = None,
    clustering_threshold: float = 0.3,
    min_segment_duration: float = 1.0,
    speaker_merge_threshold: float = 0.7,
    use_whisperx_builtin: bool = False
):
    """
    Run diarization on already transcribed audio.
    
    Args:
        audio_path: Path to audio file
        transcription_json: Path to existing transcription JSON file
        output_dir: Directory to save results (default: output/)
        max_speakers: Maximum number of speakers (None = auto). Pyannote and Resemblyzer.
        min_speakers: Minimum number of speakers (None = no constraint). Pyannote only.
        num_speakers: Exact number of speakers (None = auto). Pyannote only.
        clustering_threshold: Distance threshold for speaker clustering (0.0-1.0, lower = more clusters)
        min_segment_duration: Minimum segment duration in seconds
        speaker_merge_threshold: Similarity threshold for merging speakers (0.0-1.0)
    """
    setup_project_logging()
    log = get_progress_logger()

    log.info("=" * 70)
    log.info("DIARIZATION-ONLY TESTING SCRIPT")
    log.info("=" * 70)

    # Resolve paths
    audio_path = os.path.abspath(audio_path)
    transcription_json = os.path.abspath(transcription_json)

    if output_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        output_dir = os.path.join(project_root, "output")

    log.info(f"Audio file: {audio_path}")
    log.info(f"Transcription file: {transcription_json}")
    log.info(f"Output directory: {output_dir}")

    if not os.path.exists(audio_path):
        log.error(f"Audio file not found: {audio_path}")
        sys.exit(1)

    if not os.path.exists(transcription_json):
        log.error(f"Transcription file not found: {transcription_json}")
        sys.exit(1)

    # Load existing transcription (supports both JSON and TXT)
    log.info("Loading existing transcription...")
    transcription = None
    transcription_segments = None
    
    if transcription_json.endswith('.json'):
        try:
            with open(transcription_json, 'r', encoding='utf-8') as f:
                transcription = json.load(f)
            transcription_segments = transcription.get('segments', [])
            log.info(f"Transcription JSON loaded: {len(transcription.get('text', ''))} chars, {len(transcription_segments)} segments (text-to-segment mapping enabled).")
        except Exception as e:
            log.error(f"Error loading transcription JSON: {e}")
            sys.exit(1)
    elif transcription_json.endswith('.txt'):
        try:
            with open(transcription_json, 'r', encoding='utf-8') as f:
                text_content = f.read()
            transcription = {'text': text_content, 'segments': []}
            transcription_segments = []
            log.info(f"Transcription TXT loaded: {len(text_content)} chars. WARNING: no timestamps; text-to-segment mapping disabled. Use JSON for full results.")
        except Exception as e:
            log.error(f"Error loading transcription TXT: {e}")
            sys.exit(1)
    else:
        log.error("Unsupported file format. Use .json or .txt")
        sys.exit(1)

    # Extract call_id from transcription filename if possible
    transcription_filename = os.path.basename(transcription_json)
    if '_transcription.json' in transcription_filename:
        call_id = transcription_filename.replace('_transcription.json', '')
    else:
        call_id = f"diarization_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log.info(f"Call ID: {call_id}")

    os.makedirs(output_dir, exist_ok=True)

    try:
        # STEP 1: Initialize AudioProcessor
        log.info("Step 1/5: Initializing AudioProcessor...")
        if num_speakers is not None:
            log.info(f"  Num speakers (exact): {num_speakers}")
        elif min_speakers is not None or max_speakers is not None:
            log.info(f"  Min speakers: {min_speakers or 'auto'}, Max speakers: {max_speakers or 'auto'}")
        else:
            log.info("  Speaker count: Auto-detect")
        log.info(f"  Clustering threshold: {clustering_threshold}, min segment: {min_segment_duration}s, merge threshold: {speaker_merge_threshold}")

        if use_whisperx_builtin and not Config.HF_TOKEN:
            log.error("HF_TOKEN is required for WhisperX 3.x built-in diarization. Set HF_TOKEN in .env")
            sys.exit(1)

        audio_processor = AudioProcessor(
            model_size=Config.WHISPER_MODEL_SIZE or "base",
            hf_token=Config.HF_TOKEN,
            max_speakers=max_speakers,
            min_speakers=min_speakers,
            num_speakers=num_speakers,
            clustering_threshold=clustering_threshold,
            min_segment_duration=min_segment_duration,
            speaker_merge_threshold=speaker_merge_threshold,
            use_whisperx_builtin_diarization=use_whisperx_builtin
        )
        
        if use_whisperx_builtin:
            log.info("  Using WhisperX 3.x built-in diarization (Pyannote.audio).")
        else:
            log.info("  Using WhisperX + Resemblyzer (faster, CPU-friendly).")
        log.info("Step 1/5: AudioProcessor initialized.")

        # STEP 2: Run Diarization
        log.info("Step 2/5: Running speaker diarization (may take several minutes)...")
        diarization_segments = audio_processor.perform_speaker_diarization(audio_path, call_id)
        unique_speakers = set(seg.get('speaker', 'Unknown') for seg in diarization_segments)
        log.info(f"Step 2/5: Diarization completed. Segments: {len(diarization_segments)}, speakers: {len(unique_speakers)}")
        for speaker in sorted(unique_speakers):
            speaker_segments = [s for s in diarization_segments if s.get('speaker') == speaker]
            total_duration = sum(s.get('end', 0) - s.get('start', 0) for s in speaker_segments)
            log.info(f"  {speaker}: {len(speaker_segments)} segments, {total_duration:.1f}s total")

        # STEP 3: Map Transcription Text to Diarization Segments
        log.info("Step 3/5: Mapping transcription text to diarization segments...")
        text_processor = TextProcessor()
        segments_with_text = 0
        if transcription_segments and len(transcription_segments) > 0:
            text_processor._assign_text_to_diarization_segments(
                diarization_segments,
                transcription_segments
            )
            segments_with_text = sum(1 for seg in diarization_segments if seg.get('text', '').strip())
            log.info(f"Step 3/5: Text mapping completed: {segments_with_text}/{len(diarization_segments)} segments have text.")
        else:
            log.warning("TXT format has no timestamps; segments will not have text. Use JSON for full functionality.")

        # STEP 4: Save Results
        log.info("Step 4/5: Saving results...")
        diarization_output_path = os.path.join(output_dir, f"{call_id}_diarization.json")
        with open(diarization_output_path, 'w', encoding='utf-8') as f:
            json.dump(diarization_segments, f, indent=2, ensure_ascii=False)
        log.info(f"Diarization saved: {diarization_output_path}")

        # Save summary
        summary = {
            'call_id': call_id,
            'audio_path': audio_path,
            'transcription_file': transcription_json,
            'transcription_format': 'json' if transcription_json.endswith('.json') else 'txt',
            'timestamp': datetime.now().isoformat(),
                'parameters': {
                    'max_speakers': max_speakers,
                    'min_speakers': min_speakers,
                    'num_speakers': num_speakers,
                    'clustering_threshold': clustering_threshold,
                    'min_segment_duration': min_segment_duration,
                    'speaker_merge_threshold': speaker_merge_threshold
                },
            'results': {
                'total_segments': len(diarization_segments),
                'unique_speakers': len(unique_speakers),
                'speakers': {speaker: len([s for s in diarization_segments if s.get('speaker') == speaker])
                            for speaker in unique_speakers},
                'segments_with_text': segments_with_text
            }
        }
        
        summary_path = os.path.join(output_dir, f"{call_id}_diarization_summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        log.info(f"Step 4/5: Summary saved: {summary_path}")

        # STEP 5: Summary
        log.info("Step 5/5: Diarization complete.")
        log.info("=" * 70)
        log.info("DIARIZATION RESULTS SUMMARY")
        log.info(f"Total segments: {len(diarization_segments)}, unique speakers: {len(unique_speakers)}, segments with text: {segments_with_text}")
        for speaker in sorted(unique_speakers):
            speaker_segments = [s for s in diarization_segments if s.get('speaker') == speaker]
            total_duration = sum(s.get('end', 0) - s.get('start', 0) for s in speaker_segments)
            avg_duration = total_duration / len(speaker_segments) if speaker_segments else 0
            log.info(f"  {speaker}: {len(speaker_segments)} segments, {total_duration:.1f}s total, avg {avg_duration:.1f}s")
        log.info("=" * 70)
        return diarization_segments, summary

    except Exception as e:
        log.error(f"Error during diarization: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description='Run diarization on already transcribed audio for testing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with JSON (RECOMMENDED - includes timestamps for text mapping)
  python run_diarization_only.py real_audio.wav output/call_20251125_153129_transcription.json
  
  # With custom parameters
  python run_diarization_only.py real_audio.wav output/call_20251125_153129_transcription.json \\
      --max-speakers 2 \\
      --clustering-threshold 0.4 \\
      --min-segment-duration 0.5
  
  # Using TXT file (NOT RECOMMENDED - no timestamps, text mapping won't work)
  python run_diarization_only.py real_audio.wav transcription_output.txt \\
      --max-speakers 2
  
  # Using WhisperX 3.x built-in diarization (more accurate, slower)
  python run_diarization_only.py real_audio.wav output/call_20251125_160302_transcription.json \\
      --use-whisperx-builtin \\
      --max-speakers 2
        """
    )
    
    parser.add_argument(
        'audio_path',
        type=str,
        help='Path to audio file (.wav, .mp3, .m4a)'
    )
    
    parser.add_argument(
        'transcription_file',
        type=str,
        help='Path to existing transcription file (.json recommended, .txt supported but limited)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory (default: output/)'
    )
    
    parser.add_argument(
        '--max-speakers',
        type=int,
        default=None,
        help='Maximum number of speakers (None = auto). Used by Pyannote and Resemblyzer.'
    )
    
    parser.add_argument(
        '--min-speakers',
        type=int,
        default=None,
        help='Minimum number of speakers (None = no constraint). Pyannote only.'
    )
    
    parser.add_argument(
        '--num-speakers',
        type=int,
        default=None,
        help='Exact number of speakers (None = auto). Pyannote only; do not use with --min-speakers/--max-speakers.'
    )
    
    parser.add_argument(
        '--clustering-threshold',
        type=float,
        default=0.3,
        help='Distance threshold for speaker clustering, 0.0-1.0, lower = more clusters (default: 0.3)'
    )
    
    parser.add_argument(
        '--min-segment-duration',
        type=float,
        default=1.0,
        help='Minimum segment duration in seconds (default: 1.0)'
    )
    
    parser.add_argument(
        '--speaker-merge-threshold',
        type=float,
        default=0.7,
        help='Similarity threshold for merging speakers, 0.0-1.0 (default: 0.7)'
    )
    
    parser.add_argument(
        '--use-whisperx-builtin',
        action='store_true',
        help='Use WhisperX 3.x built-in diarization (Pyannote.audio) instead of Resemblyzer. '
             'More accurate but slower. Requires HF_TOKEN.'
    )
    
    args = parser.parse_args()
    
    run_diarization_only(
        audio_path=args.audio_path,
        transcription_json=args.transcription_file,
        output_dir=args.output_dir,
        max_speakers=args.max_speakers,
        min_speakers=args.min_speakers,
        num_speakers=args.num_speakers,
        clustering_threshold=args.clustering_threshold,
        min_segment_duration=args.min_segment_duration,
        speaker_merge_threshold=args.speaker_merge_threshold,
        use_whisperx_builtin=args.use_whisperx_builtin
    )


if __name__ == '__main__':
    main()

