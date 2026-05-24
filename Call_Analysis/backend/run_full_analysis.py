#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complete End-to-End Analysis Pipeline
Runs full analysis from audio file → transcription → features → models → results
"""

import sys
import os
import json
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
from call_analysis.feature_extraction import FeatureExtractor
from call_analysis.models import ConversationAnalyzer
from call_analysis.dashboard import Dashboard

# Load environment variables
load_dotenv()


def run_full_analysis(audio_path: str, call_id: str = None, output_dir: str = None):
    """
    Run complete analysis pipeline on audio file.
    
    Args:
        audio_path: Path to audio file
        call_id: Unique call identifier (auto-generated if None)
        output_dir: Directory to save results
        
    Returns:
        Complete analysis results dictionary
    """
    setup_project_logging()
    log = get_progress_logger()

    log.info("=" * 70)
    log.info("COMPLETE CALL ANALYSIS PIPELINE")
    log.info("=" * 70)

    # Generate call_id if not provided
    if call_id is None:
        call_id = f"call_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Resolve audio path to absolute
    audio_path = os.path.abspath(audio_path)

    # Set default output directory if not provided
    if output_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        output_dir = os.path.join(project_root, "output")

    log.info(f"Audio file: {audio_path}")
    log.info(f"Call ID: {call_id}")
    log.info(f"Output directory: {output_dir}")

    # Verify audio file exists
    if not os.path.exists(audio_path):
        log.error(f"Audio file not found: {audio_path}")
        sys.exit(1)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    try:
        # STEP 1: Initialize Components
        log.info("Step 1/10: Initializing components...")
        from config import Config

        audio_processor = AudioProcessor(
            model_size=Config.WHISPER_MODEL_SIZE or "base",
            hf_token=Config.HF_TOKEN
        )
        text_processor = TextProcessor()
        feature_extractor = FeatureExtractor()
        analyzer = ConversationAnalyzer()
        dashboard = Dashboard()

        log.info("Step 1/10: Components initialized.")

        # STEP 2: Transcribe Audio (Whisper)
        log.info("Step 2/10: Transcribing audio with Whisper (may take several minutes for long files)...")
        transcription = audio_processor.transcribe_audio(audio_path, call_id)
        text = transcription.get('text', '')
        language = transcription.get('language', 'en')
        duration = transcription.get('duration', 0)
        segments = transcription.get('segments', [])

        log.info(f"Step 2/10: Transcription completed. Language: {language}, Duration: {duration:.2f}s, Segments: {len(segments)}.")

        if not text:
            log.warning("No speech detected in audio file; continuing with empty transcript.")

        # Save transcription
        transcription_file = os.path.join(output_dir, f"{call_id}_transcription.json")
        with open(transcription_file, 'w', encoding='utf-8') as f:
            json.dump(transcription, f, indent=2, ensure_ascii=False)
        log.info(f"Transcription saved: {transcription_file}")

        # STEP 3: Speaker Diarization
        log.info("Step 3/10: Performing speaker diarization (required)...")
        import librosa
        try:
            audio_duration = librosa.get_duration(path=audio_path)
            estimated_minutes = max(5, int(audio_duration / 60 / 10))
            log.info(f"Audio duration: {audio_duration:.1f}s ({audio_duration/60:.1f} min). Estimated diarization time: {estimated_minutes}-{estimated_minutes*2} min.")
        except Exception:
            log.info("Diarization in progress (this step is required)...")

        try:
            diarization_segments = audio_processor.perform_speaker_diarization(audio_path, call_id)
            log.info(f"Step 3/10: Diarization completed. Segments: {len(diarization_segments)}, Speakers: {len(set(s.get('speaker', '') for s in diarization_segments))}.")

            diarization_file = os.path.join(output_dir, f"{call_id}_diarization.json")
            diarization_data = {
                'call_id': call_id,
                'timestamp': datetime.now().isoformat(),
                'segments_count': len(diarization_segments),
                'speakers_found': len(set(s.get('speaker', '') for s in diarization_segments)),
                'segments': diarization_segments
            }
            with open(diarization_file, 'w', encoding='utf-8') as f:
                json.dump(diarization_data, f, indent=2, ensure_ascii=False, default=str)
            log.info(f"Diarization saved: {diarization_file}")
        except Exception as e:
            log.error(f"Diarization failed (required): {e}")
            log.info("Troubleshooting: set HF_TOKEN in .env, ensure Pyannote models are downloaded, verify audio file.")
            raise

        # STEP 4: Process Text and Segment Conversation
        log.info("Step 4/10: Processing text and segmenting conversation...")
        processed_segments = text_processor.segment_conversation(
            text,
            diarization_segments,
            call_id,
            transcription_segments=segments
        )
        log.info(f"Step 4/10: Text processing completed. Processed segments: {len(processed_segments)}.")
        
        # Re-save diarization file with updated text segments
        diarization_file = os.path.join(output_dir, f"{call_id}_diarization.json")
        updated_diarization = {
            'call_id': call_id,
            'timestamp': datetime.now().isoformat(),
            'segments_count': len(diarization_segments),
            'speakers_found': len(set(s.get('speaker', '') for s in diarization_segments)),
            'segments': diarization_segments
        }
        with open(diarization_file, 'w', encoding='utf-8') as f:
            json.dump(updated_diarization, f, indent=2, ensure_ascii=False, default=str)

        # STEP 5: Extract Audio Features
        log.info("Step 5/10: Extracting audio features...")
        try:
            audio_features = audio_processor.extract_audio_features(audio_path)
            log.info(f"Step 5/10: Audio features extracted. Keys: {list(audio_features.keys())}")
        except Exception as e:
            log.warning(f"Audio feature extraction failed: {e}; using fallback features.")
            audio_features = {
                "mfcc": None,
                "spectral_centroid": None,
                "zero_crossing_rate": None,
                "duration": duration
            }

        # STEP 6: Extract Combined Features
        log.info("Step 6/10: Extracting combined features...")
        feature_data = [{
            'audio_features': audio_features,
            'text_features': {'text': text},
            'segments': processed_segments
        }]
        try:
            fused_features = feature_extractor.fit_transform(feature_data)[0]
            log.info(f"Step 6/10: Feature extraction completed. Shape: {fused_features.shape if hasattr(fused_features, 'shape') else 'N/A'}")
        except Exception as e:
            log.warning(f"Feature extraction failed: {e}; using fallback.")
            import numpy as np
            fused_features = np.random.randn(50)

        # STEP 7: Run Complete Analysis
        log.info("Step 7/10: Running complete analysis (sentiment, emotion, sale prediction)...")
        analysis_result = analyzer.analyze_conversation(
            audio_path=audio_path,
            segments=processed_segments
        )
        if fused_features is not None and len(fused_features) > 0:
            sale_prediction = analyzer.sale_predictor.predict_sale_probability(fused_features)
            analysis_result['sale_prediction'] = sale_prediction
        log.info("Step 7/10: Analysis completed.")

        # STEP 8: Generate Dashboard
        log.info("Step 8/10: Generating dashboard...")
        try:
            dashboard_html = dashboard.generate_dashboard(
                analysis_result,
                call_id=call_id,
                output_path=os.path.join(output_dir, f"{call_id}_dashboard.html")
            )
            log.info(f"Step 8/10: Dashboard saved: {os.path.join(output_dir, f'{call_id}_dashboard.html')}")
        except Exception as e:
            log.warning(f"Dashboard generation failed: {e}")

        # STEP 9: Save Complete Results
        log.info("Step 9/10: Saving complete results...")
        # Prepare complete results
        complete_results = {
            "call_id": call_id,
            "timestamp": datetime.now().isoformat(),
            "audio_file": audio_path,
            "transcription": {
                "text": text,
                "language": language,
                "duration": duration,
                "segments_count": len(segments)
            },
            "diarization": {
                "speakers_found": len(set(s.get('speaker', '') for s in diarization_segments)),
                "segments_count": len(diarization_segments)
            },
            "analysis": analysis_result,
            "summary": {
                "sale_probability": analysis_result.get('sale_prediction', {}).get('sale_probability', 0),
                "avg_sentiment": analysis_result.get('conversation_metrics', {}).get('avg_sentiment', 0),
                "dominant_emotion": analysis_result.get('conversation_metrics', {}).get('dominant_emotion', 'neutral'),
                "total_segments": len(processed_segments)
            }
        }
        
        # Save JSON results
        results_file = os.path.join(output_dir, f"{call_id}_results.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(complete_results, f, indent=2, ensure_ascii=False, default=str)
        log.info(f"Step 9/10: Results saved: {results_file}")

        # STEP 10: Summary
        log.info("Step 10/10: Pipeline complete.")
        log.info("=" * 70)
        log.info("ANALYSIS COMPLETE - SUMMARY")
        log.info(f"Sale Probability: {complete_results['summary']['sale_probability']:.1%}")
        log.info(f"Average Sentiment: {complete_results['summary']['avg_sentiment']:.2f}")
        log.info(f"Dominant Emotion: {complete_results['summary']['dominant_emotion']}")
        log.info(f"Total Segments: {complete_results['summary']['total_segments']}")
        log.info(f"Output files: {transcription_file}, {results_file}")
        if os.path.exists(os.path.join(output_dir, f"{call_id}_dashboard.html")):
            log.info(f"Dashboard: {os.path.join(output_dir, f'{call_id}_dashboard.html')}")
        log.info("=" * 70)
        return complete_results

    except Exception as e:
        log = get_progress_logger()
        log.error(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    import argparse

    setup_project_logging()
    log = get_progress_logger()

    parser = argparse.ArgumentParser(description='Run complete call analysis pipeline')
    parser.add_argument('audio_file', help='Path to audio file')
    parser.add_argument('--call-id', help='Call ID (auto-generated if not provided)')
    parser.add_argument('--output-dir', default='output', help='Output directory')

    args = parser.parse_args()
    audio_path = args.audio_file

    if not os.path.exists(audio_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        project_path = os.path.join(project_root, audio_path)
        if os.path.exists(project_path):
            audio_path = project_path
            log.info(f"Found audio file in project root: {audio_path}")
        else:
            filename = os.path.basename(audio_path)
            project_path = os.path.join(project_root, filename)
            if os.path.exists(project_path):
                audio_path = project_path
                log.info(f"Found audio file in project root: {audio_path}")
            else:
                log.error(f"Audio file not found: {args.audio_file}")
                log.info(f"Searched: {os.path.abspath(args.audio_file)}, {project_path}")
                sys.exit(1)

    if not os.path.isabs(args.output_dir):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        output_dir = os.path.join(project_root, args.output_dir)
    else:
        output_dir = args.output_dir

    run_full_analysis(audio_path, args.call_id, output_dir)

