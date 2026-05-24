#!/usr/bin/env python3
"""
Analyze Pyannote (WhisperX 3.x built-in) diarization output.
"""

import json
import sys
import io
import os
import argparse
from collections import defaultdict
from datetime import timedelta

# Fix Windows encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

def analyze_diarization_output(diarization_file: str, summary_file: str = None):
    """
    Analyze diarization output files.
    
    Args:
        diarization_file: Path to diarization JSON file
        summary_file: Optional path to summary JSON file
    """
    # Get project root (go up from backend/scripts/ to project root)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    backend_dir = os.path.dirname(script_dir)
    project_root = os.path.dirname(backend_dir)
    
    # If paths are relative, make them relative to project root
    if not os.path.isabs(diarization_file):
        diarization_file = os.path.join(project_root, diarization_file)
    
    if summary_file and not os.path.isabs(summary_file):
        summary_file = os.path.join(project_root, summary_file)
    
    # Load diarization results
    if not os.path.exists(diarization_file):
        print(f"❌ Error: Diarization file not found: {diarization_file}")
        sys.exit(1)
    
    with open(diarization_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Handle both formats: direct segments array or nested structure
    if isinstance(data, list):
        segments = data
    elif isinstance(data, dict):
        segments = data.get('segments', [])
    else:
        print("❌ Error: Invalid diarization file format")
        sys.exit(1)
    
    # Load summary if provided
    summary = None
    if summary_file:
        if os.path.exists(summary_file):
            with open(summary_file, 'r', encoding='utf-8') as f:
                summary = json.load(f)
        else:
            print(f"⚠️  Warning: Summary file not found: {summary_file}")
    
    print("=" * 80)
    print("PYANNOTE (WHISPERX 3.X BUILT-IN) DIARIZATION ANALYSIS")
    print("=" * 80)
    print()
    
    # Basic statistics
    agent_segments = [s for s in segments if s.get('speaker') == 'AGENT']
    customer_segments = [s for s in segments if s.get('speaker') == 'CUSTOMER']
    
    print("📊 OVERALL STATISTICS")
    print("-" * 80)
    print(f"Total segments: {len(segments)}")
    if summary:
        print(f"Unique speakers: {summary.get('results', {}).get('unique_speakers', 'N/A')}")
    else:
        unique_speakers = len(set(s.get('speaker', '') for s in segments))
        print(f"Unique speakers: {unique_speakers}")
    print(f"AGENT segments: {len(agent_segments)} ({len(agent_segments)/len(segments)*100:.1f}%)" if segments else "AGENT segments: 0")
    print(f"CUSTOMER segments: {len(customer_segments)} ({len(customer_segments)/len(segments)*100:.1f}%)" if segments else "CUSTOMER segments: 0")
    print()
    
    if not segments:
        print("⚠️  No segments found in diarization file")
        return
    
    # Duration analysis
    agent_duration = sum(s.get('end', 0) - s.get('start', 0) for s in agent_segments)
    customer_duration = sum(s.get('end', 0) - s.get('start', 0) for s in customer_segments)
    total_duration = max((s.get('end', 0) for s in segments), default=0)
    
    print("⏱️  DURATION ANALYSIS")
    print("-" * 80)
    print(f"Total audio duration: {total_duration:.1f} seconds ({total_duration/60:.1f} minutes)")
    if total_duration > 0:
        print(f"AGENT speaking time: {agent_duration:.1f} seconds ({agent_duration/60:.1f} minutes) - {agent_duration/total_duration*100:.1f}%")
        print(f"CUSTOMER speaking time: {customer_duration:.1f} seconds ({customer_duration/60:.1f} minutes) - {customer_duration/total_duration*100:.1f}%")
        print(f"Silence/overlap: {total_duration - agent_duration - customer_duration:.1f} seconds")
    print()
    
    # Segment length analysis
    if agent_segments:
        agent_lengths = [s.get('end', 0) - s.get('start', 0) for s in agent_segments]
        print("📏 SEGMENT LENGTH ANALYSIS")
        print("-" * 80)
        print("AGENT segments:")
        print(f"  Average: {sum(agent_lengths)/len(agent_lengths):.2f} seconds")
        print(f"  Min: {min(agent_lengths):.2f} seconds")
        print(f"  Max: {max(agent_lengths):.2f} seconds")
        print()
    
    if customer_segments:
        customer_lengths = [s.get('end', 0) - s.get('start', 0) for s in customer_segments]
        print("CUSTOMER segments:")
        print(f"  Average: {sum(customer_lengths)/len(customer_lengths):.2f} seconds")
        print(f"  Min: {min(customer_lengths):.2f} seconds")
        print(f"  Max: {max(customer_lengths):.2f} seconds")
        print()
    
    # Text analysis
    agent_texts = [s.get('text', '').strip() for s in agent_segments if s.get('text', '').strip()]
    customer_texts = [s.get('text', '').strip() for s in customer_segments if s.get('text', '').strip()]
    
    print("📝 TEXT ANALYSIS")
    print("-" * 80)
    print(f"AGENT segments with text: {len(agent_texts)}/{len(agent_segments)}")
    print(f"CUSTOMER segments with text: {len(customer_texts)}/{len(customer_segments)}")
    print()
    
    # Check for repetition issues
    agent_repeated = [t for t in agent_texts if len(t.split()) > 0 and len(set(t.split())) < len(t.split()) * 0.5]
    customer_repeated = [t for t in customer_texts if len(t.split()) > 0 and len(set(t.split())) < len(t.split()) * 0.5]
    
    print("🔄 REPETITION CHECK")
    print("-" * 80)
    print(f"AGENT segments with potential repetition: {len(agent_repeated)}")
    if agent_repeated:
        print("  Examples:")
        for text in agent_repeated[:3]:
            print(f"    - {text[:100]}...")
    print()
    print(f"CUSTOMER segments with potential repetition: {len(customer_repeated)}")
    if customer_repeated:
        print("  Examples:")
        for text in customer_repeated[:3]:
            print(f"    - {text[:100]}...")
    print()
    
    # Conversation flow analysis
    print("🗣️  CONVERSATION FLOW")
    print("-" * 80)
    print("First 10 segments:")
    for i, seg in enumerate(segments[:10]):
        speaker = seg.get('speaker', 'UNKNOWN')
        start = seg.get('start', 0)
        text = seg.get('text', '')[:60]
        print(f"  {i+1:2d}. [{start:6.1f}s] {speaker:8s}: {text}...")
    print()
    
    # Customer segment distribution
    if customer_segments:
        print("👤 CUSTOMER SEGMENT DISTRIBUTION")
        print("-" * 80)
        print("Customer segments throughout the conversation:")
        for i, seg in enumerate(customer_segments[:20]):
            start = seg.get('start', 0)
            end = seg.get('end', 0)
            duration = end - start
            text = seg.get('text', '')[:70]
            print(f"  {i+1:2d}. [{start:6.1f}s - {end:6.1f}s] ({duration:4.1f}s): {text}...")
        if len(customer_segments) > 20:
            print(f"  ... and {len(customer_segments) - 20} more customer segments")
        print()
    
    # Quality assessment
    print("✅ QUALITY ASSESSMENT")
    print("-" * 80)
    issues = []
    if customer_segments and len(customer_segments) < 20:
        issues.append("⚠️  Low customer segment count (might indicate under-detection)")
    if agent_segments and customer_segments and len(agent_segments) / len(customer_segments) > 10:
        issues.append("⚠️  Very imbalanced speaker distribution (agent >> customer)")
    if agent_segments and len(agent_repeated) > len(agent_segments) * 0.1:
        issues.append("⚠️  High repetition in agent segments")
    if customer_segments and len(customer_repeated) > len(customer_segments) * 0.1:
        issues.append("⚠️  High repetition in customer segments")
    
    if not issues:
        print("✅ No major issues detected!")
        print("✅ Good speaker balance")
        print("✅ Reasonable segment distribution")
        print("✅ Text mapping appears correct")
    else:
        for issue in issues:
            print(issue)
    print()
    
    print("=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze Pyannote diarization output')
    parser.add_argument('diarization_file', help='Path to diarization JSON file')
    parser.add_argument('--summary', help='Path to summary JSON file (optional)')
    
    args = parser.parse_args()
    analyze_diarization_output(args.diarization_file, args.summary)

