#!/usr/bin/env python3
"""
Generate ~5 minutes of 2-speaker audio and matching transcription for diarization testing.
Uses distinct male and female voices so diarization can separate speakers.
Creates:
  - 5min_diarization_test.wav
  - 5min_diarization_test_transcription.json
  - 5min_diarization_test_ground_truth.json

Preferred: edge-tts (male + female). Install: pip install edge-tts pydub
Fallback: pyttsx3 with David (male) / Zira (female) on Windows. Install: pip install pyttsx3 pydub
"""

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

# Dialogue: list of (speaker_id, text). "SPEAKER_00" and "SPEAKER_01" for ground truth.
# Target ~5 min at TTS speed (~2–3 words/sec). Adjust list length to get desired duration.
DIALOGUE = [
    ("SPEAKER_00", "Hello, thanks for calling support. How can I help you today?"),
    ("SPEAKER_01", "Hi, I have a question about my account."),
    ("SPEAKER_00", "Sure. Can I have your account number or email?"),
    ("SPEAKER_01", "Yes, it's john dot smith at email dot com."),
    ("SPEAKER_00", "Thank you. I see your account. What would you like to do?"),
    ("SPEAKER_01", "I want to update my billing address."),
    ("SPEAKER_00", "No problem. What is your new address?"),
    ("SPEAKER_01", "One two three Main Street, Austin Texas, seven eight seven zero one."),
    ("SPEAKER_00", "Got it. I will update that now. Anything else?"),
    ("SPEAKER_01", "Yes. Can you explain the last charge on my bill?"),
    ("SPEAKER_00", "Of course. The last charge was for your monthly plan on November first."),
    ("SPEAKER_01", "Okay. And what about the extra five dollar fee?"),
    ("SPEAKER_00", "That was for paperless billing opt-out. You can switch to paperless to avoid it."),
    ("SPEAKER_01", "I will do that. How do I switch?"),
    ("SPEAKER_00", "In the app, go to Settings, then Billing, then choose Paperless."),
    ("SPEAKER_01", "Thanks. I think that's all for now."),
    ("SPEAKER_00", "Is there anything else I can help with today?"),
    ("SPEAKER_01", "No, that was everything. Thank you."),
    ("SPEAKER_00", "You are welcome. Have a great day. Goodbye."),
    ("SPEAKER_01", "Goodbye."),
    ("SPEAKER_00", "Hello again. This is support. How can I help?"),
    ("SPEAKER_01", "I need help with my password. I cannot log in."),
    ("SPEAKER_00", "I can help with that. Have you tried the reset link?"),
    ("SPEAKER_01", "Yes, but the link expired."),
    ("SPEAKER_00", "I will send a new reset link to your email now."),
    ("SPEAKER_01", "Thank you. What email will you use?"),
    ("SPEAKER_00", "The one on file: j smith at email dot com."),
    ("SPEAKER_01", "That is correct. I will check my inbox."),
    ("SPEAKER_00", "The link is valid for one hour. Let me know if you need another."),
    ("SPEAKER_01", "Will do. Thanks."),
    ("SPEAKER_00", "You are welcome. Anything else?"),
    ("SPEAKER_01", "No. Bye."),
    ("SPEAKER_00", "Bye."),
    ("SPEAKER_01", "Hi, I am calling about my order."),
    ("SPEAKER_00", "Sure. What is your order number?"),
    ("SPEAKER_01", "Order number is four five six seven eight nine."),
    ("SPEAKER_00", "I see it. The order shipped yesterday. You should get it by Friday."),
    ("SPEAKER_01", "Can I change the delivery address?"),
    ("SPEAKER_00", "It may be too late. Let me check. One moment please."),
    ("SPEAKER_01", "Okay."),
    ("SPEAKER_00", "I can redirect it. What is the new address?"),
    ("SPEAKER_01", "Four five six Oak Lane, Denver Colorado, eight zero two zero two."),
    ("SPEAKER_00", "I have updated the delivery address. You will get a new tracking email."),
    ("SPEAKER_01", "Perfect. Thank you so much."),
    ("SPEAKER_00", "My pleasure. Have a good one."),
    ("SPEAKER_01", "You too. Bye."),
    ("SPEAKER_00", "Thanks for calling. Goodbye."),
]
# Actual duration depends on TTS speed (typically 3–8 min for this many turns). For longer audio, duplicate DIALOGUE or add more turns.


def get_output_dir():
    """Output to project output/ folder if we can find it, else cwd."""
    script_dir = Path(__file__).resolve().parent
    backend_dir = script_dir.parent
    project_root = backend_dir.parent
    out = project_root / "output"
    out.mkdir(parents=True, exist_ok=True)
    return out


# Edge-TTS: male and female English voices (agent = female, customer = male for clear contrast)
EDGE_VOICE_FEMALE = "en-US-JennyNeural"   # SPEAKER_00 (agent)
EDGE_VOICE_MALE = "en-US-GuyNeural"       # SPEAKER_01 (customer)


async def _generate_with_edge_tts(out_dir, wav_path, trans_path, truth_path):
    """Use edge-tts for distinct male/female voices. Requires: pip install edge-tts pydub"""
    import edge_tts
    from pydub import AudioSegment

    combined = AudioSegment.empty()
    segments_transcription = []
    segments_ground_truth = []
    current_time = 0.0
    silence_between = AudioSegment.silent(duration=300)  # 0.3 s

    for i, (speaker_id, text) in enumerate(DIALOGUE):
        voice = EDGE_VOICE_FEMALE if speaker_id == "SPEAKER_00" else EDGE_VOICE_MALE
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            tmp = f.name
        try:
            communicate = edge_tts.Communicate(text.strip(), voice)
            await communicate.save(tmp)
        except Exception as e:
            print(f"edge-tts error for turn {i + 1}: {e}")
            os.remove(tmp)
            continue
        try:
            seg = AudioSegment.from_mp3(tmp)
        except Exception as e:
            print(f"Could not load MP3 for turn {i + 1}: {e}")
            os.remove(tmp)
            continue
        os.remove(tmp)

        start = current_time
        duration_sec = len(seg) / 1000.0
        end = start + duration_sec
        current_time = end + 0.3

        segments_transcription.append({"start": round(start, 2), "end": round(end, 2), "text": text.strip()})
        segments_ground_truth.append({
            "start": round(start, 2), "end": round(end, 2), "speaker": speaker_id, "text": text.strip()
        })
        combined += seg + silence_between

    combined = combined[:-300]
    combined.export(str(wav_path), format="wav")
    return len(combined) / 1000.0, segments_transcription, segments_ground_truth


def _pick_pyttsx3_male_female_voices(engine):
    """Pick one male and one female voice by name (Windows: David, Zira; macOS: Alex, Samantha)."""
    voices = engine.getProperty("voices")
    male_id = female_id = None
    for v in voices:
        name_lower = (v.name or "").lower()
        id_lower = (v.id or "").lower()
        if "david" in name_lower or "david" in id_lower or "male" in name_lower:
            male_id = v.id
        if "zira" in name_lower or "zira" in id_lower or "samantha" in name_lower or "female" in name_lower:
            female_id = v.id
    if male_id and female_id:
        return male_id, female_id
    if len(voices) >= 2:
        return voices[0].id, voices[1].id
    return (voices[0].id, voices[0].id) if voices else (None, None)


def _generate_with_pyttsx3(out_dir, wav_path, trans_path, truth_path):
    """Fallback: pyttsx3 with male/female voice selection where possible."""
    import pyttsx3
    from pydub import AudioSegment

    engine = pyttsx3.init()
    voice_male, voice_female = _pick_pyttsx3_male_female_voices(engine)
    if voice_male == voice_female:
        print("WARNING: Could not find distinct male/female voices. Install edge-tts for best results: pip install edge-tts")
    else:
        print("Using pyttsx3: male and female voices (David/Zira on Windows).")

    combined = AudioSegment.empty()
    segments_transcription = []
    segments_ground_truth = []
    current_time = 0.0
    silence_between = AudioSegment.silent(duration=300)

    for i, (speaker_id, text) in enumerate(DIALOGUE):
        engine.setProperty("voice", voice_female if speaker_id == "SPEAKER_00" else voice_male)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp = f.name
        try:
            engine.save_to_file(text, tmp)
            engine.runAndWait()
        except Exception as e:
            print(f"TTS error for turn {i + 1}: {e}")
            continue
        try:
            seg = AudioSegment.from_wav(tmp)
        except Exception as e:
            os.remove(tmp)
            continue
        os.remove(tmp)

        start = current_time
        duration_sec = len(seg) / 1000.0
        end = start + duration_sec
        current_time = end + 0.3

        segments_transcription.append({"start": round(start, 2), "end": round(end, 2), "text": text.strip()})
        segments_ground_truth.append({
            "start": round(start, 2), "end": round(end, 2), "speaker": speaker_id, "text": text.strip()
        })
        combined += seg + silence_between

    combined = combined[:-300]
    combined.export(str(wav_path), format="wav")
    return len(combined) / 1000.0, segments_transcription, segments_ground_truth


def generate_audio_and_transcription():
    try:
        from pydub import AudioSegment
    except ImportError:
        print("ERROR: pydub is required. Install with: pip install pydub")
        sys.exit(1)

    out_dir = get_output_dir()
    wav_path = out_dir / "5min_diarization_test.wav"
    trans_path = out_dir / "5min_diarization_test_transcription.json"
    truth_path = out_dir / "5min_diarization_test_ground_truth.json"

    print("Generating 2-speaker test audio (male + female voices, ~5 min target)...")
    print(f"Output directory: {out_dir}")
    print()

    try:
        import edge_tts
        print("Using edge-tts (Jenny = female/agent, Guy = male/customer).")
        total_sec, segments_transcription, segments_ground_truth = asyncio.run(
            _generate_with_edge_tts(out_dir, wav_path, trans_path, truth_path)
        )
    except ImportError:
        print("edge-tts not found. Fallback: pyttsx3 (install edge-tts for male/female: pip install edge-tts).")
        total_sec, segments_transcription, segments_ground_truth = _generate_with_pyttsx3(
            out_dir, wav_path, trans_path, truth_path
        )

    full_text = " ".join(s["text"] for s in segments_transcription)
    with open(trans_path, "w", encoding="utf-8") as f:
        json.dump({"text": full_text, "segments": segments_transcription}, f, indent=2, ensure_ascii=False)
    with open(truth_path, "w", encoding="utf-8") as f:
        json.dump(
            {"total_duration_sec": round(total_sec, 2), "num_segments": len(segments_ground_truth), "segments": segments_ground_truth},
            f, indent=2, ensure_ascii=False,
        )

    print(f"Created: {wav_path} ({total_sec:.1f} s)")
    print(f"Created: {trans_path}")
    print(f"Created: {truth_path}")
    print()
    print("Run diarization with:")
    print(f"  python run_diarization_only.py \"{wav_path}\" \"{trans_path}\" --num-speakers 2 --output-dir \"{out_dir}\"")
    print()
    print("Then compare diarization output to ground_truth.json to verify speaker labels and timing.")


if __name__ == "__main__":
    generate_audio_and_transcription()
