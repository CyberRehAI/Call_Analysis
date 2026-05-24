# Preprocessing Module (`preprocessing.py`)

This document explains the **preprocessing** module, which turns raw call audio into **transcribed text** and **speaker-labeled segments** that the rest of the system uses for sentiment, emotion, and sale prediction.

---

## What This Module Does

1. **AudioProcessor**: Loads audio, transcribes it with **Whisper**, performs **speaker diarization** (who spoke when), and extracts **audio features** (mel-spectrogram, MFCC) for the emotion model.
2. **TextProcessor**: Cleans text, masks PII, and **merges transcription with diarization** so each segment has `start_time`, `end_time`, `speaker`, and `text`.

Without preprocessing, the pipeline would not have per-speaker, time-aligned text or audio features.

---

## Main Classes and Flow

```
Audio file (.wav, .mp3, .m4a)
    → AudioProcessor.validate_audio_format()   # Convert to .wav if needed
    → AudioProcessor.transcribe_audio()       # Whisper → full text + segments
    → AudioProcessor.perform_speaker_diarization()  # Who spoke when
    → TextProcessor.segment_conversation()    # Merge text + diarization → segments
    → AudioProcessor.extract_audio_features() # Mel + MFCC for emotion model
```

The **segments** and **audio features** are then passed to the conversation analyzer (models + feature extraction).

---

## 1. AudioProcessor – Initialization and Model Loading

The `AudioProcessor` can use different backends for diarization:

- **WhisperX + Resemblyzer**: Faster, CPU-friendly; uses voice embeddings and clustering.
- **WhisperX 3.x built-in (Pyannote.audio)**: More accurate, requires `HF_TOKEN`.
- **Pyannote.audio only**: Fallback when WhisperX is not available.

**Code: Initialization and device selection**

```python
def __init__(self, model_size: str = "base", hf_token: str = None, use_whisperx: bool = True,
             chunk_duration: float = 300.0, enable_chunking: bool = True,
             max_speakers: int = None, num_speakers: int = None, ...):
    self.model_size = model_size
    self.hf_token = hf_token
    self.target_sample_rate = 16000
    self._device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Inference device: {self._device}")
    self._load_models()
```

**Explanation:**  
- `model_size`: Whisper size (e.g. `tiny`, `base`, `small`).  
- `hf_token`: Needed for Pyannote / WhisperX built-in diarization.  
- `_device`: Enables GPU if available; otherwise CPU.  
- `_load_models()` loads Whisper (required) and the chosen diarization pipeline (WhisperX+Resemblyzer, WhisperX built-in, or Pyannote).

---

## 2. Validating and Converting Audio Format

Only certain formats are supported. Others are converted to WAV using `pydub`.

**Code: Format validation**

```python
def validate_audio_format(self, audio_path: str) -> str:
    try:
        if not audio_path.endswith(('.wav', '.mp3', '.m4a')):
            logger.info(f"Converting {audio_path} to .wav")
            audio = AudioSegment.from_file(audio_path)
            new_path = audio_path.rsplit('.', 1)[0] + '.wav'
            audio.export(new_path, format='wav')
            return new_path
        return audio_path
    except Exception as e:
        logger.error(f"Audio format validation failed: {e}")
        raise ValueError(f"Unsupported audio format: {audio_path}")
```

**Explanation:**  
If the file is not `.wav`, `.mp3`, or `.m4a`, it is loaded with `AudioSegment` and exported as `.wav`. The returned path is what the rest of the pipeline uses. This keeps Whisper and librosa on a known format.

---

## 3. Transcription with Whisper

Transcription converts speech to text and (optionally) word-level segments. Long files can be processed in chunks to save memory and time.

**Code: Standard transcription (no chunking)**

```python
result = self.whisper_model.transcribe(
    audio_path,
    language="en",
    task="transcribe",
    verbose=False
)

transcription = {
    "text": result.get("text", ""),
    "segments": result.get("segments", []),
    "language": result.get("language", "en"),
    "duration": result.get("duration", 0)
}
```

**Explanation:**  
- `text`: Full transcript.  
- `segments`: List of `{ "start", "end", "text" }` used later to align with diarization.  
- If the file is longer than `chunk_duration`, `_transcribe_with_chunking()` splits the audio, transcribes each chunk, and merges segments with adjusted timestamps.

---

## 4. Speaker Diarization

Diarization answers: “Who spoke in which time range?” The module supports three methods; the code chooses based on config and availability.

**Code: Choosing the diarization method**

```python
def perform_speaker_diarization(self, audio_path: str, call_id: str) -> List[Dict]:
    audio_path = self.validate_audio_format(audio_path)
    # ...
    if self.use_whisperx_builtin_diarization and self.hf_token:
        return self._diarize_with_whisperx_builtin(audio_path, call_id, audio_duration)
    elif self.use_whisperx and self.voice_encoder is not None:
        return self._diarize_with_whisperx_resemblyzer(audio_path, call_id, audio_duration)
    elif self.diarization_pipeline is not None:
        return self._diarize_with_pyannote(audio_path, call_id, audio_duration)
    else:
        raise ValueError("Speaker diarization is required but no diarization method is available. ...")
```

**Explanation:**  
1. **WhisperX built-in**: Uses Pyannote via WhisperX; most accurate, needs `HF_TOKEN`.  
2. **WhisperX + Resemblyzer**: Transcribe with WhisperX, get voice embeddings per segment with Resemblyzer, cluster embeddings to get speakers; faster on CPU.  
3. **Pyannote only**: Direct Pyannote pipeline; used when WhisperX/Resemblyzer are not available.

Each method returns a list of segments with `speaker`, `start`, `end`, and (after text mapping) `text`.

**Code: Resemblyzer clustering (simplified idea)**

```python
# After extracting embeddings for each segment:
distances = pdist(segment_embeddings, metric='cosine')
linkage_matrix = linkage(distances, method='average')
cluster_labels = fcluster(linkage_matrix, threshold, criterion='distance')
# Then each segment gets SPEAKER_00, SPEAKER_01, ... based on cluster_labels
```

**Explanation:**  
Segments are converted to fixed-size vectors (embeddings). Similar voices end up in the same cluster; `fcluster` assigns a speaker ID per segment. Optional post-step merges very similar clusters to reduce over-segmentation.

---

## 5. Speaker Role Identification (Agent vs Customer)

For call-center use, generic labels like `SPEAKER_00` are replaced with **Agent** and **Customer** using simple heuristics.

**Code: Role identification heuristics**

```python
# Heuristic 1: First speaker is often the agent (greeting)
first_speaker = first_segment.get('speaker', 'Unknown')

# Heuristic 2 & 3: Keyword scoring
agent_keywords = ['thank you for calling', 'how can i help', 'may i have your', ...]
customer_keywords = ['i want', 'i need', 'i\'m calling about', ...]
# Score each speaker by how many of their segments contain these keywords

# Heuristic 4: Question patterns (agents often ask more questions)
# Count segments containing '?' or question starters like 'what', 'how', 'can you'
```

**Explanation:**  
Each speaker gets a score from: keyword match (agent vs customer), question count, speaking time, and whether they speak first. The highest-scoring speaker is labeled **Agent**, the other main speaker **Customer**. Segments are then updated so `speaker` is either `"AGENT"` or `"CUSTOMER"`.

---

## 6. Extracting Audio Features for the Emotion Model

The emotion model (CNN+LSTM) needs **mel-spectrogram** and **MFCC** for the whole file. This is done in preprocessing so the same audio is not loaded again later.

**Code: Audio feature extraction**

```python
def extract_audio_features(self, audio_path: str, n_mfcc: int = 40) -> Dict:
    audio_path = self.validate_audio_format(audio_path)
    y, sr = librosa.load(audio_path, sr=16000)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=128, fmax=8000, hop_length=512
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    features = {
        "mfcc": mfcc,
        "mel_spectrogram": mel_spec_db,
        "chroma": librosa.feature.chroma_stft(y=y, sr=sr, hop_length=512),
        "sample_rate": sr,
        "duration": librosa.get_duration(y=y, sr=sr),
        # ... other optional features
    }
    return features
```

**Explanation:**  
- Audio is loaded at 16 kHz for consistency with Whisper.  
- **MFCC**: 40 coefficients, used with mel in the emotion model.  
- **Mel-spectrogram**: 128 mel bands, log (dB) scale; the emotion model uses segment-level slices of this (slicing is done in `feature_extraction.py`).  
- `mel_spectrogram` and `mfcc` are the critical keys; other keys are optional for analytics.

---

## 7. TextProcessor – Merging Transcription and Diarization

The **TextProcessor** takes the full transcript and the diarization segments, assigns text to each segment, and returns a single list of “processed segments” used by sentiment, emotion, and dynamics.

**Code: segment_conversation**

```python
def segment_conversation(self, text: str, segments: List[Dict], call_id: str,
                        transcription_segments: Optional[List[Dict]] = None) -> List[Dict]:
    processed_segments = []
    text = self.mask_pii(text)

    # Fill diarization segments with actual text (align by time)
    self._assign_text_to_diarization_segments(segments, transcription_segments)

    for segment in segments:
        segment_text = self.mask_pii(segment.get('text', ''))
        processed_segment = {
            'start_time': segment.get('start', 0),
            'end_time': segment.get('end', 0),
            'speaker': segment.get('speaker', 'Unknown'),
            'text': segment_text,
            'features': self.extract_text_features(segment_text)
        }
        processed_segments.append(processed_segment)

    self.save_segments(processed_segments, call_id)
    return processed_segments
```

**Explanation:**  
- `_assign_text_to_diarization_segments()` matches transcription segments (with start/end) to diarization segments and fills each segment’s `text`.  
- Each processed segment has `start_time`, `end_time`, `speaker` (Agent/Customer), PII-masked `text`, and optional BERT-friendly `features`.  
- This list is what `ConversationAnalyzer` and `FeatureExtractor` expect.

---

## 8. TextProcessor – PII Masking and Cleaning

Before storing or sending text to the frontend, PII is masked. If spaCy is available, it is used; otherwise a simple regex fallback is used.

**Code: PII masking (spaCy path)**

```python
def mask_pii(self, text: str) -> str:
    if not SPACY_AVAILABLE or nlp is None:
        # Fallback: regex for phone and email
        text = re.sub(phone_pattern, '[PHONE]', text)
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@...\b', '[EMAIL]', text)
        return text
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ in ['PERSON', 'PHONE', 'EMAIL', 'GPE', 'ORG']:
            text = text.replace(ent.text, '[REDACTED]')
    return text
```

**Explanation:**  
- Protects privacy and supports compliance by replacing names, phones, emails, and orgs with placeholders.  
- Same idea is used in the dashboard and export so outputs do not expose raw PII.

---

## 9. Saving to MongoDB and Files

Transcription, diarization, and segments can be saved to JSON files and (if enabled) to MongoDB for debugging and auditing.

**Code: Saving transcription**

```python
def save_transcription(self, transcription: Dict, call_id: str):
    # Always save to JSON file
    output_dir = script_dir / "output"
    transcription_file = output_dir / f"{call_id}_transcription.json"
    with open(transcription_file, 'w', encoding='utf-8') as f:
        json.dump(transcription, f, indent=2, ...)

    # Optionally save to MongoDB
    if self.mongo_enabled:
        db['transcriptions'].insert_one({
            'call_id': call_id,
            'transcription': transcription,
            'timestamp': datetime.now()
        })
```

**Explanation:**  
- File save is always attempted so you have a local record even if MongoDB is down.  
- MongoDB is optional and controlled by `MONGODB_ENABLED` and connection settings.

---

## Summary Table

| Component | Responsibility |
|----------|----------------|
| **AudioProcessor.validate_audio_format** | Ensure .wav/.mp3/.m4a or convert to .wav |
| **AudioProcessor.transcribe_audio** | Whisper ASR → full text + segments (with optional chunking) |
| **AudioProcessor.perform_speaker_diarization** | Who spoke when (WhisperX+Resemblyzer, WhisperX built-in, or Pyannote) |
| **AudioProcessor._identify_speaker_roles** | Map SPEAKER_XX → Agent / Customer |
| **AudioProcessor.extract_audio_features** | Mel-spectrogram, MFCC, sample_rate, duration for emotion model |
| **TextProcessor.segment_conversation** | Merge transcript + diarization → segments with start_time, end_time, speaker, text |
| **TextProcessor.mask_pii** | Replace PII with [REDACTED] / [PHONE] / [EMAIL] |
| **save_transcription / save_diarization / save_segments** | Persist to JSON and optionally MongoDB |

After preprocessing, you have **processed_segments** and **audio_features** ready for the rest of the Call Analysis pipeline.
