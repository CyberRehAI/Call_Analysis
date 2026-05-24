# Feature Extraction Module (`feature_extraction.py`)

This document explains the **feature_extraction** module. It builds the **audio and conversation-level features** used by the emotion model and the sale predictor, and provides **segment-level mel-spectrograms** for per-segment emotion detection.

---

## What This Module Does

1. **Mel-spectrogram normalization** – Prepares mel-spectrograms for the CNN+LSTM emotion model (CMVN, z-score, min-max, etc.).
2. **FeatureExtractor** – Loads audio and computes mel, MFCC, chroma, pitch, etc.; extracts **conversational dynamics** (silence, interruptions, talk–listen ratio, turn-taking, filler words) for the sale predictor.
3. **Segment-level mel extraction** – Slices the full mel-spectrogram by time so each conversation segment can get its own emotion prediction.
4. **PII masking** – Optional text anonymization for security.

The sale predictor’s **fused feature vector** (in `models.py`) is built from sentiment stats, emotion probabilities, and the **conversational dynamics** produced here.

---

## High-Level Flow

```
Audio path + Processed segments
    → FeatureExtractor.load_audio_features()     # Mel, MFCC, chroma, pitch, etc.
    → normalize_mel_spectrogram()                # Used by emotion model
    → extract_segment_mel_spectrogram()         # Per-segment mel for emotion
    → FeatureExtractor.extract_conversational_dynamics()  # For sale predictor
```

---

## 1. Mel-Spectrogram Normalization

The emotion model expects normalized mel-spectrograms. This module supports several schemes; **CMVN** (Cepstral Mean and Variance Normalization) is the default and works well for speaker-independent emotion.

**Code: CMVN normalization**

```python
def apply_cmvn_normalization(mel_spec: np.ndarray) -> np.ndarray:
    """
    Apply Cepstral Mean and Variance Normalization (CMVN) per frequency band.
    Normalizes each frequency band independently, good for speaker-independent emotion.
    """
    mel_spec_norm = mel_spec.copy()
    for i in range(mel_spec.shape[0]):
        band = mel_spec[i, :]
        mean = np.mean(band)
        std = np.std(band)
        if std > 1e-8:
            mel_spec_norm[i, :] = (band - mean) / std
        else:
            mel_spec_norm[i, :] = 0.0
    return mel_spec_norm
```

**Explanation:**  
- For each **frequency band** (each row of the mel-spectrogram), we subtract the mean and divide by the standard deviation.  
- This reduces speaker-specific level differences while keeping relative spectral shape, which helps the emotion model generalize across speakers.

**Code: Wrapper to choose normalization method**

```python
def normalize_mel_spectrogram(mel_spec: np.ndarray, method: str = 'cmvn',
                              stats: Optional[Dict] = None) -> np.ndarray:
    if method == 'cmvn':
        return apply_cmvn_normalization(mel_spec)
    elif method == 'zscore':
        mean = stats.get('mean') if stats else None
        std = stats.get('std') if stats else None
        return apply_zscore_normalization(mel_spec, mean, std)
    elif method == 'logmel':
        return apply_logmel_normalization(mel_spec)
    elif method == 'minmax':
        return apply_minmax_normalization(mel_spec, stats.get('min'), stats.get('max'))
    # ...
```

**Explanation:**  
- **cmvn**: No extra stats; computed per-band as above.  
- **zscore**: Global mean/std (e.g. from training) for the whole mel.  
- **logmel**: Clip and scale dB values.  
- **minmax**: Scale to [0, 1] using min/max (can remove level information).  
The emotion model in `models.py` uses the same `method` and `stats` as at training time.

---

## 2. FeatureExtractor – Loading Audio Features

`load_audio_features` loads an audio file and computes mel-spectrogram, MFCC, and optional features (chroma, pitch, spectral centroid, etc.). This is the main entry for “full audio” features used by the emotion model and for segment slicing.

**Code: Core audio feature extraction**

```python
def load_audio_features(self, audio_path: str, n_mfcc: int = 40) -> Dict:
    y, sr = librosa.load(audio_path, sr=16000)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=128, fmax=8000, hop_length=512
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=1.0)

    # Optional: chroma, spectral centroid, pitch, etc.
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=512)
    # ...
    return {
        "mfcc": mfcc,
        "mel_spectrogram": mel_spec_db,
        "chroma": chroma,
        "sample_rate": sr,
        "duration": librosa.get_duration(y=y, sr=sr),
        # ...
    }
```

**Explanation:**  
- **16000 Hz** matches the rest of the pipeline (Whisper, preprocessing).  
- **Mel-spectrogram**: 128 mel bands, hop 512; converted to dB. Used as-is for full-file or sliced per segment for emotion.  
- **MFCC**: 40 coefficients; the emotion model can use both mel and MFCC.  
- **chroma**, pitch, etc. are optional for analytics or future features; the sale predictor currently uses dynamics, not raw chroma.

---

## 3. Conversational Dynamics (for Sale Predictor)

The sale predictor does not use raw audio; it uses a **fused feature vector** that includes **conversational dynamics**: silence ratio, interruption frequency, talk–listen ratio, turn-taking frequency, and filler-word frequency. These are computed here.

**Code: extract_conversational_dynamics**

```python
def extract_conversational_dynamics(self, segments: List[Dict],
                                   total_duration: float) -> Dict:
    if not segments or total_duration <= 0:
        return {
            'silence_ratio': 0.0,
            'interruption_frequency': 0.0,
            'talk_listen_ratio': 1.0,
            'turn_taking_frequency': 0.0,
            'filler_word_frequency': 0.0
        }

    speaking_time = sum(
        seg.get('end_time', 0) - seg.get('start_time', 0)
        for seg in segments
    )

    # 1. Silence Ratio
    silence_time = total_duration - speaking_time
    silence_ratio = silence_time / total_duration if total_duration > 0 else 0.0

    # 2. Interruption Frequency (overlapping speech from different speakers)
    interruptions = self._count_interruptions(segments)
    interruption_frequency = interruptions / total_duration if total_duration > 0 else 0.0

    # 3. Talk-to-Listen Ratio (agent time / customer time)
    agent_time = sum(duration for seg in segments
                     if seg.get('speaker', '').lower() in ['agent', 'speaker_0'])
    customer_time = sum(duration for seg in segments
                        if seg.get('speaker', '').lower() in ['customer', 'speaker_1'])
    talk_listen_ratio = agent_time / customer_time if customer_time > 0 else 1.0

    # 4. Turn-taking frequency (speaker changes per minute)
    turn_taking_frequency = self._calculate_turn_taking_frequency(segments, total_duration)

    # 5. Filler word frequency (e.g. "um", "uh", "like" per minute)
    filler_word_frequency = self._calculate_filler_word_frequency(segments, total_duration)

    return {
        'silence_ratio': silence_ratio,
        'interruption_frequency': interruption_frequency,
        'talk_listen_ratio': talk_listen_ratio,
        'turn_taking_frequency': turn_taking_frequency,
        'filler_word_frequency': filler_word_frequency
    }
```

**Explanation:**  
- **Silence ratio**: Fraction of the call with no speech; high value can mean long pauses or disengagement.  
- **Interruption frequency**: Pairs of segments from different speakers that overlap in time; indicates overlap/interruptions.  
- **Talk–listen ratio**: Agent speaking time / customer speaking time; balance of who dominates.  
- **Turn-taking frequency**: How often the speaker changes per minute.  
- **Filler word frequency**: Count of filler words per minute from segment text.  

These five (plus sentiment and emotion from `models.py`) form the inputs to the XGBoost sale predictor.

**Code: Counting interruptions (overlapping different speakers)**

```python
def _count_interruptions(self, segments: List[Dict]) -> int:
    interruptions = 0
    for i in range(len(segments)):
        for j in range(i + 1, len(segments)):
            seg1, seg2 = segments[i], segments[j]
            if seg1.get('speaker') == seg2.get('speaker'):
                continue
            if (seg1.get('end_time', 0) > seg2.get('start_time', 0) and
                seg1.get('start_time', 0) < seg2.get('end_time', 0)):
                interruptions += 1
    return interruptions
```

**Explanation:**  
Two segments from **different speakers** that overlap in time count as one interruption (overlapping speech).

---

## 4. Segment-Level Mel-Spectrogram (for Per-Segment Emotion)

The emotion model can run on the **whole** file or on **each segment** separately. For per-segment emotion, we need a mel-spectrogram that covers only that segment’s time range. That is done by slicing the full mel-spectrogram by frame indices.

**Code: extract_segment_mel_spectrogram**

```python
def extract_segment_mel_spectrogram(full_mel: np.ndarray, start_time: float, end_time: float,
                                   sample_rate: int = 16000, hop_length: int = 512) -> np.ndarray:
    """
    Extract the part of the full mel-spectrogram that corresponds to [start_time, end_time].
    """
    # time (seconds) -> frame index: frame = time * sample_rate / hop_length
    start_frame = int(start_time * sample_rate / hop_length)
    end_frame = int(end_time * sample_rate / hop_length)

    start_frame = max(0, min(start_frame, full_mel.shape[1]))
    end_frame = max(start_frame, min(end_frame, full_mel.shape[1]))

    if start_frame >= end_frame:
        return np.zeros((full_mel.shape[0], 1))

    segment_mel = full_mel[:, start_frame:end_frame]
    return segment_mel
```

**Explanation:**  
- Mel-spectrogram has shape `(n_mels, time_frames)`. Each frame corresponds to `hop_length` samples.  
- So frame index = `time * sample_rate / hop_length`. We clamp to valid indices and slice `full_mel[:, start_frame:end_frame]`.  
- The emotion model in `models.py` uses this to get one mel slice per segment and runs the CNN+LSTM on each slice to get per-segment emotions.

---

## 5. Filler Word Frequency

Filler words (“um”, “uh”, “like”, etc.) are counted in segment text and normalized by call duration to get a rate (e.g. per minute).

**Code: Filler word calculation (concept)**

```python
def _calculate_filler_word_frequency(self, segments: List[Dict], total_duration: float) -> float:
    FILLER_WORDS = {'um', 'uh', 'like', 'you know', 'actually', 'basically', ...}
    count = 0
    for seg in segments:
        text = seg.get('text', '').lower()
        words = text.split()
        for word in words:
            if word in FILLER_WORDS:
                count += 1
    # Per minute
    return (count / total_duration) * 60.0 if total_duration > 0 else 0.0
```

**Explanation:**  
- Simple word-level match against a fixed set of fillers.  
- Result is **filler count per minute**, which is one of the five conversational dynamics fed to the sale predictor.

---

## 6. PII Masking in Feature Extraction

The module can mask PII in text (e.g. before logging or storing derived text). Same idea as in preprocessing: spaCy NER or regex for phone/email.

**Code: Masking with spaCy**

```python
def mask_pii(self, text: str) -> str:
    if not SPACY_AVAILABLE or nlp is None:
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
- Used for any text that might be stored or exposed (e.g. in features or logs).  
- Keeps the pipeline consistent with preprocessing and dashboard PII handling.

---

## 7. How This Fits in the Pipeline

| Step | Module / function | Output |
|------|-------------------|--------|
| Preprocessing | `AudioProcessor.extract_audio_features()` | Basic mel/MFCC (preprocessing also has this; feature_extraction’s version is richer and used for dynamics + segment mel). |
| Feature extraction | `FeatureExtractor.load_audio_features(audio_path)` | Full mel, MFCC, chroma, sample_rate, duration. |
| Feature extraction | `normalize_mel_spectrogram(mel, method, stats)` | Normalized mel for the emotion model. |
| Feature extraction | `extract_segment_mel_spectrogram(full_mel, start, end, sr, hop)` | Per-segment mel for per-segment emotion. |
| Feature extraction | `FeatureExtractor.extract_conversational_dynamics(segments, total_duration)` | silence_ratio, interruption_frequency, talk_listen_ratio, turn_taking_frequency, filler_word_frequency. |
| Models | `SalePredictor.create_fused_feature_vector(sentiment_results, emotion_results, conversational_dynamics)` | Single vector for XGBoost. |

So: **feature_extraction** supplies the **conversational dynamics** and the **segment mel extraction**; **models** combines them with sentiment and emotion into the fused vector and runs the sale and emotion models.

---

## Summary Table

| Function / method | Purpose |
|-------------------|--------|
| **apply_cmvn_normalization** | Per-band mean/variance normalization for mel (default for emotion). |
| **normalize_mel_spectrogram** | Selects and applies one of CMVN, zscore, logmel, minmax. |
| **FeatureExtractor.load_audio_features** | Load audio; compute mel, MFCC, chroma, pitch, etc. |
| **FeatureExtractor.extract_conversational_dynamics** | Silence ratio, interruptions, talk–listen ratio, turn-taking, filler frequency. |
| **extract_segment_mel_spectrogram** | Slice full mel by [start_time, end_time] for one segment. |
| **FeatureExtractor.mask_pii** | Anonymize PII in text. |

Together, these support **emotion detection** (normalized mel, segment mel) and **sale prediction** (conversational dynamics as part of the fused feature vector).
