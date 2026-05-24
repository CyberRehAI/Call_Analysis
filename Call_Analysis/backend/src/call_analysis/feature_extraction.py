"""
Feature extraction module for audio and text features, aligned with SRS1 (Purely ML-Based System).
Includes BERT integration for text features (FR-4), improved sentiment calculation,
audio preprocessing from raw files (FR-1/FR-5), database storage (FR-7),
PII masking for security, and enhanced error handling.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import librosa
from transformers import BertTokenizer, BertModel
import torch
from pymongo import MongoClient
from datetime import datetime
import spacy
import unicodedata
import os
import json
import time
from functools import wraps

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Structured logging utility
def log_performance(func):
    """Decorator to log function performance metrics."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        start_memory = None
        try:
            import psutil
            import os as os_module
            process = psutil.Process(os_module.getpid())
            start_memory = process.memory_info().rss / 1024 / 1024  # MB
        except ImportError:
            pass
        
        try:
            result = func(*args, **kwargs)
            elapsed_time = time.time() - start_time
            
            end_memory = None
            if start_memory is not None:
                try:
                    end_memory = process.memory_info().rss / 1024 / 1024  # MB
                    memory_delta = end_memory - start_memory
                except:
                    pass
            
            # Structured log entry (only log at DEBUG level to reduce noise during training)
            log_entry = {
                'function': func.__name__,
                'status': 'success',
                'latency_ms': elapsed_time * 1000,
                'timestamp': time.time()
            }
            if start_memory is not None and end_memory is not None:
                log_entry['memory_mb'] = end_memory
                log_entry['memory_delta_mb'] = memory_delta
            
            logger.debug(f"PERF: {json.dumps(log_entry)}")  # Changed from logger.info to logger.debug
            return result
        except Exception as e:
            elapsed_time = time.time() - start_time
            log_entry = {
                'function': func.__name__,
                'status': 'error',
                'error': str(e),
                'latency_ms': elapsed_time * 1000,
                'timestamp': time.time()
            }
            logger.error(f"PERF: {json.dumps(log_entry)}")  # Keep errors at ERROR level
            raise
    
    return wrapper

MONGO_ENABLED = os.getenv('MONGODB_ENABLED', 'false').lower() == 'true'
MONGO_URI = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/')
MONGO_DB_NAME = os.getenv('MONGODB_DATABASE', 'call_center_db')

_BERT_TOKENIZER = None
_BERT_MODEL = None

# Load spaCy model for PII masking
# Load spaCy for PII masking (optional)
try:
    nlp = spacy.load('en_core_web_sm')
    SPACY_AVAILABLE = True
    logger.info("spaCy model loaded successfully")
except OSError:
    nlp = None
    SPACY_AVAILABLE = False
    logger.warning("spaCy model 'en_core_web_sm' not found. PII masking will be limited.")


# ============================================================================
# Normalization Functions for Mel-Spectrograms
# ============================================================================

def apply_cmvn_normalization(mel_spec: np.ndarray) -> np.ndarray:
    """
    Apply Cepstral Mean and Variance Normalization (CMVN) per frequency band.
    
    CMVN normalizes each frequency band independently, preserving relative energy
    differences between bands while removing speaker-dependent variations.
    This is preferred for speaker-independent emotion recognition.
    
    Args:
        mel_spec: Mel-spectrogram array (n_mels, time_frames)
    
    Returns:
        Normalized mel-spectrogram with same shape
    """
    # Normalize each frequency band (each row) independently
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


def apply_zscore_normalization(mel_spec: np.ndarray, mean: Optional[float] = None, 
                               std: Optional[float] = None) -> np.ndarray:
    """
    Apply Z-score normalization using training set statistics.
    
    Args:
        mel_spec: Mel-spectrogram array (n_mels, time_frames)
        mean: Pre-computed mean from training set (if None, computed from mel_spec)
        std: Pre-computed std from training set (if None, computed from mel_spec)
    
    Returns:
        Normalized mel-spectrogram
    """
    if mean is None:
        mean = np.mean(mel_spec)
    if std is None:
        std = np.std(mel_spec)
    
    if std > 1e-8:
        return (mel_spec - mean) / std
    else:
        return mel_spec - mean


def apply_logmel_normalization(mel_spec: np.ndarray) -> np.ndarray:
    """
    Apply log-mel normalization without min-max scaling.
    
    This preserves absolute energy levels which are important for emotion recognition.
    Only applies basic scaling to prevent extreme values.
    
    Args:
        mel_spec: Mel-spectrogram array (n_mels, time_frames) in dB scale
    
    Returns:
        Normalized mel-spectrogram
    """
    # Clip extreme values but preserve relative energy
    mel_spec_clipped = np.clip(mel_spec, -80, 0)  # Typical dB range
    # Scale to [0, 1] without min-max (preserves energy differences)
    mel_spec_norm = (mel_spec_clipped + 80) / 80.0
    return mel_spec_norm


def apply_minmax_normalization(mel_spec: np.ndarray, min_val: Optional[float] = None,
                               max_val: Optional[float] = None) -> np.ndarray:
    """
    Apply min-max normalization (current approach, kept for comparison).
    
    WARNING: This removes absolute energy cues which are important for emotion.
    Use only for comparison or when energy differences are not relevant.
    
    Args:
        mel_spec: Mel-spectrogram array (n_mels, time_frames)
        min_val: Minimum value for normalization (if None, uses mel_spec.min())
        max_val: Maximum value for normalization (if None, uses mel_spec.max())
    
    Returns:
        Normalized mel-spectrogram in [0, 1] range
    """
    if min_val is None:
        min_val = mel_spec.min()
    if max_val is None:
        max_val = mel_spec.max()
    
    if max_val - min_val > 1e-8:
        return (mel_spec - min_val) / (max_val - min_val)
    else:
        return np.zeros_like(mel_spec)


def normalize_mel_spectrogram(mel_spec: np.ndarray, method: str = 'cmvn', 
                              stats: Optional[Dict] = None) -> np.ndarray:
    """
    Wrapper function to select normalization method.
    
    Args:
        mel_spec: Mel-spectrogram array (n_mels, time_frames)
        method: Normalization method ('cmvn', 'zscore', 'logmel', 'minmax')
        stats: Optional statistics dictionary with 'mean', 'std', 'min', 'max' keys
    
    Returns:
        Normalized mel-spectrogram
    """
    if method == 'cmvn':
        return apply_cmvn_normalization(mel_spec)
    elif method == 'zscore':
        mean = stats.get('mean') if stats else None
        std = stats.get('std') if stats else None
        return apply_zscore_normalization(mel_spec, mean, std)
    elif method == 'logmel':
        return apply_logmel_normalization(mel_spec)
    elif method == 'minmax':
        min_val = stats.get('min') if stats else None
        max_val = stats.get('max') if stats else None
        return apply_minmax_normalization(mel_spec, min_val, max_val)
    else:
        logger.warning(f"Unknown normalization method '{method}', using CMVN")
        return apply_cmvn_normalization(mel_spec)


class FeatureExtractor:
    """Extracts and combines features from audio and text data, with SRS1 alignment."""
    
    def __init__(self, load_bert: bool = False):
        """
        Initialize feature extractor with BERT and scalers.
        
        Args:
            load_bert: If True, load BERT model immediately. If False, load lazily when needed.
                      Set to False for audio-only feature extraction (e.g., emotion training).
        """
        self.tfidf_vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        self.audio_scaler = StandardScaler()
        self.text_scaler = StandardScaler()
        self.is_fitted = False
        
        # Lazy BERT loading - only load when actually needed (for text features)
        # This speeds up audio-only workflows like emotion model training
        self._bert_tokenizer = None
        self._bert_model = None
        self._load_bert = load_bert
        
        if load_bert:
            self._ensure_bert_loaded()
        
        # Placeholder for sentiment classifier (e.g., fine-tuned on top of BERT)
        # In practice, train this with your dataset
        self.sentiment_classifier = lambda emb: np.mean(emb)  # Dummy; replace with actual classifier
    
    def _ensure_bert_loaded(self):
        """Lazy load BERT model only when needed."""
        if self._bert_tokenizer is None or self._bert_model is None:
            global _BERT_TOKENIZER, _BERT_MODEL
            if _BERT_TOKENIZER is None:
                logger.info("Loading BERT tokenizer (lazy loading)...")
                _BERT_TOKENIZER = BertTokenizer.from_pretrained('bert-base-uncased')
            if _BERT_MODEL is None:
                logger.info("Loading BERT model (lazy loading)...")
                _BERT_MODEL = BertModel.from_pretrained('bert-base-uncased')
            self._bert_tokenizer = _BERT_TOKENIZER
            self._bert_model = _BERT_MODEL
    
    @property
    def bert_tokenizer(self):
        """Lazy access to BERT tokenizer."""
        self._ensure_bert_loaded()
        return self._bert_tokenizer
    
    @property
    def bert_model(self):
        """Lazy access to BERT model."""
        self._ensure_bert_loaded()
        return self._bert_model
    
    def mask_pii(self, text: str) -> str:
        """
        Mask personally identifiable information (PII) in text for security (SRS1 regulatory constraints).
        
        Args:
            text: Raw text to anonymize.
            
        Returns:
            Anonymized text.
        """
        if not text or not text.strip():
            return text
        
        if not SPACY_AVAILABLE or nlp is None:
            # Fallback to simple regex-based masking if spaCy not available
            import re
            phone_pattern = r'(?:(?:\+?\d{1,2}[\s.-]?)?(?:\(\d{3}\)|\d{3})[\s.-]\d{3}[\s.-]\d{4})'
            text = re.sub(phone_pattern, '[PHONE]', text)
            # Mask email addresses
            text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
            return text
        
        try:
            doc = nlp(text)
            for ent in doc.ents:
                if ent.label_ in ['PERSON', 'PHONE', 'EMAIL', 'GPE', 'ORG']:
                    text = text.replace(ent.text, '[REDACTED]')
            return text
        except Exception as e:
            logger.error(f"Error in PII masking: {e}")
            return text  # Fallback to original if error
    
    @log_performance
    def load_audio_features(self, audio_path: str, n_mfcc: int = 40) -> Dict:
        """
        Load and extract raw audio features optimized for CNN+LSTM.
        
        Args:
            audio_path: Path to audio file
            n_mfcc: Number of MFCC coefficients (13-40, default: 40)
        
        Returns:
            Dictionary with:
            - mfcc: (n_mfcc, time_frames) array
            - mel_spectrogram: (n_mels, time_frames) array (2D format)
            - chroma: (12, time_frames) array
            - Other spectral features
        """
        # Load audio - this is critical, so if it fails, return empty dict
        try:
            y, sr = librosa.load(audio_path, sr=16000)  # Resample to 16kHz for consistency
        except Exception as e:
            logger.error(f"Failed to load audio file {audio_path}: {e}")
            return {}  # Can't proceed without audio data
        
        # Extract core features (MFCC and Mel-Spectrogram) - these are REQUIRED
        try:
            # MFCC: 13-40 coefficients (per guide.txt requirement)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
            
            # Mel-Spectrogram: 2D format for CNN input
            mel_spec = librosa.feature.melspectrogram(
                y=y, sr=sr,
                n_mels=128,  # Standard for emotion recognition
                fmax=8000,
                hop_length=512
            )
            # Convert to dB scale (log-mel) with absolute reference
            mel_spec_db = librosa.power_to_db(mel_spec, ref=1.0)
        except Exception as e:
            logger.error(f"Failed to extract core features (MFCC/Mel) from {audio_path}: {e}")
            return {}  # Can't proceed without these
        
        # Pitch (F0) extraction - OPTIONAL, can fail without breaking the function
        try:
            f0_result = librosa.pyin(
                y,
                fmin=librosa.note_to_hz('C2'),  # ~65 Hz (lowest human voice)
                fmax=librosa.note_to_hz('C7'),  # ~2093 Hz (highest human voice)
                frame_length=2048,
                hop_length=512
            )
            # Handle different return signatures across librosa versions
            if isinstance(f0_result, tuple):
                f0 = f0_result[0]
            else:
                f0 = f0_result

            # Replace NaN values with 0 (unvoiced frames)
            f0 = np.nan_to_num(f0, nan=0.0)

            # Calculate pitch statistics for voiced frames only
            if np.any(f0 > 0):
                voiced_f0 = f0[f0 > 0]
                pitch_stats = {
                    'mean': float(np.mean(voiced_f0)),
                    'std': float(np.std(voiced_f0)),
                    'min': float(np.min(voiced_f0)),
                    'max': float(np.max(voiced_f0)),
                    'median': float(np.median(voiced_f0)),
                }
            else:
                pitch_stats = {
                    'mean': 0.0,
                    'std': 0.0,
                    'min': 0.0,
                    'max': 0.0,
                    'median': 0.0,
                }
        except Exception as e:
            # PYIN can fail on some files - log warning but continue with zero pitch
            logger.warning(
                f"PYIN pitch extraction failed for {audio_path}: {e}. "
                "Using zero-valued pitch features instead."
            )
            # Use a zero contour aligned with mel time frames
            f0 = np.zeros(mel_spec_db.shape[1], dtype=np.float32)
            pitch_stats = {
                'mean': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0,
                'median': 0.0,
            }
        
        # Extract additional features - these are also optional
        try:
            # Chroma features (optimized)
            chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=512)
            
            # Existing spectral features
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
            duration = librosa.get_duration(y=y, sr=sr)
            
            # Energy envelope (RMS energy over time)
            energy = librosa.feature.rms(y=y, hop_length=512)[0]
            
            # Speaking rate approximation (syllables per second)
            zcr_mean = np.mean(zero_crossing_rate)
            speaking_rate = zcr_mean * 2.0  # Approximate scaling factor
            
            # Formant frequencies (F1, F2, F3) using spectral peaks
            spectral_envelope = np.mean(mel_spec_db, axis=0)  # Average across frequency
            try:
                peak_result = librosa.util.peak_pick(spectral_envelope, pre_max=3, post_max=3, pre_avg=3, post_avg=5, delta=0.1, wait=10)
                # Handle different return types across librosa versions
                if isinstance(peak_result, tuple):
                    peaks = peak_result[0]
                else:
                    peaks = peak_result
            except Exception:
                peaks = np.array([], dtype=int)
            
            formant_freqs = []
            if len(peaks) > 0:
                # Convert peak indices to frequencies (simplified)
                freqs = librosa.mel_frequencies(n_mels=128, fmax=8000)
                for i, peak_idx in enumerate(peaks[:3]):  # Top 3 peaks
                    if peak_idx < len(freqs):
                        formant_freqs.append(float(freqs[peak_idx]))
            # Pad to 3 formants if needed
            while len(formant_freqs) < 3:
                formant_freqs.append(0.0)
        except Exception as e:
            # If additional features fail, use defaults but still return core features
            logger.warning(f"Some additional features failed for {audio_path}: {e}. Using defaults.")
            chroma = np.zeros((12, mel_spec_db.shape[1]))
            spectral_centroid = np.zeros((1, mel_spec_db.shape[1]))
            spectral_rolloff = np.zeros((1, mel_spec_db.shape[1]))
            zero_crossing_rate = np.zeros((1, mel_spec_db.shape[1]))
            duration = librosa.get_duration(y=y, sr=sr) if 'y' in locals() else 0.0
            energy = np.zeros(mel_spec_db.shape[1])
            speaking_rate = 0.0
            formant_freqs = [0.0, 0.0, 0.0]
        
        # Always return at least the core features (mfcc and mel_spectrogram)
        return {
            'mfcc': mfcc,  # (n_mfcc, time_frames) - REQUIRED
            'mel_spectrogram': mel_spec_db,  # (n_mels, time_frames) - REQUIRED
            'chroma': chroma,
            'spectral_centroid': spectral_centroid,
            'spectral_rolloff': spectral_rolloff,
            'zero_crossing_rate': zero_crossing_rate,
            'duration': duration,
            'sample_rate': sr,
            'pitch': f0,  # (time_frames,) - F0 contour
            'pitch_stats': pitch_stats,  # Dictionary with pitch statistics
            'energy': energy,  # (time_frames,) - RMS energy envelope
            'speaking_rate': speaking_rate,  # Scalar - syllables per second approximation
            'formants': formant_freqs[:3]  # List of 3 formant frequencies [F1, F2, F3]
        }
    
    def extract_audio_features(self, audio_features: Dict) -> np.ndarray:
        """
        Extract numerical features from audio data for fusion.
        Now handles 13-40 MFCC coefficients and Mel-Spectrogram.
        
        Args:
            audio_features: Dictionary of audio features.
            
        Returns:
            Numpy array of audio features.
        """
        if not audio_features:
            logger.warning("Empty audio features provided")
            return np.zeros(100)  # Increased default size
        
        features = []
        
        # MFCC features (now 13-40 coefficients)
        if 'mfcc' in audio_features:
            mfcc = audio_features['mfcc']
            n_mfcc = mfcc.shape[0]  # Dynamic based on extraction
            features.extend([
                np.mean(mfcc, axis=1),  # Mean of each MFCC coefficient
                np.std(mfcc, axis=1)    # Std of each MFCC coefficient
            ])
        
        # Mel-Spectrogram statistics (for fusion layer)
        if 'mel_spectrogram' in audio_features:
            mel_spec = audio_features['mel_spectrogram']
            features.extend([
                np.mean(mel_spec, axis=1),  # Mean across time
                np.std(mel_spec, axis=1),   # Std across time
                np.mean(mel_spec),          # Global mean
                np.std(mel_spec)            # Global std
            ])
        
        # Spectral features
        if 'spectral_centroid' in audio_features:
            sc = audio_features['spectral_centroid']
            features.extend([np.mean(sc), np.std(sc)])
        
        if 'spectral_rolloff' in audio_features:
            sr = audio_features['spectral_rolloff']
            features.extend([np.mean(sr), np.std(sr)])
        
        if 'zero_crossing_rate' in audio_features:
            zcr = audio_features['zero_crossing_rate']
            features.extend([np.mean(zcr), np.std(zcr)])
        
        # Chroma features
        if 'chroma' in audio_features:
            chroma = audio_features['chroma']
            features.extend([
                np.mean(chroma, axis=1),  # Mean chroma values
                np.std(chroma, axis=1)    # Std chroma values
            ])
        
        # Duration and sample rate
        features.extend([
            audio_features.get('duration', 0),
            audio_features.get('sample_rate', 0) / 1000  # Normalize sample rate
        ])
        
        # Pitch statistics (5 features: mean, std, min, max, median)
        pitch_stats = audio_features.get('pitch_stats', {})
        if pitch_stats:
            features.extend([
                pitch_stats.get('mean', 0.0),
                pitch_stats.get('std', 0.0),
                pitch_stats.get('min', 0.0),
                pitch_stats.get('max', 0.0),
                pitch_stats.get('median', 0.0)
            ])
        else:
            features.extend([0.0, 0.0, 0.0, 0.0, 0.0])
        
        # Energy statistics (2 features: mean, std)
        energy = audio_features.get('energy', None)
        if energy is not None and len(energy) > 0:
            features.extend([
                np.mean(energy),
                np.std(energy)
            ])
        else:
            features.extend([0.0, 0.0])
        
        # Speaking rate (1 feature)
        speaking_rate = audio_features.get('speaking_rate', 0.0)
        features.append(float(speaking_rate))
        
        # Flatten and return
        flat_features = np.concatenate([f.flatten() if hasattr(f, 'flatten') else [f] for f in features])
        return flat_features
    
    def extract_text_features(self, text_features: Dict) -> np.ndarray:
        """
        Extract BERT embeddings from text data (FR-4: BERT for sentiment).
        
        Args:
            text_features: Dictionary with 'text' key (raw text).
            
        Returns:
            Numpy array of BERT embeddings.
        """
        text = self.mask_pii(text_features.get('text', ''))  # Mask PII first
        if not text:
            logger.warning("Empty text provided")
            return np.zeros(768)  # BERT embedding size
        
        try:
            inputs = self.bert_tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding=True)
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
            # Use [CLS] token embedding
            embedding = outputs.last_hidden_state[:, 0, :].numpy().flatten()
            return embedding
        except Exception as e:
            logger.error(f"Error in BERT extraction: {e}")
            return np.zeros(768)
    
    def extract_conversational_dynamics(self, segments: List[Dict], 
                                       total_duration: float) -> Dict:
        """
        Extract conversational dynamics features required for XGBoost.
        
        Per guide.txt:
        - Silence Ratio (Total silence / Call duration)
        - Interruption Frequency (Overlapping speech counts)
        - Talk-to-Listen Ratio (Agent speaking time / Customer speaking time)
        - Filler Word Frequency (Enhanced: FR5.2)
        
        Args:
            segments: List of segments with start_time, end_time, speaker, text
            total_duration: Total call duration in seconds
        
        Returns:
            Dictionary with dynamics metrics including filler word frequency
        """
        if not segments or total_duration <= 0:
            return {
                'silence_ratio': 0.0,
                'interruption_frequency': 0.0,
                'talk_listen_ratio': 1.0,
                'turn_taking_frequency': 0.0,
                'filler_word_frequency': 0.0
            }
        
        # Calculate total speaking time
        speaking_time = sum(
            seg.get('end_time', 0) - seg.get('start_time', 0)
            for seg in segments
        )
        
        # 1. Silence Ratio
        silence_time = total_duration - speaking_time
        silence_ratio = silence_time / total_duration if total_duration > 0 else 0.0
        
        # 2. Interruption Frequency (overlapping speech)
        interruptions = self._count_interruptions(segments)
        interruption_frequency = interruptions / total_duration if total_duration > 0 else 0.0
        
        # 3. Talk-to-Listen Ratio
        agent_time = sum(
            seg.get('end_time', 0) - seg.get('start_time', 0)
            for seg in segments
            if seg.get('speaker', '').lower() in ['agent', 'speaker_0']
        )
        customer_time = sum(
            seg.get('end_time', 0) - seg.get('start_time', 0)
            for seg in segments
            if seg.get('speaker', '').lower() in ['customer', 'speaker_1']
        )
        talk_listen_ratio = agent_time / customer_time if customer_time > 0 else 1.0
        
        # 4. Speaker Turn-taking Frequency
        turn_taking_frequency = self._calculate_turn_taking_frequency(segments, total_duration)
        
        # 5. Filler Word Frequency (Enhanced: FR5.2)
        filler_word_frequency = self._calculate_filler_word_frequency(segments, total_duration)
        
        return {
            'silence_ratio': silence_ratio,
            'interruption_frequency': interruption_frequency,
            'talk_listen_ratio': talk_listen_ratio,
            'turn_taking_frequency': turn_taking_frequency,
            'filler_word_frequency': filler_word_frequency
        }
    
    def _count_interruptions(self, segments: List[Dict]) -> int:
        """
        Count overlapping speech segments (interruptions).
        
        Two segments overlap if:
        - seg1.end_time > seg2.start_time AND seg1.start_time < seg2.end_time
        - AND they have different speakers
        """
        interruptions = 0
        for i in range(len(segments)):
            for j in range(i + 1, len(segments)):
                seg1 = segments[i]
                seg2 = segments[j]
                
                # Check if different speakers
                if seg1.get('speaker') == seg2.get('speaker'):
                    continue
                
                # Check for overlap
                seg1_start = seg1.get('start_time', 0)
                seg1_end = seg1.get('end_time', 0)
                seg2_start = seg2.get('start_time', 0)
                seg2_end = seg2.get('end_time', 0)
                
                if (seg1_end > seg2_start and seg1_start < seg2_end):
                    interruptions += 1
        
        return interruptions
    
    def _calculate_turn_taking_frequency(self, segments: List[Dict], 
                                         total_duration: float) -> float:
        """
        Calculate speaker turn-taking frequency (transitions per minute).
        
        Returns: Number of speaker changes per minute
        """
        if len(segments) < 2 or total_duration <= 0:
            return 0.0
        
        speaker_changes = sum(
            1 for i in range(1, len(segments))
            if segments[i].get('speaker') != segments[i-1].get('speaker')
        )
        
        # Convert to frequency (changes per minute)
        turn_taking_frequency = (speaker_changes / total_duration) * 60.0
        
        return turn_taking_frequency
    
    def _calculate_filler_word_frequency(self, segments: List[Dict], 
                                         total_duration: float) -> float:
        """
        Calculate filler word frequency (filler words per minute).
        
        Detects common filler words: um, uh, like, you know, well, so, actually, basically, etc.
        
        Args:
            segments: List of segments with text field
            total_duration: Total call duration in seconds
        
        Returns:
            Filler word frequency (filler words per minute)
        """
        if not segments or total_duration <= 0:
            return 0.0
        
        # Common filler words and phrases (case-insensitive matching)
        filler_words = [
            'um', 'uh', 'er', 'ah', 'eh',  # Hesitation sounds
            'like', 'you know', 'you see', 'i mean',  # Discourse markers
            'well', 'so', 'actually', 'basically', 'literally',  # Filler phrases
            'kind of', 'sort of', 'pretty much',  # Approximators
            'right', 'okay', 'ok', 'yeah', 'yep', 'yup'  # Backchannel fillers
        ]
        
        total_filler_count = 0
        total_words = 0
        
        for segment in segments:
            text = segment.get('text', '')
            if not text or not isinstance(text, str):
                continue
            
            # Convert to lowercase for matching
            text_lower = text.lower()
            
            # Count filler words (word boundaries to avoid partial matches)
            import re
            for filler in filler_words:
                # Use word boundaries to match whole words only
                pattern = r'\b' + re.escape(filler) + r'\b'
                matches = len(re.findall(pattern, text_lower))
                total_filler_count += matches
            
            # Count total words in segment
            words = text.split()
            total_words += len(words)
        
        # Calculate filler word frequency (fillers per minute)
        if total_duration > 0:
            filler_frequency = (total_filler_count / total_duration) * 60.0
        else:
            filler_frequency = 0.0
        
        # Also calculate filler word ratio (fillers per 100 words)
        filler_ratio = (total_filler_count / total_words * 100.0) if total_words > 0 else 0.0
        
        # Return frequency (fillers per minute) as primary metric
        # Ratio can be added to output if needed
        return filler_frequency
    
    def extract_temporal_features(self, segments: List[Dict]) -> Dict:
        """
        Extract temporal features including conversational dynamics.
        
        Args:
            segments: List of conversation segments (from diarization, FR-3).
            
        Returns:
            Dictionary of temporal features.
        """
        if not segments:
            logger.warning("Empty segments provided")
            return {}
        
        try:
            total_duration = max([s.get('end_time', 0) for s in segments])
            
            # Existing metrics
            speaker_changes = len([i for i in range(1, len(segments)) 
                                  if segments[i].get('speaker') != segments[i-1].get('speaker')])
            
            # Calculate speaking time by speaker
            speaker_times = {}
            for segment in segments:
                speaker = segment.get('speaker', 'Unknown')
                duration = segment.get('end_time', 0) - segment.get('start_time', 0)
                speaker_times[speaker] = speaker_times.get(speaker, 0) + duration
            
            # Sentiment progression (existing)
            sentiment_scores = []
            for segment in segments:
                text = self.mask_pii(segment.get('text', ''))
                bert_embedding = self.extract_text_features({'text': text})
                sentiment = self.sentiment_classifier(bert_embedding)  # Use classifier
                sentiment_scores.append(sentiment)
            
            # NEW: Conversational dynamics
            dynamics = self.extract_conversational_dynamics(segments, total_duration)
            
            temporal_features = {
                'total_duration': total_duration,
                'speaker_changes': speaker_changes,
                'avg_segment_duration': total_duration / len(segments) if segments else 0,
                'speaker_balance': min(speaker_times.values()) / max(speaker_times.values()) if speaker_times else 0,
                'sentiment_trend': np.polyfit(range(len(sentiment_scores)), sentiment_scores, 1)[0] if len(sentiment_scores) > 1 else 0,
                'sentiment_volatility': np.std(sentiment_scores) if sentiment_scores else 0,
                'final_sentiment': sentiment_scores[-1] if sentiment_scores else 0,
                # NEW: Conversational dynamics metrics
                'silence_ratio': dynamics['silence_ratio'],
                'interruption_frequency': dynamics['interruption_frequency'],
                'talk_listen_ratio': dynamics['talk_listen_ratio'],
                'turn_taking_frequency': dynamics['turn_taking_frequency']
            }
            
            return temporal_features
        except Exception as e:
            logger.error(f"Error in temporal features: {e}")
            return {}
    
    def combine_features(self, audio_features: np.ndarray, text_features: np.ndarray, 
                         temporal_features: Dict) -> np.ndarray:
        """
        Combine all feature types into a single feature vector (FR-6: fused features).
        
        Args:
            audio_features: Audio feature vector.
            text_features: Text feature vector.
            temporal_features: Dictionary of temporal features.
            
        Returns:
            Combined feature vector.
        """
        # Convert temporal features to array
        temporal_array = np.array(list(temporal_features.values()))
        
        # Combine all features
        combined = np.concatenate([audio_features, text_features, temporal_array])
        
        return combined
    
    def fit_transform(self, feature_data: List[Dict]) -> np.ndarray:
        """
        Fit scalers and transform features (optimized for batches).
        
        Args:
            feature_data: List of feature dictionaries.
            
        Returns:
            Transformed feature matrix.
        """
        if not feature_data:
            return np.array([])
        
        # Vectorized extraction for performance
        all_features = np.array([self.combine_features(
            self.extract_audio_features(d.get('audio_features', {})),
            self.extract_text_features(d.get('text_features', {})),
            self.extract_temporal_features(d.get('segments', []))
        ) for d in feature_data])
        
        # Fit scalers if not already fitted
        if not self.is_fitted:
            self.audio_scaler.fit(all_features)
            self.is_fitted = True
        
        # Scale features
        scaled_features = self.audio_scaler.transform(all_features)
        
        return scaled_features
    
    def transform(self, feature_data: Dict) -> np.ndarray:
        """
        Transform single sample features.
        
        Args:
            feature_data: Feature dictionary for single sample.
            
        Returns:
            Transformed feature vector.
        """
        audio_feat = self.extract_audio_features(feature_data.get('audio_features', {}))
        text_feat = self.extract_text_features(feature_data.get('text_features', {}))
        temporal_feat = self.extract_temporal_features(feature_data.get('segments', []))
        
        combined = self.combine_features(audio_feat, text_feat, temporal_feat)
        
        # Reshape for scaler
        combined = combined.reshape(1, -1)
        
        if self.is_fitted:
            scaled = self.audio_scaler.transform(combined)
            return scaled.flatten()
        else:
            return combined.flatten()
    
    def save_features(self, features: np.ndarray, call_id: str):
        """
        Save features to MongoDB (FR-7: store data in PostgreSQL/MongoDB).
        
        Args:
            features: Feature vector.
            call_id: Unique call identifier.
        """
        if not MONGO_ENABLED:
            logger.debug("MongoDB disabled - skipping feature persistence")
            return
        try:
            client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=2000)
            db = client[MONGO_DB_NAME]
            collection = db['features']
            collection.insert_one({
                'call_id': call_id,
                'features': features.tolist(),
                'timestamp': datetime.now()
            })
            logger.info(f"Features saved for call_id: {call_id}")
        except Exception as e:
            logger.error(f"Error saving features: {e}")
    
    def extract_segment_features(self, audio_path: str, start_time: float, end_time: float) -> Dict:
        """
        Extract features for a specific time window from audio file.
        
        This method uses the segment extraction functions to extract features
        for a specific time segment, enabling per-segment emotion detection.
        
        Args:
            audio_path: Path to audio file
            start_time: Start time of segment in seconds
            end_time: End time of segment in seconds
        
        Returns:
            Dictionary of audio features for the segment
        """
        return extract_segment_features(audio_path, start_time, end_time, self)
    
    def get_feature_names(self, n_mfcc: int = 40) -> List[str]:
        """
        Get names of all features for interpretability.
        
        Args:
            n_mfcc: Number of MFCC coefficients used (default: 40)
        
        Returns:
            List of feature names
        """
        feature_names = []
        
        # Audio features - MFCC (13-40 coefficients)
        feature_names.extend([f'mfcc_mean_{i}' for i in range(n_mfcc)])
        feature_names.extend([f'mfcc_std_{i}' for i in range(n_mfcc)])
        
        # Mel-Spectrogram statistics (NEW)
        feature_names.extend([f'mel_spec_mean_{i}' for i in range(128)])  # 128 mel bands
        feature_names.extend([f'mel_spec_std_{i}' for i in range(128)])
        feature_names.extend(['mel_spec_global_mean', 'mel_spec_global_std'])
        
        # Existing spectral features
        feature_names.extend(['spectral_centroid_mean', 'spectral_centroid_std'])
        feature_names.extend(['spectral_rolloff_mean', 'spectral_rolloff_std'])
        feature_names.extend(['zcr_mean', 'zcr_std'])
        feature_names.extend([f'chroma_mean_{i}' for i in range(12)])
        feature_names.extend([f'chroma_std_{i}' for i in range(12)])
        feature_names.extend(['duration', 'sample_rate'])
        
        # BERT text features (768 dimensions)
        feature_names.extend([f'bert_embedding_{i}' for i in range(768)])
        
        # Temporal features (including NEW conversational dynamics)
        feature_names.extend([
            'total_duration', 'speaker_changes', 'avg_segment_duration',
            'speaker_balance', 'sentiment_trend', 'sentiment_volatility', 
            'final_sentiment',
            # NEW: Conversational dynamics
            'silence_ratio', 'interruption_frequency', 
            'talk_listen_ratio', 'turn_taking_frequency'
        ])
        
        return feature_names


# ============================================================================
# Segment Extraction Functions for Per-Segment Emotion Detection
# ============================================================================

def extract_segment_mel_spectrogram(full_mel: np.ndarray, start_time: float, end_time: float,
                                   sample_rate: int = 16000, hop_length: int = 512) -> np.ndarray:
    """
    Extract segment-specific mel-spectrogram from full audio mel-spectrogram.
    
    This function extracts the time frames corresponding to a specific segment
    from the full conversation mel-spectrogram, enabling per-segment emotion detection.
    
    Args:
        full_mel: Full mel-spectrogram array (n_mels, total_time_frames)
        start_time: Start time of segment in seconds
        end_time: End time of segment in seconds
        sample_rate: Audio sample rate (default: 16000)
        hop_length: Hop length used in mel-spectrogram computation (default: 512)
    
    Returns:
        Segment mel-spectrogram (n_mels, segment_time_frames)
    """
    # Convert time to frame indices
    # frames = time * sample_rate / hop_length
    start_frame = int(start_time * sample_rate / hop_length)
    end_frame = int(end_time * sample_rate / hop_length)
    
    # Ensure valid indices
    start_frame = max(0, min(start_frame, full_mel.shape[1]))
    end_frame = max(start_frame, min(end_frame, full_mel.shape[1]))
    
    if start_frame >= end_frame:
        # Empty segment, return zero array with same frequency dimension
        return np.zeros((full_mel.shape[0], 1))
    
    # Extract segment
    segment_mel = full_mel[:, start_frame:end_frame]
    
    return segment_mel


def extract_segment_features(audio_path: str, start_time: float, end_time: float,
                            feature_extractor: Optional['FeatureExtractor'] = None) -> Dict:
    """
    Extract features for a specific time window from audio file.
    
    This is a wrapper function that loads the full audio, extracts the segment,
    and computes features for that segment only.
    
    Args:
        audio_path: Path to audio file
        start_time: Start time of segment in seconds
        end_time: End time of segment in seconds
        feature_extractor: Optional FeatureExtractor instance (creates new if None)
    
    Returns:
        Dictionary of audio features for the segment
    """
    if feature_extractor is None:
        feature_extractor = FeatureExtractor()
    
    # Load full audio features
    full_features = feature_extractor.load_audio_features(audio_path)
    
    if 'mel_spectrogram' not in full_features:
        logger.warning(f"Could not extract mel-spectrogram from {audio_path}")
        return {}
    
    # Extract segment mel-spectrogram
    full_mel = full_features['mel_spectrogram']
    sample_rate = full_features.get('sample_rate', 16000)
    hop_length = 512  # Standard hop length used in load_audio_features
    
    segment_mel = extract_segment_mel_spectrogram(
        full_mel, start_time, end_time, sample_rate, hop_length
    )
    
    # Create segment features dictionary
    segment_features = full_features.copy()
    segment_features['mel_spectrogram'] = segment_mel
    segment_features['duration'] = end_time - start_time
    
    return segment_features


# ============================================================================
# SpecAugment Functions for Data Augmentation
# ============================================================================

def apply_specaugment_time_mask(mel_spec: np.ndarray, num_masks: int = 2, 
                                mask_size: int = 27) -> np.ndarray:
    """
    Apply time masking to mel-spectrogram (SpecAugment).
    
    Masks consecutive time steps to improve model robustness to temporal variations.
    
    Args:
        mel_spec: Mel-spectrogram array (n_mels, time_frames)
        num_masks: Number of time masks to apply (default: 2)
        mask_size: Maximum size of each mask in time frames (default: 27)
    
    Returns:
        Augmented mel-spectrogram with same shape
    """
    mel_spec_aug = mel_spec.copy()
    n_mels, time_frames = mel_spec.shape
    
    for _ in range(num_masks):
        # Random mask size (0 to mask_size)
        t = np.random.randint(0, mask_size + 1)
        if t == 0:
            continue
        
        # Random start position
        t0 = np.random.randint(0, time_frames - t + 1)
        
        # Apply mask (set to minimum value)
        mel_spec_aug[:, t0:t0+t] = mel_spec_aug.min()
    
    return mel_spec_aug


def apply_specaugment_freq_mask(mel_spec: np.ndarray, num_masks: int = 2,
                                mask_size: int = 13) -> np.ndarray:
    """
    Apply frequency masking to mel-spectrogram (SpecAugment).
    
    Masks consecutive mel bands to improve model robustness to frequency variations.
    
    Args:
        mel_spec: Mel-spectrogram array (n_mels, time_frames)
        num_masks: Number of frequency masks to apply (default: 2)
        mask_size: Maximum size of each mask in mel bands (default: 13)
    
    Returns:
        Augmented mel-spectrogram with same shape
    """
    mel_spec_aug = mel_spec.copy()
    n_mels, time_frames = mel_spec.shape
    
    for _ in range(num_masks):
        # Random mask size (0 to mask_size)
        f = np.random.randint(0, mask_size + 1)
        if f == 0:
            continue
        
        # Random start position
        f0 = np.random.randint(0, n_mels - f + 1)
        
        # Apply mask (set to minimum value)
        mel_spec_aug[f0:f0+f, :] = mel_spec_aug.min()
    
    return mel_spec_aug


def apply_specaugment(mel_spec: np.ndarray, time_masks: int = 2, freq_masks: int = 2,
                     time_mask_size: int = 27, freq_mask_size: int = 13) -> np.ndarray:
    """
    Apply complete SpecAugment augmentation (time + frequency masking).
    
    SpecAugment is a data augmentation technique that masks time and frequency
    dimensions, improving model robustness. This is more realistic than Gaussian noise
    for acoustic data.
    
    Args:
        mel_spec: Mel-spectrogram array (n_mels, time_frames)
        time_masks: Number of time masks to apply (default: 2)
        freq_masks: Number of frequency masks to apply (default: 2)
        time_mask_size: Maximum size of time mask in frames (default: 27)
        freq_mask_size: Maximum size of frequency mask in mel bands (default: 13)
    
    Returns:
        Augmented mel-spectrogram with same shape
    """
    mel_spec_aug = mel_spec.copy()
    
    # Apply time masking
    if time_masks > 0:
        mel_spec_aug = apply_specaugment_time_mask(mel_spec_aug, time_masks, time_mask_size)
    
    # Apply frequency masking
    if freq_masks > 0:
        mel_spec_aug = apply_specaugment_freq_mask(mel_spec_aug, freq_masks, freq_mask_size)
    
    return mel_spec_aug