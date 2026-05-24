"""
Data preprocessing module for audio and text processing, aligned with SRS1 (Purely ML-Based System).
Handles audio loading (FR-1), Whisper transcription (FR-2), Pyannote diarization (FR-3),
audio feature extraction (FR-5), text preprocessing for BERT (FR-4), and MongoDB storage (FR-7).
"""

import os
import torch
import librosa
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import whisper
import unicodedata
import soundfile as sf
from pydub import AudioSegment
import logging
from jiwer import wer
import spacy
from pymongo import MongoClient
from datetime import datetime
from transformers import BertTokenizer
import warnings

# Suppress noisy but harmless third-party warnings as early as possible,
# before importing libraries that may emit them during import.
warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")
warnings.filterwarnings("ignore", message=".*torchaudio.*deprecated.*")
# pyannote / torchcodec on Windows – decoding fallback warnings
warnings.filterwarnings("ignore", message=".*torchcodec is not installed correctly.*")

from pyannote.audio import Pipeline

_BERT_TOKENIZER = None
from concurrent.futures import ThreadPoolExecutor

# WhisperX + Resemblyzer imports (CPU-friendly alternative)
try:
    import whisperx
    WHISPERX_AVAILABLE = True
except ImportError:
    WHISPERX_AVAILABLE = False
    whisperx = None

try:
    from resemblyzer import VoiceEncoder, preprocess_wav
    from pathlib import Path
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import pdist
    RESEMBLYZER_AVAILABLE = True
except ImportError:
    RESEMBLYZER_AVAILABLE = False
    VoiceEncoder = None
    preprocess_wav = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mongo configuration (disabled by default unless explicitly enabled)
MONGO_ENABLED = os.getenv('MONGODB_ENABLED', 'false').lower() == 'true'
MONGO_URI = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/')
MONGO_DB_NAME = os.getenv('MONGODB_DATABASE', 'call_center_db')

# Load spaCy for PII masking (optional)
try:
    nlp = spacy.load('en_core_web_sm')
    SPACY_AVAILABLE = True
    logger.info("spaCy model loaded successfully")
    from spacy.lang.en.stop_words import STOP_WORDS as SPACY_STOP_WORDS
except OSError:
    nlp = None
    SPACY_AVAILABLE = False
    SPACY_STOP_WORDS = set()
    logger.warning("spaCy model 'en_core_web_sm' not found. PII masking will be limited.")


class AudioProcessor:
    """Handles audio preprocessing including transcription and speaker diarization."""
    
    def __init__(self, model_size: str = "base", hf_token: str = None, use_whisperx: bool = True, 
                 chunk_duration: float = 300.0, enable_chunking: bool = True,
                 max_speakers: int = None, min_speakers: int = None, num_speakers: int = None,
                 clustering_threshold: float = 0.3, min_segment_duration: float = 1.0,
                 speaker_merge_threshold: float = 0.7, use_whisperx_builtin_diarization: bool = False):
        """
        Initialize audio processor.
        
        Args:
            model_size: Whisper model size (tiny, base, small, medium, large).
            hf_token: Hugging Face token for Pyannote.audio (required for WhisperX 3.x built-in diarization).
            use_whisperx: Use WhisperX + Resemblyzer (faster, CPU-friendly) instead of Pyannote.
            chunk_duration: Duration of each chunk in seconds (default: 300 = 5 minutes).
            enable_chunking: Enable chunking for long audio files (default: True).
            max_speakers: Maximum number of speakers (None = auto-detect). Used by Pyannote and Resemblyzer.
            min_speakers: Minimum number of speakers (None = no constraint). Pyannote only; cannot mix with num_speakers.
            num_speakers: Exact number of speakers (None = auto). Pyannote only; cannot mix with min/max_speakers.
            clustering_threshold: Distance threshold for speaker clustering (0.0-1.0, lower = more clusters).
            min_segment_duration: Minimum segment duration in seconds for reliable embeddings (default: 1.0).
            use_whisperx_builtin_diarization: Use WhisperX 3.x built-in diarization (Pyannote.audio) instead of Resemblyzer.
                                             More accurate but slower. Requires HF_TOKEN.
        """
        self.model_size = model_size
        self.hf_token = hf_token
        self.use_whisperx_builtin_diarization = use_whisperx_builtin_diarization
        # Use WhisperX built-in if requested, otherwise use Resemblyzer
        if use_whisperx_builtin_diarization:
            self.use_whisperx = use_whisperx_builtin_diarization and WHISPERX_AVAILABLE and hf_token is not None
        else:
            self.use_whisperx = use_whisperx and WHISPERX_AVAILABLE and RESEMBLYZER_AVAILABLE
        self.chunk_duration = chunk_duration
        self.enable_chunking = enable_chunking
        self.max_speakers = max_speakers
        self.min_speakers = min_speakers
        self.num_speakers = num_speakers
        self.clustering_threshold = clustering_threshold
        self.min_segment_duration = min_segment_duration
        self.whisper_model = None
        self.diarization_pipeline = None
        self.whisperx_model = None
        self.voice_encoder = None
        self.whisperx_align_models = {}  # cache per language
        self.whisperx_diarize_model = None  # WhisperX 3.x built-in diarization model
        self.target_sample_rate = 16000
        self.speaker_merge_threshold = speaker_merge_threshold
        self.mongo_enabled = MONGO_ENABLED
        self.mongo_uri = MONGO_URI
        self.mongo_db = MONGO_DB_NAME
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Inference device: {self._device}")
        self._load_models()
    
    def _load_models(self):
        """Load Whisper and diarization models (FR-2, FR-3)."""
        # Load Whisper (required for transcription)
        try:
            logger.info(f"Loading Whisper model: {self.model_size}")
            self.whisper_model = whisper.load_model(self.model_size)
            logger.info(f"Whisper model '{self.model_size}' loaded successfully")
        except Exception as e:
            logger.error(f"Whisper model loading failed: {e}")
            logger.warning("Whisper transcription will not be available")
            self.whisper_model = None
        
        # Load diarization model
        if self.use_whisperx_builtin_diarization and self.hf_token:
            try:
                logger.info("Loading WhisperX 3.x built-in diarization (Pyannote.audio, more accurate)")
                # WhisperX model will be loaded on first use
                self.whisperx_model = None  # Lazy load
                # WhisperX diarization model will be loaded on first use
                self.whisperx_diarize_model = None  # Lazy load
                logger.info("✅ WhisperX 3.x built-in diarization ready (will load on first use)")
                logger.info("   This uses Pyannote.audio for state-of-the-art accuracy")
            except Exception as e:
                logger.warning(f"WhisperX 3.x built-in diarization setup failed: {e}")
                logger.warning("Falling back to Resemblyzer method")
                self.use_whisperx_builtin_diarization = False
                self.use_whisperx = use_whisperx and WHISPERX_AVAILABLE and RESEMBLYZER_AVAILABLE
        
        if not self.use_whisperx_builtin_diarization and self.use_whisperx:
            try:
                logger.info("Loading WhisperX + Resemblyzer (CPU-friendly, 10x faster)")
                # WhisperX model will be loaded on first use
                self.whisperx_model = None  # Lazy load
                # Load Resemblyzer voice encoder
                self.voice_encoder = VoiceEncoder()
                logger.info("✅ WhisperX + Resemblyzer loaded successfully (CPU-optimized)")
                logger.info("   This is 10x faster than Pyannote on CPU!")
            except Exception as e:
                logger.warning(f"WhisperX + Resemblyzer loading failed: {e}")
                logger.warning("Falling back to Pyannote.audio")
                self.use_whisperx = False
                self.voice_encoder = None
        
        # Load Pyannote as fallback (if WhisperX not available or disabled)
        if not self.use_whisperx and not self.use_whisperx_builtin_diarization:
            try:
                logger.info("Loading Pyannote.audio diarization pipeline (fallback)")
                if self.hf_token:
                    self.diarization_pipeline = Pipeline.from_pretrained(
                        "pyannote/speaker-diarization-3.1",
                        token=self.hf_token,
                    )
                    if self._device == "cuda":
                        self.diarization_pipeline = self.diarization_pipeline.to(torch.device("cuda"))
                        logger.info("Pyannote pipeline moved to GPU")
                    # Tune segmentation to reduce fragmentation (call-center style: merge short pauses)
                    try:
                        _params = getattr(self.diarization_pipeline, "parameters", None)
                        if callable(_params):
                            _instantiate = getattr(self.diarization_pipeline, "instantiate", None)
                            if callable(_instantiate):
                                self.diarization_pipeline.instantiate({
                                    "segmentation": {
                                        "min_duration_on": 0.1,
                                        "min_duration_off": 0.3,
                                    }
                                })
                                logger.info("Pyannote pipeline tuned (min_duration_on=0.1s, min_duration_off=0.3s)")
                    except Exception as inst_err:
                        logger.debug(f"Pyannote pipeline.instantiate skipped: {inst_err}")
                    logger.info("Pyannote.audio diarization pipeline loaded successfully")
                else:
                    logger.warning("No Hugging Face token provided. Pyannote diarization will not be available.")
                    logger.info("Set HF_TOKEN in .env file to enable speaker diarization")
                    self.diarization_pipeline = None
            except Exception as e:
                logger.warning(f"Pyannote model loading failed: {e}")
                logger.warning("Speaker diarization will not be available, but transcription will still work")
                self.diarization_pipeline = None
    
    def validate_audio_format(self, audio_path: str) -> str:
        """
        Validate and convert audio format (FR-1: .wav, .mp3, .m4a).
        
        Args:
            audio_path: Path to audio file.
            
        Returns:
            Path to validated/converted audio.
        """
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
    
    def transcribe_audio(self, audio_path: str, call_id: str) -> Dict:
        """
        Transcribe audio file to text using Whisper ASR (FR-2: Whisper, WER ≤15%).
        Supports chunking for long audio files to save time and memory.
        
        Args:
            audio_path: Path to audio file.
            call_id: Unique call identifier for storage.
            
        Returns:
            Dictionary with transcription results.
        """
        if self.whisper_model is None:
            raise ValueError("Whisper model not loaded. Cannot transcribe audio.")
        
        try:
            # Validate and convert audio format if needed
            audio_path = self.validate_audio_format(audio_path)
            
            # Check if chunking should be used
            try:
                import librosa
                audio_duration = librosa.get_duration(path=audio_path)
                use_chunking = self.enable_chunking and audio_duration > self.chunk_duration
            except:
                use_chunking = False
                audio_duration = 0
            
            if use_chunking:
                logger.info(f"📦 Using chunking for long audio file ({audio_duration/60:.1f} minutes)")
                logger.info(f"   Chunk size: {self.chunk_duration/60:.1f} minutes")
                return self._transcribe_with_chunking(audio_path, call_id, audio_duration)
            else:
                # Standard transcription (no chunking)
                logger.info(f"Transcribing audio: {audio_path}")
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
                
                text_length = len(transcription['text'])
                logger.info(f"Transcription completed. Text length: {text_length} characters")
                
                if text_length == 0:
                    logger.warning("⚠️  Transcription is empty - no speech detected in audio file")
                
                try:
                    self.save_transcription(transcription, call_id)
                except Exception as db_error:
                    logger.warning(f"Failed to save transcription to database: {db_error}")
                
                return transcription
                
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise ValueError(f"Failed to transcribe audio: {str(e)}")
    
    def _transcribe_with_chunking(self, audio_path: str, call_id: str, audio_duration: float) -> Dict:
        """
        Transcribe audio by processing it in chunks for faster processing.
        
        Args:
            audio_path: Path to audio file.
            call_id: Unique call identifier.
            audio_duration: Total duration of audio in seconds.
            
        Returns:
            Combined transcription results from all chunks.
        """
        import librosa
        import soundfile as sf
        import tempfile
        import os
        
        logger.info(f"📦 Processing {audio_duration/60:.1f} minutes of audio in chunks")
        
        # Calculate number of chunks
        num_chunks = int(np.ceil(audio_duration / self.chunk_duration))
        logger.info(f"   Splitting into {num_chunks} chunks (~{self.chunk_duration/60:.1f} min each)")
        
        # Load audio
        audio_data, sample_rate = librosa.load(audio_path, sr=self.target_sample_rate)
        chunk_size_samples = int(self.chunk_duration * sample_rate)
        
        all_segments = []
        all_text_parts = []
        language = "en"
        
        # Process each chunk
        for chunk_idx in range(num_chunks):
            start_sample = chunk_idx * chunk_size_samples
            end_sample = min((chunk_idx + 1) * chunk_size_samples, len(audio_data))
            chunk_audio = audio_data[start_sample:end_sample]
            chunk_start_time = chunk_idx * self.chunk_duration
            
            if len(chunk_audio) < sample_rate * 0.5:  # Skip very short chunks
                continue
            
            logger.info(f"   Processing chunk {chunk_idx + 1}/{num_chunks} "
                       f"({chunk_start_time/60:.1f} - {chunk_start_time/60 + len(chunk_audio)/sample_rate/60:.1f} min)")
            
            # Save chunk to temporary file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                tmp_path = tmp_file.name
                sf.write(tmp_path, chunk_audio, sample_rate)
            
            try:
                # Transcribe chunk
                result = self.whisper_model.transcribe(
                    tmp_path,
                    language="en",
                    task="transcribe",
                    verbose=False
                )
                
                # Adjust timestamps to account for chunk offset
                chunk_segments = result.get("segments", [])
                for segment in chunk_segments:
                    segment["start"] += chunk_start_time
                    segment["end"] += chunk_start_time
                    all_segments.append(segment)
                
                chunk_text = result.get("text", "")
                if chunk_text.strip():
                    all_text_parts.append(chunk_text)
                
                if chunk_idx == 0:
                    language = result.get("language", "en")
                
            finally:
                # Clean up temporary file
                try:
                    os.unlink(tmp_path)
                except:
                    pass
        
        # Combine results
        cleaned_parts = []
        for part in all_text_parts:
            chunk_text = part.strip()
            if chunk_text and chunk_text[-1] not in ".!?":
                chunk_text += "."
            cleaned_parts.append(chunk_text)
        combined_text = " ".join(cleaned_parts).replace("  ", " ")
        transcription = {
            "text": combined_text,
            "segments": sorted(all_segments, key=lambda x: x.get("start", 0)),
            "language": language,
            "duration": audio_duration
        }
        
        text_length = len(transcription['text'])
        logger.info(f"✅ Chunked transcription completed. Text length: {text_length} characters")
        logger.info(f"   Processed {num_chunks} chunks successfully")
        
        try:
            self.save_transcription(transcription, call_id)
        except Exception as db_error:
            logger.warning(f"Failed to save transcription to database: {db_error}")
        
        return transcription
    
    def validate_transcription(self, transcription: str, reference: str) -> float:
        """
        Validate transcription quality (FR-2: WER ≤15%).
        
        Args:
            transcription: Transcribed text.
            reference: Ground-truth text.
            
        Returns:
            Word Error Rate (WER).
        """
        try:
            return wer(reference, transcription)
        except Exception as e:
            logger.error(f"WER calculation failed: {e}")
            return 1.0  # Assume worst-case
    
    def perform_speaker_diarization(self, audio_path: str, call_id: str) -> List[Dict]:
        """
        Perform speaker diarization to identify speakers (FR-3).
        Uses WhisperX + Resemblyzer (fast, CPU-friendly) or Pyannote.audio (fallback).
        REQUIRED: This is a project requirement and cannot be skipped.
        
        Args:
            audio_path: Path to audio file.
            call_id: Unique call identifier for storage.
            
        Returns:
            List of speaker segments with timestamps.
        """
        audio_path = self.validate_audio_format(audio_path)
        
        # Get audio duration for progress estimation
        try:
            import librosa
            audio_duration = librosa.get_duration(path=audio_path)
            logger.info(f"Performing speaker diarization on: {audio_path}")
            logger.info(f"Audio duration: {audio_duration:.1f} seconds ({audio_duration/60:.1f} minutes)")
        except:
            logger.info(f"Performing speaker diarization on: {audio_path}")
        
        # Choose diarization method
        if self.use_whisperx_builtin_diarization and self.hf_token:
            # WhisperX 3.x built-in diarization (Pyannote.audio)
            return self._diarize_with_whisperx_builtin(audio_path, call_id, audio_duration)
        elif self.use_whisperx and self.voice_encoder is not None:
            # WhisperX + Resemblyzer (faster, CPU-friendly)
            return self._diarize_with_whisperx_resemblyzer(audio_path, call_id, audio_duration)
        # Fallback to Pyannote.audio
        elif self.diarization_pipeline is not None:
            return self._diarize_with_pyannote(audio_path, call_id, audio_duration)
        else:
            raise ValueError(
                "Speaker diarization is required but no diarization method is available. "
                "Please install WhisperX + Resemblyzer (recommended) or set HF_TOKEN for WhisperX 3.x built-in/Pyannote.audio."
            )
    
    def _diarize_with_whisperx_resemblyzer(self, audio_path: str, call_id: str, audio_duration: float) -> List[Dict]:
        """
        Fast CPU-friendly diarization using WhisperX + Resemblyzer.
        10x faster than Pyannote on CPU. Supports chunking for long files.
        """
        try:
            # Check if chunking should be used
            use_chunking = self.enable_chunking and audio_duration > self.chunk_duration
            
            if use_chunking:
                logger.info("🚀 Using WhisperX + Resemblyzer with chunking (CPU-optimized, 10x faster)")
                logger.info(f"📦 Processing {audio_duration/60:.1f} minutes in chunks of {self.chunk_duration/60:.1f} minutes")
                return self._diarize_chunked_whisperx_resemblyzer(audio_path, call_id, audio_duration)
            
            logger.info("🚀 Using WhisperX + Resemblyzer (CPU-optimized, 10x faster)")
            estimated_minutes = max(1, int(audio_duration / 60 / 10))
            logger.info(f"⏱️  Estimated time: {estimated_minutes}-{estimated_minutes*2} minutes (vs 20-40 min with Pyannote)")
            logger.info("🔄 Processing... This is much faster than Pyannote!")
            
            # Load WhisperX model (lazy load)
            if self.whisperx_model is None:
                logger.info(f"Loading WhisperX model: {self.model_size}")
                self.whisperx_model = whisperx.load_model(self.model_size, self._device, compute_type="int8")
                logger.info(f"WhisperX model loaded on {self._device}")
            
            # Step 1: Transcribe with WhisperX (faster, word-level timestamps)
            logger.info("Step 1: Transcribing with WhisperX...")
            audio = whisperx.load_audio(audio_path)
            result = self.whisperx_model.transcribe(audio, batch_size=16)
            
            # Step 2: Align timestamps
            logger.info("Step 2: Aligning word-level timestamps...")
            language_code = result.get("language", "en")
            model_a, metadata = self._get_whisperx_align_model(language_code)
            result = whisperx.align(result["segments"], model_a, metadata, audio, device=self._device, return_char_alignments=False)
            
            # Step 3: Extract speaker embeddings using Resemblyzer
            # IMPROVED: Use transcription segments line-by-line, but merge very short ones
            logger.info("Step 3: Extracting speaker embeddings with Resemblyzer...")
            logger.info("   Using transcription segments line-by-line for better accuracy")
            
            # Load and preprocess audio for Resemblyzer
            wav = preprocess_wav(Path(audio_path))
            sample_rate = 16000  # Resemblyzer uses 16kHz
            
            # IMPROVED: Merge very short transcription segments together for better embeddings
            # This prevents too many tiny segments that cause poor clustering
            merged_segments = []
            current_merged = None
            min_merge_duration = 2.0  # Merge segments shorter than 2 seconds
            
            for segment in result["segments"]:
                start_time = segment["start"]
                end_time = segment["end"]
                duration = end_time - start_time
                text = segment.get("text", "").strip()
                
                if duration < min_merge_duration and current_merged is not None:
                    # Merge with previous segment if both are short
                    current_merged["end"] = end_time
                    current_merged["text"] = (current_merged.get("text", "") + " " + text).strip()
                else:
                    # Start new segment
                    if current_merged is not None:
                        merged_segments.append(current_merged)
                    current_merged = {
                        "start": start_time,
                        "end": end_time,
                        "text": text
                    }
            
            # Add last segment
            if current_merged is not None:
                merged_segments.append(current_merged)
            
            logger.info(f"   Merged {len(result['segments'])} transcription segments into {len(merged_segments)} segments for diarization")
            
            # Get embeddings for each merged segment
            segment_embeddings = []
            segment_times = []
            segment_texts = []  # Store text for later mapping
            
            for segment in merged_segments:
                start_time = segment["start"]
                end_time = segment["end"]
                duration = end_time - start_time
                
                # PRODUCTION FIX: Use configurable minimum segment duration
                if duration < self.min_segment_duration:
                    continue
                
                # Extract audio segment (Resemblyzer expects 16kHz)
                start_sample = int(start_time * sample_rate)
                end_sample = int(end_time * sample_rate)
                
                if end_sample > len(wav):
                    end_sample = len(wav)
                if start_sample >= len(wav):
                    continue
                
                segment_audio = wav[start_sample:end_sample]
                
                # PRODUCTION FIX: Use configurable minimum segment duration
                min_samples = int(self.min_segment_duration * sample_rate)
                if len(segment_audio) > min_samples:
                    try:
                        embedding = self.voice_encoder.embed_utterance(segment_audio)
                        segment_embeddings.append(embedding)
                        segment_times.append((start_time, end_time))
                        segment_texts.append(segment.get("text", ""))  # Store text for mapping
                    except Exception as e:
                        logger.warning(f"Failed to embed segment {start_time:.1f}-{end_time:.1f}: {e}")
                        continue
            
            if len(segment_embeddings) < 2:
                logger.warning("Not enough segments for diarization. Using single speaker.")
                return [{
                    "speaker": "SPEAKER_00",
                    "start": 0.0,
                    "end": audio_duration,
                    "text": ""
                }]
            
            # Step 4: Cluster speakers (PRODUCTION FIX: Improved clustering)
            logger.info("Step 4: Clustering speakers...")
            segment_embeddings = np.array(segment_embeddings)
            
            if len(segment_embeddings) < 2:
                logger.warning("Not enough segments for clustering. Using single speaker.")
                cluster_labels = np.array([1])
            else:
                # Calculate distance matrix
                distances = pdist(segment_embeddings, metric='cosine')
                
                # Hierarchical clustering
                linkage_matrix = linkage(distances, method='average')
                
                # PRODUCTION FIX 1: Use distance threshold with adaptive adjustment
                # Start with configured threshold
                threshold = self.clustering_threshold
                
                # Try to get desired number of speakers
                cluster_labels = fcluster(linkage_matrix, threshold, criterion='distance')
                num_clusters = len(set(cluster_labels))
                
                # PRODUCTION FIX 2: Use clustering threshold to naturally determine speakers
                # The threshold determines how similar embeddings need to be to be the same speaker
                # Lower threshold = more speakers, higher threshold = fewer speakers
                # Let the system automatically determine the optimal number
                
                # Only apply max_speakers as a safety check (warn if exceeded, but don't force)
                if self.max_speakers is not None and num_clusters > self.max_speakers:
                    logger.warning(f"Detected {num_clusters} speakers (exceeds max_speakers={self.max_speakers}). "
                                 f"This may indicate over-segmentation. Consider increasing clustering_threshold.")
                    # Optionally: increase threshold slightly to reduce speakers
                    # But don't force - let the data speak
                    if num_clusters > self.max_speakers * 2:  # Only if way too many
                        logger.info(f"Attempting to reduce speakers by increasing threshold...")
                        threshold *= 1.3
                        cluster_labels = fcluster(linkage_matrix, threshold, criterion='distance')
                        num_clusters = len(set(cluster_labels))
                        logger.info(f"After adjustment: {num_clusters} speakers")
                
                # PRODUCTION FIX 4: Post-process to merge very similar clusters
                cluster_labels = self._merge_similar_speakers(
                    segment_embeddings, cluster_labels, similarity_threshold=self.speaker_merge_threshold
                )
                num_clusters = len(set(cluster_labels))
                
                max_info = f", max: {self.max_speakers}" if self.max_speakers is not None else ", auto-detect"
                logger.info(f"Clustering completed: {num_clusters} speakers detected "
                          f"(threshold: {threshold:.3f}{max_info})")
            
            # Step 5: Create segments with text from merged transcription segments
            segments = []
            for i, (start, end) in enumerate(segment_times):
                speaker_id = f"SPEAKER_{cluster_labels[i]-1:02d}"
                # IMPROVED: Use text from merged segments directly (avoids repetition)
                segment_text = segment_texts[i] if i < len(segment_texts) else ""
                segments.append({
                    "speaker": speaker_id,
                    "start": start,
                    "end": end,
                    "text": segment_text  # Use text from merged transcription segments
                })
            
            logger.info(f"✅ WhisperX + Resemblyzer diarization completed. Found {len(set(cluster_labels))} speakers, {len(segments)} segments")
            
            # Step 6: Text is already mapped from merged segments, but refine if needed
            # IMPROVED: Segments already have text from merged transcription segments (line-by-line)
            transcription_segments = result.get("segments", [])
            segments_with_text = sum(1 for seg in segments if seg.get('text', '').strip())
            logger.info(f"   Segments with text: {segments_with_text}/{len(segments)} (from merged transcription segments)")
            
            # If some segments are missing text, map from original transcription
            if segments_with_text < len(segments) * 0.9 and transcription_segments:
                logger.info("Step 6a: Mapping missing text from original transcription segments...")
                from call_analysis.preprocessing import TextProcessor
                text_processor = TextProcessor()
                text_processor._assign_text_to_diarization_segments(segments, transcription_segments)
                segments_with_text = sum(1 for seg in segments if seg.get('text', '').strip())
                logger.info(f"   After mapping: {segments_with_text}/{len(segments)} segments have text")
            
            # Step 6b: Identify customer vs agent roles (for call centers)
            # Segments have text, so role identification can use keyword analysis
            segments = self._identify_speaker_roles(segments, transcription_segments=transcription_segments)
            
            # Save to database if available
            try:
                self.save_diarization(segments, call_id)
            except Exception as db_error:
                logger.warning(f"Failed to save diarization to database: {db_error}")
            
            return segments
            
        except Exception as e:
            logger.error(f"WhisperX + Resemblyzer diarization failed: {e}")
            logger.warning("Falling back to Pyannote.audio...")
            # Fallback to Pyannote if available
            if self.diarization_pipeline is not None:
                return self._diarize_with_pyannote(audio_path, call_id, audio_duration)
            raise ValueError(f"Speaker diarization failed: {str(e)}")
    
    def _diarize_with_whisperx_builtin(self, audio_path: str, call_id: str, audio_duration: float) -> List[Dict]:
        """
        WhisperX 3.x built-in diarization using Pyannote.audio.
        More accurate than Resemblyzer but slower. Uses assign_word_speakers.
        
        Args:
            audio_path: Path to audio file
            call_id: Unique call identifier
            audio_duration: Audio duration in seconds
            
        Returns:
            List of speaker segments with timestamps
        """
        try:
            if not self.hf_token:
                raise ValueError("HF_TOKEN is required for WhisperX 3.x built-in diarization")
            
            logger.info("🎯 Using WhisperX 3.x built-in diarization (Pyannote.audio, state-of-the-art accuracy)")
            estimated_time = max(5, int(audio_duration / 10))
            logger.info(f"⏱️  Estimated time: {estimated_time}-{estimated_time*2} minutes")
            logger.info(f"🔄 Processing... This may take 20-40 minutes for long audio files on CPU")
            
            # Load WhisperX model if not already loaded
            if self.whisperx_model is None:
                logger.info(f"Loading WhisperX model: {self.model_size}")
                self.whisperx_model = whisperx.load_model(self.model_size, self._device, compute_type="int8")
                logger.info(f"WhisperX model loaded on {self._device}")
            
            # Step 1: Transcribe with WhisperX
            logger.info("Step 1: Transcribing with WhisperX...")
            audio = whisperx.load_audio(audio_path)
            result = self.whisperx_model.transcribe(audio, batch_size=16)
            
            # Step 2: Align timestamps
            logger.info("Step 2: Aligning word-level timestamps...")
            language_code = result.get("language", "en")
            model_a, metadata = self._get_whisperx_align_model(language_code)
            result = whisperx.align(result["segments"], model_a, metadata, audio, device=self._device, return_char_alignments=False)
            
            # Step 3: Load WhisperX diarization model (Pyannote.audio)
            if self.whisperx_diarize_model is None:
                logger.info("Step 3: Loading WhisperX diarization model (Pyannote.audio)...")
                try:
                    from whisperx.diarize import DiarizationPipeline
                    # DiarizationPipeline API: newer uses token=, older uses use_auth_token=; no revision param
                    try:
                        self.whisperx_diarize_model = DiarizationPipeline(
                            token=self.hf_token,
                            device=self._device,
                        )
                    except TypeError as te:
                        if "token" in str(te):
                            self.whisperx_diarize_model = DiarizationPipeline(
                                use_auth_token=self.hf_token,
                                device=self._device,
                            )
                        else:
                            raise
                    logger.info("✅ WhisperX DiarizationPipeline loaded successfully")
                except ImportError as e:
                    logger.error(f"Failed to import WhisperX DiarizationPipeline: {e}")
                    logger.error("Make sure you have WhisperX 3.7+ installed: pip install --upgrade whisperx")
                    raise
                except Exception as e:
                    logger.error(f"Failed to load WhisperX diarization model: {e}")
                    logger.error("Make sure you have:")
                    logger.error("  1. Valid HF_TOKEN in .env file")
                    logger.error("  2. Accepted user agreements for:")
                    logger.error("     - pyannote/segmentation-3.0")
                    logger.error("     - pyannote/speaker-diarization-3.1")
                    raise
            
            # Step 4: Perform diarization
            logger.info("Step 4: Performing speaker diarization (this may take a while)...")
            logger.info("   This is CPU-intensive and may take 20-40 minutes for long audio files")
            logger.info("   Progress will be logged every minute...")
            
            # Add periodic logging for long-running diarization
            import threading
            import time
            
            diarization_result = [None]
            diarization_error = [None]
            is_running = [True]
            
            def log_progress():
                elapsed = 0
                while is_running[0]:
                    time.sleep(60)  # Log every minute
                    elapsed += 1
                    if is_running[0]:
                        # Use print to ensure visibility even if logging is filtered
                        remaining_lo = max(5, int(audio_duration / 10) - elapsed)
                        remaining_hi = max(10, int(audio_duration / 10) * 2 - elapsed)
                        logger.info(f"Diarization progress: {elapsed} min elapsed (Pyannote is CPU-intensive). Est. remaining: {remaining_lo}-{remaining_hi} min")
                        logger.info(f"   This is normal - Pyannote.audio is CPU-intensive, please continue waiting...")
            
            progress_thread = threading.Thread(target=log_progress, daemon=True)
            progress_thread.start()
            
            try:
                # Use WhisperX DiarizationPipeline (returns a DataFrame)
                if self.max_speakers:
                    logger.info(f"   Running WhisperX diarization with max_speakers={self.max_speakers}")
                    diarize_df = self.whisperx_diarize_model(
                        audio_path, 
                        min_speakers=self.max_speakers, 
                        max_speakers=self.max_speakers
                    )
                else:
                    logger.info("   Running WhisperX diarization with auto-detection of speaker count")
                    diarize_df = self.whisperx_diarize_model(audio_path)
                
                # Convert DataFrame to format expected by assign_word_speakers
                diarization_result[0] = diarize_df
            except Exception as e:
                diarization_error[0] = e
            finally:
                is_running[0] = False
            
            if diarization_error[0]:
                raise diarization_error[0]
            
            diarize_df = diarization_result[0]
            logger.info("   ✅ Diarization completed!")
            
            # Step 5: Assign speakers to words using WhisperX's assign_word_speakers
            logger.info("Step 5: Assigning speakers to words...")
            try:
                from whisperx.diarize import assign_word_speakers
                # assign_word_speakers expects: (diarize_df: DataFrame, transcript_result: TranscriptionResult)
                # The result from WhisperX transcribe should be in the correct format
                result = assign_word_speakers(diarize_df, result)
            except (KeyError, TypeError, AttributeError) as e:
                logger.error(f"assign_word_speakers failed with error: {e}")
                logger.error(f"This is a known issue with WhisperX assign_word_speakers and certain diarization formats")
                logger.warning("Skipping word-level speaker assignment, using segment-level diarization instead")
                # Don't fail - just use the diarization DataFrame directly
                # We'll convert it to our format without word-level assignment
                import pandas as pd
                if isinstance(diarize_df, pd.DataFrame):
                    logger.info("Converting diarization DataFrame to segment format...")
                    # Convert DataFrame to segments format
                    segments = []
                    for _, row in diarize_df.iterrows():
                        segments.append({
                            "speaker": f"SPEAKER_{row.get('speaker', '00'):02d}" if isinstance(row.get('speaker'), (int, float)) else str(row.get('speaker', 'SPEAKER_00')),
                            "start": float(row.get('start', 0)),
                            "end": float(row.get('end', 0)),
                            "text": ""  # Will be filled by text mapping
                        })
                    # Use the original result segments for text mapping
                    transcription_segments = result.get("segments", [])
                    # Skip to Step 7 (role identification) - we'll do text mapping there
                    logger.info(f"Converted {len(segments)} segments from DataFrame")
                    # Continue to role identification
                    segments = self._identify_speaker_roles(segments, transcription_segments=transcription_segments)
                    # Save and return
                    try:
                        self.save_diarization(segments, call_id)
                    except Exception as db_error:
                        logger.warning(f"Failed to save diarization to database: {db_error}")
                    return segments
                else:
                    logger.error(f"diarize_df is not a DataFrame: {type(diarize_df)}")
                    raise ValueError(f"Expected DataFrame from DiarizationPipeline, got {type(diarize_df)}")
            except Exception as e:
                logger.error(f"Unexpected error in assign_word_speakers: {e}")
                raise
            
            # Step 6: Convert to our format
            logger.info("Step 6: Converting to segment format...")
            segments = []
            for segment in result.get("segments", []):
                speaker = segment.get("speaker", "SPEAKER_00")
                # Convert to consistent format
                if not speaker.startswith("SPEAKER_"):
                    # Ensure format is SPEAKER_XX
                    try:
                        speaker_num = int(speaker.replace("SPEAKER_", "").replace("SPK_", ""))
                        speaker = f"SPEAKER_{speaker_num:02d}"
                    except:
                        speaker = f"SPEAKER_{speaker}"
                
                segments.append({
                    "speaker": speaker,
                    "start": segment.get("start", 0),
                    "end": segment.get("end", 0),
                    "text": segment.get("text", "")
                })
            
            logger.info(f"✅ WhisperX 3.x built-in diarization completed. Found {len(set(seg['speaker'] for seg in segments))} speakers, {len(segments)} segments")
            
            # Step 7: Identify customer vs agent roles (for call centers)
            segments = self._identify_speaker_roles(segments, transcription_segments=result.get("segments", []))
            
            # Save to database if available
            try:
                self.save_diarization(segments, call_id)
            except Exception as db_error:
                logger.warning(f"Failed to save diarization to database: {db_error}")
            
            return segments
            
        except Exception as e:
            logger.error(f"WhisperX 3.x built-in diarization failed: {e}")
            logger.warning("Falling back to Resemblyzer method...")
            # Fallback to Resemblyzer if available
            if self.voice_encoder is not None:
                return self._diarize_with_whisperx_resemblyzer(audio_path, call_id, audio_duration)
            # Fallback to Pyannote if available
            elif self.diarization_pipeline is not None:
                return self._diarize_with_pyannote(audio_path, call_id, audio_duration)
            raise ValueError(f"Speaker diarization failed: {str(e)}")
    
    def _identify_speaker_roles(self, segments: List[Dict], transcription_segments: Optional[List[Dict]] = None) -> List[Dict]:
        """
        Identify customer vs agent roles for call center conversations.
        Uses heuristics: first speaker, speaking time, keywords, question patterns.
        
        Args:
            segments: List of diarization segments with speaker IDs
            transcription_segments: Optional transcription segments with text for keyword analysis
            
        Returns:
            Segments with speaker labels updated to "CUSTOMER" or "AGENT"
        """
        if not segments:
            return segments
        
        # Get unique speakers
        unique_speakers = list(set(seg.get('speaker', 'Unknown') for seg in segments))
        
        # For call centers, expect 2-3 speakers max
        if len(unique_speakers) > 3:
            logger.warning(f"More than 3 speakers detected ({len(unique_speakers)}). "
                         f"Using top 3 by speaking time for role identification.")
            # Keep only top 3 speakers by total speaking time
            speaker_times = {}
            for seg in segments:
                speaker = seg.get('speaker', 'Unknown')
                duration = seg.get('end', 0) - seg.get('start', 0)
                speaker_times[speaker] = speaker_times.get(speaker, 0) + duration
            unique_speakers = sorted(speaker_times.items(), key=lambda x: x[1], reverse=True)[:3]
            unique_speakers = [sp[0] for sp in unique_speakers]
        
        if len(unique_speakers) < 2:
            # Single speaker or no clear distinction
            logger.info("Only 1 speaker detected. Cannot distinguish customer/agent.")
            return segments
        
        # Heuristic 1: First speaker is usually the agent (greeting)
        first_segment = min(segments, key=lambda x: x.get('start', 0))
        first_speaker = first_segment.get('speaker', 'Unknown')
        
        # Heuristic 2: Calculate speaking time per speaker
        speaker_times = {}
        speaker_segments = {sp: [] for sp in unique_speakers}
        
        for seg in segments:
            speaker = seg.get('speaker', 'Unknown')
            if speaker not in unique_speakers:
                continue
            duration = seg.get('end', 0) - seg.get('start', 0)
            speaker_times[speaker] = speaker_times.get(speaker, 0) + duration
            speaker_segments[speaker].append(seg)
        
        # Heuristic 3: Keyword analysis (if transcription available)
        agent_keywords = [
            'thank you for calling', 'how can i help', 'may i have your', 'can i get your',
            'i understand', 'let me help', 'i can assist', 'is there anything else',
            'have a great day', 'thank you for your time', 'i appreciate', 'we can'
        ]
        customer_keywords = [
            'i want', 'i need', 'i would like', 'i\'m calling about', 'i have a question',
            'can you help me', 'i\'m looking for', 'do you have', 'how much', 'what is'
        ]
        
        speaker_agent_scores = {sp: 0 for sp in unique_speakers}
        speaker_customer_scores = {sp: 0 for sp in unique_speakers}
        
        # Map transcription text to segments if available
        if transcription_segments:
            for seg in segments:
                speaker = seg.get('speaker', 'Unknown')
                if speaker not in unique_speakers:
                    continue
                
                # Find matching transcription text
                seg_start = seg.get('start', 0)
                seg_end = seg.get('end', 0)
                text = seg.get('text', '').lower()
                
                if not text:
                    # Try to find text from transcription segments
                    for trans_seg in transcription_segments:
                        t_start = trans_seg.get('start', 0)
                        t_end = trans_seg.get('end', 0)
                        if t_start >= seg_start and t_end <= seg_end:
                            text = trans_seg.get('text', '').lower()
                            break
                
                # Score keywords
                for keyword in agent_keywords:
                    if keyword in text:
                        speaker_agent_scores[speaker] += 1
                for keyword in customer_keywords:
                    if keyword in text:
                        speaker_customer_scores[speaker] += 1
        
        # Heuristic 4: Question patterns (agents ask more questions)
        speaker_question_counts = {sp: 0 for sp in unique_speakers}
        for seg in segments:
            speaker = seg.get('speaker', 'Unknown')
            if speaker not in unique_speakers:
                continue
            text = seg.get('text', '').lower()
            # Also check transcription segments if text not in segment
            if not text and transcription_segments:
                seg_start = seg.get('start', 0)
                seg_end = seg.get('end', 0)
                for trans_seg in transcription_segments:
                    t_start = trans_seg.get('start', 0)
                    t_end = trans_seg.get('end', 0)
                    if t_start >= seg_start and t_end <= seg_end:
                        text = trans_seg.get('text', '').lower()
                        break
            if text and ('?' in text or any(text.startswith(q) for q in ('what', 'how', 'when', 'where', 'why', 'can you', 'do you', 'would you'))):
                speaker_question_counts[speaker] += 1
        
        # Determine roles
        # Agent is usually: first speaker OR most speaking time OR most agent keywords OR most questions
        # Customer is usually: second speaker OR less speaking time OR more customer keywords
        agent_candidates = []
        for speaker in unique_speakers:
            score = 0
            speaking_time = speaker_times.get(speaker, 0)
            max_speaking_time = max(speaker_times.values())
            min_speaking_time = min(speaker_times.values())
            
            # IMPROVED: Keyword analysis is most reliable - give it highest weight
            agent_keyword_score = speaker_agent_scores.get(speaker, 0)
            customer_keyword_score = speaker_customer_scores.get(speaker, 0)
            if agent_keyword_score > customer_keyword_score:
                score += 5  # Strong indicator - more agent keywords
            elif customer_keyword_score > agent_keyword_score:
                score -= 3  # Strong indicator - more customer keywords
            
            # IMPROVED: Question patterns (agents ask more questions) - second most reliable
            question_count = speaker_question_counts.get(speaker, 0)
            max_questions = max(speaker_question_counts.values()) if speaker_question_counts.values() else 0
            if question_count == max_questions and max_questions > 0:
                score += 3  # Most questions (agents ask more)
            elif question_count == 0 and max_questions > 0:
                score -= 1  # No questions (customers ask fewer)
            
            # IMPROVED: Speaking time analysis - but less weight
            # In call centers, agents often talk more, but not always
            speaking_ratio = speaking_time / max_speaking_time if max_speaking_time > 0 else 0
            if speaking_ratio > 0.6:
                score += 2  # Speaks significantly more (likely agent)
            elif speaking_ratio < 0.4:
                score -= 1  # Speaks significantly less (might be customer)
            
            # IMPROVED: First speaker - least reliable, but still a hint
            # Don't assume first speaker is always agent
            if speaker == first_speaker:
                score += 1  # Weak indicator - first speaker might be agent
            
            agent_candidates.append((speaker, score))
        
        # Sort by score and assign roles
        agent_candidates.sort(key=lambda x: x[1], reverse=True)
        agent_speaker = agent_candidates[0][0]
        
        logger.info(f"Role identification scores: {[(s[0], s[1]) for s in agent_candidates]}")
        
        # For call centers with 2 speakers: assign agent and customer
        # For 3+ speakers: assign agent, and merge others into customer or agent based on similarity
        speaker_role_map = {agent_speaker: "AGENT"}
        
        if len(unique_speakers) == 2:
            # Simple case: 2 speakers
            for speaker in unique_speakers:
                if speaker != agent_speaker:
                    speaker_role_map[speaker] = "CUSTOMER"
        else:
            # 3+ speakers: assign second most speaking time as customer, merge rest
            remaining_speakers = [s for s in unique_speakers if s != agent_speaker]
            remaining_speakers.sort(key=lambda x: speaker_times.get(x, 0), reverse=True)
            
            if remaining_speakers:
                # Second most speaking time = customer
                customer_speaker = remaining_speakers[0]
                speaker_role_map[customer_speaker] = "CUSTOMER"
                
                # Others (likely noise) will be merged later in the update loop
        
        # Update segments with role labels
        # CRITICAL: Ensure ALL segments get labeled as AGENT or CUSTOMER (no leftover SPEAKER_XX)
        for seg in segments:
            original_speaker = seg.get('speaker', 'Unknown')
            if original_speaker in speaker_role_map:
                seg['speaker'] = speaker_role_map[original_speaker]
                seg['original_speaker_id'] = original_speaker  # Keep original ID for reference
            else:
                # Leftover speaker not in top 2-3 - merge into closest main speaker
                seg_duration = seg.get('end', 0) - seg.get('start', 0)
                
                # Very short segments (< 2s) are likely noise - merge into agent
                if seg_duration < 2.0:
                    seg['speaker'] = "AGENT"
                    seg['original_speaker_id'] = original_speaker
                    logger.debug(f"Merged short segment ({seg_duration:.1f}s) from {original_speaker} into AGENT")
                else:
                    # For longer segments, merge into the speaker with most speaking time
                    # Usually this is the agent, but check if we have a customer identified
                    if "CUSTOMER" in speaker_role_map.values():
                        # We have both agent and customer - merge based on speaking time ratio
                        agent_speaking_time = speaker_times.get(agent_speaker, 0)
                        total_main_time = sum(speaker_times.get(s, 0) for s in unique_speakers)
                        if agent_speaking_time > total_main_time * 0.6:
                            # Agent speaks much more - merge into agent
                            seg['speaker'] = "AGENT"
                        else:
                            # More balanced - merge into customer (less common speaker)
                            seg['speaker'] = "CUSTOMER"
                    else:
                        # Only agent identified - merge into agent
                        seg['speaker'] = "AGENT"
                    seg['original_speaker_id'] = original_speaker
                    logger.debug(f"Merged {original_speaker} ({seg_duration:.1f}s) into {seg['speaker']}")
        
        # Recalculate final speaker distribution
        final_speakers = {}
        for seg in segments:
            speaker = seg.get('speaker', 'Unknown')
            duration = seg.get('end', 0) - seg.get('start', 0)
            if speaker not in final_speakers:
                final_speakers[speaker] = {'count': 0, 'duration': 0}
            final_speakers[speaker]['count'] += 1
            final_speakers[speaker]['duration'] += duration
        
        logger.info(f"Speaker role identification completed:")
        for speaker, stats in final_speakers.items():
            logger.info(f"  {speaker}: {stats['count']} segments, {stats['duration']:.1f}s total")
        
        return segments
    
    def _is_valid_embedding(self, embedding: np.ndarray) -> bool:
        """
        Check if an embedding is valid (not None, not empty, no NaN/inf, proper shape).
        
        Args:
            embedding: Embedding array to validate
            
        Returns:
            True if valid, False otherwise
        """
        if embedding is None:
            return False
        embedding = np.asarray(embedding)
        if embedding.size == 0:
            return False
        if np.any(np.isnan(embedding)) or np.any(np.isinf(embedding)):
            return False
        if embedding.ndim == 0:
            return False
        return True
    
    def _merge_similar_speakers(self, embeddings: np.ndarray, cluster_labels: np.ndarray, 
                                similarity_threshold: float = 0.7) -> np.ndarray:
        """
        PRODUCTION FIX: Merge clusters with very similar average embeddings.
        This reduces over-segmentation by combining clusters that are too similar.
        
        Args:
            embeddings: Array of segment embeddings (N x embedding_dim).
            cluster_labels: Current cluster labels (N,).
            similarity_threshold: Cosine similarity threshold for merging (0.0-1.0).
            
        Returns:
            Updated cluster labels with similar clusters merged.
        """
        from scipy.spatial.distance import cosine
        
        unique_clusters = np.unique(cluster_labels)
        if len(unique_clusters) <= 1:
            return cluster_labels
        
        # Calculate average embedding per cluster (skip invalid embeddings)
        cluster_centroids = {}
        for cluster_id in unique_clusters:
            cluster_mask = cluster_labels == cluster_id
            cluster_embeddings = embeddings[cluster_mask]
            
            # Filter out invalid embeddings
            valid_embeddings = []
            for emb in cluster_embeddings:
                if self._is_valid_embedding(emb):
                    valid_embeddings.append(emb)
            
            if not valid_embeddings:
                logger.warning(f"Skipping cluster {cluster_id}: no valid embeddings")
                continue
            
            valid_embeddings = np.array(valid_embeddings)
            centroid = np.mean(valid_embeddings, axis=0)
            
            # Validate centroid before storing
            if not self._is_valid_embedding(centroid):
                logger.warning(f"Skipping invalid centroid for cluster {cluster_id}")
                continue
            
            cluster_centroids[cluster_id] = centroid
        
        # Find clusters to merge (high similarity)
        merge_map = {}  # Maps cluster_id -> target_cluster_id
        processed = set()
        
        for i, cluster_id_i in enumerate(unique_clusters):
            if cluster_id_i in processed:
                continue
            
            for cluster_id_j in unique_clusters[i+1:]:
                if cluster_id_j in processed:
                    continue
                
                # Calculate cosine similarity between centroids
                centroid_i = cluster_centroids[cluster_id_i]
                centroid_j = cluster_centroids[cluster_id_j]
                similarity = 1 - cosine(centroid_i, centroid_j)
                
                # If very similar, merge into the smaller cluster ID
                if similarity >= similarity_threshold:
                    target_id = min(cluster_id_i, cluster_id_j)
                    source_id = max(cluster_id_i, cluster_id_j)
                    merge_map[source_id] = target_id
                    processed.add(source_id)
                    processed.add(cluster_id_i)
                    break
        
        # Apply merges
        if merge_map:
            logger.info(f"Merging {len(merge_map)} similar speaker clusters...")
            new_labels = cluster_labels.copy()
            for source_id, target_id in merge_map.items():
                new_labels[cluster_labels == source_id] = target_id
            
            # Renumber clusters to be consecutive (0, 1, 2, ...)
            unique_new = np.unique(new_labels)
            label_map = {old_id: new_id for new_id, old_id in enumerate(unique_new)}
            final_labels = np.array([label_map[label] for label in new_labels])
            
            return final_labels + 1  # Make 1-indexed like original
        
        return cluster_labels
    
    def _get_whisperx_align_model(self, language_code: str):
        """
        Load (or retrieve cached) WhisperX alignment model for a given language.
        """
        if language_code in self.whisperx_align_models:
            return self.whisperx_align_models[language_code]
        
        if whisperx is None:
            raise RuntimeError("WhisperX is not available")
        
        model_a, metadata = whisperx.load_align_model(language_code=language_code, device=self._device)
        self.whisperx_align_models[language_code] = (model_a, metadata)
        return model_a, metadata
    
    def _match_speakers_across_chunks(self, chunk_speaker_embeddings: Dict, 
                                      similarity_threshold: float = 0.8) -> Tuple[Dict, int]:
        """
        PRODUCTION FIX: Match speakers across chunks using embedding similarity.
        Prevents duplicate speaker IDs when the same person appears in multiple chunks.
        
        Args:
            chunk_speaker_embeddings: Dict mapping chunk_idx -> {local_speaker_id: [embeddings]}
            similarity_threshold: Cosine similarity threshold for matching (0.0-1.0)
            
        Returns:
            Tuple of (global_speaker_map, next_global_speaker_id)
            global_speaker_map: Maps (chunk_idx, local_speaker_id) -> global_speaker_id
        """
        from scipy.spatial.distance import cosine
        
        global_speaker_map = {}
        next_global_speaker_id = 0
        
        # Calculate average embedding per speaker per chunk (skip invalid embeddings)
        chunk_speaker_centroids = {}
        for chunk_idx, speakers in chunk_speaker_embeddings.items():
            chunk_speaker_centroids[chunk_idx] = {}
            for local_speaker_id, embeddings in speakers.items():
                if not embeddings:
                    continue
                
                # Filter out invalid embeddings
                valid_embeddings = []
                for emb in embeddings:
                    if self._is_valid_embedding(emb):
                        valid_embeddings.append(emb)
                
                if not valid_embeddings:
                    logger.warning(f"Skipping chunk {chunk_idx}, speaker {local_speaker_id}: no valid embeddings")
                    continue
                
                # Average embedding for this speaker in this chunk
                embeddings_array = np.array(valid_embeddings)
                if len(embeddings_array.shape) == 1:
                    # Single embedding, use as-is
                    centroid = embeddings_array
                else:
                    # Multiple embeddings, average them
                    centroid = np.mean(embeddings_array, axis=0)
                
                # Validate centroid before storing
                if not self._is_valid_embedding(centroid):
                    logger.warning(f"Skipping invalid centroid for chunk {chunk_idx}, speaker {local_speaker_id}")
                    continue
                
                # Ensure centroid is 1-D
                chunk_speaker_centroids[chunk_idx][local_speaker_id] = np.atleast_1d(centroid).flatten()
        
        # Process chunks in order, matching with previous chunks
        processed_speakers = {}  # Maps global_speaker_id -> centroid
        
        for chunk_idx in sorted(chunk_speaker_centroids.keys()):
            for local_speaker_id, centroid in chunk_speaker_centroids[chunk_idx].items():
                chunk_speaker_key = (chunk_idx, local_speaker_id)
                
                # Try to match with speakers from previous chunks
                best_match_id = None
                best_similarity = 0.0
                
                # Ensure centroid is 1-D for cosine distance calculation
                centroid_1d = np.atleast_1d(centroid).flatten()
                
                for global_id, prev_centroid in processed_speakers.items():
                    # Ensure prev_centroid is also 1-D
                    prev_centroid_1d = np.atleast_1d(prev_centroid).flatten()
                    try:
                        similarity = 1 - cosine(centroid_1d, prev_centroid_1d)
                        if similarity > best_similarity and similarity >= similarity_threshold:
                            best_similarity = similarity
                            best_match_id = global_id
                    except ValueError as e:
                        logger.warning(f"Error calculating similarity: {e}, skipping match")
                        continue
                
                # Assign speaker ID
                if best_match_id is not None:
                    # Match found - use existing global ID
                    global_speaker_map[chunk_speaker_key] = best_match_id
                    # Update centroid (weighted average with previous chunks)
                    prev_centroid_1d = np.atleast_1d(processed_speakers[best_match_id]).flatten()
                    processed_speakers[best_match_id] = (
                        0.7 * prev_centroid_1d + 0.3 * centroid_1d
                    )
                else:
                    # New speaker - assign new global ID
                    global_speaker_map[chunk_speaker_key] = next_global_speaker_id
                    processed_speakers[next_global_speaker_id] = centroid_1d
                    next_global_speaker_id += 1
        
        logger.info(f"Cross-chunk matching: {len(processed_speakers)} unique speakers identified across all chunks")        
        return global_speaker_map, next_global_speaker_id
    
    def _diarize_chunked_whisperx_resemblyzer(self, audio_path: str, call_id: str, audio_duration: float) -> List[Dict]:
        """
        Diarize long audio files by processing in chunks and combining results.
        Much faster for long files.
        """
        import librosa
        import soundfile as sf
        import tempfile
        import os
        
        logger.info(f"📦 Processing diarization in chunks")
        
        # Calculate number of chunks
        num_chunks = int(np.ceil(audio_duration / self.chunk_duration))
        logger.info(f"   Splitting into {num_chunks} chunks (~{self.chunk_duration/60:.1f} min each)")
        
        # Load audio
        audio_data, sample_rate = librosa.load(audio_path, sr=self.target_sample_rate)
        chunk_size_samples = int(self.chunk_duration * sample_rate)
        
        all_segments = []
        # PRODUCTION FIX: Store embeddings per chunk for cross-chunk speaker re-identification
        chunk_speaker_embeddings = {}  # {chunk_idx: {local_speaker_id: [embeddings]}}
        chunk_speaker_segments = {}  # {chunk_idx: {local_speaker_id: [segments]}}
        global_speaker_map = {}  # Map (chunk_idx, local_speaker_id) to global IDs
        next_global_speaker_id = 0
        
        # Load models once
        if self.whisperx_model is None:
            logger.info(f"Loading WhisperX model: {self.model_size}")
            self.whisperx_model = whisperx.load_model(self.model_size, self._device, compute_type="int8")
            logger.info(f"WhisperX model loaded on {self._device}")
        
        if self.voice_encoder is None:
            self.voice_encoder = VoiceEncoder()
        
        # Process each chunk
        for chunk_idx in range(num_chunks):
            start_sample = chunk_idx * chunk_size_samples
            end_sample = min((chunk_idx + 1) * chunk_size_samples, len(audio_data))
            chunk_audio = audio_data[start_sample:end_sample]
            chunk_start_time = chunk_idx * self.chunk_duration
            
            if len(chunk_audio) < sample_rate * 0.5:
                continue
            
            logger.info(f"   Processing chunk {chunk_idx + 1}/{num_chunks} "
                       f"({chunk_start_time/60:.1f} - {chunk_start_time/60 + len(chunk_audio)/sample_rate/60:.1f} min)")
            
            # Save chunk to temporary file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                tmp_path = tmp_file.name
                sf.write(tmp_path, chunk_audio, sample_rate)
            
            try:
                # Process chunk with WhisperX + Resemblyzer
                audio_chunk = whisperx.load_audio(tmp_path)
                result = self.whisperx_model.transcribe(audio_chunk, batch_size=16)
                
                # Align timestamps
                language_code = result.get("language", "en")
                model_a, metadata = self._get_whisperx_align_model(language_code)
                result = whisperx.align(result["segments"], model_a, metadata, audio_chunk, device=self._device, return_char_alignments=False)
                
                # IMPROVED: Merge transcription segments before extracting embeddings (same as non-chunked)
                # Extract embeddings and cluster
                wav = preprocess_wav(Path(tmp_path))
                
                # Merge very short transcription segments together for better embeddings
                merged_segments = []
                current_merged = None
                min_merge_duration = 2.0  # Merge segments shorter than 2 seconds
                
                for segment in result["segments"]:
                    start_time = segment["start"]
                    end_time = segment["end"]
                    duration = end_time - start_time
                    text = segment.get("text", "").strip()
                    
                    if duration < min_merge_duration and current_merged is not None:
                        # Merge with previous segment if both are short
                        current_merged["end"] = end_time
                        current_merged["text"] = (current_merged.get("text", "") + " " + text).strip()
                    else:
                        # Start new segment
                        if current_merged is not None:
                            merged_segments.append(current_merged)
                        current_merged = {
                            "start": start_time,
                            "end": end_time,
                            "text": text
                        }
                
                # Add last segment
                if current_merged is not None:
                    merged_segments.append(current_merged)
                
                segment_embeddings = []
                segment_times = []
                segment_texts = []  # Store text for later mapping
                
                for segment in merged_segments:
                    start_time = segment["start"]
                    end_time = segment["end"]
                    duration = end_time - start_time
                    
                    # PRODUCTION FIX: Use configurable minimum segment duration
                    if duration < self.min_segment_duration:
                        continue
                    
                    start_sample_local = int(start_time * 16000)
                    end_sample_local = int(end_time * 16000)
                    
                    if end_sample_local > len(wav):
                        end_sample_local = len(wav)
                    if start_sample_local >= len(wav):
                        continue
                    
                    segment_audio = wav[start_sample_local:end_sample_local]
                    
                    # PRODUCTION FIX: Use configurable minimum segment duration
                    min_samples = int(self.min_segment_duration * 16000)
                    if len(segment_audio) > min_samples:
                        try:
                            embedding = self.voice_encoder.embed_utterance(segment_audio)
                            segment_embeddings.append(embedding)
                            segment_times.append((start_time, end_time))
                            segment_texts.append(segment.get("text", ""))  # Store text
                        except Exception as e:
                            logger.warning(f"Failed to embed segment in chunk {chunk_idx+1}: {e}")
                            continue
                
                if len(segment_embeddings) < 2:
                    # Single speaker in chunk - store for cross-chunk matching
                    chunk_speaker_embeddings[chunk_idx] = {0: segment_embeddings}
                    chunk_speaker_segments[chunk_idx] = {0: []}
                    for i, (start, end) in enumerate(segment_times):
                        segment_text = segment_texts[i] if i < len(segment_texts) else ""
                        chunk_speaker_segments[chunk_idx][0].append({
                            "start": start + chunk_start_time,
                            "end": end + chunk_start_time,
                            "text": segment_text  # Use text from merged transcription segments
                        })
                    continue
                
                # PRODUCTION FIX: Cluster speakers in chunk with improved algorithm
                segment_embeddings = np.array(segment_embeddings)
                distances = pdist(segment_embeddings, metric='cosine')
                linkage_matrix = linkage(distances, method='average')
                
                # Use adaptive clustering (same as non-chunked version)
                threshold = self.clustering_threshold
                cluster_labels = fcluster(linkage_matrix, threshold, criterion='distance')
                num_clusters = len(set(cluster_labels))
                
                # Only apply max_speakers as safety check (same as non-chunked version)
                if self.max_speakers is not None and num_clusters > self.max_speakers * 2:
                    logger.warning(f"Chunk detected {num_clusters} speakers (exceeds max_speakers={self.max_speakers}). "
                                 f"Adjusting threshold...")
                    threshold *= 1.3
                    cluster_labels = fcluster(linkage_matrix, threshold, criterion='distance')
                    num_clusters = len(set(cluster_labels))
                
                # Merge similar speakers within chunk
                cluster_labels = self._merge_similar_speakers(
                    segment_embeddings, cluster_labels, similarity_threshold=self.speaker_merge_threshold
                )
                
                # PRODUCTION FIX: Store embeddings per speaker for cross-chunk matching
                chunk_speaker_embeddings[chunk_idx] = {}
                chunk_speaker_segments[chunk_idx] = {}
                
                for i, (start, end) in enumerate(segment_times):
                    local_speaker_id = cluster_labels[i] - 1
                    
                    # Store embedding for this speaker
                    if local_speaker_id not in chunk_speaker_embeddings[chunk_idx]:
                        chunk_speaker_embeddings[chunk_idx][local_speaker_id] = []
                        chunk_speaker_segments[chunk_idx][local_speaker_id] = []
                    
                    # IMPROVED: Use text from merged segments directly (avoids repetition)
                    segment_text = segment_texts[i] if i < len(segment_texts) else ""
                    chunk_speaker_embeddings[chunk_idx][local_speaker_id].append(segment_embeddings[i])
                    chunk_speaker_segments[chunk_idx][local_speaker_id].append({
                        "start": start + chunk_start_time,
                        "end": end + chunk_start_time,
                        "text": segment_text  # Use text from merged transcription segments
                    })
                
            finally:
                try:
                    os.unlink(tmp_path)
                except:
                    pass
        
        # PRODUCTION FIX: Cross-chunk speaker re-identification
        logger.info("Matching speakers across chunks...")
        global_speaker_map, next_global_speaker_id = self._match_speakers_across_chunks(
            chunk_speaker_embeddings, similarity_threshold=0.8
        )
        
        # Build final segments with merged speaker IDs
        for chunk_idx, speaker_segments in chunk_speaker_segments.items():
            for local_speaker_id, segments in speaker_segments.items():
                chunk_speaker_key = (chunk_idx, local_speaker_id)
                global_speaker_id = global_speaker_map.get(chunk_speaker_key, next_global_speaker_id)
                if chunk_speaker_key not in global_speaker_map:
                    global_speaker_map[chunk_speaker_key] = global_speaker_id
                    next_global_speaker_id += 1
                
                speaker_id = f"SPEAKER_{global_speaker_id:02d}"
                for segment in segments:
                    segment["speaker"] = speaker_id
                    all_segments.append(segment)
        
        # Sort segments by start time
        all_segments.sort(key=lambda x: x["start"])
        
        unique_speakers = len(set(seg["speaker"] for seg in all_segments))
        logger.info(f"✅ Chunked diarization completed. Found {unique_speakers} speakers (after cross-chunk matching), {len(all_segments)} segments")
        
        # Identify customer vs agent roles (for call centers)
        all_segments = self._identify_speaker_roles(all_segments, transcription_segments=None)
        
        try:
            self.save_diarization(all_segments, call_id)
        except Exception as db_error:
            logger.warning(f"Failed to save diarization to database: {db_error}")
        
        return all_segments
    
    def _diarize_with_pyannote(self, audio_path: str, call_id: str, audio_duration: float) -> List[Dict]:
        """
        Diarization using Pyannote.audio (slower but accurate fallback).
        """
        try:
            estimated_time = max(5, int(audio_duration / 10))
            logger.info("Using Pyannote.audio diarization (slower, but accurate)")
            logger.info(f"⏱️  Estimated time: {estimated_time}-{estimated_time*2} minutes")
            logger.info(f"🔄 Processing... Please wait (this may take 15-30 minutes for long audio files)")
            
            # Add periodic logging
            import threading
            import time
            
            diarization_result = [None]
            diarization_error = [None]
            is_running = [True]
            
            def log_progress():
                elapsed = 0
                while is_running[0]:
                    time.sleep(60)
                    elapsed += 1
                    if is_running[0]:
                        logger.info(f"🔄 Diarization still processing... ({elapsed} minutes elapsed)")
            
            progress_thread = threading.Thread(target=log_progress, daemon=True)
            progress_thread.start()
            
            try:
                logger.info("Starting Pyannote.audio pipeline...")
                # Dynamic speaker constraints: num_speakers (exact) OR min_speakers/max_speakers (range)
                pipeline_kwargs = {}
                if self.num_speakers is not None:
                    pipeline_kwargs["num_speakers"] = self.num_speakers
                    logger.info(f"   Speaker count: exact num_speakers={self.num_speakers}")
                elif self.min_speakers is not None or self.max_speakers is not None:
                    if self.min_speakers is not None:
                        pipeline_kwargs["min_speakers"] = self.min_speakers
                    if self.max_speakers is not None:
                        pipeline_kwargs["max_speakers"] = self.max_speakers
                    if pipeline_kwargs:
                        logger.info(f"   Speaker range: min_speakers={pipeline_kwargs.get('min_speakers', 'auto')}, max_speakers={pipeline_kwargs.get('max_speakers', 'auto')}")
                diarization = self.diarization_pipeline(audio_path, **pipeline_kwargs)
                diarization_result[0] = diarization
            except Exception as e:
                diarization_error[0] = e
            finally:
                is_running[0] = False
            
            if diarization_error[0]:
                raise diarization_error[0]
            
            diarization = diarization_result[0]
            segments = []
            for turn, _, speaker in diarization.speaker_diarization.itertracks(yield_label=True):
                segments.append({
                    "speaker": speaker,
                    "start": turn.start,
                    "end": turn.end,
                    "text": ""
                })
            
            logger.info(f"✅ Pyannote.audio diarization completed. Found {len(segments)} speaker segments")
            
            try:
                self.save_diarization(segments, call_id)
            except Exception as db_error:
                logger.warning(f"Failed to save diarization to database: {db_error}")
            
            return segments
        except Exception as e:
            logger.error(f"Pyannote.audio diarization failed: {e}")
            raise ValueError(f"Speaker diarization failed: {str(e)}")
    
    def extract_audio_features(self, audio_path: str, n_mfcc: int = 40) -> Dict:
        """
        Extract audio features for emotion detection (FR-5: CNN+LSTM).
        
        Args:
            audio_path: Path to audio file.
            n_mfcc: Number of MFCC coefficients (13-40, default: 40)
            
        Returns:
            Dictionary of audio features.
        """
        try:
            audio_path = self.validate_audio_format(audio_path)
            y, sr = librosa.load(audio_path, sr=16000)  # Match Whisper's default
            
            # MFCC: 13-40 coefficients (per guide.txt requirement)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
            
            # Mel-Spectrogram: 2D format for CNN input
            mel_spec = librosa.feature.melspectrogram(
                y=y, sr=sr,
                n_mels=128,  # Standard for emotion recognition
                fmax=8000,
                hop_length=512
            )
            # Convert to dB scale (log-mel)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Chroma features (optimized)
            chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=512)
            
            features = {
                "mfcc": mfcc,  # (n_mfcc, time_frames)
                "mel_spectrogram": mel_spec_db,  # (n_mels, time_frames) - 2D format
                "spectral_centroid": librosa.feature.spectral_centroid(y=y, sr=sr),
                "spectral_rolloff": librosa.feature.spectral_rolloff(y=y, sr=sr),
                "zero_crossing_rate": librosa.feature.zero_crossing_rate(y),
                "chroma": chroma,
                "duration": librosa.get_duration(y=y, sr=sr),
                "sample_rate": sr,
                "waveform": y,  # raw waveform for wav2vec2-based emotion model
            }
            return features
        except Exception as e:
            logger.error(f"Audio feature extraction failed: {e}")
            raise
    
    def save_transcription(self, transcription: Dict, call_id: str):
        """
        Save transcription to MongoDB (FR-7) and to JSON file.
        
        Args:
            transcription: Transcription dictionary.
            call_id: Unique call identifier.
        """
        # Save to JSON file (always, even if MongoDB fails)
        try:
            import json
            from pathlib import Path
            
            # Determine output directory
            script_dir = Path(__file__).parent.parent.parent.parent
            output_dir = script_dir / "output"
            output_dir.mkdir(exist_ok=True)
            
            transcription_file = output_dir / f"{call_id}_transcription.json"
            
            with open(transcription_file, 'w', encoding='utf-8') as f:
                json.dump(transcription, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"Transcription saved to file: {transcription_file}")
        except Exception as file_error:
            logger.warning(f"Failed to save transcription to file: {file_error}")
        
        # Also try to save to MongoDB (optional)
        if not self.mongo_enabled:
            logger.debug("MongoDB disabled - skipping transcription DB save")
            return
        
        try:
            client = MongoClient(self.mongo_uri, serverSelectionTimeoutMS=2000)
            client.server_info()
            db = client[self.mongo_db]
            collection = db['transcriptions']
            collection.insert_one({
                'call_id': call_id,
                'transcription': transcription,
                'timestamp': datetime.now()
            })
            logger.info(f"Transcription saved to MongoDB for call_id: {call_id}")
            client.close()
        except Exception as e:
            logger.warning(f"MongoDB not available, transcription not saved to database: {e}")
            logger.info("Transcription completed successfully (saved to file only)")
    
    def save_diarization(self, segments: List[Dict], call_id: str):
        """
        Save diarization segments to MongoDB (FR-7) and to JSON file.
        
        Args:
            segments: List of diarization segments.
            call_id: Unique call identifier.
        """
        # Save to JSON file (always, even if MongoDB fails)
        try:
            import json
            import os
            from pathlib import Path
            
            # Determine output directory - find project root
            # preprocessing.py is at: backend/src/call_analysis/preprocessing.py
            # Project root is 4 levels up
            current = Path(__file__).parent  # call_analysis
            current = current.parent  # src
            current = current.parent  # backend
            project_root = current.parent  # project root
            
            # Try to verify it's the project root
            if not (project_root / "pyproject.toml").exists():
                # Try alternative: maybe we're already at project root level
                if (current / "pyproject.toml").exists():
                    project_root = current
            
            output_dir = project_root / "output"
            output_dir.mkdir(exist_ok=True)
            
            diarization_file = output_dir / f"{call_id}_diarization.json"
            
            diarization_data = {
                'call_id': call_id,
                'timestamp': datetime.now().isoformat(),
                'segments_count': len(segments),
                'speakers_found': len(set(s.get('speaker', '') for s in segments)),
                'segments': segments
            }
            
            with open(diarization_file, 'w', encoding='utf-8') as f:
                json.dump(diarization_data, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"Diarization saved to file: {diarization_file}")
        except Exception as file_error:
            logger.warning(f"Failed to save diarization to file: {file_error}")
        
        # Also try to save to MongoDB (optional)
        if not self.mongo_enabled:
            logger.debug("MongoDB disabled - skipping diarization DB save")
            return
        
        try:
            client = MongoClient(self.mongo_uri, serverSelectionTimeoutMS=2000)
            client.server_info()
            db = client[self.mongo_db]
            collection = db['diarization']
            collection.insert_one({
                'call_id': call_id,
                'segments': segments,
                'timestamp': datetime.now()
            })
            logger.info(f"Diarization saved to MongoDB for call_id: {call_id}")
            client.close()
        except Exception as e:
            logger.warning(f"MongoDB not available, diarization not saved to database: {e}")
            logger.info("Diarization completed successfully (saved to file only)")
    
    def process_batch(self, audio_paths: List[str], call_ids: List[str]) -> List[Dict]:
        """
        Process multiple audio files in parallel (performance optimization).
        
        Args:
            audio_paths: List of audio file paths.
            call_ids: List of unique call identifiers.
            
        Returns:
            List of transcription dictionaries.
        """
        try:
            with ThreadPoolExecutor() as executor:
                results = list(executor.map(self.transcribe_audio, audio_paths, call_ids))
            return results
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            raise


class TextProcessor:
    """Handles text preprocessing and cleaning."""
    
    def __init__(self):
        """Initialize text processor with BERT tokenizer and stop words."""
        base_stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'
        }
        self.stop_words = set(base_stop_words).union(SPACY_STOP_WORDS)
        global _BERT_TOKENIZER
        if _BERT_TOKENIZER is None:
            _BERT_TOKENIZER = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_tokenizer = _BERT_TOKENIZER
    
    def mask_pii(self, text: str) -> str:
        """
        Mask PII in text for security (SRS1 regulatory constraints).
        Handles cases where spaCy model is not available.
        
        Args:
            text: Raw text input.
            
        Returns:
            Anonymized text.
        """
        if not text or not text.strip():
            return text
        
        if not SPACY_AVAILABLE or nlp is None:
            # Fallback to simple regex-based masking if spaCy not available
            import re
            # Mask phone numbers (require separators or parentheses to reduce false positives)
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
    
    def _normalize_text_preserve(self, text: str) -> str:
        """Normalize text while preserving emojis/fillers."""
        import re
        text = unicodedata.normalize('NFKC', text)
        text = ''.join(ch for ch in text if unicodedata.category(ch)[0] != 'C')
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def clean_text(self, text: str) -> str:
        """
        Clean and preprocess text.
        
        Args:
            text: Raw text input.
            
        Returns:
            Cleaned text.
        """
        text = self.mask_pii(text)  # Mask PII first
        text = self._normalize_text_preserve(text)
        return text.lower()
    
    def tokenize_text(self, text: str) -> Dict:
        """
        Tokenize text for BERT (FR-4).
        
        Args:
            text: Input text.
            
        Returns:
            Dictionary with BERT tokens.
        """
        masked_text = self.mask_pii(text)
        normalized = self._normalize_text_preserve(masked_text)
        tokenized = self.bert_tokenizer(
            normalized,
            return_tensors='pt',
            max_length=512,
            truncation=True,
            padding=True
        )
        return {'tokens': tokenized, 'raw_text': normalized, 'masked_text': masked_text}
    
    def extract_text_features(self, text: str) -> Dict:
        """
        Extract text features for sentiment analysis (FR-4: BERT-compatible).
        
        Args:
            text: Input text.
            
        Returns:
            Dictionary with BERT tokens and basic features.
        """
        token_package = self.tokenize_text(text)
        tokenized = token_package['tokens']
        masked_text = token_package.get('masked_text', '')
        raw_text = token_package['raw_text']
        word_tokens = raw_text.split()
        capital_ratio = sum(1 for c in masked_text if c.isupper()) / len(masked_text) if masked_text else 0
        
        features = {
            'tokens': tokenized,  # For BERT input
            'word_count': len(word_tokens),
            'char_count': len(raw_text),
            'avg_word_length': np.mean([len(word) for word in word_tokens]) if word_tokens else 0,
            'sentence_count': len([s for s in raw_text.split('.') if s.strip()]),
            'exclamation_count': raw_text.count('!'),
            'question_count': raw_text.count('?'),
            'capital_ratio': capital_ratio
        }
        return features
    
    def _assign_text_to_diarization_segments(self, diar_segments: List[Dict], transcription_segments: Optional[List[Dict]]) -> None:
        """
        Align Whisper transcription text with diarization segments.
        Uses transcription segments line-by-line to avoid repetition.
        Mutates diar_segments in-place by filling the "text" field.
        """
        if not diar_segments or not transcription_segments:
            return
        
        # Ensure segments are sorted by time
        diar_segments.sort(key=lambda x: x.get('start', 0))
        transcription_segments = sorted(transcription_segments, key=lambda x: x.get('start', 0))
        
        # IMPROVED: Use transcription segments more intelligently
        # Map each diarization segment to the best-matching transcription segment(s)
        transcript_idx = 0
        num_transcripts = len(transcription_segments)
        
        # Track which transcription segments have been used to avoid repetition
        used_transcript_indices = set()
        
        for segment in diar_segments:
            seg_start = segment.get('start', 0)
            seg_end = segment.get('end', 0)
            collected_text_parts = []
            collected_indices = []
            
            # Advance pointer to first transcript overlapping this segment
            while transcript_idx < num_transcripts and transcription_segments[transcript_idx].get('end', 0) <= seg_start:
                transcript_idx += 1
            
            # Find all transcription segments that overlap with this diarization segment
            idx = transcript_idx
            while idx < num_transcripts:
                transcript = transcription_segments[idx]
                t_start = transcript.get('start', 0)
                t_end = transcript.get('end', 0)
                
                if t_start >= seg_end:
                    break  # Past this diarization segment
                
                # Calculate overlap
                overlap_start = max(seg_start, t_start)
                overlap_end = min(seg_end, t_end)
                overlap_duration = overlap_end - overlap_start
                
                # Only use segments with significant overlap (> 0.1 seconds)
                if overlap_duration > 0.1:
                    text_piece = transcript.get('text', '').strip()
                    if text_piece and idx not in used_transcript_indices:
                        # Calculate overlap ratio to prioritize better matches
                        transcript_duration = t_end - t_start
                        overlap_ratio = overlap_duration / max(transcript_duration, 0.1)
                        
                        collected_text_parts.append((text_piece, overlap_ratio, idx))
                
                idx += 1
            
            # IMPROVED: Select best-matching transcription segments to avoid repetition
            if collected_text_parts:
                # Sort by overlap ratio (best matches first)
                collected_text_parts.sort(key=lambda x: x[1], reverse=True)
                
                # Use the best-matching segment(s) - prefer segments with >50% overlap
                selected_texts = []
                for text_piece, overlap_ratio, idx in collected_text_parts:
                    if overlap_ratio > 0.5:  # Strong overlap - use this segment
                        if idx not in used_transcript_indices:
                            selected_texts.append(text_piece)
                            used_transcript_indices.add(idx)
                    elif len(selected_texts) == 0:  # No strong matches yet, use best available
                        if idx not in used_transcript_indices:
                            selected_texts.append(text_piece)
                            used_transcript_indices.add(idx)
                            break  # Use only one if overlap is weak
                
                # IMPROVED: Deduplicate repeated phrases
                if selected_texts:
                    # Remove exact duplicates
                    unique_texts = []
                    seen_phrases = set()
                    for text in selected_texts:
                        # Normalize text for comparison (lowercase, remove extra spaces)
                        normalized = " ".join(text.lower().split())
                        if normalized not in seen_phrases:
                            unique_texts.append(text)
                            seen_phrases.add(normalized)
                    
                    # Join with space, but avoid repetition
                    segment_text = " ".join(unique_texts).strip()
                    
                    # IMPROVED: Remove obvious repetitions (e.g., "Hello. Hello. Hello.")
                    import re
                    # Remove repeated phrases (2+ consecutive identical phrases)
                    segment_text = re.sub(r'\b(\w+(?:\s+\w+)*)\s+\1\s+\1+', r'\1', segment_text)
                    # Remove repeated single words (3+ times)
                    segment_text = re.sub(r'\b(\w+)\s+\1\s+\1+', r'\1', segment_text)
                    
                    if segment_text:
                        segment["text"] = segment_text
    
    def segment_conversation(self, text: str, segments: List[Dict], call_id: str, transcription_segments: Optional[List[Dict]] = None) -> List[Dict]:
        """
        Segment conversation by speaker and time (supports FR-8).
        
        Args:
            text: Full conversation text.
            segments: List of diarization segments.
            call_id: Unique call identifier.
            
        Returns:
            List of processed segments.
        """
        processed_segments = []
        text = self.mask_pii(text)
        
        # Fill diarization segments with actual text before processing
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
        
        self.save_segments(processed_segments, call_id)  # FR-7
        return processed_segments
    
    def save_segments(self, segments: List[Dict], call_id: str):
        """
        Save conversation segments to MongoDB (FR-7).
        
        Args:
            segments: List of processed segments.
            call_id: Unique call identifier.
        """
        if not MONGO_ENABLED:
            logger.debug("MongoDB disabled - skipping segment storage")
            return
        try:
            client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=2000)
            db = client[MONGO_DB_NAME]
            collection = db['segments']
            collection.insert_one({
                'call_id': call_id,
                'segments': segments,
                'timestamp': datetime.now()
            })
            logger.info(f"Segments saved for call_id: {call_id}")
        except Exception as e:
            logger.error(f"Error saving segments: {e}")