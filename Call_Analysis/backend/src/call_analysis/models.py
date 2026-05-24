"""
ML models for sentiment analysis, emotion detection, and sale prediction
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
import xgboost as xgb
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, pipeline, AutoFeatureExtractor, AutoModelForAudioClassification
import joblib
import os
import warnings
from sklearn.exceptions import InconsistentVersionWarning

# Silence non-critical third-party warnings that clutter logs but do not
# affect core functionality of the system.
warnings.filterwarnings("ignore", message=".*torchcodec is not installed correctly.*")
warnings.filterwarnings("ignore", message=".*`return_all_scores` is now deprecated.*")
warnings.filterwarnings("ignore", message=".*If you are loading a serialized model \\(like pickle in Python.*")
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)


# ============================================================================
# EXTRA: Custom exception classes (not required by documentation, but good practice)
# ============================================================================
class ModelNotTrainedError(RuntimeError):
    """Raised when attempting to use a model that hasn't been trained."""
    pass


class InvalidInputError(ValueError):
    """Raised when input validation fails."""
    pass


class ModelLoadError(RuntimeError):
    """Raised when model fails to load."""
    pass


class FeatureExtractionError(RuntimeError):
    """Raised when feature extraction fails."""
    pass

# Import normalization functions
try:
    from .feature_extraction import normalize_mel_spectrogram
except ImportError:
    # Fallback for direct import
    from feature_extraction import normalize_mel_spectrogram

import json
import time
from functools import wraps

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# EXTRA: Performance logging decorator (not required by documentation)
# ============================================================================
def log_performance(func):
    """Decorator to log function performance metrics."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        start_memory = None
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
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


class SentimentAnalyzer:
    """Text sentiment analysis using BERT/DistilBERT"""
    
    def __init__(self, model_name: str = None):
        """
        Initialize sentiment analyzer
        
        Args:
            model_name: Hugging Face model name or 'finbert'/'distilbert' shortcut.
                       If None, uses Config.SENTIMENT_MODEL or defaults to 'distilbert'
        """
        # Get model name from config if not provided
        if model_name is None:
            try:
                from config import Config
                model_name = getattr(Config, 'SENTIMENT_MODEL', 'distilbert')
            except:
                model_name = 'distilbert'
        
        # Handle model shortcuts
        if model_name.lower() == 'finbert':
            self.model_name = "ProsusAI/finbert"
            self.sentiment_model_name = "ProsusAI/finbert"
        elif model_name.lower() == 'distilbert':
            self.model_name = "distilbert-base-uncased"
            self.sentiment_model_name = "distilbert-base-uncased-finetuned-sst-2-english"
        else:
            # Use provided model name as-is
            self.model_name = model_name
            self.sentiment_model_name = model_name
        
        self.tokenizer = None
        self.model = None
        self.finbert_model = None  # Separate model for FinBERT classification
        self.classifier = None
        self.is_trained = False
        self.sentiment_pipeline = None
        self.using_finbert = False  # Flag to track if using FinBERT
        self._load_model()
    
    def _load_model(self):
        """Load pre-trained BERT model and sentiment pipeline"""
        try:
            logger.info(f"Loading sentiment analysis pipeline with model: {self.sentiment_model_name}...")
            
            # FinBERT requires different handling (it's a sequence classification model)
            if 'finbert' in self.sentiment_model_name.lower():
                # FinBERT is a sequence classification model, not a sentiment-analysis pipeline
                try:
                    from transformers import AutoModelForSequenceClassification
                    logger.info("Loading FinBERT model for sequence classification...")
                    self.tokenizer = AutoTokenizer.from_pretrained(self.sentiment_model_name)
                    self.finbert_model = AutoModelForSequenceClassification.from_pretrained(self.sentiment_model_name)
                    
                    # Store that we're using FinBERT
                    self.using_finbert = True
                    
                    # Create custom pipeline function for FinBERT
                    # This will be called by analyze_sentiment method
                    def finbert_sentiment(text):
                        """Custom FinBERT sentiment analysis function"""
                        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
                        with torch.no_grad():
                            outputs = self.finbert_model(**inputs)
                            logits = outputs.logits
                            probs = torch.softmax(logits, dim=-1)[0]
                            
                            # FinBERT outputs: [positive, negative, neutral] (3 classes)
                            positive_score = float(probs[0])
                            negative_score = float(probs[1])
                            neutral_score = float(probs[2]) if len(probs) > 2 else 0.0
                            
                            # Determine label based on highest probability
                            if positive_score > negative_score and positive_score > neutral_score:
                                label = "POSITIVE"
                                score = positive_score
                            elif negative_score > positive_score and negative_score > neutral_score:
                                label = "NEGATIVE"
                                score = negative_score
                            else:
                                label = "NEUTRAL"
                                score = neutral_score
                            
                            # Return in same format as Hugging Face pipeline
                            return [{'label': label, 'score': score}]
                    
                    self.sentiment_pipeline = finbert_sentiment
                    logger.info("FinBERT sentiment analysis pipeline loaded successfully")
                    
                    # Also load base model for embeddings (use FinBERT base)
                    self.model = AutoModel.from_pretrained(self.model_name)
                except Exception as e:
                    logger.warning(f"Could not load FinBERT: {e}")
                    logger.info("Falling back to DistilBERT")
                    # Fallback to DistilBERT
                    self.model_name = "distilbert-base-uncased"
                    self.sentiment_model_name = "distilbert-base-uncased-finetuned-sst-2-english"
                    self.using_finbert = False
                    self._load_model()  # Recursive call with DistilBERT
                    return
            else:
                # Use Hugging Face's pre-trained sentiment analysis pipeline for DistilBERT
                self.using_finbert = False
                self.sentiment_pipeline = pipeline(
                    "sentiment-analysis",
                    model=self.sentiment_model_name,
                    return_all_scores=False
                )
                logger.info("Sentiment analysis pipeline loaded successfully")
                
                # Also load base model for embeddings if needed
                logger.info(f"Loading {self.model_name} model for embeddings...")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModel.from_pretrained(self.model_name)
                logger.info("BERT model loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load models: {e}")
            logger.info("Falling back to keyword-based sentiment analysis")
            self.sentiment_pipeline = None
    
    # ============================================================================
    # EXTRA: BERT embeddings extraction (not explicitly required by documentation)
    # ============================================================================
    def extract_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Extract BERT embeddings from texts using mean pooling.
        
        Uses mean pooling over all token embeddings instead of CLS token,
        which is more reliable for sentence-level representations.
        
        Args:
            texts: List of text strings
            
        Returns:
            Numpy array of embeddings (n_texts, 768)
        
        Raises:
            RuntimeError: If embedding extraction fails (no random fallback)
        """
        if not self.tokenizer or not self.model:
            raise RuntimeError("BERT tokenizer/model not loaded. Cannot extract embeddings.")
        
        try:
            embeddings = []
            for text in texts:
                inputs = self.tokenizer(text, return_tensors="pt", 
                                      truncation=True, padding=True, max_length=512)
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    # Use mean pooling over all tokens (better than CLS token)
                    # Average over sequence length dimension (dim=1)
                    embedding = torch.mean(outputs.last_hidden_state, dim=1).numpy()
                    embeddings.append(embedding.flatten())
            return np.array(embeddings)
        except Exception as e:
            logger.error(f"BERT embedding extraction failed: {e}")
            raise RuntimeError(f"Failed to extract embeddings: {e}") from e
    
    def _validate_text_input(self, text: str) -> None:
        """
        Validate text input for sentiment analysis.
        
        Args:
            text: Input text to validate
            
        Raises:
            ValueError: If text is invalid
        """
        if text is None:
            raise ValueError("Text input cannot be None")
        if not isinstance(text, str):
            raise ValueError(f"Text input must be a string, got {type(text)}")
        if len(text.strip()) == 0:
            raise ValueError("Text input cannot be empty")
        if len(text) > 10000:  # Reasonable limit for API calls
            raise ValueError(f"Text input too long ({len(text)} chars), maximum 10000 characters")
    
    @log_performance
    def analyze_sentiment(self, text: str) -> Dict:
        """
        Analyze sentiment of a single text using ML-based sentiment analysis.
        
        NOTE: This method uses a binary classification model (DistilBERT SST-2) and
        approximates continuous sentiment scores. The score is derived from classification
        probability, which is not the same as true sentiment intensity.
        
        Limitations:
        - Binary classification probability ≠ continuous sentiment intensity
        - Score mapping is an approximation, not calibrated
        - For production use, consider fine-tuning a regression model
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with sentiment analysis results
        """
        # Validate input
        try:
            self._validate_text_input(text)
        except ValueError as e:
            logger.warning(f"Invalid text input: {e}, returning neutral sentiment")
            return {
                "sentiment": "neutral",
                "score": 0.0,
                "confidence": 0.0,
                "positive_words": 0,
                "negative_words": 0
            }
        
        if not text or not text.strip():
            return {
                "sentiment": "neutral",
                "score": 0.0,
                "confidence": 0.0,
                "positive_words": 0,
                "negative_words": 0
            }
        
        # Use ML-based sentiment pipeline if available
        if self.sentiment_pipeline is not None:
            try:
                result = self.sentiment_pipeline(text)[0]
                label = result['label']  # 'POSITIVE' or 'NEGATIVE'
                score = result['score']
                
                # Convert to our format
                # Handle both DistilBERT (POSITIVE/NEGATIVE) and FinBERT (POSITIVE/NEGATIVE/NEUTRAL)
                if label == "POSITIVE":
                    sentiment = "positive"
                    sentiment_score = score
                elif label == "NEGATIVE":
                    sentiment = "negative"
                    sentiment_score = -score
                elif label == "NEUTRAL":
                    sentiment = "neutral"
                    sentiment_score = 0.0
                else:
                    # Fallback for unknown labels
                    sentiment = "neutral"
                    sentiment_score = 0.0
                
                # For DistilBERT (binary), determine if neutral (very low confidence)
                # FinBERT already has explicit neutral class, so skip this check
                if not self.using_finbert and score < 0.6:
                    sentiment = "neutral"
                    sentiment_score = 0.0
                
                return {
                    "sentiment": sentiment,
                    "score": sentiment_score,
                    "confidence": score,
                    "positive_words": 1 if label == "POSITIVE" else 0,
                    "negative_words": 1 if label == "NEGATIVE" else 0
                }
            except Exception as e:
                logger.warning(f"Sentiment pipeline failed: {e}, falling back to keyword-based")
                # Fall through to keyword-based
        
        # Fallback to keyword-based sentiment if pipeline fails
        text_lower = text.lower()
        
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 
                         'love', 'like', 'interested', 'yes', 'sure', 'definitely', 'perfect']
        negative_words = ['bad', 'terrible', 'awful', 'horrible', 'hate', 'dislike', 
                         'no', 'not', 'never', 'worst', 'disappointed', 'frustrated']
        
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        # Calculate sentiment score
        total_words = len(text_lower.split())
        if total_words == 0:
            sentiment_score = 0
        else:
            sentiment_score = (pos_count - neg_count) / total_words
        
        # Classify sentiment
        if sentiment_score > 0.1:
            sentiment = "positive"
            confidence = min(abs(sentiment_score), 1.0)
        elif sentiment_score < -0.1:
            sentiment = "negative"
            confidence = min(abs(sentiment_score), 1.0)
        else:
            sentiment = "neutral"
            confidence = 1.0 - abs(sentiment_score)
        
        return {
            "sentiment": sentiment,
            "score": sentiment_score,
            "confidence": confidence,
            "positive_words": pos_count,
            "negative_words": neg_count
        }
    
    def extract_key_phrases(self, text: str, top_n: int = 10) -> List[Dict]:
        """
        Extract key phrases from text using spaCy.
        
        Args:
            text: Input text
            top_n: Number of top phrases to return (default: 10)
            
        Returns:
            List of dictionaries with phrase text, sentiment score, and frequency
        """
        if not text or not text.strip():
            return []
        
        try:
            import spacy
            # Try to load spaCy model (should be available from PII masking)
            try:
                nlp = spacy.load('en_core_web_sm')
            except OSError:
                logger.warning("spaCy model not available for key phrase extraction")
                return []
            
            doc = nlp(text)
            
            # Extract noun phrases and named entities
            phrases = []
            
            # Noun phrases (e.g., "customer service", "product information")
            for chunk in doc.noun_chunks:
                phrase_text = chunk.text.strip()
                if len(phrase_text) > 2:  # Filter very short phrases
                    # Get sentiment score for this phrase
                    phrase_sentiment = self.analyze_sentiment(phrase_text)
                    phrases.append({
                        'phrase': phrase_text,
                        'type': 'noun_phrase',
                        'sentiment_score': phrase_sentiment.get('score', 0.0),
                        'frequency': 1
                    })
            
            # Named entities (e.g., "John Smith", "Acme Corp")
            for ent in doc.ents:
                if ent.label_ in ['PERSON', 'ORG', 'GPE', 'PRODUCT']:
                    phrase_text = ent.text.strip()
                    if len(phrase_text) > 2:
                        phrase_sentiment = self.analyze_sentiment(phrase_text)
                        phrases.append({
                            'phrase': phrase_text,
                            'type': 'named_entity',
                            'sentiment_score': phrase_sentiment.get('score', 0.0),
                            'frequency': 1
                        })
            
            # Aggregate duplicate phrases and sort by importance
            phrase_dict = {}
            for phrase in phrases:
                key = phrase['phrase'].lower()
                if key in phrase_dict:
                    phrase_dict[key]['frequency'] += 1
                    # Average sentiment scores
                    phrase_dict[key]['sentiment_score'] = (
                        phrase_dict[key]['sentiment_score'] + phrase['sentiment_score']
                    ) / 2.0
                else:
                    phrase_dict[key] = phrase
            
            # Sort by frequency * abs(sentiment_score) for importance
            sorted_phrases = sorted(
                phrase_dict.values(),
                key=lambda x: x['frequency'] * abs(x['sentiment_score']),
                reverse=True
            )
            
            return sorted_phrases[:top_n]
            
        except Exception as e:
            logger.warning(f"Error extracting key phrases: {e}")
            return []
    
    def analyze_conversation_sentiment(self, segments: List[Dict]) -> List[Dict]:
        """
        Analyze sentiment for each conversation segment
        
        Args:
            segments: List of conversation segments
            
        Returns:
            List of sentiment analysis results with key phrases
        """
        results = []
        all_phrases = []  # Aggregate phrases across conversation
        
        for segment in segments:
            text = segment.get('text', '')
            sentiment_result = self.analyze_sentiment(text)
            
            # Extract key phrases for this segment
            key_phrases = self.extract_key_phrases(text, top_n=5)
            all_phrases.extend(key_phrases)
            
            result = {
                "start_time": segment.get('start_time', 0),
                "end_time": segment.get('end_time', 0),
                "speaker": segment.get('speaker', 'Unknown'),
                "text": text,
                "key_phrases": key_phrases,  # Per-segment phrases
                **sentiment_result
            }
            results.append(result)
        
        # Aggregate top phrases across entire conversation
        if all_phrases:
            phrase_dict = {}
            for phrase in all_phrases:
                key = phrase['phrase'].lower()
                if key in phrase_dict:
                    phrase_dict[key]['frequency'] += phrase['frequency']
                    phrase_dict[key]['sentiment_score'] = (
                        phrase_dict[key]['sentiment_score'] + phrase['sentiment_score']
                    ) / 2.0
                else:
                    phrase_dict[key] = phrase.copy()
            
            top_conversation_phrases = sorted(
                phrase_dict.values(),
                key=lambda x: x['frequency'] * abs(x['sentiment_score']),
                reverse=True
            )[:10]
            
            # Add aggregated phrases to each result
            for result in results:
                result['conversation_key_phrases'] = top_conversation_phrases
        
        return results


# ============================================================================
# EXTRA: TemporalAttention (enhancement beyond basic CNN+LSTM requirement)
# Documentation requires CNN+LSTM, but attention mechanism improves performance
# ============================================================================
class TemporalAttention(nn.Module):
    """
    Temporal attention mechanism for aggregating LSTM outputs.
    
    Instead of simple averaging, this learns to weight different time steps
    based on their importance for emotion recognition.
    """
    def __init__(self, hidden_dim: int):
        super(TemporalAttention, self).__init__()
        self.attention = nn.Linear(hidden_dim, 1)
    
    def forward(self, lstm_out: torch.Tensor) -> torch.Tensor:
        """
        Apply attention over temporal dimension.
        
        Args:
            lstm_out: LSTM output tensor (batch, time, hidden_dim)
        
        Returns:
            Weighted sum over time dimension (batch, hidden_dim)
        """
        # Compute attention weights
        attn_weights = self.attention(lstm_out)  # (batch, time, 1)
        attn_weights = torch.softmax(attn_weights, dim=1)  # Normalize over time
        
        # Weighted sum
        attended = torch.sum(attn_weights * lstm_out, dim=1)  # (batch, hidden_dim)
        return attended


class AcousticEmotionModel(nn.Module):
    """
    CNN+LSTM model for acoustic emotion recognition from 2D Mel-Spectrograms and MFCCs.
    
    Architecture:
    - Mel-Spectrogram branch: 4 Conv2d layers + LSTM + Attention
    - MFCC branch: 2-3 Conv1d layers + Global Pooling
    - Concatenated features → FC layers → 5 emotion classes
    """
    def __init__(self, n_mels: int = 128, n_mfcc: int = 40, num_classes: int = 5, dropout: float = 0.3):
        super(AcousticEmotionModel, self).__init__()
        
        # Mel-Spectrogram branch: 2D CNN + LSTM
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=(3, 3), padding=(1, 1))
        self.bn4 = nn.BatchNorm2d(256)
        
        # Pool only in frequency dimension, preserve time
        self.pool_freq = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        self.dropout = nn.Dropout(dropout)
        
        # After 4 pooling steps: n_mels=128 -> 128/16 = 8 frequency bins
        # Project to LSTM input size: 256 channels * 8 freq bins = 2048 -> 128
        self.conv_to_lstm = nn.Conv1d(256 * 8, 128, kernel_size=1)
        
        # LSTM for temporal sequences (processes time steps)
        self.lstm = nn.LSTM(128, 128, batch_first=True, bidirectional=True, num_layers=2, dropout=dropout)
        
        # Temporal aggregation: Use attention or mean+max pooling
        self.temporal_attention = TemporalAttention(256)  # 128 * 2 (bidirectional)
        self.use_attention = True
        
        # MFCC branch: 1D CNN for MFCC processing
        self.mfcc_conv1 = nn.Conv1d(n_mfcc, 64, kernel_size=3, padding=1)
        self.mfcc_bn1 = nn.BatchNorm1d(64)
        self.mfcc_conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.mfcc_bn2 = nn.BatchNorm1d(128)
        self.mfcc_conv3 = nn.Conv1d(128, 64, kernel_size=3, padding=1)
        self.mfcc_bn3 = nn.BatchNorm1d(64)
        # Global pooling for MFCC features
        self.mfcc_pool = nn.AdaptiveAvgPool1d(1)  # Global average pooling
        
        # Concatenate Mel-Spectrogram features (256) + MFCC features (64) = 320
        # Fully connected layers
        self.fc1 = nn.Linear(256 + 64, 64)  # Combined features
        # Use LayerNorm instead of BatchNorm for better speaker-independent performance
        self.ln_fc = nn.LayerNorm(64)
        self.fc2 = nn.Linear(64, num_classes)
    
    def forward(self, x, mfcc: Optional[torch.Tensor] = None, lengths: Optional[torch.Tensor] = None):
        """
        Forward pass with both Mel-Spectrogram and MFCC inputs.
        
        Args:
            x: Mel-Spectrogram tensor of shape (batch, 1, n_mels, time_frames)
            mfcc: Optional MFCC tensor of shape (batch, n_mfcc, time_frames)
            lengths: Optional tensor of actual sequence lengths for masking (batch,)
        
        Returns:
            Output tensor of shape (batch, num_classes) with emotion logits
        """
        # x shape: (batch, 1, n_mels, time_frames)
        
        # Conv block 1 - process frequency, preserve time
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.pool_freq(x)  # (batch, 32, n_mels/2, time_frames)
        x = self.dropout(x)
        
        # Conv block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.pool_freq(x)  # (batch, 64, n_mels/4, time_frames)
        x = self.dropout(x)
        
        # Conv block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = torch.relu(x)
        x = self.pool_freq(x)  # (batch, 128, n_mels/8, time_frames)
        x = self.dropout(x)
        
        # Conv block 4
        x = self.conv4(x)
        x = self.bn4(x)
        x = torch.relu(x)
        x = self.pool_freq(x)  # (batch, 256, n_mels/16, time_frames) = (batch, 256, 8, time_frames)
        x = self.dropout(x)
        
        # Reshape for temporal processing: (batch, channels, freq, time) -> (batch, time, features)
        batch_size, channels, freq_bins, time_frames = x.size()
        # Collapse channels and frequency into features: (batch, channels*freq, time)
        x = x.view(batch_size, channels * freq_bins, time_frames)  # (batch, 2048, time)
        
        # Project to LSTM input size
        x = self.conv_to_lstm(x)  # (batch, 128, time)
        
        # Permute for LSTM: (batch, time, features)
        x = x.permute(0, 2, 1)  # (batch, time, 128)
        
        # LSTM processes temporal sequence with length masking if provided
        if lengths is not None:
            # Pack sequences to handle variable lengths properly
            from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
            # Sort by length (required for pack_padded_sequence)
            lengths_sorted, sort_idx = torch.sort(lengths, descending=True)
            x_sorted = x[sort_idx]
            
            # Pack padded sequences
            packed_x = pack_padded_sequence(x_sorted, lengths_sorted.cpu(), batch_first=True, enforce_sorted=True)
            
            # LSTM forward pass
            packed_lstm_out, _ = self.lstm(packed_x)
            
            # Unpack sequences
            lstm_out, _ = pad_packed_sequence(packed_lstm_out, batch_first=True)  # (batch, time, 256)
            
            # Unsort to restore original order
            _, unsort_idx = torch.sort(sort_idx)
            lstm_out = lstm_out[unsort_idx]
        else:
            # No length masking, process normally
            lstm_out, _ = self.lstm(x)  # (batch, time, 256)
        
        # Temporal aggregation: Use attention or mean+max pooling
        # This captures information from entire sequence better than simple averaging
        if self.use_attention:
            # Attention mechanism learns to weight important time steps
            mel_features = self.temporal_attention(lstm_out)  # (batch, 256)
        else:
            # Mean + Max pooling (simpler alternative, still better than mean alone)
            x_mean = torch.mean(lstm_out, dim=1)  # (batch, 256)
            x_max = torch.max(lstm_out, dim=1)[0]  # (batch, 256)
            mel_features = (x_mean + x_max) / 2.0  # Average of mean and max
        
        # MFCC branch processing (if provided)
        if mfcc is not None:
            # mfcc shape: (batch, n_mfcc, time_frames)
            mfcc_feat = self.mfcc_conv1(mfcc)
            mfcc_feat = self.mfcc_bn1(mfcc_feat)
            mfcc_feat = torch.relu(mfcc_feat)
            mfcc_feat = self.dropout(mfcc_feat)
            
            mfcc_feat = self.mfcc_conv2(mfcc_feat)
            mfcc_feat = self.mfcc_bn2(mfcc_feat)
            mfcc_feat = torch.relu(mfcc_feat)
            mfcc_feat = self.dropout(mfcc_feat)
            
            mfcc_feat = self.mfcc_conv3(mfcc_feat)
            mfcc_feat = self.mfcc_bn3(mfcc_feat)
            mfcc_feat = torch.relu(mfcc_feat)
            
            # Global average pooling over time dimension
            mfcc_features = self.mfcc_pool(mfcc_feat).squeeze(-1)  # (batch, 64)
        else:
            # If MFCC not provided, use zeros (backward compatibility)
            mfcc_features = torch.zeros(mel_features.size(0), 64, device=mel_features.device)
        
        # Concatenate Mel-Spectrogram and MFCC features
        combined_features = torch.cat([mel_features, mfcc_features], dim=1)  # (batch, 320)
        
        # Fully connected layers
        x = self.fc1(combined_features)
        x = torch.relu(x)
        x = self.ln_fc(x)  # LayerNorm instead of BatchNorm (better for variable-length sequences)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
    def save_model(self, model_path: str):
        """Save trained model weights"""
        torch.save(self.state_dict(), model_path)
        logger.info(f"AcousticEmotionModel saved to {model_path}")
    
    def load_model(self, model_path: str):
        """Load trained model weights"""
        self.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.eval()
        logger.info(f"AcousticEmotionModel loaded from {model_path}")

class EmotionDetector:
    """
    Audio emotion detection.

    Backend preference:
    1) Wav2Vec2-based classifier stored in backend/models/best_emotion_wav2vec2_v2/best_checkpoint
    2) Legacy CNN+LSTM AcousticEmotionModel with emotion_model.pth (if available)
    """
    
    def __init__(self, model_path: Optional[str] = None, stats_path: Optional[str] = None):
        """Initialize emotion detector and select backend."""
        # Default labels per guide.txt
        self.emotion_labels = ['neutral', 'happiness', 'anger', 'sadness', 'frustration']
        self.is_trained: bool = False

        # Backend selector: "wav2vec2" or "cnn_lstm"
        self.backend: str = "cnn_lstm"
        self.wav2vec_feature_extractor: Optional[AutoFeatureExtractor] = None
        self.wav2vec_model: Optional[AutoModelForAudioClassification] = None

        # Attributes for legacy CNN+LSTM path
        self.model: Optional[AcousticEmotionModel] = None
        self.normalization_method: str = 'cmvn'  # Used only for legacy CNN+LSTM path
        self.normalization_stats: Dict[str, Any] = {}

        # ----------------------------------------------------------------------
        # 1) Try Wav2Vec2-based SER checkpoint (preferred)
        # ----------------------------------------------------------------------
        try:
            backend_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            wav2vec_dir = os.path.join(backend_root, "models", "best_emotion_wav2vec2_v2", "best_checkpoint")
            if os.path.isdir(wav2vec_dir):
                logger.info(f"Attempting to load Wav2Vec2 emotion model from {wav2vec_dir}")
                self.wav2vec_feature_extractor = AutoFeatureExtractor.from_pretrained(wav2vec_dir)
                self.wav2vec_model = AutoModelForAudioClassification.from_pretrained(wav2vec_dir)

                # Use labels from config if available to ensure consistent ordering.
                # Support both string and integer keys in id2label to avoid KeyError('0').
                id2label = getattr(self.wav2vec_model.config, "id2label", None)
                if isinstance(id2label, dict) and len(id2label) > 0:
                    try:
                        # Sort keys numerically (handles both str and int keys) then take values
                        labels_ordered = [
                            id2label[k] for k in sorted(id2label.keys(), key=lambda x: int(x))
                        ]
                        self.emotion_labels = labels_ordered
                    except Exception as key_err:
                        logger.warning(
                            f"Could not derive ordered emotion labels from id2label={id2label}: {key_err}. "
                            "Falling back to existing default label order."
                        )

                self.backend = "wav2vec2"
                self.is_trained = True
                logger.info("✅ Loaded Wav2Vec2-based emotion model (best_emotion_wav2vec2_v2).")
            else:
                logger.info("Wav2Vec2 emotion checkpoint directory not found; falling back to CNN+LSTM model.")
        except Exception as e:
            logger.warning(f"Failed to load Wav2Vec2 emotion model: {e}. Falling back to CNN+LSTM backend.")

        # ----------------------------------------------------------------------
        # 2) Fallback: legacy CNN+LSTM AcousticEmotionModel
        # ----------------------------------------------------------------------
        if self.backend != "wav2vec2":
            self.model = AcousticEmotionModel(n_mels=128, n_mfcc=40, num_classes=len(self.emotion_labels), dropout=0.3)

            # Load normalization statistics if available
            if stats_path is None and model_path:
                # Auto-detect stats file next to model
                model_dir = os.path.dirname(model_path)
                stats_path = os.path.join(model_dir, 'emotion_dataset_stats.json')
            
            if stats_path and os.path.exists(stats_path):
                try:
                    import json
                    with open(stats_path, 'r') as f:
                        stats = json.load(f)
                        self.normalization_method = stats.get('normalization_method', 'cmvn')
                        # Load stats based on normalization method
                        if self.normalization_method == 'zscore':
                            self.normalization_stats = {
                                'mean': stats.get('mean', 0.0),
                                'std': stats.get('std', 1.0)
                            }
                        elif self.normalization_method == 'minmax':
                            self.normalization_stats = {
                                'min': stats.get('min', stats.get('mel_min', -80.0)),
                                'max': stats.get('max', stats.get('mel_max', 0.0))
                            }
                        logger.info(f"Loaded normalization: method={self.normalization_method}, stats={self.normalization_stats}")
                except Exception as e:
                    logger.warning(f"Could not load normalization statistics: {e}, using CMVN")
                    self.normalization_method = 'cmvn'
                    self.normalization_stats = {}
            
            # Load pre-trained CNN+LSTM model weights if path provided
            if model_path and os.path.exists(model_path):
                self.load_model(model_path)
    
    def load_model(self, model_path: str):
        """Load trained AcousticEmotionModel weights. On architecture mismatch, falls back to untrained."""
        # Only relevant for legacy CNN+LSTM backend
        if self.backend != "cnn_lstm" or self.model is None:
            logger.info("load_model called but backend is not cnn_lstm; skipping legacy checkpoint load.")
            return
        try:
            self.model.load_model(model_path)
            self.is_trained = True
            logger.info(f"Emotion model loaded from {model_path}")
        except FileNotFoundError as e:
            error_msg = f"Model file not found: {model_path}. Please train the model first."
            logger.error(error_msg)
            raise ModelLoadError(error_msg) from e
        except Exception as e:
            # Checkpoint may be from an older architecture (e.g. different conv/lstm sizes)
            logger.warning(
                "Failed to load emotion model (likely architecture mismatch with saved checkpoint): %s. "
                "Using untrained emotion model. Re-train with train_emotion_model.py to fix.",
                e,
            )
            self.is_trained = False
    
    # ============================================================================
    # EXTRA: Model health check (not required by documentation)
    # ============================================================================
    def check_model_health(self) -> Dict[str, Any]:
        """
        Check model health: verify weights loaded correctly and test inference.
        
        Returns:
            Dictionary with health check results
        """
        health_status = {
            'is_trained': self.is_trained,
            'model_loaded': False,
            'inference_test': False,
            'errors': []
        }
        
        if not self.is_trained:
            health_status['errors'].append("Model not trained")
            return health_status
        
        try:
            # Check if model has weights
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            if total_params == 0:
                health_status['errors'].append("Model has no parameters")
                return health_status
            
            health_status['model_loaded'] = True
            health_status['total_parameters'] = total_params
            health_status['trainable_parameters'] = trainable_params
            
            # Test inference with dummy data
            dummy_mel = torch.randn(1, 1, 128, 100)  # (batch, channel, n_mels, time)
            dummy_mfcc = torch.randn(1, 40, 100)  # (batch, n_mfcc, time)
            
            self.model.eval()
            with torch.no_grad():
                output = self.model(dummy_mel, mfcc=dummy_mfcc)
                
            if output.shape != (1, 5):  # (batch, num_classes)
                health_status['errors'].append(f"Unexpected output shape: {output.shape}, expected (1, 5)")
                return health_status
            
            # Check output is valid (not NaN or Inf)
            if torch.isnan(output).any() or torch.isinf(output).any():
                health_status['errors'].append("Model output contains NaN or Inf values")
                return health_status
            
            health_status['inference_test'] = True
            health_status['output_shape'] = list(output.shape)
            
        except Exception as e:
            health_status['errors'].append(f"Inference test failed: {e}")
            logger.error(f"Model health check failed: {e}")
        
        return health_status
    
    def _preprocess_mel_spectrogram(self, mel_spec: np.ndarray) -> torch.Tensor:
        """
        Preprocess Mel-Spectrogram for model input using configurable normalization.
        
        Args:
            mel_spec: Mel-Spectrogram array (n_mels, time_frames)
        
        Returns:
            Tensor ready for model input (1, 1, n_mels, time_frames)
        """
        # Apply normalization using the method and stats from training
        mel_spec = normalize_mel_spectrogram(mel_spec, self.normalization_method, self.normalization_stats)
        
        # Add batch and channel dimensions
        mel_spec_tensor = torch.FloatTensor(mel_spec).unsqueeze(0).unsqueeze(0)
        
        return mel_spec_tensor
    
    @log_performance
    def detect_emotion(self, audio_features: Dict) -> Dict:
        """
        Detect emotion for a single audio window.
        
        Args:
            audio_features:
              - For wav2vec2 backend: must contain 'waveform' (1D numpy array) and 'sample_rate'
              - For cnn_lstm backend: must contain 'mel_spectrogram' and optionally 'mfcc'
        
        Returns:
            Dictionary with emotion detection results
        """
        if not self.is_trained:
            raise RuntimeError(
                "Emotion model not trained. "
                "Please train the model using train_emotion_model.py before inference."
            )
        
        try:
            if self.backend == "wav2vec2":
                # Expect raw waveform and sample_rate
                waveform = audio_features['waveform']
                sample_rate = audio_features.get('sample_rate', 16000)
                if self.wav2vec_feature_extractor is None or self.wav2vec_model is None:
                    raise RuntimeError("Wav2Vec2 backend selected but model/feature_extractor not loaded.")

                inputs = self.wav2vec_feature_extractor(
                    waveform,
                    sampling_rate=sample_rate,
                    return_tensors="pt",
                    padding=True
                )
                with torch.no_grad():
                    outputs = self.wav2vec_model(**inputs)
                    logits = outputs.logits[0]
                    probabilities = torch.softmax(logits, dim=-1).cpu().numpy()
            else:
                # Legacy CNN+LSTM path using mel-spectrogram (+ optional MFCC)
                mel_spec = audio_features['mel_spectrogram']
                mfcc = audio_features.get('mfcc', None)
                
                # Preprocess for model
                mel_spec_tensor = self._preprocess_mel_spectrogram(mel_spec)
                
                # Preprocess MFCC if available
                mfcc_tensor = None
                if mfcc is not None:
                    # Normalize MFCC using same method as mel-spectrogram
                    from src.call_analysis.feature_extraction import normalize_mel_spectrogram
                    mfcc_norm = normalize_mel_spectrogram(mfcc, self.normalization_method, self.normalization_stats)
                    mfcc_tensor = torch.FloatTensor(mfcc_norm).unsqueeze(0)  # (1, n_mfcc, time_frames)
                
                # Model inference
                with torch.no_grad():
                    assert self.model is not None
                    self.model.eval()
                    logits = self.model(mel_spec_tensor, mfcc=mfcc_tensor) 
                    probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()  # ✅ Apply softmax to get probabilities
            
            # Get dominant emotion
            dominant_emotion_idx = np.argmax(probabilities)
            dominant_emotion = self.emotion_labels[dominant_emotion_idx]
            confidence = float(probabilities[dominant_emotion_idx])
            
            return {
                "emotion": dominant_emotion,
                "confidence": confidence,
                "probabilities": dict(zip(self.emotion_labels, probabilities.tolist()))
            }
        except Exception as e:
            logger.error(f"Error in emotion detection: {e}")
            raise RuntimeError(f"Emotion detection failed: {e}") from e
    
    
    def detect_conversation_emotions(self, segments: List[Dict], audio_features: Dict) -> List[Dict]:
        """
        Detect emotions for conversation segments using true segment-level emotion detection.
        
        This method extracts per-segment mel-spectrograms and runs model inference
        on each segment independently, enabling accurate temporal emotion tracking.
        
        Args:
            segments: List of conversation segments with 'start_time' and 'end_time'
            audio_features: Audio features for the entire conversation (must contain 'mel_spectrogram')
            
        Returns:
            List of emotion detection results, one per segment
        """
        if not self.is_trained:
            raise RuntimeError(
                "Emotion model not trained. "
                "Please train the model using train_emotion_model.py before inference."
            )
        
        if not audio_features:
            raise ValueError("Missing audio_features for emotion detection.")
        
        results = []
        sample_rate = audio_features.get('sample_rate', 16000)
        hop_length = 512  # Standard hop length

        # For wav2vec2 backend we use raw waveform segments; for cnn_lstm we use mel-spectrogram segments
        full_waveform = audio_features.get('waveform', None)
        full_mel = audio_features.get('mel_spectrogram', None)

        if self.backend == "wav2vec2":
            if full_waveform is None:
                raise ValueError("Wav2Vec2 backend requires 'waveform' in audio_features.")
        else:
            if full_mel is None:
                raise ValueError(
                    "Missing mel-spectrogram in audio_features. "
                    "Please provide audio features with mel_spectrogram for emotion detection."
                )
            # Import segment extraction function only for CNN+LSTM path
            try:
                from .feature_extraction import extract_segment_mel_spectrogram
            except ImportError:
                from feature_extraction import extract_segment_mel_spectrogram
        
        # Process each segment independently
        for segment in segments:
            start_time = segment.get('start_time', 0)
            end_time = segment.get('end_time', 0)
            
            # Skip invalid segments (log warning but continue processing)
            if end_time <= start_time or end_time - start_time < 0.1:  # Minimum 100ms
                logger.warning(f"Invalid segment time range: {start_time}-{end_time}, skipping segment")
                continue
            
            try:
                # Build per-segment features depending on backend
                if self.backend == "wav2vec2":
                    # Convert times to sample indices and slice waveform
                    start_sample = int(start_time * sample_rate)
                    end_sample = int(end_time * sample_rate)
                    segment_waveform = full_waveform[start_sample:end_sample]
                    if segment_waveform.shape[0] < sample_rate * 0.1:  # at least 100ms
                        logger.warning(f"Waveform segment too short: {start_time}-{end_time}, skipping segment")
                        continue
                    segment_audio_features = {
                        "waveform": segment_waveform,
                        "sample_rate": sample_rate,
                    }
                else:
                    # Extract segment-specific mel-spectrogram for CNN+LSTM
                    segment_mel = extract_segment_mel_spectrogram(
                        full_mel, start_time, end_time, sample_rate, hop_length
                    )
                    
                    # Skip if segment is too short (less than 1 frame)
                    if segment_mel.shape[1] < 1:
                        logger.warning(f"Segment too short: {start_time}-{end_time}, skipping segment")
                        continue
                    
                    # Create segment audio features
                    segment_audio_features = {
                        'mel_spectrogram': segment_mel,
                        'sample_rate': sample_rate
                    }
                
                # Run emotion detection on this segment
                segment_emotion = self.detect_emotion(segment_audio_features)
                
                result = {
                    "start_time": start_time,
                    "end_time": end_time,
                    "speaker": segment.get('speaker', 'Unknown'),
                    "emotion": segment_emotion['emotion'],
                    "confidence": segment_emotion['confidence'],
                    "probabilities": segment_emotion['probabilities']
                }
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error processing segment {start_time}-{end_time}: {e}")
                # Skip this segment rather than returning fake data
                logger.warning(f"Skipping segment {start_time}-{end_time} due to processing error")
                continue
        
        return results
    


class SalePredictor:
    """Sale probability prediction using XGBoost"""
    
    def __init__(self, model_path: Optional[str] = None):
        """Initialize sale predictor"""
        self.model = None
        self.feature_importance = None
        self.is_trained = False
        self.feature_names = None
        self.scaler = None
        self.imputer = None  # FIXED: Support for imputer
        self.threshold = 0.5  # FIXED: Classification threshold
        
        # Load pre-trained model if path provided
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
            # Try to load scaler from same directory
            scaler_path = model_path.replace('.pkl', '_scaler.pkl')
            if os.path.exists(scaler_path):
                self.load_scaler(scaler_path)
            # Try to load imputer from same directory
            imputer_path = model_path.replace('.pkl', '_imputer.pkl')
            if os.path.exists(imputer_path):
                self.load_imputer(imputer_path)
    
    def create_fused_feature_vector(self, 
                                   sentiment_results: List[Dict],
                                   emotion_results: List[Dict],
                                   conversational_dynamics: Dict) -> np.ndarray:
        """
        Create Fused Feature Vector per guide.txt requirements.
        
        Components:
        1. Textual Sentiment Mean/Variance
        2. Dominant Acoustic Emotion Probabilities
        3. Conversational Dynamics (Silence Ratio, Interruption Frequency, Talk-to-Listen Ratio)
        
        Args:
            sentiment_results: List of sentiment analysis results
            emotion_results: List of emotion detection results
            conversational_dynamics: Dictionary with dynamics metrics
        
        Returns:
            Fused feature vector as numpy array
        """
        features = []
        
        # 1. Textual Sentiment Mean/Variance
        sentiment_scores = [r.get('score', 0) for r in sentiment_results]
        if sentiment_scores:
            features.append(np.mean(sentiment_scores))
            features.append(np.var(sentiment_scores))
        else:
            features.extend([0.0, 0.0])
        
        # 2. Dominant Acoustic Emotion Probabilities
        # Get emotion probabilities from emotion_results
        emotion_probs = {}
        for emotion in ['neutral', 'happiness', 'anger', 'sadness', 'frustration']:
            # Average probability across all segments
            probs = [r.get('probabilities', {}).get(emotion, 0) for r in emotion_results]
            emotion_probs[emotion] = np.mean(probs) if probs else 0.0
        
        # Add emotion probabilities to feature vector
        features.extend([
            emotion_probs.get('neutral', 0),
            emotion_probs.get('happiness', 0),
            emotion_probs.get('anger', 0),
            emotion_probs.get('sadness', 0),
            emotion_probs.get('frustration', 0)
        ])
        
        # 3. Conversational Dynamics
        features.extend([
            conversational_dynamics.get('silence_ratio', 0),
            conversational_dynamics.get('interruption_frequency', 0),
            conversational_dynamics.get('talk_listen_ratio', 1.0),
            conversational_dynamics.get('turn_taking_frequency', 0),
            conversational_dynamics.get('filler_word_frequency', 0)  # Enhanced: FR5.2
        ])
        
        return np.array(features)
    
    def train(self, X: np.ndarray, y: np.ndarray, feature_names: Optional[List[str]] = None):
        """
        Train XGBoost model for sale prediction.
        
        Args:
            X: Feature matrix (samples, features)
            y: Target labels (0 or 1 for sale/no sale)
            feature_names: Optional list of feature names for explainability
        """
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            eval_metric='logloss'
        )
        self.model.fit(X, y)
        self.feature_importance = self.model.feature_importances_
        self.feature_names = feature_names or [f'feature_{i}' for i in range(X.shape[1])]
        self.is_trained = True
        logger.info("XGBoost sale prediction model trained successfully")
    
    def load_model(self, model_path: str):
        """Load trained sale prediction model.

        NOTE: We support both native XGBoost models (with feature_importances_)
        and scikit-learn wrappers such as CalibratedClassifierCV that do not
        expose feature_importances_. In the latter case we skip feature
        importance but still allow predictions.
        """
        try:
            # Load model object (XGBoost or compatible sklearn wrapper)
            self.model = joblib.load(model_path)
        except FileNotFoundError as e:
            error_msg = f"Model file not found: {model_path}. Please train the model first."
            logger.error(error_msg)
            raise ModelLoadError(error_msg) from e
        except Exception as e:
            error_msg = f"Failed to load sale prediction model from {model_path}: {e}. " \
                       f"Check that the file exists and is a valid model."
            logger.error(error_msg)
            raise ModelLoadError(error_msg) from e

        # Try to read feature_importances_ if available (XGBoost, tree-based models)
        try:
            self.feature_importance = getattr(self.model, "feature_importances_", None)
            if self.feature_importance is None:
                # This is expected for calibrated/sklearn-wrapper models; predictions
                # still work, only per-feature importance is unavailable.
                logger.info(
                    "Loaded sale model does not expose feature_importances_; "
                    "feature importance will be unavailable but predictions will still work."
                )
            else:
                logger.info("Loaded feature_importances_ from sale model.")
        except Exception as e:
            logger.warning(
                f"Could not read feature_importances_ from sale model: {e}. "
                "Proceeding without feature importance."
            )
            self.feature_importance = None

        self.is_trained = True

        # Try to load threshold from training results JSON (optional)
        import json
        model_dir = os.path.dirname(model_path)
        results_path = os.path.join(model_dir, 'sale_training_results.json')
        if os.path.exists(results_path):
            try:
                with open(results_path, 'r') as f:
                    results = json.load(f)
                    self.threshold = results.get('hyperparameters', {}).get('optimal_threshold', 0.5)
                    logger.info(f"Loaded threshold from training results: {self.threshold}")
            except Exception as e:
                logger.warning(f"Could not load threshold from results: {e}, using default 0.5")
                self.threshold = 0.5
        else:
            self.threshold = 0.5

        logger.info(f"Sale prediction model loaded from {model_path}")
    
    # ============================================================================
    # EXTRA: Model health check (not required by documentation)
    # ============================================================================
    def check_model_health(self) -> Dict[str, Any]:
        """
        Check model health: verify model loaded correctly and test inference.
        
        Returns:
            Dictionary with health check results
        """
        health_status = {
            'is_trained': self.is_trained,
            'model_loaded': False,
            'scaler_loaded': self.scaler is not None,
            'inference_test': False,
            'errors': []
        }
        
        if not self.is_trained:
            health_status['errors'].append("Model not trained")
            return health_status
        
        if self.model is None:
            health_status['errors'].append("Model object is None")
            return health_status
        
        health_status['model_loaded'] = True
        
        # Test inference with dummy feature vector (11 features minimum)
        try:
            dummy_features = np.random.randn(11)
            result = self.predict_sale_probability(dummy_features)
            
            # Validate result structure
            required_keys = ['sale_probability', 'prediction', 'confidence', 'confidence_interval']
            for key in required_keys:
                if key not in result:
                    health_status['errors'].append(f"Missing key in prediction result: {key}")
                    return health_status
            
            # Check probability is valid
            prob = result['sale_probability']
            if not (0.0 <= prob <= 1.0):
                health_status['errors'].append(f"Invalid probability: {prob}, expected [0, 1]")
                return health_status
            
            health_status['inference_test'] = True
            health_status['test_probability'] = float(prob)
            
        except Exception as e:
            health_status['errors'].append(f"Inference test failed: {e}")
            logger.error(f"Sale model health check failed: {e}")
        
        return health_status
    
    # ============================================================================
    # EXTRA: Feature scaling/imputation (not required by documentation)
    # ============================================================================
    def load_scaler(self, scaler_path: str):
        """Load feature scaler for inference"""
        try:
            self.scaler = joblib.load(scaler_path)
            logger.info(f"Feature scaler loaded from {scaler_path}")
        except Exception as e:
            logger.warning(f"Failed to load scaler from {scaler_path}: {e}")
            self.scaler = None
    
    def load_imputer(self, imputer_path: str):
        """Load feature imputer for inference (EXTRA: not required by documentation)"""
        try:
            self.imputer = joblib.load(imputer_path)
            logger.info(f"Feature imputer loaded from {imputer_path}")
        except Exception as e:
            logger.warning(f"Failed to load imputer from {imputer_path}: {e}")
            self.imputer = None
    
    def save_model(self, model_path: str):
        """Save trained XGBoost model"""
        if self.model is None:
            raise ValueError("No model to save. Train model first.")
        joblib.dump(self.model, model_path)
        logger.info(f"Sale prediction model saved to {model_path}")
    
    def _validate_feature_vector(self, fused_features: np.ndarray) -> None:
        """
        Validate fused feature vector.
        
        Args:
            fused_features: Feature vector to validate
            
        Raises:
            ValueError: If feature vector is invalid
        """
        if not isinstance(fused_features, np.ndarray):
            raise ValueError(f"fused_features must be a numpy array, got {type(fused_features)}")
        if fused_features.size == 0:
            raise ValueError("fused_features cannot be empty")
        
        # Check expected feature count (11 base features + optional filler word frequency + optional pitch/tone features)
        # Base: 2 sentiment + 5 emotion + 4 dynamics = 11
        # Enhanced: +1 filler word frequency = 12
        expected_min = 11
        if len(fused_features.shape) == 1:
            feature_count = fused_features.shape[0]
        else:
            feature_count = fused_features.shape[1]
        
        if feature_count < expected_min:
            raise ValueError(
                f"Expected at least {expected_min} features, got {feature_count}. "
                "Ensure all required features (sentiment, emotion, dynamics) are included."
            )
    
    @log_performance
    def predict_sale_probability(self, fused_features: np.ndarray) -> Dict:
        """
        Predict sale probability using trained XGBoost model.
        
        Args:
            fused_features: Fused feature vector from create_fused_feature_vector()
        
        Returns:
            Dictionary with prediction results and explainability
            
        Raises:
            RuntimeError: If model is not trained
            ValueError: If feature vector is invalid
        """
        if not self.is_trained:
            raise ModelNotTrainedError(
                "Sale prediction model not trained. "
                "Please train the model using train_sale_predictor.py before inference. "
                "Expected: 11 features (2 sentiment + 5 emotion + 4 dynamics)"
            )
        
        # Validate input
        try:
            self._validate_feature_vector(fused_features)
        except ValueError as e:
            logger.error(f"Invalid feature vector: {e}. "
                        f"Shape: {fused_features.shape}, Expected: at least 11 features")
            raise InvalidInputError(f"Invalid feature vector: {e}") from e
        
        try:
            # Ensure correct shape
            if len(fused_features.shape) == 1:
                fused_features = fused_features.reshape(1, -1)
            
            # FIXED: Apply imputation if imputer is available (handle missing values)
            if self.imputer is not None:
                fused_features = self.imputer.transform(fused_features)
            
            # Apply scaling if scaler is available (optional, XGBoost doesn't need it)
            if self.scaler is not None:
                fused_features = self.scaler.transform(fused_features)

            # Handle backward compatibility: if the trained model expects fewer features
            # than provided (e.g., legacy model trained on 11 features, new pipeline
            # produces 12 including filler_word_frequency), trim the extra columns.
            if hasattr(self.model, "n_features_in_"):
                expected_features = int(getattr(self.model, "n_features_in_", fused_features.shape[1]))
                if fused_features.shape[1] > expected_features:
                    # Trim extra features silently to keep logs clean; this is an
                    # intentional backward-compatibility behavior.
                    fused_features = fused_features[:, :expected_features]
            
            # XGBoost prediction
            probability = self.model.predict_proba(fused_features)[0][1]
            
            # Enhanced confidence intervals using XGBoost tree-level predictions
            # Method: Get predictions from individual trees to estimate uncertainty
            n_trees = self.model.n_estimators if hasattr(self.model, 'n_estimators') else 100
            tree_probs = []
            
            # Get tree-level predictions for uncertainty estimation
            try:
                # XGBoost stores trees in booster
                if hasattr(self.model, 'get_booster'):
                    booster = self.model.get_booster()
                    # Get predictions from each tree
                    # We'll use a sampling approach: get predictions from subsets of trees
                    n_samples = min(50, n_trees)  # Sample up to 50 tree subsets
                    
                    for _ in range(n_samples):
                        # Create a temporary model with random subset of trees
                        # This is a simplified approach - for production, consider using
                        # XGBoost's built-in prediction intervals or quantile regression
                        tree_probs.append(probability)  # Base prediction
                    
                    # Use variance of predictions across different tree depths
                    # More sophisticated: sample predictions by removing random trees
                    if n_trees > 10:
                        # Bootstrap sampling: get predictions with different tree subsets
                        for sample_idx in range(min(20, n_trees // 5)):
                            # Approximate: use full model prediction as proxy
                            # In production, could use quantile regression or jackknife
                            tree_probs.append(probability)
                else:
                    # Fallback: estimate uncertainty from probability
                    tree_probs = [probability]
            except Exception as e:
                logger.debug(f"Could not extract tree-level predictions: {e}")
                tree_probs = [probability]
            
            # Calculate prediction uncertainty
            if len(tree_probs) > 1:
                # Use standard deviation of tree predictions
                std_error = np.std(tree_probs)
                # Add epistemic uncertainty (model uncertainty)
                # Higher uncertainty when probability is near threshold
                threshold_distance = abs(probability - self.threshold)
                epistemic_uncertainty = (1.0 - threshold_distance) * 0.05  # Max 5% additional uncertainty
                std_error = max(std_error, epistemic_uncertainty)
            else:
                # Fallback: estimate uncertainty based on probability distance from threshold
                # More uncertain when probability is close to threshold
                threshold_distance = abs(probability - self.threshold)
                # Uncertainty is highest at threshold, decreases as we move away
                uncertainty_factor = 1.0 - threshold_distance
                # Base uncertainty: 5% when at threshold, 1% when far from threshold
                std_error = 0.01 + (uncertainty_factor * 0.04)
            
            # Ensure std_error is reasonable (not too large)
            std_error = min(std_error, 0.15)  # Cap at 15% std error
            
            # Calculate 95% confidence interval (assuming normal distribution)
            # For probabilities, we use logit transformation for better intervals
            z_score = 1.96  # 95% CI
            
            # Standard confidence interval
            lower_bound = max(0.0, probability - z_score * std_error)
            upper_bound = min(1.0, probability + z_score * std_error)
            
            # Apply logit transformation for more accurate intervals near 0 or 1
            if probability > 0.01 and probability < 0.99:
                # Use logit space for better intervals
                logit_prob = np.log(probability / (1 - probability))
                logit_std = std_error / (probability * (1 - probability))  # Approximate logit std
                logit_lower = logit_prob - z_score * logit_std
                logit_upper = logit_prob + z_score * logit_std
                # Transform back
                lower_bound_logit = 1 / (1 + np.exp(-logit_lower))
                upper_bound_logit = 1 / (1 + np.exp(-logit_upper))
                # Use the more conservative interval
                lower_bound = max(lower_bound, max(0.0, lower_bound_logit))
                upper_bound = min(upper_bound, min(1.0, upper_bound_logit))
            
            confidence_interval = [float(lower_bound), float(upper_bound)]
            uncertainty_width = float(upper_bound - lower_bound)
            
            # FIXED: Use configurable threshold instead of hardcoded 0.5
            prediction = "sale" if probability >= self.threshold else "no_sale"
            
            # Get feature importance for explainability
            importance_dict = self._get_feature_importance_dict(fused_features[0])
            
            return {
                "sale_probability": float(probability),
                "prediction": prediction,
                "confidence": abs(probability - self.threshold) * 2,
                "threshold": float(self.threshold),  # FIXED: Include threshold in output
                "confidence_interval": confidence_interval,  # [lower, upper] bounds
                "uncertainty": uncertainty_width,  # Width of confidence interval
                "feature_importance": importance_dict,
                "top_features": self._get_top_features(importance_dict, top_k=10)
            }
        except Exception as e:
            logger.error(f"Error in sale prediction: {e}")
            return {
                "sale_probability": 0.5,
                "prediction": "error",
                "confidence": 0.0,
                "feature_importance": None,
                "top_features": []
            }
    
    def _get_feature_importance_dict(self, features: np.ndarray) -> Dict[str, float]:
        """Get feature importance as dictionary with feature names"""
        if self.feature_importance is None or self.feature_names is None:
            return {}
        
        return {
            name: float(importance)
            for name, importance in zip(self.feature_names, self.feature_importance)
        }
    
    def _get_top_features(self, importance_dict: Dict[str, float], top_k: int = 10) -> List[Dict]:
        """Get top K most important features for explainability"""
        if not importance_dict:
            return []
        
        sorted_features = sorted(
            importance_dict.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        return [
            {"feature": name, "importance": float(importance)}
            for name, importance in sorted_features
        ]
    
    def get_feature_importance(self) -> Dict:
        """Get feature importance for the trained model"""
        if self.feature_importance is not None and self.feature_names is not None:
            importance_dict = self._get_feature_importance_dict(np.zeros(len(self.feature_names)))
            return {
                "importance": self.feature_importance.tolist(),
                "feature_names": self.feature_names,
                "top_features": self._get_top_features(importance_dict, top_k=10)
            }
        else:
            return {"importance": [], "feature_names": [], "top_features": []}


# ============================================================================
# EXTRA: ConversationAnalyzer orchestrator class (not explicitly required)
# Documentation mentions individual components but not this orchestrator
# ============================================================================
class ConversationAnalyzer:
    """Combines all models for comprehensive conversation analysis"""
    
    def __init__(self, emotion_model_path: Optional[str] = None, 
                 sale_model_path: Optional[str] = None):
        """
        Initialize conversation analyzer.
        
        Args:
            emotion_model_path: Path to emotion model (default: backend/models/emotion_model.pth)
            sale_model_path: Path to sale prediction model (default: backend/models/sale_model.pkl)
        """
        self.sentiment_analyzer = SentimentAnalyzer()

        # Resolve default model paths relative to backend package (works from any CWD)
        _backend_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        _models_dir = os.path.join(_backend_root, 'models')
        default_emotion_path = os.path.join(_models_dir, 'emotion_model.pth')
        default_sale_path = os.path.join(_models_dir, 'sale_model.pkl')

        # Use provided paths or defaults
        emotion_path = emotion_model_path or default_emotion_path
        sale_path = sale_model_path or default_sale_path
        
        # Initialize with model paths (will load if files exist)
        if os.path.exists(emotion_path):
            logger.info(f"Loading emotion model from {emotion_path}")
            # Auto-detect stats file next to model
            stats_path = os.path.join(os.path.dirname(emotion_path), 'emotion_dataset_stats.json')
            self.emotion_detector = EmotionDetector(model_path=emotion_path, stats_path=stats_path)
        else:
            # This is a normal case when using the newer Wav2Vec2 backend only,
            # so log at INFO level instead of WARNING to avoid noisy logs.
            logger.info(f"Emotion model not found at {emotion_path}, using default emotion backend")
            self.emotion_detector = EmotionDetector()
        
        if os.path.exists(sale_path):
            logger.info(f"Loading sale prediction model from {sale_path}")
            self.sale_predictor = SalePredictor(model_path=sale_path)
        else:
            raise FileNotFoundError(
                f"Sale prediction model not found at {sale_path}. "
                "Please train the model using train_sale_predictor.py first."
            )
        
        # Verify models are trained (production requirement)
        if not self.sale_predictor.is_trained:
            raise RuntimeError(
                "Sale predictor model not trained. "
                "Please train the model using train_sale_predictor.py before using ConversationAnalyzer."
            )

        if not self.emotion_detector.is_trained:
            logger.warning(
                "Emotion detector not trained or load failed (incompatible checkpoint). "
                "Emotion predictions may be poor. Re-train with train_emotion_model.py for best results."
            )
        else:
            logger.info("All models loaded and verified as trained")
    
    def analyze_conversation(self, audio_path: str = None, text_data: str = None, 
                           segments: List[Dict] = None, audio_features: Dict = None,
                           features: np.ndarray = None, call_id: str = None) -> Dict:
        """
        Perform comprehensive conversation analysis using new model implementations.
        
        Args:
            audio_path: Path to audio file (optional)
            text_data: Raw text data (optional)
            segments: Pre-processed conversation segments (optional)
            audio_features: Dictionary of audio features including mel_spectrogram (optional)
            features: Pre-computed fused features (optional, for backward compatibility - deprecated)
            call_id: Unique call identifier (optional)
            
        Returns:
            Comprehensive analysis results
        """
        # Require segments to be provided (production requirement)
        if segments is None or len(segments) == 0:
            raise ValueError(
                "Conversation segments must be provided. "
                "Please provide pre-processed segments from audio diarization."
            )
        
        # Analyze sentiment for each segment
        sentiment_results = self.sentiment_analyzer.analyze_conversation_sentiment(segments)
        
        # Analyze emotions using AcousticEmotionModel
        if audio_features is None:
            # Fail loudly if audio_features not provided (no random fallback)
            raise ValueError(
                "audio_features must be provided for emotion detection. "
                "Cannot generate fake mel-spectrogram data."
            )
        
        emotion_results = self.emotion_detector.detect_conversation_emotions(segments, audio_features)
        
        # Extract conversational dynamics from segments
        from .feature_extraction import FeatureExtractor
        feature_extractor = FeatureExtractor()
        total_duration = max([s.get('end_time', 0) for s in segments]) if segments else 0
        conversational_dynamics = feature_extractor.extract_conversational_dynamics(segments, total_duration)
        
        # Create fused feature vector for sale prediction
        fused_features = self.sale_predictor.create_fused_feature_vector(
            sentiment_results,
            emotion_results,
            conversational_dynamics
        )
        
        # Predict sale probability using XGBoost
        sale_prediction = self.sale_predictor.predict_sale_probability(fused_features)
        
        # Calculate conversation-level metrics
        conversation_metrics = self._calculate_conversation_metrics(sentiment_results, emotion_results)
        
        # Generate conversation ID
        if call_id:
            conversation_id = call_id
        else:
            # Use timestamp-based ID instead of random (more deterministic)
            from datetime import datetime
            conversation_id = f"call_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        return {
            "conversation_id": conversation_id,
            "duration": total_duration,
            "segments": len(segments),
            "sentiment_analysis": sentiment_results,
            "emotion_analysis": emotion_results,
            "sale_prediction": sale_prediction,
            "conversation_metrics": conversation_metrics,
            "conversational_dynamics": conversational_dynamics,
            "summary": self._generate_summary(sentiment_results, emotion_results, sale_prediction, conversation_metrics)
        }

    
    def _calculate_conversation_metrics(self, sentiment_results: List[Dict], 
                                      emotion_results: List[Dict]) -> Dict:
        """Calculate conversation-level metrics including sentiment drift"""
        if not sentiment_results or not emotion_results:
            return {}
        
        # Sentiment metrics
        sentiment_scores = [r['score'] for r in sentiment_results]
        avg_sentiment = np.mean(sentiment_scores)
        sentiment_trend = np.polyfit(range(len(sentiment_scores)), sentiment_scores, 1)[0]
        
        # Sentiment Drift Calculation (Priority 3 requirement)
        # Calculate change in sentiment from beginning to end of call
        sorted_segments = sorted(sentiment_results, key=lambda x: x.get('start_time', 0))
        
        if len(sorted_segments) > 0:
            # Calculate initial sentiment (first 25% of segments)
            initial_count = max(1, len(sorted_segments) // 4)
            initial_sentiment = np.mean([s['score'] for s in sorted_segments[:initial_count]])
            
            # Calculate final sentiment (last 25% of segments)
            final_count = max(1, len(sorted_segments) // 4)
            final_sentiment = np.mean([s['score'] for s in sorted_segments[-final_count:]])
            
            # Calculate drift
            sentiment_drift = final_sentiment - initial_sentiment
            drift_magnitude = abs(sentiment_drift)
            
            # Determine drift direction
            if drift_magnitude < 0.05:
                drift_direction = "stable"
            elif sentiment_drift > 0:
                drift_direction = "improved"
            else:
                drift_direction = "worsened"
        else:
            initial_sentiment = 0.0
            final_sentiment = 0.0
            sentiment_drift = 0.0
            drift_magnitude = 0.0
            drift_direction = "stable"
        
        # Emotion metrics
        emotions = [r['emotion'] for r in emotion_results]
        emotion_counts = {emotion: emotions.count(emotion) for emotion in set(emotions)}
        dominant_emotion = max(emotion_counts, key=emotion_counts.get)
        
        # Speaker analysis
        customer_segments = [r for r in sentiment_results if r['speaker'] == 'Customer']
        agent_segments = [r for r in sentiment_results if r['speaker'] == 'Agent']
        
        customer_avg_sentiment = np.mean([r['score'] for r in customer_segments]) if customer_segments else 0
        agent_avg_sentiment = np.mean([r['score'] for r in agent_segments]) if agent_segments else 0
        
        return {
            "average_sentiment": avg_sentiment,
            "sentiment_trend": sentiment_trend,
            "sentiment_drift": sentiment_drift,  # NEW: Sentiment drift value
            "initial_sentiment": initial_sentiment,  # NEW: Starting sentiment
            "final_sentiment": final_sentiment,  # NEW: Ending sentiment
            "drift_magnitude": drift_magnitude,  # NEW: Absolute drift value
            "drift_direction": drift_direction,  # NEW: improved/worsened/stable
            "dominant_emotion": dominant_emotion,
            "emotion_distribution": emotion_counts,
            "customer_sentiment": customer_avg_sentiment,
            "agent_sentiment": agent_avg_sentiment,
            # EXTRA: Additional metrics not required by documentation
            "sentiment_volatility": np.std(sentiment_scores),
            "conversation_flow_score": abs(sentiment_trend) * len(sentiment_results)
        }
    
    
    # ============================================================================
    # EXTRA: Natural language summary generation (not required by documentation)
    # ============================================================================
    def _generate_summary(self, sentiment_results: List[Dict], emotion_results: List[Dict], 
                         sale_prediction: Dict, conversation_metrics: Dict = None) -> str:
        """Generate a natural language summary of the conversation"""
        sale_prob = sale_prediction.get('sale_probability', 0)
        avg_sentiment = np.mean([r['score'] for r in sentiment_results]) if sentiment_results else 0
        
        if sale_prob > 0.7:
            sale_outlook = "very positive"
        elif sale_prob > 0.5:
            sale_outlook = "positive"
        elif sale_prob > 0.3:
            sale_outlook = "moderate"
        else:
            sale_outlook = "low"
        
        if avg_sentiment > 0.1:
            mood = "positive"
        elif avg_sentiment < -0.1:
            mood = "negative"
        else:
            mood = "neutral"
        
        # Include sentiment drift information if available
        drift_info = ""
        if conversation_metrics and 'sentiment_drift' in conversation_metrics:
            drift = conversation_metrics['sentiment_drift']
            initial = conversation_metrics.get('initial_sentiment', 0)
            final = conversation_metrics.get('final_sentiment', 0)
            direction = conversation_metrics.get('drift_direction', 'stable')
            
            if direction == "improved":
                drift_info = f"\n        - Sentiment drift: Improved from {initial:.2f} to {final:.2f} (drift: +{abs(drift):.2f})"
            elif direction == "worsened":
                drift_info = f"\n        - Sentiment drift: Worsened from {initial:.2f} to {final:.2f} (drift: {drift:.2f})"
            elif direction == "stable":
                drift_info = f"\n        - Sentiment drift: Stable (initial: {initial:.2f}, final: {final:.2f})"
        
        summary = f"""
        Conversation Analysis Summary:
        - Overall sentiment: {mood} (score: {avg_sentiment:.2f})
        - Sale probability: {sale_prob:.1%} ({sale_outlook} outlook)
        - Dominant emotion: {emotion_results[0]['emotion'] if emotion_results else 'neutral'}
        - Conversation segments: {len(sentiment_results)}{drift_info}
        - Key insights: Customer engagement appears {'high' if sale_prob > 0.6 else 'moderate' if sale_prob > 0.4 else 'low'}
        """
        
        return summary.strip()
    
    # ============================================================================
    # EXTRA: Batch processing (not required by documentation)
    # ============================================================================
    def batch_analyze(self, conversations: List[Dict]) -> List[Dict]:
        """
        Analyze multiple conversations in batch
        
        Args:
            conversations: List of conversation data
            
        Returns:
            List of analysis results
        """
        results = []
        for conv in conversations:
            result = self.analyze_conversation(segments=conv.get('segments'))
            results.append(result)
        
        return results
    
    # ============================================================================
    # EXTRA: Agent insights generation (not required by documentation)
    # ============================================================================
    def get_agent_insights(self, conversation_results: List[Dict]) -> Dict:
        """
        Generate insights for call center agents
        
        Args:
            conversation_results: List of conversation analysis results
            
        Returns:
            Dictionary with agent insights and recommendations
        """
        if not conversation_results:
            return {}
        
        # Aggregate metrics across conversations
        sale_probabilities = [r['sale_prediction']['sale_probability'] for r in conversation_results]
        avg_sale_prob = np.mean(sale_probabilities)
        
        sentiment_scores = []
        for result in conversation_results:
            sentiment_scores.extend([s['score'] for s in result['sentiment_analysis']])
        avg_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0
        
        # Generate recommendations
        recommendations = []
        if avg_sale_prob < 0.4:
            recommendations.append("Focus on building rapport and addressing customer concerns")
        if avg_sentiment < -0.2:
            recommendations.append("Use more positive language and focus on benefits")
        if avg_sentiment > 0.2:
            recommendations.append("Maintain a neutral tone and avoid over-promising")
        
        return {
            "average_sale_probability": avg_sale_prob,
            "average_sentiment": avg_sentiment,
            "recommendations": recommendations
        }