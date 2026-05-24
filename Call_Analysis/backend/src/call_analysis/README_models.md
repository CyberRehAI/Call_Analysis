## Models Module (`models.py`)

This document explains the **models** module, which contains all the machine‑learning logic for:

- **Text sentiment & key phrases** – `SentimentAnalyzer`
- **Acoustic emotion detection** – `AcousticEmotionModel` + `EmotionDetector`
- **Sale probability prediction** – `SalePredictor`
- **End‑to‑end conversation analysis** – `ConversationAnalyzer`

Together, these models take **speaker‑segmented transcription + audio features** and produce:

- Per‑segment sentiment and emotions
- Conversation‑level metrics
- A sale probability (0–1) with uncertainty
- A natural‑language summary

---

## 1. `SentimentAnalyzer` – Text Sentiment & Key Phrases

### Purpose

**Goal:** Given a piece of text (utterance/segment), estimate:

- Sentiment label: `positive`, `negative`, `neutral`
- Sentiment score: signed value (\>0 = positive, \<0 = negative)
- Confidence
- Simple positive/negative word counts
- Key phrases with sentiment

It wraps Hugging Face models (DistilBERT or FinBERT) and falls back to a keyword‑based method if the transformer pipeline fails.

### Model Selection and Loading

```12:45:C:\Users\PMYLS\Desktop\fyp\Call_Analysis\backend\src\call_analysis\models.py
class SentimentAnalyzer:
    """Text sentiment analysis using BERT/DistilBERT"""
    
    def __init__(self, model_name: str = None):
        ...
        if model_name is None:
            from config import Config
            model_name = getattr(Config, 'SENTIMENT_MODEL', 'distilbert')
        
        if model_name.lower() == 'finbert':
            self.model_name = "ProsusAI/finbert"
            self.sentiment_model_name = "ProsusAI/finbert"
        elif model_name.lower() == 'distilbert':
            self.model_name = "distilbert-base-uncased"
            self.sentiment_model_name = "distilbert-base-uncased-finetuned-sst-2-english"
        else:
            self.model_name = model_name
            self.sentiment_model_name = model_name
        ...
        self._load_model()
```

**Explanation:**

- If no model name is passed, it reads `Config.SENTIMENT_MODEL` (env‑driven).
- Two shortcuts:
  - **`distilbert`** → general sentiment (SST‑2).
  - **`finbert`** → financial sentiment (ProsusAI/finbert).

The `_load_model` method:

- For **DistilBERT**, uses `pipeline("sentiment-analysis")`.
- For **FinBERT**, loads `AutoModelForSequenceClassification` and wraps it in a small helper that returns the same `{label, score}` format as a pipeline.
- If loading fails, sets `sentiment_pipeline = None` and logs a warning, so the code can gracefully fall back to a keyword method.

### Sentiment for a Single Text

```230:276:C:\Users\PMYLS\Desktop\fyp\Call_Analysis\backend\src\call_analysis\models.py
    @log_performance
    def analyze_sentiment(self, text: str) -> Dict:
        """
        Analyze sentiment of a single text using ML-based sentiment analysis.
        """
        # Validate input (none/empty/too long) – returns neutral if invalid.
        self._validate_text_input(text)
        
        if self.sentiment_pipeline is not None:
            result = self.sentiment_pipeline(text)[0]
            label = result['label']  # 'POSITIVE', 'NEGATIVE', maybe 'NEUTRAL'
            score = result['score']
            
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
                sentiment = "neutral"
                sentiment_score = 0.0
            
            # For DistilBERT (binary), very low confidence is treated as neutral
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
        
        # Fallback: keyword-based sentiment
        ...
```

**Explanation:**

- Uses the HF pipeline’s `label` and `score` to compute:
  - `"positive"` → positive `score`
  - `"negative"` → negative `score`
  - `"neutral"` → `score = 0.0`
- For DistilBERT (binary POSITIVE/NEGATIVE), low‑confidence outputs (`score < 0.6`) are treated as **neutral**.
- If the pipeline is unavailable, a simple positive/negative keyword count is used as fallback.

### Conversation‑Level Sentiment and Key Phrases

```389:433:C:\Users\PMYLS\Desktop\fyp\Call_Analysis\backend\src\call_analysis\models.py
    def analyze_conversation_sentiment(self, segments: List[Dict]) -> List[Dict]:
        """
        Analyze sentiment for each conversation segment
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
                "key_phrases": key_phrases,
                **sentiment_result
            }
            results.append(result)
        
        # Aggregate top phrases for the entire conversation and attach them
        ...
        for result in results:
            result['conversation_key_phrases'] = top_conversation_phrases
        return results
```

**Explanation:**

- Loops over each segment (from preprocessing) and:
  - Runs `analyze_sentiment`.
  - Extracts key phrases via spaCy (noun chunks + entities) plus sentiment score for each phrase.
- Then computes **global top phrases** over the entire call and attaches them to each segment under `conversation_key_phrases`.

---

## 2. Acoustic Emotion – `AcousticEmotionModel` and `EmotionDetector`

### Purpose

- The **AcousticEmotionModel** is a CNN+LSTM network that processes:
  - 2D mel‑spectrograms (128 mel bands over time)
  - 1D MFCC features
- The **EmotionDetector** wraps this model:
  - Loads weights + normalization stats.
  - Applies the right normalization (CMVN, z‑score, etc.).
  - Runs per‑segment emotion inference.

The output for each segment is a probability distribution over:

`['neutral', 'happiness', 'anger', 'sadness', 'frustration']`.

### Model Architecture (High Level)

```472:555:C:\Users\PMYLS\Desktop\fyp\Call_Analysis\backend\src\call_analysis\models.py
class AcousticEmotionModel(nn.Module):
    """
    CNN+LSTM model for acoustic emotion recognition from 2D Mel-Spectrograms and MFCCs.
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
        
        # After pooling → project to LSTM input size
        self.conv_to_lstm = nn.Conv1d(256 * 8, 128, kernel_size=1)
        
        # Bi-LSTM over time
        self.lstm = nn.LSTM(128, 128, batch_first=True, bidirectional=True, num_layers=2, dropout=dropout)
        
        # Temporal attention
        self.temporal_attention = TemporalAttention(256)
        self.use_attention = True
        
        # MFCC branch: 1D CNN + global pooling
        self.mfcc_conv1 = nn.Conv1d(n_mfcc, 64, kernel_size=3, padding=1)
        ...
        self.mfcc_pool = nn.AdaptiveAvgPool1d(1)
        
        # Fusion + fully connected layers
        self.fc1 = nn.Linear(256 + 64, 64)
        self.ln_fc = nn.LayerNorm(64)
        self.fc2 = nn.Linear(64, num_classes)
```

**Explanation (in plain language):**

- **Mel branch**:
  - 4 convolution blocks over the 2D mel image.
  - Pool only along frequency, to keep the **time axis** intact.
  - Reshape into a sequence of feature vectors and feed a bi‑LSTM.
  - Use an **attention layer** to emphasize time steps that are most informative for emotion.
- **MFCC branch**:
  - Convolutions over MFCC sequences.
  - Global average pooling to get a single vector.
- **Fusion**:
  - Concatenate mel and MFCC features → fully connected layers → logits over 5 emotions.

### EmotionDetector – From Audio Features to Emotion

```620:681:C:\Users\PMYLS\Desktop\fyp\Call_Analysis\backend\src\call_analysis\models.py
class EmotionDetector:
    """Audio emotion detection using CNN+LSTM"""
    
    def __init__(self, model_path: Optional[str] = None, stats_path: Optional[str] = None):
        self.emotion_labels = ['neutral', 'happiness', 'anger', 'sadness', 'frustration']
        self.model = AcousticEmotionModel(n_mels=128, n_mfcc=40, num_classes=5, dropout=0.3)
        self.is_trained = False
        self.normalization_method = 'cmvn'
        self.normalization_stats = {}
        ...
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
            self.is_trained = True
    
    @log_performance
    def detect_emotion(self, audio_features: Dict) -> Dict:
        """
        Detect emotion using AcousticEmotionModel.
        Expects 'mel_spectrogram' and optionally 'mfcc'.
        """
        if not self.is_trained:
            raise RuntimeError("Emotion model not trained. ...")
        
        mel_spec = audio_features['mel_spectrogram']
        mfcc = audio_features.get('mfcc', None)
        
        # Normalize mel-spectrogram
        mel_spec_tensor = self._preprocess_mel_spectrogram(mel_spec)
        
        # Normalize MFCC if provided
        if mfcc is not None:
            mfcc_tensor = torch.FloatTensor(...).unsqueeze(0)
        else:
            mfcc_tensor = None
        
        with torch.no_grad():
            self.model.eval()
            logits = self.model(mel_spec_tensor, mfcc=mfcc_tensor)
            probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()
        
        dominant_idx = np.argmax(probabilities)
        dominant_emotion = self.emotion_labels[dominant_idx]
        confidence = float(probabilities[dominant_idx])
        
        return {
            "emotion": dominant_emotion,
            "confidence": confidence,
            "probabilities": dict(zip(self.emotion_labels, probabilities.tolist()))
        }
```

**Explanation:**

- Validates that the model is trained.
- Normalizes mel‑spectrogram using the training‑time method (e.g. CMVN).
- Optionally uses MFCC.
- Runs the model, softmaxes the logits, and returns:
  - The top emotion label.
  - The associated probability as confidence.
  - Probabilities for all 5 emotions.

### Per‑Segment Emotion over a Conversation

```732:803:C:\Users\PMYLS\Desktop\fyp\Call_Analysis\backend\src\call_analysis\models.py
    def detect_conversation_emotions(self, segments: List[Dict], audio_features: Dict) -> List[Dict]:
        """
        Detect emotions for conversation segments using true segment-level emotion detection.
        """
        if not self.is_trained:
            raise RuntimeError("Emotion model not trained. ...")
        if not audio_features or 'mel_spectrogram' not in audio_features:
            raise ValueError("Missing mel-spectrogram in audio_features.")
        
        results = []
        full_mel = audio_features['mel_spectrogram']
        sample_rate = audio_features.get('sample_rate', 16000)
        hop_length = 512
        
        from .feature_extraction import extract_segment_mel_spectrogram
        
        for segment in segments:
            start_time = segment.get('start_time', 0)
            end_time = segment.get('end_time', 0)
            if end_time <= start_time or end_time - start_time < 0.1:
                continue
            
            segment_mel = extract_segment_mel_spectrogram(
                full_mel, start_time, end_time, sample_rate, hop_length
            )
            if segment_mel.shape[1] < 1:
                continue
            
            segment_audio_features = {
                'mel_spectrogram': segment_mel,
                'sample_rate': sample_rate
            }
            segment_emotion = self.detect_emotion(segment_audio_features)
            
            results.append({
                "start_time": start_time,
                "end_time": end_time,
                "speaker": segment.get('speaker', 'Unknown'),
                "emotion": segment_emotion['emotion'],
                "confidence": segment_emotion['confidence'],
                "probabilities": segment_emotion['probabilities']
            })
```

**Explanation:**

- For each segment:
  - Slices the **full** mel‑spectrogram into a **segment‑specific** mel window.
  - Runs `detect_emotion` on that slice.
  - Returns a list of per‑segment emotions with times and speakers.

---

## 3. `SalePredictor` – XGBoost Sale Probability

### Purpose

**Goal:** Predict the **probability of a successful sale** using:

- Textual sentiment statistics (from `SentimentAnalyzer`)
- Aggregated acoustic emotion probabilities (from `EmotionDetector`)
- Conversational dynamics (from `FeatureExtractor`: silence, interruptions, talk–listen ratio, etc.)

It also:

- Applies an **optimal classification threshold** (read from training results when available).
- Estimates **uncertainty** and a **95% confidence interval** around the probability.
- Exposes **feature importance** for explainability.

### Fused Feature Vector

```882:920:C:\Users\PMYLS\Desktop\fyp\Call_Analysis\backend\src\call_analysis\models.py
    def create_fused_feature_vector(self, 
                                   sentiment_results: List[Dict],
                                   emotion_results: List[Dict],
                                   conversational_dynamics: Dict) -> np.ndarray:
        """
        Create Fused Feature Vector per guide.txt requirements.
        
        Components:
        1. Textual Sentiment Mean/Variance
        2. Dominant Acoustic Emotion Probabilities
        3. Conversational Dynamics
        """
        features = []
        
        # 1. Sentiment mean/variance
        sentiment_scores = [r.get('score', 0) for r in sentiment_results]
        if sentiment_scores:
            features.append(np.mean(sentiment_scores))
            features.append(np.var(sentiment_scores))
        else:
            features.extend([0.0, 0.0])
        
        # 2. Emotion probabilities (average across segments)
        emotion_probs = {}
        for emotion in ['neutral', 'happiness', 'anger', 'sadness', 'frustration']:
            probs = [r.get('probabilities', {}).get(emotion, 0) for r in emotion_results]
            emotion_probs[emotion] = np.mean(probs) if probs else 0.0
        features.extend([
            emotion_probs.get('neutral', 0),
            emotion_probs.get('happiness', 0),
            emotion_probs.get('anger', 0),
            emotion_probs.get('sadness', 0),
            emotion_probs.get('frustration', 0)
        ])
        
        # 3. Conversational dynamics
        features.extend([
            conversational_dynamics.get('silence_ratio', 0),
            conversational_dynamics.get('interruption_frequency', 0),
            conversational_dynamics.get('talk_listen_ratio', 1.0),
            conversational_dynamics.get('turn_taking_frequency', 0),
            conversational_dynamics.get('filler_word_frequency', 0)
        ])
        
        return np.array(features)
```

**Explanation:**

- Fused vector includes:
  - 2 sentiment stats.
  - 5 averaged emotion probabilities.
  - 5 conversational dynamics metrics.
- This fixed‑length vector is what XGBoost consumes.

### Prediction with Uncertainty

```1049:1135:C:\Users\PMYLS\Desktop\fyp\Call_Analysis\backend\src\call_analysis\models.py
    @log_performance
    def predict_sale_probability(self, fused_features: np.ndarray) -> Dict:
        """
        Predict sale probability using trained XGBoost model.
        """
        if not self.is_trained:
            raise ModelNotTrainedError("Sale prediction model not trained. ...")
        
        self._validate_feature_vector(fused_features)
        
        if len(fused_features.shape) == 1:
            fused_features = fused_features.reshape(1, -1)
        if self.imputer is not None:
            fused_features = self.imputer.transform(fused_features)
        if self.scaler is not None:
            fused_features = self.scaler.transform(fused_features)
        
        probability = self.model.predict_proba(fused_features)[0][1]
        
        # Estimate uncertainty and 95% confidence interval
        # (combination of tree-sampling and distance-from-threshold heuristic)
        ...
        prediction = "sale" if probability >= self.threshold else "no_sale"
        
        importance_dict = self._get_feature_importance_dict(fused_features[0])
        
        return {
            "sale_probability": float(probability),
            "prediction": prediction,
            "confidence": abs(probability - self.threshold) * 2,
            "threshold": float(self.threshold),
            "confidence_interval": confidence_interval,
            "uncertainty": uncertainty_width,
            "feature_importance": importance_dict,
            "top_features": self._get_top_features(importance_dict, top_k=10)
        }
```

**Explanation:**

- Applies optional **imputer** and **scaler** from training.
- Uses `predict_proba` to get the probability of sale.
- Computes a rough **uncertainty** and **confidence interval** by:
  - Using tree‑level info when possible, or
  - Falling back to how far the probability is from the threshold.
- Outputs both a scalar probability and a label (`"sale"`/`"no_sale"`), plus:
  - The threshold used.
  - A 95% interval `[lower, upper]`.
  - Feature importance and top contributing features.

---

## 4. `ConversationAnalyzer` – End‑to‑End Orchestration

### Purpose

**Goal:** Pull everything together:

- Segment‑level sentiment (from `SentimentAnalyzer`)
- Segment‑level emotions (from `EmotionDetector`)
- Conversational dynamics (from `FeatureExtractor`)
- Sale probability (from `SalePredictor`)
- Aggregate conversation metrics and a text summary

It is the **main entry point** used by the FastAPI backend when you call `/api/analyze`.

### Initialization

```1365:1411:C:\Users\PMYLS\Desktop\fyp\Call_Analysis\backend\src\call_analysis\models.py
class ConversationAnalyzer:
    """Combines all models for comprehensive conversation analysis"""
    
    def __init__(self, emotion_model_path: Optional[str] = None, 
                 sale_model_path: Optional[str] = None):
        self.sentiment_analyzer = SentimentAnalyzer()

        # Resolve default model paths relative to backend package
        _backend_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        _models_dir = os.path.join(_backend_root, 'models')
        default_emotion_path = os.path.join(_models_dir, 'emotion_model.pth')
        default_sale_path = os.path.join(_models_dir, 'sale_model.pkl')

        emotion_path = emotion_model_path or default_emotion_path
        sale_path = sale_model_path or default_sale_path
        
        if os.path.exists(emotion_path):
            self.emotion_detector = EmotionDetector(model_path=emotion_path, ...)
        else:
            logger.warning("Emotion model not found, using untrained model")
            self.emotion_detector = EmotionDetector()
        
        if os.path.exists(sale_path):
            self.sale_predictor = SalePredictor(model_path=sale_path)
        else:
            raise FileNotFoundError("Sale prediction model not found. Train first.")
        
        if not self.sale_predictor.is_trained:
            raise RuntimeError("Sale predictor model not trained. ...")
        if not self.emotion_detector.is_trained:
            logger.warning("Emotion detector not trained or incompatible; results may be poor.")
```

**Explanation:**

- Locates model files relative to the backend root (`backend/models/…`), so it works regardless of current working directory.
- Fails fast if the sale model is missing or untrained.
- Warns (but does not crash) if the emotion model is missing; that lets the rest of the system still work for text/sale outputs if needed.

### analyze_conversation – Main Pipeline

```1413:1483:C:\Users\PMYLS\Desktop\fyp\Call_Analysis\backend\src\call_analysis\models.py
    def analyze_conversation(self, audio_path: str = None, text_data: str = None, 
                           segments: List[Dict] = None, audio_features: Dict = None,
                           features: np.ndarray = None, call_id: str = None) -> Dict:
        """
        Perform comprehensive conversation analysis using new model implementations.
        """
        if segments is None or len(segments) == 0:
            raise ValueError("Conversation segments must be provided.")
        
        # 1) Sentiment per segment
        sentiment_results = self.sentiment_analyzer.analyze_conversation_sentiment(segments)
        
        # 2) Emotions per segment (requires mel-spectrogram in audio_features)
        if audio_features is None:
            raise ValueError("audio_features must be provided for emotion detection.")
        emotion_results = self.emotion_detector.detect_conversation_emotions(segments, audio_features)
        
        # 3) Conversational dynamics (silence ratio, interruptions, etc.)
        from .feature_extraction import FeatureExtractor
        feature_extractor = FeatureExtractor()
        total_duration = max([s.get('end_time', 0) for s in segments]) if segments else 0
        conversational_dynamics = feature_extractor.extract_conversational_dynamics(segments, total_duration)
        
        # 4) Fused feature vector and sale prediction
        fused_features = self.sale_predictor.create_fused_feature_vector(
            sentiment_results,
            emotion_results,
            conversational_dynamics
        )
        sale_prediction = self.sale_predictor.predict_sale_probability(fused_features)
        
        # 5) Conversation-level metrics and summary
        conversation_metrics = self._calculate_conversation_metrics(sentiment_results, emotion_results)
        
        if call_id:
            conversation_id = call_id
        else:
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
```

**Explanation:**

1. **Input check:** Requires preprocessed `segments` and `audio_features`; does not fabricate data.
2. **Sentiment:** Runs `SentimentAnalyzer` on each segment (per‑speaker, per‑time range).
3. **Emotions:** Runs `EmotionDetector` on per‑segment mel slices from `audio_features`.
4. **Dynamics:** Uses `FeatureExtractor` to compute conversation‑level dynamics.
5. **Sale prediction:** Builds a fused vector and predicts sale probability + explanation.
6. **Metrics + summary:** Computes averages, trends, sentiment drift, and a short natural‑language summary.

The result is exactly what the FastAPI backend stores in MongoDB under `calls.result` and partially reshapes for the `/api/results/{call_id}` endpoint.

---

## Quick “Explain to Someone Else” Summary

- **`SentimentAnalyzer`**: Runs DistilBERT or FinBERT over each utterance to get sentiment scores and key phrases. Falls back to a keyword method if the transformer pipeline is unavailable.
- **`AcousticEmotionModel` + `EmotionDetector`**: A CNN+LSTM with temporal attention over mel‑spectrograms, plus an MFCC branch, to classify each segment into 5 emotions with probabilities.
- **`SalePredictor`**: An XGBoost model that takes a fused feature vector (sentiment stats, emotion probabilities, conversational dynamics) and outputs a sale probability, label, confidence interval, and feature importances.
- **`ConversationAnalyzer`**: The orchestrator that ties everything together for a single call. It takes segments and audio features from preprocessing, calls all three models, aggregates metrics, and returns a single structured analysis used by the API and frontend.

