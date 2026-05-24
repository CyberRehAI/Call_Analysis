
### Chapter 4 – System Design and Architecture  

#### 4.1 Overall System Architecture  

The implemented backend follows a modular, pipeline-oriented architecture that matches the multimodal design described in the proposal and interim report \(`Project_Proposal_(2)[1].docx 10-49-48-725.pdf`, `Finalized.docx 10-49-44-364.pdf`\). The core stages are:

- **Audio ingestion and ASR**: Telephonic calls are provided as audio files (`.wav`, `.mp3`, `.m4a`, `.flac`). The `AudioProcessor` class in `preprocessing.py` validates/normalizes formats, then uses OpenAI Whisper for automatic speech recognition to produce a time-stamped transcript.  
- **Speaker diarization and conversational structuring**: The same `AudioProcessor` performs required speaker diarization using:
  - WhisperX + Resemblyzer (fast, CPU-friendly) as the primary path, and  
  - WhisperX 3.x built‑in diarization (Pyannote.audio) or Pyannote’s pipeline as fallbacks, depending on `HF_TOKEN` and installation.  
  This yields speaker-labeled segments with start/end times.
- **Text preprocessing and segment enrichment**: `TextProcessor` in `preprocessing.py` masks PII, aligns Whisper transcription segments to diarization segments, normalizes text, and attaches BERT‑ready token-level features to each segment.
- **Multimodal feature extraction and fusion**: `FeatureExtractor` in `feature_extraction.py` combines:
  - Acoustic descriptors (MFCCs, spectral features, chroma, duration, sample rate),
  - BERT-based text embeddings and simple textual statistics, and
  - Temporal/conversational features (speaker changes, sentiment trend, balance of speaking time),
  into a single fused feature vector per call.
- **Machine learning analysis layer**:
  - `SentimentAnalyzer` (in `models.py`) runs Transformer-based sentiment analysis (using `distilbert-base-uncased-finetuned-sst-2-english`, with a keyword-based fallback) over each segment.
  - `EmotionDetector` infers per-segment and conversation-level emotions from acoustic features using rule-based logic designed to approximate CNN+LSTM behavior.
  - `SalePredictor` uses an XGBoost-based interface trained on synthetic data plus a heuristics-based `_demo_prediction` to map fused features to sale probability.
  - `ConversationAnalyzer` orchestrates these components and aggregates conversation-level metrics.
- **Visualization and reporting**: `Dashboard` in `dashboard.py` turns the analysis result into:
  - Interactive Plotly dashboards saved as standalone HTML,
  - Structured textual summary reports,
  - Feature-importance visualizations for interpretability.
- **Persistence and APIs**:
  - MongoDB is used as the primary persistent store for calls, analysis results, dashboards, and intermediate artifacts when enabled.
  - A FastAPI application (`web_app_fastapi.py`) exposes REST endpoints that power the Next.js frontend and allow uploads, analysis, history retrieval, and exports.  
- **End-to-end orchestration**: CLI scripts (`run_full_analysis.py`, `run_diarization_only.py`) provide end-to-end and diagnostic pipelines for offline experimentation.

This architecture directly implements the multimodal workflow described in the proposal—audio + text fusion, conversation dynamics, and sale prediction—while staying within a research-prototype scope.  

---

#### 4.2 Backend Architectural Layers  

The backend is structured into clearly separated layers that correspond to the logical architecture discussed in Chapters 1–3 of the report:

- **Configuration and environment layer**  
  - `config.py` defines a `Config` base class and environment-specific subclasses (`DevelopmentConfig`, `ProductionConfig`, `TestingConfig`).  
  - It captures:
    - **Model configuration**: Whisper model size (`WHISPER_MODEL_SIZE`), BERT model name (`BERT_MODEL_NAME`), Pyannote model ID.
    - **Database configuration**: MongoDB URI and database, Postgres connection parameters (currently configured but not used by the code paths).
    - **Web-server configuration**: Host/port, debug mode, Flask-style keys (reused by FastAPI/uvicorn).
    - **File, logging, and performance settings**: upload folder, maximum payload size, log file path, batch size, worker limits, cache TTL, demo toggles.
  - `Config.init_app` ensures that required directories (`uploads`, `logs`, `output`, `templates`) exist at runtime.  
  - These settings align with the tool and environment choices listed in the proposal’s “Tools and Technologies” section.

- **Data ingestion and preprocessing layer** (`preprocessing.py`)  
  - `AudioProcessor` encapsulates audio-related operations (format validation, chunking, Whisper ASR, diarization, audio-feature extraction, persistence).
  - `TextProcessor` encapsulates text preparation for downstream BERT-based analysis, including PII masking, normalization, tokenization, and alignment between ASR segments and diarization segments.

- **Feature engineering layer** (`feature_extraction.py`)  
  - `FeatureExtractor` converts raw audio features, text embeddings, and temporal metrics into standardized numerical vectors. It uses scikit‑learn scalers and a shared BERT model/tokenizer for efficiency.

- **Model layer** (`models.py`)  
  - `SentimentAnalyzer`, `EmotionDetector`, `SalePredictor`, and `ConversationAnalyzer` implement the ML logic.  
  - The code structure corresponds to the conceptual models described in the documents: text-based sentiment (DistilBERT/FinBERT‑style), acoustic emotion recognition (CNN+LSTM‑inspired logic), and XGBoost‑style sale prediction.

- **Visualization and reporting layer** (`dashboard.py`)  
  - Responsible for interactive charts (sentiment timeline, emotion distribution, sale-probability gauge, conversation flow, feature importance), summary reports, and optional metadata persistence.

- **Application/control layer**  
  - `DemoSystem` in `demo.py` wires the above into a self-contained demo pipeline with dummy audio and demo conversations, used for POC and testing.
  - `web_app_fastapi.py` configures and exposes the HTTP API surface used by the frontend.
  - CLI scripts (`run_full_analysis.py`, `run_diarization_only.py`) act as orchestrators for batch or offline workflows.

This layered architecture provides a clear mapping from requirements (ASR, diarization, multimodal fusion, dashboards) to concrete modules and classes.

---

#### 4.3 Module–Level Design  

This section links key classes and functions in the backend to the conceptual components described in the project documentation.

- **Audio preprocessing and ASR (`preprocessing.AudioProcessor`)**  
  - Constructor parameters (`model_size`, `hf_token`, `use_whisperx`, `chunk_duration`, `max_speakers`, `clustering_threshold`, etc.) are explicitly designed to support different diarization strategies and performance trade-offs.
  - `_load_models()`:
    - Loads Whisper (`whisper.load_model`), optionally WhisperX, Resemblyzer, and Pyannote pipelines depending on availability and `HF_TOKEN`.
    - Implements “fast CPU-friendly” versus “more accurate but slower” diarization paths as described in the methodology.
  - `validate_audio_format()` enforces supported formats (FR‑1) and converts other formats to `.wav` using `pydub`.
  - `transcribe_audio()` and `_transcribe_with_chunking()`:
    - Perform full-file or chunked transcription depending on duration (`chunk_duration`), matching the requirement to handle long calls efficiently.
    - Save transcriptions both to JSON in the project `output/` directory and, if enabled, MongoDB (`transcriptions` collection).  
  - `validate_transcription()` provides an explicit WER computation function using `jiwer`, aligning with the “WER ≤ 15%” target mentioned in the requirements, although it does not compute dataset‑level aggregates by itself.
  - `perform_speaker_diarization()` plus helper methods select between:
    - WhisperX + Resemblyzer (`_diarize_with_whisperx_resemblyzer` and `_diarize_chunked_whisperx_resemblyzer`),
    - WhisperX 3.x built-in diarization (`_diarize_with_whisperx_builtin` using Pyannote internally), or
    - Pure Pyannote (`_diarize_with_pyannote`),
    ensuring diarization is always attempted and treated as required, as stated in the docs.
  - `_identify_speaker_roles()` infers AGENT vs CUSTOMER roles using speaking time, keyword patterns, and questions, implementing the “conversational dynamics” requirement from the proposal (interruptions, speaker balance, etc.).
  - `extract_audio_features()` builds a feature dictionary used later for both emotion and multimodal fusion.

- **Text preprocessing (`preprocessing.TextProcessor`)**  
  - Uses spaCy where available (and regex fallbacks otherwise) to **mask PII** (names, phone numbers, emails, organizations) before any downstream processing.
  - `clean_text`, `tokenize_text`, and `extract_text_features` provide normalized, BERT-token-ready representations plus basic statistics (word counts, sentence counts, punctuation-based cues).
  - `_assign_text_to_diarization_segments()` aligns Whisper transcription segments to diarization intervals, ensuring each diarization segment gets non-duplicated text where possible, which is a prerequisite for segment‑level sentiment/emotion analysis.
  - `segment_conversation()` produces a standardized segment schema used throughout the pipeline (`start_time`, `end_time`, `speaker`, `text`, and precomputed text features) and calls `save_segments()` to persist segments when MongoDB is active.

- **Feature extraction (`FeatureExtractor`)**  
  - `load_audio_features()` and `extract_audio_features()` compute MFCCs, spectral centroid/rolloff, zero-crossing rate, chroma, duration, and sample rate, mirroring the prosodic feature types discussed in the literature and proposal.
  - `extract_text_features()` produces 768‑dimensional BERT embeddings from PII-masked text using `bert-base-uncased`.
  - `extract_temporal_features()` derives aggregate conversational metrics such as:
    - total duration and speaker changes,
    - speaking-time balance,
    - sentiment trend and volatility,
    - final sentiment.  
    These directly correspond to “conversation flow” and “sentiment drift” modeling described earlier in the report.
  - `combine_features()`, `fit_transform()`, and `transform()` standardize and concatenate the three modalities into a single feature vector, which becomes the input to the sale-prediction layer.

- **Model layer (`models.py`)**  
  - `SentimentAnalyzer`:
    - Wraps a Hugging Face sentiment pipeline based on DistilBERT and falls back to keyword-based scoring if the pipeline fails.
    - `analyze_conversation_sentiment()` applies this per segment to obtain sentiment labels, scores, and basic counts of positive/negative cues.
  - `EmotionDetector`:
    - Defines a CNN+LSTM architecture but, in the current prototype, uses a rule-based analysis over acoustic features to produce an emotion probability distribution and a dominant label (neutral/happy/sad/angry/etc.), consistent with the SER concept but without a fully trained deep model in this codebase.
  - `SalePredictor`:
    - Exposes an XGBoost-based API and trains on synthetic data, but marks `is_trained=False` such that the deployed probability comes from `_demo_prediction`, a deterministic heuristic over the fused feature vector.
    - This keeps the code consistent with the proposal’s XGBoost component while acknowledging the prototype nature in this repository.
  - `ConversationAnalyzer`:
    - Coordinates segment-level sentiment and emotion analysis, aggregates conversation metrics, calls the sale predictor, and produces a textual summary.
    - Also offers `batch_analyze` and `get_agent_insights` to support dashboard-style aggregation over many calls.

- **Visualization (`dashboard.py`)**  
  - Implements Plotly-based charts:
    - Sentiment timeline over time,
    - Emotion distribution,
    - Sale-probability gauge,
    - Conversation flow by speaker,
    - Feature-importance bar chart (when importance data are available).  
  - `create_dashboard_html()` and `generate_dashboard()` combine those figures into a single HTML dashboard, which is written to disk and optionally mirrored into MongoDB (`dashboards` collection) as metadata.  
  - `create_summary_report()` gives a text-only version of the same information.

- **Demo integration (`demo.py`)**  
  - `DemoSystem` wires together `ConversationAnalyzer`, `Dashboard`, `AudioProcessor`, `TextProcessor`, and `FeatureExtractor` into a full demonstration pipeline that:
    - Creates synthetic audio if demo files are missing,
    - Defines example conversation scenarios with PII-masked text,
    - Runs the full pipeline (ASR, diarization, segmentation, features, ML analysis, dashboards, reports).
  - This module realizes the “student-friendly, low-cost POC” objective from the proposal.

- **API layer (`web_app_fastapi.py`)**  
  - Exposes:
    - **Demo analysis endpoints** (e.g., `/api/conversations`, `/api/analyze/{conversation_id}`, `/api/analyze-all`, `/api/dashboard/{conversation_id}`, `/api/insights`) that work purely against `DemoSystem`.
    - **Upload and real-call analysis**:
      - `POST /api/upload` accepts a user audio file, saves it, and runs the same full pipeline (ASR, diarization, segmentation, multimodal features, analysis), saving into MongoDB `calls` plus `analyses`.
      - `POST /api/analyze` is used as a status update hook for uploaded calls (compatible with the frontend workflow).
      - `GET /api/results/{call_id}`, `/api/status/{call_id}`, `/api/history` provide the data required by the Next.js dashboards.
    - **Exports**:
      - `GET /api/export/{conversation_id}` writes JSON exports to `output/`,
      - `GET /api/export/{call_id}/pdf` generates a PDF report via ReportLab,
      - `GET /api/export/{call_id}/csv` emits a CSV export of summary metrics, time series, and key phrases.

These APIs map directly to the user-facing scenarios described in earlier chapters: call upload, automatic analysis, visualization, and export for managerial use.

---

#### 4.4 Data Flow and Control Flow  

There are two primary execution paths implemented in the backend:

- **CLI end-to-end offline analysis (`run_full_analysis.py`)**  
  - `run_full_analysis(audio_path, call_id, output_dir)` implements a 10-step pipeline:
    1. **Initialize components**: `AudioProcessor`, `TextProcessor`, `FeatureExtractor`, `ConversationAnalyzer`, `Dashboard` are instantiated using `Config` for model size and tokens.
    2. **Transcription (Whisper)**: `audio_processor.transcribe_audio` produces a JSON with `text`, `segments`, `language`, and `duration`, stored under `output/{call_id}_transcription.json`.
    3. **Diarization**: `audio_processor.perform_speaker_diarization` yields diarization segments and an updated diarization JSON.
    4. **Text segmentation and alignment**: `text_processor.segment_conversation` merges transcription and diarization into speaker-labeled text segments with extracted text features.
    5. **Acoustic feature extraction**: `audio_processor.extract_audio_features` computes MFCCs and other low-level descriptors.
    6. **Multimodal feature fusion**: `feature_extractor.fit_transform` converts audio features, text, and segments into a fused vector.
    7. **Conversation analysis**: `analyzer.analyze_conversation` produces sentiment, emotion, sale-probability, and conversation metrics. If fused features are present, sale probability is recomputed using `SalePredictor`.
    8. **Dashboard generation**: `dashboard.generate_dashboard` saves `{call_id}_dashboard.html`.
    9. **Result packaging**: the code composes a `complete_results` dictionary summarizing transcription, diarization, and analysis, and saves it as `{call_id}_results.json`.
    10. **Summary output**: key metrics (sale probability, average sentiment, dominant emotion, total segments and file paths) are printed to console.

- **Online analysis via FastAPI (`/api/upload`, `/api/results/{call_id}`, etc.)**  
  - `POST /api/upload`:
    - Validates upload type against `Config.ALLOWED_EXTENSIONS`.
    - Saves the audio file into the configured `UPLOAD_FOLDER`.
    - Executes the same core pipeline as above (ASR, diarization, segmentation, acoustic features, fused features, analysis).
    - Constructs a `call_doc` record capturing duration, average sentiment, sale probability, emotion distribution, key phrases, and raw result, and inserts it into `db.calls`.
    - Also stores an analysis document in `db.analyses` for backward compatibility with earlier data models.
  - Follow-up endpoints (`/api/results/{call_id}`, `/api/status/{call_id}`, `/api/history`) reference and expose these stored records for consumption by the frontend.  

This flow matches the “upload → analysis → dashboard → export” lifecycle described conceptually in the earlier chapters and mockups.

---

#### 4.5 Data Storage Design  

The backend uses MongoDB as the primary persistence layer when enabled (via `MONGODB_ENABLED`), with the following effective collections:

- **`transcriptions`**: Created in `AudioProcessor.save_transcription`, storing `call_id`, transcription object, and timestamp.
- **`diarization`**: Created in `AudioProcessor.save_diarization`, storing `call_id`, segments, and metadata; also mirrored as JSON files in `output/`.
- **`segments`**: Created in `TextProcessor.save_segments`, grouping all processed segments per call along with timestamps.
- **`features`**: Created in `FeatureExtractor.save_features`, storing fused numerical features, `call_id`, and timestamp.
- **`calls`** (FastAPI path):
  - Inserts performed in `/api/upload`, capturing:
    - `call_id`, filename, audio path, timestamps,
    - status/progress, duration, participants,
    - aggregated sentiment and sale-probability metrics,
    - emotion distributions and key phrases, plus full `result`.
- **`analyses`**, **`batch_analyses`**, **`insights`**, **`exports`**, **`dashboards`**:
  - Used by demo and export paths to store:
    - Individual conversation analyses (`analyses`),
    - Batch-analysis sets (`batch_analyses`),
    - Agent-level aggregated insights (`insights`),
    - JSON export metadata (`exports`),
    - Dashboard metadata (`dashboards`).

PostgreSQL parameters exist in `Config`, matching the original plan to support SQL-based storage, but no active PostgreSQL CRUD logic is present in the current backend implementation; MongoDB is the only live DB path in code.  

JSON export files (transcriptions, diarization, results, dashboards, comparison reports) are systematically stored under the root-level `output/` and can be used as a filesystem‑based fallback when MongoDB is disabled.

---

#### 4.6 Security, Privacy, and Regulatory Considerations  

The backend incorporates several measures aligned with the security and PII-handling requirements described in the documentation:

- **PII masking at multiple layers**:
  - `TextProcessor.mask_pii` uses spaCy NER (when available) and regex fallbacks to redact names, phone numbers, emails, locations, and organizations.  
  - `FeatureExtractor.mask_pii` and `Dashboard.mask_pii` apply similar logic before text is used for embeddings or visualized.  
  - `DemoSystem.mask_pii` ensures synthetic demo conversations also respect masking, reflecting the same constraints during demonstrations.
  - FastAPI routes explicitly mask text (`dashboard.mask_pii`) before returning sentiment and segment details in JSON.
- **Configuration-level controls**:
  - `Config.PII_MASKING_ENABLED` and `Config.DATA_RETENTION_DAYS` are defined, giving a configuration hook for masking and retention, even though retention enforcement logic is not yet implemented in the code.
- **Data separation and minimal exposure**:
  - Raw transcriptions and segments are stored server-side; exported dashboards and PDFs only show redacted text.
  - The CSV and JSON export functions explicitly compute high-level metrics (sale probability, sentiment scores, emotion distributions, key phrases) without exposing unmasked raw conversations.
- **Licensing and dependency constraints**:
  - The code uses open-source libraries consistent with the licensing requirements in the report (Whisper, Pyannote.audio, Hugging Face Transformers, librosa, XGBoost, ReportLab, etc.), all under permissive or research-friendly terms as documented in the references \[`https://github.com/openai/whisper`, `https://github.com/pyannote/pyannote-audio`, `https://huggingface.co/transformers`, `https://xgboost.ai`\].

Overall, the implemented backend reflects the architectural and non-functional design described up to Chapter 3, with explicit attention to modularity, multimodality, and privacy-aware processing.

---

### Chapter 5 – System Implementation  

#### 5.1 Development Environment and Technology Stack  

Based on `requirements.txt`, `pyproject.toml`, and the imports in the backend modules, the implementation uses:

- **Programming language and core stack**:
  - Python 3.10 (as specified in the proposal and project docs),
  - FastAPI for the HTTP API and uvicorn as the ASGI server (`run_web_app.py`),
  - MongoDB as the operational data store, with optional environmental toggles.
- **Machine learning and signal processing libraries**:
  - PyTorch and Torch-based models (Whisper, Transformers, simple CNN/LSTM definitions),
  - OpenAI Whisper for ASR,
  - WhisperX for diarization alignment (where installed),
  - Resemblyzer for speaker embeddings,
  - Pyannote.audio as alternative diarization backend,
  - Librosa and SoundFile for audio processing,
  - scikit‑learn and XGBoost for feature scaling and tree-based sale-prediction scaffolding.
- **NLP and PII masking**:
  - Hugging Face Transformers (BERT and DistilBERT models),
  - spaCy (`en_core_web_sm`) for NER-based PII redaction, with regex fallbacks when spaCy is missing.
- **Visualization and reporting**:
  - Plotly for interactive charts,
  - ReportLab for PDF report generation.

This stack strictly matches the tool choices listed in the “Tools and Technologies” sections of the proposal and report and is aligned with the target of a low‑cost, open-source stack.

---

#### 5.2 Core Backend Implementation  

This section describes how each major functional requirement from the earlier chapters is concretely realized in code.

##### 5.2.1 ASR and Long-Audio Handling  

- **ASR implementation**:
  - Whisper is loaded in `AudioProcessor._load_models()`. The chosen model size is determined via `Config.WHISPER_MODEL_SIZE` and defaults to `"base"`.
  - `transcribe_audio()` calls `whisper_model.transcribe` with English language and transcription task. Output is normalized to a dictionary containing:
    - `text` (full transcription),
    - `segments` (per-phrase time-coded segments),
    - `language`,
    - `duration`.
- **Long-audio chunking**:
  - `_transcribe_with_chunking()` uses `librosa` to load the full audio at a target sampling rate and splits it into contiguous windows of `chunk_duration` seconds (default: 300s / 5 minutes).
  - Each chunk is saved temporarily, transcribed separately, and time-stamps are adjusted by chunk offsets before segments are recombined.
  - The resulting transcript is a time-ordered list of segments with consistent global times, plus a stitched text document, ensuring long recordings can be processed on limited hardware.

This implementation is consistent with the project’s requirement to support telephonic conversations of realistic call-center lengths while remaining “student-friendly” in terms of compute.

##### 5.2.2 Speaker Diarization and Role Assignment  

- **WhisperX + Resemblyzer path**:
  - `_diarize_with_whisperx_resemblyzer()`:
    - Loads audio via WhisperX, produces aligned word-level segments, then merges short transcription segments to improve the stability of speaker embeddings.
    - Extracts speaker embeddings using Resemblyzer (`VoiceEncoder.embed_utterance`) for segments longer than `min_segment_duration`.
    - Clusters segments with hierarchical clustering and a configurable cosine distance threshold (`clustering_threshold`) to derive speaker IDs.
    - `_merge_similar_speakers()` merges over-fragmented speaker clusters based on centroid similarity, limiting artificial speaker proliferation.
    - Outputs diarization segments with `speaker`, `start`, `end`, and preliminary `text` from the merged WhisperX segments.
- **Chunked diarization**:
  - `_diarize_chunked_whisperx_resemblyzer()` extends the above logic to multi-chunk diarization, then `_match_speakers_across_chunks()` merges speaker identities across chunks using centroid similarity, ensuring the same person keeps the same ID throughout the call.
- **WhisperX built‑in and Pyannote fallback**:
  - `_diarize_with_whisperx_builtin()` leverages WhisperX’s `DiarizationPipeline` (internally using Pyannote models) and `assign_word_speakers` for fine-grained speaker assignment, with extensive error handling for known edge cases.
  - `_diarize_with_pyannote()` invokes the Pyannote pipeline directly if WhisperX paths are not available.
- **Role identification (customer vs agent)**:
  - `_identify_speaker_roles()` analyzes:
    - Speaking time per speaker,
    - Keyword presence in text (agent vs customer phrases),
    - Question frequency and patterns,
    - Order of appearance in the conversation.
  - It converts raw `SPEAKER_XX` identifiers into semantic labels (`AGENT`, `CUSTOMER`), preserving `original_speaker_id` for traceability.

Together, these functions implement the “conversation dynamics” requirements (speaker turns, interruptions, agent vs customer behavior) using robust heuristics grounded in the actual audio/transcript content.

##### 5.2.3 Text Processing, PII Masking, and Segment Construction  

- **Text normalization and tokenization**:
  - `TextProcessor` methods (`_normalize_text_preserve`, `tokenize_text`, `extract_text_features`) lower-case text, preserve relevant prosodic cues (fillers/emphasis characters), and prepare BERT tokens.
- **PII handling**:
  - PII is masked **before** creating embeddings or feature vectors, ensuring that BERT never ingests raw sensitive text.
  - Masking is applied consistently in the preprocessing, feature extraction, dashboard, and demo components, reflecting the data-protection requirements in the documentation.
- **Segment construction**:
  - `segment_conversation()`:
    - Fills diarization segments with aligned, non-duplicated text via `_assign_text_to_diarization_segments`.
    - Produces normalized segment dictionaries containing both raw text and precomputed text features.
    - Optionally writes segments into MongoDB (`segments` collection) to support dataset-level evaluation and analysis.

This implementation turns unstructured conversation audio into a structured sequence of semantically rich segments, aligning closely with the “multi-faceted sentiment” and “dialogue flow” concepts in the proposal.

##### 5.2.4 Multimodal Feature Fusion  

- **Audio features**:
  - MFCC: capture timbral and prosodic structure,
  - Spectral centroid/rolloff: approximate perceived brightness and spectral distribution,
  - Zero-crossing rate: proxy for noisiness/voicing,
  - Chroma and mel-spectrogram: represent harmonic and energy distributions,
  - Duration and sample rate: global descriptors.
- **Text features**:
  - 768‑dimensional BERT embeddings from `bert-base-uncased`,
  - Basic counts (word length, punctuation, upper-case ratio) approximating emphasis and intensity.
- **Temporal features**:
  - Speaker changes and balance,
  - Sentiment trend (via polynomial fit over segment-level scores),
  - Sentiment volatility,
  - Final sentiment.
- **Scaling and fusion**:
  - `fit_transform()` computes combined vectors for a batch and uses a `StandardScaler` to standardize them.
  - `transform()` handles a single call and applies the same scaler (if already fit), ensuring feature distribution consistency between training/analysis runs.

This design effectively realizes the “multimodal fusion” requirement discussed in the literature review and methodology sections of the existing report.

##### 5.2.5 Sentiment, Emotion, and Sale Prediction  

- **Sentiment analysis**:
  - `SentimentAnalyzer.analyze_conversation_sentiment()` uses the Hugging Face `distilbert-base-uncased-finetuned-sst-2-english` pipeline to obtain sentiment labels and confidence per segment.
  - When the pipeline is unavailable (e.g., offline environments), a fallback keyword-based scoring is applied to maintain functional behavior, which is clearly a prototype mechanism.
- **Emotion detection**:
  - While a CNN+LSTM architecture is defined in `EmotionDetector._create_model`, the current implementation uses statistically informed rules on spectral and MFCC features plus zero-crossing rate to infer high-level emotion categories, matching the SER concept but without a fully trained deep model.
- **Sale prediction**:
  - `SalePredictor.train()` constructs an XGBoost model but trains it only on synthetic data for demonstration. It then intentionally sets `is_trained=False` so the deployed method, `predict_sale_probability`, uses `_demo_prediction`, which:
    - Normalizes features,
    - Computes heuristic sentiment, emotion, and temporal sub-scores from different parts of the fused vector, and
    - Combines them linearly to a sale probability range [0.1, 0.9], with an associated confidence score.
  - This preserves the architectural position of an XGBoost-like model and a well-defined API, while acknowledging that empirical training on real call-center data is out of scope for this codebase.

These implementations concretely connect the backend to the conceptual models described in the proposal, but they clearly remain in a research-prototype state for sale prediction and SER.

##### 5.2.6 Visualization and Exports  

- **Dashboards**:
  - `Dashboard.create_sentiment_timeline()` visualizes segment-level sentiment over time with color-coding by sentiment and speaker.
  - `create_emotion_distribution()` aggregates dominant emotions into a donut chart.
  - `create_sale_probability_gauge()` displays sale probability as a gauge.
  - `create_conversation_flow()` plots speaker turns along the time axis.
  - `create_feature_importance_chart()` presents the top N most influential features when importances are available.
- **HTML dashboards and reports**:
  - `create_dashboard_html()` composes these figures and summary metrics into a styled HTML page for a single call.
  - `create_batch_dashboards()` iterates over multiple calls to produce a set of HTML dashboards.
  - `create_summary_report()` generates a text-only report summarizing key metrics and PII-masked recommendations.

- **Exports via FastAPI**:
  - JSON exports (`/api/export/{conversation_id}`), PDF exports (`/api/export/{call_id}/pdf`), and CSV exports (`/api/export/{call_id}/csv`) convert MongoDB `calls` documents into files compatible with managerial workflows and third-party tools.

This layer fully implements the dashboard and reporting requirements articulated in the mockups and scope sections of the existing documentation.

---

#### 5.3 Integration with the Frontend and Execution Scripts  

- **FastAPI entrypoint**:
  - `run_web_app.py` loads configuration and runs uvicorn with `call_analysis.web_app_fastapi:app`, enabling autoreload in debug mode.
  - CORS is explicitly configured for `http://localhost:3000`, matching the provided Next.js frontend.
- **API surface**:
  - The routes in `web_app_fastapi.py` are aligned with the endpoints expected by the frontend (`/api/upload`, `/api/analyze`, `/api/results/{callId}`, `/api/history`, `/api/export` variants), guaranteeing compatibility between the documented architecture and the running system.
- **Batch and diagnostic tools**:
  - `run_full_analysis.py` offers a CLI for end‑to‑end analysis of arbitrary audio files, ideal for offline evaluation.
  - `run_diarization_only.py` focuses on diarization using existing transcripts and allows parameter tuning (max speakers, clustering thresholds) as part of experimentation with diarization quality and performance.

These scripts operationalize the system in ways that mirror the incremental development and evaluation phases described in earlier chapters (from preprocessing to dashboard integration).

---

#### 5.4 Error Handling, Logging, and Configuration Management  

- The code systematically uses Python’s `logging` module with informative messages for:
  - Model loading (Whisper, WhisperX, Resemblyzer, Pyannote, spaCy),
  - Long-running diarization (including periodic console progress logs),
  - MongoDB connectivity issues (falling back to file-based storage when DB is unavailable),
  - PII masking failures and fallback to unmodified text.
- Configuration is centralized in `.env` (as documented in `CONFIGURATION.md`) and `Config` classes, following the guidelines to avoid hardcoding sensitive information.
- Many operations (e.g., diarization methods) adopt a multi-level fallback strategy (WhisperX built-in → WhisperX+Resemblyzer → Pyannote), which is consistent with the project’s emphasis on cost-effective but robust solutions.

---

### Chapter 6 – Testing and Evaluation  

#### 6.1 Testing Strategy Reflected in the Codebase  

The current backend does not include an explicit `tests/` directory or unit-test suite; instead, it relies on **functional and integration testing** conducted through:

- The **CLI scripts**:
  - `run_full_analysis.py` for end-to-end pipeline testing on sample or real audio files,
  - `run_diarization_only.py` for focused evaluation of diarization behavior under different parameter settings and file formats (JSON vs TXT transcripts).
- The **demo harness**:
  - `DemoSystem.run_demo()` orchestrates multiple end-to-end runs on synthetic demo calls, printing key metrics and generating dashboards and reports.
- The **FastAPI endpoints**:
  - Manual or automated HTTP calls (e.g., via the Next.js frontend or API tools) verify correctness of upload handling, analysis triggering, result retrieval, and export operations.

This functional testing approach is appropriate for a research prototype and aligns with the iterative methodology outlined in the existing documentation, though it leaves room for more formalized unit testing in future work.

---

#### 6.2 Functional Testing Scenarios  

From the implemented scripts and routes, the following test scenarios are explicitly supported:

- **End-to-end single-call test (`run_full_analysis.py`)**:
  - Input: A telephonic conversation audio file and optional `call_id`/`output_dir`.
  - Expected behavior:
    - ASR produces a non-empty transcript (or logs a warning when silent),
    - Diarization completes with a non-zero number of segments and at least one speaker,
    - Processed segments contain aligned text where possible,
    - Multimodal feature extraction completes without shape/NaN errors,
    - Conversation analysis returns sale probability, sentiment metrics, and emotion distribution,
    - Dashboard HTML and JSON result files are successfully written to `output/`.  
  - This test checks the complete multimodal pipeline on real or demo data.

- **Diarization-only parameter tests (`run_diarization_only.py`)**:
  - Input: audio file path, transcript JSON or TXT, plus optional parameters:
    - `--max-speakers`, `--clustering-threshold`, `--min-segment-duration`, `--speaker-merge-threshold`, `--use-whisperx-builtin`.
  - Expected behavior:
    - The script prints and saves:
      - Total segments, unique speakers, and segment counts per speaker,
      - Segment durations and average durations,
      - Successful mapping of text to segments when JSON transcripts with timestamps are used.
    - Output JSON (`*_diarization.json` and `*_diarization_summary.json`) can be inspected to evaluate the impact of threshold changes and diarization modes.  
  - This supports empirical evaluation of different diarization strategies and hyperparameters as described in the methodology.

- **Demo-system evaluation (`demo.py`)**:
  - `DemoSystem.run_demo()`:
    - Runs the full pipeline on predefined demo scenarios,
    - Prints sale probabilities, average sentiments, dominant emotions, and durations for each demo call,
    - Generates HTML dashboards and a textual comparison report.
  - This scenario verifies integration across all layers and gives qualitative evidence of system behavior.

- **API-level verification (`web_app_fastapi.py`)**:
  - Upload tests:
    - Ensure that invalid file types are rejected with appropriate HTTP errors.
    - Confirm that accepted files result in a new `call_id`, stored metadata, and accessible results via `/api/results/{call_id}` and `/api/history`.
  - Export tests:
    - Validate that JSON, PDF, and CSV exports are produced and contain consistent metrics with the API responses.  

These scenarios collectively exercise the core functional requirements under different usage patterns.

---

#### 6.3 Embedded Evaluation Procedures  

Although there is no dedicated evaluation notebook or script, the backend contains built-in support for several evaluation aspects:

- **ASR quality (WER)**:
  - `AudioProcessor.validate_transcription()` computes the Word Error Rate between a generated transcription and a reference string using `jiwer`.  
  - This facilitates offline experiments where the same codebase is applied to labeled datasets, consistent with the “ASR WER ≤ 15%” benchmark referenced in the documentation; however, such datasets and metrics are not embedded in this repository.

- **Diarization quality and performance**:
  - Logging in `_diarize_with_whisperx_resemblyzer`, `_diarize_chunked_whisperx_resemblyzer`, `_diarize_with_whisperx_builtin`, and `_diarize_with_pyannote` informs:
    - Estimated processing time based on audio length,
    - Number of detected speakers before and after cluster merging,
    - Number of segments with text vs total segments,
    - Any fallback triggers between diarization methods.
  - These logs support qualitative evaluation of diarization performance across different parameter configurations and hardware setups.

- **Conversation-level metrics**:
  - `ConversationAnalyzer._calculate_conversation_metrics()` yields quantitative metrics (average sentiment, sentiment trend, emotion distribution, speaker sentiment breakdown, volatility, and a simple flow score), which can be aggregated across calls (e.g., via `batch_analyze` and `get_agent_insights`) to assess model behavior at scale.

Together, these mechanisms allow for systematic evaluation of each pipeline stage using external datasets, even though such datasets and numeric benchmark results are not included in this codebase.

---

#### 6.4 Limitations of Current Testing and Evaluation  

Based strictly on the repository contents:

- There are **no automated unit or integration test suites** (e.g., `pytest`), so regression detection must be done manually via scripts and HTTP tests.
- **Quantitative benchmarks** (exact WER, diarization DER, emotion F1, sale-prediction AUC) are not present; the code provides tools for computing some of these metrics but not the datasets or scripts to run standardized experiments.
- The **sale predictor** currently uses synthetic training data and heuristic inference, so any reported sale probability should be interpreted as **relative** rather than empirically validated.
- The **emotion detector** is partially heuristic and has not been validated against a labeled SER corpus in this repository.

These gaps are consistent with the project’s current stage as a research and demonstration prototype and point directly to future evaluation work needed for production-grade deployment.

---

### Chapter 7 – Results, Discussion, and Future Work  

#### 7.1 Qualitative Results of the Implemented Prototype  

From the implemented backend and demo workflows, the following **qualitative** observations can be made (without introducing quantitative values beyond what the code actually computes at runtime):

- The pipeline successfully transforms raw call audio into:
  - Time-aligned transcripts (with optional chunking for long calls),
  - Multi-speaker diarization with AGENT vs CUSTOMER labeling,
  - Segment-level sentiment and emotion profiles,
  - A call-level sale‑probability estimate and aggregated sentiment/emotion metrics.
- The **interactive dashboards** (HTML generated by `Dashboard`) provide:
  - A visual timeline of sentiment variation,
  - A global distribution of emotions,
  - A summary indication of sale probability,
  - A simple but useful depiction of conversation flow by speaker.  
  This directly supports the dashboard-focused objectives outlined in the proposal and mockups.
- The **FastAPI backend** and **Next.js frontend** integration allow:
  - End users (e.g., call center managers) to upload audio,
  - Observe analysis status and history,
  - View results in both chart-based and tabular forms,
  - Export data to PDF/CSV/JSON for reporting or further analysis.

These behaviors align with the system’s intended role as an explainable, low-cost alternative to proprietary call-center analytics tools such as IBM Watson Tone Analyzer, Amazon Contact Lens, and Google Cloud Speech + Sentiment \[`https://www.ibm.com/watson/services/tone-analyzer/`, `https://aws.amazon.com/connect/contact-lens/`, `https://cloud.google.com/`\].

---

#### 7.2 Discussion of Design Choices  

Several design decisions evident in the backend have direct implications for system behavior:

- **Multimodal over unimodal**:
  - By combining acoustic, textual, and temporal features (through `FeatureExtractor` and the model layer), the system goes beyond pure text-based sentiment analysis, capturing cues like pitch, tempo, and speaker balance—exactly addressing the “paralinguistic cue” gap highlighted in the problem statement.
- **Flexible diarization backends**:
  - Supporting WhisperX+Resemblyzer, WhisperX built-in (Pyannote), and Pyannote standalone allows deployment on a range of hardware, from student laptops (CPU) to more capable servers, while still honoring the requirement that diarization is a “required” step.
- **Heuristic vs fully trained models**:
  - The current repository uses heuristics and synthetic training for emotion detection and sale prediction, while still exposing interfaces compatible with fully trained CNN+LSTM and XGBoost models.  
  - This confirms the architectural feasibility of the proposed models while leaving room for future empirical training once annotated call-center data is available.
- **MongoDB-centric persistence**:
  - The implemented code uses MongoDB as the primary data store, despite the presence of PostgreSQL configuration options. This choice simplifies early development and aligns with the need for flexible document storage of heterogeneous analysis outputs.

These decisions represent pragmatic compromises between the ideal architecture described in the proposal and the constraints of a final-year project (compute, data availability, and time).

---

#### 7.3 Limitations and Threats to Validity  

The following limitations can be directly inferred from the code and should be acknowledged explicitly:

- **Prototype-level training**:
  - Emotion detection and sale-prediction components are not trained on real call-center data within this repository; they rely on heuristics and synthetic training, limiting the external validity of their outputs.
- **Absence of labeled evaluation datasets**:
  - Without embedded reference datasets and scripts, ASR, diarization, and SER performance cannot be quantified or compared against the literature benchmarks directly in this codebase.
- **Reliance on third-party models and tokens**:
  - System performance and reproducibility depend on external services and models (Whisper, WhisperX, Pyannote, Hugging Face), as well as a valid `HF_TOKEN`. Changes in these dependencies can impact results over time.
- **Database and deployment configuration**:
  - Only MongoDB is fully integrated; PostgreSQL remains unused, and deployment scripts for production-scale environments (Docker, orchestration, monitoring) are not included, limiting immediate real-world deployment.

These limitations are typical for an academic prototype and should be considered when interpreting results and planning further work.

---

#### 7.4 Future Work Aligned with the Current Implementation  

Based on the existing backend and the scope outlined in the project documents, the following future directions are well aligned and do not require speculative assumptions:

- **Model training on real datasets**:
  - Replace heuristic SER and sale-prediction components with fully trained CNN+LSTM and XGBoost models using labeled telephonic conversation datasets, reusing the `FeatureExtractor`, `EmotionDetector`, and `SalePredictor` interfaces already in place.
- **Comprehensive evaluation suite**:
  - Add structured evaluation scripts and unit tests to:
    - Compute WER, DER, emotion classification metrics, and sale-prediction accuracy on held-out datasets,
    - Automate regression testing for every pipeline stage (`preprocessing`, `feature_extraction`, `models`, `dashboard`, `web_app_fastapi`).
- **Database and scalability enhancements**:
  - Implement the PostgreSQL-backed storage implied by `Config` and integrate it as an alternative or complement to MongoDB for better transactional guarantees and reporting.
  - Add background job queues and task scheduling (e.g., for asynchronous long-call processing) to support higher concurrency and production workloads.
- **Real-time and streaming extensions**:
  - Extend the current file-based pipeline to support chunked streaming inputs (e.g., websockets or gRPC), leveraging the existing chunking logic in `AudioProcessor` to approximate real-time analysis without major architectural changes.
- **Explainability and XAI**:
  - Use `FeatureExtractor.get_feature_names()` together with real feature importance data from a trained XGBoost model to construct more detailed explainability dashboards, helping managers understand which conversational aspects drive sale probability in practice.