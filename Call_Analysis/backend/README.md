# Call Analysis System

An AI-powered sentiment analysis system that predicts sale probability using multimodal data from call center recordings.

## Features

- **Multimodal Analysis**: Combines audio emotion detection and text sentiment analysis
- **Sale Prediction**: Predicts probability of sale (0-100%) based on conversation analysis
- **Speaker Diarization**: Identifies and separates different speakers in calls
- **Real-time Dashboard**: Visual insights with sentiment curves and conversion metrics
- **Cost-Effective**: Uses open-source tools and student-friendly technologies

## System Architecture

### Modules
1. **Data Preprocessing**: Audio transcription and speaker separation
2. **Feature Extraction**: Audio and text feature engineering
3. **Sentiment & Emotion Models**: BERT/DistilBERT for text, CNN+LSTM for audio
4. **Sale Prediction Model**: XGBoost/LSTM for probability prediction
5. **Visualization Dashboard**: Web-based analytics interface

### Technologies
- **Backend**: Python 3.10, PyTorch 2.2, Flask/FastAPI
- **ML Models**: Hugging Face Transformers, Whisper, Pyannote.audio
- **Frontend**: React 18, Plotly.js
- **Database**: PostgreSQL 15, MongoDB

## Quick Start

### Option 1: Automated Setup
```bash
python setup.py
```

### Option 2: Manual Setup
1. Install dependencies:
```bash
pip install -r requirements.txt
# OR
pip install -e .
```

2. Set up environment variables:
```bash
# Copy the environment template
cp env_template.txt .env

# Edit .env file with your configuration
# At minimum, set your Hugging Face token:
# HF_TOKEN=your_token_here
```

3. Install spaCy model (for PII masking):
```bash
python -m spacy download en_core_web_sm
```

4. Run the demo:
```bash
python run_demo.py
```

5. Start the web dashboard:
```bash
python run_web_app.py
```

### Option 3: Stakeholder Presentation
```bash
python presentation_demo.py
```

### Access the System
- **Demo Mode**: Run `python run_demo.py` for command-line demo
- **Web Dashboard**: Run `python run_web_app.py` and visit http://localhost:5000
- **Output Files**: Check the `output/` directory for generated reports and dashboards

## Environment Configuration

The system uses environment variables for configuration. Create a `.env` file from the template:

```bash
cp env_template.txt .env
```

### Key Environment Variables:

- **HF_TOKEN**: Hugging Face token for accessing models (required for real transcription)
- **MONGODB_URI**: MongoDB connection string (default: mongodb://localhost:27017/)
- **DEMO_MODE**: Set to `True` for demo mode with simulated data
- **USE_SIMULATED_AUDIO**: Set to `True` to use simulated audio instead of real processing
- **PII_MASKING_ENABLED**: Set to `True` to mask personally identifiable information

### For Production Use:
1. Set `DEMO_MODE=False`
2. Set `USE_SIMULATED_AUDIO=False`
3. Provide valid `HF_TOKEN` for real model access
4. Configure MongoDB/PostgreSQL connections
5. Set secure `FLASK_SECRET_KEY`

## Demo Features

### Core Functionality
- **Multimodal Analysis**: Combines text sentiment and audio emotion detection
- **Sale Prediction**: Predicts probability of sale (0-100%) using XGBoost model
- **Speaker Diarization**: Identifies and separates customer vs agent speech
- **Real-time Visualization**: Interactive charts and dashboards

### Demo Data
- **3 Sample Conversations**: Different scenarios (high/moderate/low sale potential)
- **Realistic Scenarios**: Insurance inquiries, service complaints, product information requests
- **Comprehensive Analysis**: Sentiment trends, emotion distribution, conversation flow

### Output Formats
- **Interactive HTML Dashboard**: Plotly-based visualizations
- **Text Reports**: Detailed analysis summaries
- **JSON Export**: Raw data for further processing
- **Comparison Reports**: Multi-conversation analysis

### Web Interface
- **Modern UI**: Responsive design with real-time updates
- **Interactive Charts**: Sentiment timelines, emotion distributions, sale probability gauges
- **Export Functionality**: Download results in multiple formats
- **API Endpoints**: RESTful API for integration

## Intended Users

- Call center managers
- Sales agents
- Sales analysts
- Researchers

## Project Timeline

- Weeks 1-2: Literature review & dataset collection
- Weeks 3-4: Preprocessing
- Weeks 5-6: Feature extraction
- Weeks 7-8: Model training
- Week 9: Dashboard development
- Week 10: Integration & testing
- Weeks 11-12: Documentation & report
