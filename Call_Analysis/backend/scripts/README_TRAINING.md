# Training Scripts Documentation

This directory contains training scripts for the Call Analysis System models.

## Scripts Overview

### 1. `train_emotion_model.py`
Trains the AcousticEmotionModel (CNN+LSTM) for emotion recognition using RAVDESS dataset format.

**Usage:**
```bash
python backend/scripts/train_emotion_model.py --data_dir data/raw/ --output_dir backend/models/ --epochs 50 --batch_size 32
```

**Arguments:**
- `--data_dir`: Path to RAVDESS dataset directory (default: `data/raw/`)
- `--output_dir`: Output directory for model (default: `backend/models/`)
- `--batch_size`: Training batch size (default: 32)
- `--epochs`: Number of training epochs (default: 50)
- `--learning_rate`: Learning rate (default: 0.001)
- `--validation_split`: Validation split ratio (default: 0.2)
- `--max_time_frames`: Maximum time frames for padding/truncation (default: 500)

**Output:**
- `backend/models/emotion_model.pth` - Trained model weights
- `backend/models/emotion_training_history.json` - Training history (loss, accuracy curves)

**Dataset Format:**
- RAVDESS format: Filenames like `03-01-01-01-01-01-01.wav`
- Emotion mapping: `01=neutral, 02=calm→neutral, 03=happiness, 04=sadness, 05=anger, 06/07=frustration`
- Supports recursive directory scanning for .wav files

### 2. `train_sale_predictor.py`
Trains the SalePredictor (XGBoost) for sale probability prediction using CSV with fused feature vectors.

**Usage:**
```bash
python backend/scripts/train_sale_predictor.py --csv_path data/training_data.csv --output_dir backend/models/
```

**Arguments:**
- `--csv_path`: Path to CSV file with fused feature vectors (required)
- `--output_dir`: Output directory for model (default: `backend/models/`)
- `--test_split`: Test split ratio (default: 0.15)
- `--validation_split`: Validation split ratio (default: 0.15)
- `--n_estimators`: XGBoost n_estimators (default: 100)
- `--max_depth`: XGBoost max_depth (default: 6)
- `--learning_rate`: XGBoost learning_rate (default: 0.1)

**CSV Format:**
Required columns:
- `sentiment_mean`, `sentiment_variance` - Textual sentiment features
- `emotion_neutral`, `emotion_happiness`, `emotion_anger`, `emotion_sadness`, `emotion_frustration` - Emotion probabilities
- `silence_ratio`, `interruption_frequency`, `talk_listen_ratio`, `turn_taking_frequency` - Conversational dynamics
- `sale_outcome` - Binary label (0=no_sale, 1=sale)

**Output:**
- `backend/models/sale_model.pkl` - Trained XGBoost model
- `backend/models/sale_model_scaler.pkl` - Feature scaler (needed for inference)
- `backend/models/sale_model_feature_importance.png` - Feature importance visualization
- `backend/models/sale_model_roc_curve.png` - ROC curve plot
- `backend/models/sale_training_results.json` - Training metrics and results

### 3. `validate_trained_models.py`
Validates that trained models can be loaded and used correctly.

**Usage:**
```bash
python backend/scripts/validate_trained_models.py
```

**What it does:**
- Tests loading `emotion_model.pth` with `EmotionDetector`
- Tests loading `sale_model.pkl` with `SalePredictor`
- Validates model inference with dummy data
- Reports validation status for each model

## Model Integration

After training, models are automatically loaded by:
- `EmotionDetector(model_path='backend/models/emotion_model.pth')` - Loads emotion model
- `SalePredictor(model_path='backend/models/sale_model.pkl')` - Loads sale prediction model

The models are used in `ConversationAnalyzer` for real-time inference.

## Notes

- **SentimentAnalyzer**: Uses pre-trained `distilbert-base-uncased-finetuned-sst-2-english`, no training needed
- **RAVDESS Dataset**: Download from https://zenodo.org/record/1188976 and organize in `data/raw/`
- **Sale Training Data**: Prepare CSV with fused feature vectors extracted from real call data using `SalePredictor.create_fused_feature_vector()`
- **GPU Support**: Training scripts automatically use GPU if available (CUDA)

## Example Workflow

1. **Prepare Emotion Dataset:**
   ```bash
   # Download RAVDESS dataset to data/raw/
   # Organize files in subdirectories (optional)
   ```

2. **Train Emotion Model:**
   ```bash
   python backend/scripts/train_emotion_model.py --data_dir data/raw/ --epochs 50
   ```

3. **Prepare Sale Training Data:**
   ```python
   # Extract features from call recordings
   # Create CSV with columns: sentiment_mean, sentiment_variance, emotion_*, dynamics_*, sale_outcome
   ```

4. **Train Sale Predictor:**
   ```bash
   python backend/scripts/train_sale_predictor.py --csv_path data/sale_training_data.csv
   ```

5. **Validate Models:**
   ```bash
   python backend/scripts/validate_trained_models.py
   ```

## Troubleshooting

- **Import Errors**: Ensure you're running from the project root or have PYTHONPATH set correctly
- **CUDA Out of Memory**: Reduce `--batch_size` in emotion training
- **Missing Columns**: Verify CSV format matches expected feature names
- **Model Loading Errors**: Check that model files exist in `backend/models/` directory

