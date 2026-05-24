"""
End-to-end test script for training pipeline validation.

This script:
1. Generates mock emotion data
2. Trains emotion model
3. Generates mock sale data
4. Trains sale predictor
5. Validates models can be loaded and used
"""

import os
import sys
import subprocess
import logging
from pathlib import Path
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.call_analysis.models import EmotionDetector, SalePredictor
from src.call_analysis.feature_extraction import FeatureExtractor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_command(cmd: list, description: str) -> bool:
    """
    Run a command and return success status.
    
    Args:
        cmd: Command as list of strings
        description: Description of what the command does
        
    Returns:
        True if successful, False otherwise
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Running: {description}")
    logger.info(f"Command: {' '.join(cmd)}")
    logger.info(f"{'='*60}")
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent
        )
        logger.info("✓ Command completed successfully")
        if result.stdout:
            logger.info(f"Output:\n{result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ Command failed with exit code {e.returncode}")
        if e.stdout:
            logger.error(f"Stdout:\n{e.stdout}")
        if e.stderr:
            logger.error(f"Stderr:\n{e.stderr}")
        return False


def test_emotion_model(model_path: str) -> bool:
    """Test that emotion model can be loaded and used."""
    logger.info(f"\nTesting emotion model at {model_path}...")
    
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        return False
    
    try:
        detector = EmotionDetector(model_path=model_path)
        
        if not detector.is_trained:
            logger.error("Model reports as not trained")
            return False
        
        # Test inference
        dummy_mel_spec = np.random.randn(128, 500)
        audio_features = {'mel_spectrogram': dummy_mel_spec}
        result = detector.detect_emotion(audio_features)
        
        if 'emotion' not in result or 'confidence' not in result:
            logger.error("Invalid result structure")
            return False
        
        logger.info(f"✓ Emotion model test passed: {result['emotion']} (confidence: {result['confidence']:.3f})")
        return True
        
    except Exception as e:
        logger.error(f"Error testing emotion model: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_sale_model(model_path: str) -> bool:
    """Test that sale model can be loaded and used."""
    logger.info(f"\nTesting sale model at {model_path}...")
    
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        return False
    
    try:
        predictor = SalePredictor(model_path=model_path)
        
        if not predictor.is_trained:
            logger.error("Model reports as not trained")
            return False
        
        # Test inference
        dummy_features = np.array([
            0.1, 0.05, 0.2, 0.3, 0.1, 0.2, 0.2, 0.1, 0.05, 1.2, 0.3
        ])
        result = predictor.predict_sale_probability(dummy_features)
        
        if 'sale_probability' not in result:
            logger.error("Invalid result structure")
            return False
        
        prob = result['sale_probability']
        if not (0.0 <= prob <= 1.0):
            logger.error(f"Invalid probability: {prob}")
            return False
        
        logger.info(f"✓ Sale model test passed: {result['prediction']} (probability: {prob:.3f})")
        return True
        
    except Exception as e:
        logger.error(f"Error testing sale model: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function."""
    logger.info("="*60)
    logger.info("Training Pipeline Validation Test")
    logger.info("="*60)
    
    # Paths
    mock_emotion_dir = 'data/mock_ravdess'
    mock_sale_csv = 'data/mock_sale_training.csv'
    emotion_model_path = 'backend/models/emotion_model.pth'
    sale_model_path = 'backend/models/sale_model.pkl'
    
    results = {}
    
    # Step 1: Generate mock emotion data
    logger.info("\n" + "="*60)
    logger.info("Step 1: Generate Mock Emotion Data")
    logger.info("="*60)
    
    cmd = [
        sys.executable,
        'backend/scripts/generate_mock_emotion_data.py',
        '--output_dir', mock_emotion_dir,
        '--samples_per_emotion', '10',  # Small dataset for quick testing
        '--duration', '2.0'  # Shorter duration for faster processing
    ]
    results['generate_emotion'] = run_command(cmd, "Generate mock emotion data")
    
    if not results['generate_emotion']:
        logger.error("Failed to generate mock emotion data. Aborting.")
        return 1
    
    # Step 2: Train emotion model
    logger.info("\n" + "="*60)
    logger.info("Step 2: Train Emotion Model")
    logger.info("="*60)
    
    cmd = [
        sys.executable,
        'backend/scripts/train_emotion_model.py',
        '--data_dir', mock_emotion_dir,
        '--output_dir', 'backend/models/',
        '--epochs', '5',  # Few epochs for quick testing
        '--batch_size', '8',
        '--validation_split', '0.2'
    ]
    results['train_emotion'] = run_command(cmd, "Train emotion model")
    
    # Step 3: Generate mock sale data
    logger.info("\n" + "="*60)
    logger.info("Step 3: Generate Mock Sale Data")
    logger.info("="*60)
    
    cmd = [
        sys.executable,
        'backend/scripts/generate_mock_sale_data.py',
        '--output_path', mock_sale_csv,
        '--n_samples', '100'  # Small dataset for quick testing
    ]
    results['generate_sale'] = run_command(cmd, "Generate mock sale data")
    
    if not results['generate_sale']:
        logger.error("Failed to generate mock sale data. Aborting.")
        return 1
    
    # Step 4: Train sale predictor
    logger.info("\n" + "="*60)
    logger.info("Step 4: Train Sale Predictor")
    logger.info("="*60)
    
    cmd = [
        sys.executable,
        'backend/scripts/train_sale_predictor.py',
        '--csv_path', mock_sale_csv,
        '--output_dir', 'backend/models/',
        '--n_estimators', '50',  # Fewer estimators for quick testing
        '--test_split', '0.15',
        '--validation_split', '0.15'
    ]
    results['train_sale'] = run_command(cmd, "Train sale predictor")
    
    # Step 5: Validate models
    logger.info("\n" + "="*60)
    logger.info("Step 5: Validate Trained Models")
    logger.info("="*60)
    
    results['test_emotion'] = test_emotion_model(emotion_model_path)
    results['test_sale'] = test_sale_model(sale_model_path)
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("Test Summary")
    logger.info("="*60)
    
    for step, success in results.items():
        status = "✓ PASSED" if success else "✗ FAILED"
        logger.info(f"{step}: {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        logger.info("\n" + "="*60)
        logger.info("✓ ALL TESTS PASSED!")
        logger.info("="*60)
        logger.info("\nModels are ready to use:")
        logger.info(f"  - Emotion model: {emotion_model_path}")
        logger.info(f"  - Sale model: {sale_model_path}")
        return 0
    else:
        logger.error("\n" + "="*60)
        logger.error("✗ SOME TESTS FAILED")
        logger.error("="*60)
        return 1


if __name__ == '__main__':
    exit(main())

