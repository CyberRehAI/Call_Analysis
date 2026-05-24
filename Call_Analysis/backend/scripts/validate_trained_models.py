"""
Validation script to test that trained models can be loaded correctly.

This script:
1. Tests loading emotion_model.pth with EmotionDetector
2. Tests loading sale_model.pkl with SalePredictor
3. Validates model inference works correctly
"""

import os
import sys
import logging
from pathlib import Path
import numpy as np
import torch

# Add parent directory to path to import call_analysis modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.call_analysis.models import EmotionDetector, SalePredictor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def validate_emotion_model(model_path: str) -> bool:
    """
    Validate that emotion model can be loaded and used.
    
    Args:
        model_path: Path to emotion_model.pth
        
    Returns:
        True if validation successful, False otherwise
    """
    logger.info(f"Validating emotion model at {model_path}")
    
    if not os.path.exists(model_path):
        logger.error(f"Emotion model file not found: {model_path}")
        return False
    
    try:
        # Initialize EmotionDetector with model path
        detector = EmotionDetector(model_path=model_path)
        
        if not detector.is_trained:
            logger.error("EmotionDetector reports model is not trained")
            return False
        
        logger.info("✓ Emotion model loaded successfully")
        
        # Test inference with dummy mel-spectrogram
        dummy_mel_spec = np.random.randn(128, 500)  # (n_mels, time_frames)
        audio_features = {'mel_spectrogram': dummy_mel_spec}
        
        result = detector.detect_emotion(audio_features)
        
        # Validate result structure
        required_keys = ['emotion', 'confidence', 'probabilities']
        for key in required_keys:
            if key not in result:
                logger.error(f"Missing key in emotion detection result: {key}")
                return False
        
        # Validate emotion is one of the expected labels
        expected_emotions = ['neutral', 'happiness', 'anger', 'sadness', 'frustration']
        if result['emotion'] not in expected_emotions:
            logger.error(f"Unexpected emotion label: {result['emotion']}")
            return False
        
        # Validate probabilities sum to ~1.0
        prob_sum = sum(result['probabilities'].values())
        if abs(prob_sum - 1.0) > 0.01:
            logger.warning(f"Probabilities sum to {prob_sum}, expected ~1.0")
        
        logger.info(f"✓ Emotion detection test passed: {result['emotion']} (confidence: {result['confidence']:.3f})")
        return True
        
    except Exception as e:
        logger.error(f"Error validating emotion model: {e}")
        import traceback
        traceback.print_exc()
        return False


def validate_sale_model(model_path: str) -> bool:
    """
    Validate that sale prediction model can be loaded and used.
    
    Args:
        model_path: Path to sale_model.pkl
        
    Returns:
        True if validation successful, False otherwise
    """
    logger.info(f"Validating sale model at {model_path}")
    
    if not os.path.exists(model_path):
        logger.error(f"Sale model file not found: {model_path}")
        return False
    
    try:
        # Initialize SalePredictor with model path
        predictor = SalePredictor(model_path=model_path)
        
        if not predictor.is_trained:
            logger.error("SalePredictor reports model is not trained")
            return False
        
        logger.info("✓ Sale model loaded successfully")
        
        # Test inference with dummy fused feature vector
        # Expected: 11 features (2 sentiment + 5 emotion + 4 dynamics)
        dummy_features = np.array([
            0.1,   # sentiment_mean
            0.05,  # sentiment_variance
            0.2,   # emotion_neutral
            0.3,   # emotion_happiness
            0.1,   # emotion_anger
            0.2,   # emotion_sadness
            0.2,   # emotion_frustration
            0.1,   # silence_ratio
            0.05,  # interruption_frequency
            1.2,   # talk_listen_ratio
            0.3    # turn_taking_frequency
        ])
        
        result = predictor.predict_sale_probability(dummy_features)
        
        # Validate result structure
        required_keys = ['sale_probability', 'prediction', 'confidence', 'feature_importance', 'top_features']
        for key in required_keys:
            if key not in result:
                logger.error(f"Missing key in sale prediction result: {key}")
                return False
        
        # Validate sale_probability is in [0, 1]
        prob = result['sale_probability']
        if not (0.0 <= prob <= 1.0):
            logger.error(f"Sale probability out of range: {prob}")
            return False
        
        # Validate prediction is valid
        valid_predictions = ['sale', 'no_sale', 'unknown', 'error']
        if result['prediction'] not in valid_predictions:
            logger.warning(f"Unexpected prediction value: {result['prediction']}")
        
        logger.info(f"✓ Sale prediction test passed: {result['prediction']} "
                   f"(probability: {result['sale_probability']:.3f}, confidence: {result['confidence']:.3f})")
        
        # Check feature importance
        if result['feature_importance']:
            logger.info(f"✓ Feature importance available ({len(result['feature_importance'])} features)")
        if result['top_features']:
            logger.info(f"✓ Top features available ({len(result['top_features'])} features)")
        
        return True
        
    except Exception as e:
        logger.error(f"Error validating sale model: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main validation function."""
    logger.info("=" * 60)
    logger.info("Validating Trained Models")
    logger.info("=" * 60)
    
    # Default model paths
    models_dir = Path(__file__).parent.parent / 'models'
    emotion_model_path = models_dir / 'emotion_model.pth'
    sale_model_path = models_dir / 'sale_model.pkl'
    
    results = {}
    
    # Validate emotion model
    logger.info("\n" + "-" * 60)
    results['emotion_model'] = validate_emotion_model(str(emotion_model_path))
    
    # Validate sale model
    logger.info("\n" + "-" * 60)
    results['sale_model'] = validate_sale_model(str(sale_model_path))
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Validation Summary")
    logger.info("=" * 60)
    
    for model_name, success in results.items():
        status = "✓ PASSED" if success else "✗ FAILED"
        logger.info(f"{model_name}: {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        logger.info("\n✓ All model validations passed!")
        return 0
    else:
        logger.error("\n✗ Some model validations failed!")
        return 1


if __name__ == '__main__':
    exit(main())

