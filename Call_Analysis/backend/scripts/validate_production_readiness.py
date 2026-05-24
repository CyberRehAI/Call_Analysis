"""
Production readiness validation script.

This script validates:
1. All models load correctly
2. Feature extraction works
3. Model health checks pass
4. Edge cases are handled
5. Input validation works
"""

import os
import sys
import logging
from pathlib import Path
import numpy as np
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.call_analysis.models import (
    EmotionDetector, 
    SalePredictor, 
    SentimentAnalyzer,
    ModelNotTrainedError,
    InvalidInputError
)
from src.call_analysis.feature_extraction import FeatureExtractor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_models_load():
    """Check that all models can be loaded."""
    logger.info("=" * 60)
    logger.info("Checking Model Loading")
    logger.info("=" * 60)
    
    results = {}
    models_dir = Path(__file__).parent.parent / 'models'
    
    # Check EmotionDetector
    emotion_model_path = models_dir / 'emotion_model.pth'
    try:
        if emotion_model_path.exists():
            detector = EmotionDetector(model_path=str(emotion_model_path))
            results['emotion_model'] = detector.is_trained
            logger.info(f"✓ Emotion model loaded: {detector.is_trained}")
        else:
            results['emotion_model'] = False
            logger.warning(f"✗ Emotion model not found: {emotion_model_path}")
    except Exception as e:
        results['emotion_model'] = False
        logger.error(f"✗ Failed to load emotion model: {e}")
    
    # Check SalePredictor
    sale_model_path = models_dir / 'sale_model.pkl'
    try:
        if sale_model_path.exists():
            predictor = SalePredictor(model_path=str(sale_model_path))
            results['sale_model'] = predictor.is_trained
            logger.info(f"✓ Sale model loaded: {predictor.is_trained}")
        else:
            results['sale_model'] = False
            logger.warning(f"✗ Sale model not found: {sale_model_path}")
    except Exception as e:
        results['sale_model'] = False
        logger.error(f"✗ Failed to load sale model: {e}")
    
    # Check SentimentAnalyzer (always available)
    try:
        analyzer = SentimentAnalyzer()
        results['sentiment_analyzer'] = True
        logger.info("✓ Sentiment analyzer initialized")
    except Exception as e:
        results['sentiment_analyzer'] = False
        logger.error(f"✗ Failed to initialize sentiment analyzer: {e}")
    
    return results


def check_model_health():
    """Check model health using health check methods."""
    logger.info("\n" + "=" * 60)
    logger.info("Checking Model Health")
    logger.info("=" * 60)
    
    results = {}
    models_dir = Path(__file__).parent.parent / 'models'
    
    # Check EmotionDetector health
    emotion_model_path = models_dir / 'emotion_model.pth'
    if emotion_model_path.exists():
        try:
            detector = EmotionDetector(model_path=str(emotion_model_path))
            health = detector.check_model_health()
            results['emotion_health'] = len(health.get('errors', [])) == 0
            if results['emotion_health']:
                logger.info("✓ Emotion model health check passed")
            else:
                logger.warning(f"✗ Emotion model health check failed: {health.get('errors', [])}")
        except Exception as e:
            results['emotion_health'] = False
            logger.error(f"✗ Emotion model health check error: {e}")
    else:
        results['emotion_health'] = None
        logger.info("⊘ Emotion model not available for health check")
    
    # Check SalePredictor health
    sale_model_path = models_dir / 'sale_model.pkl'
    if sale_model_path.exists():
        try:
            predictor = SalePredictor(model_path=str(sale_model_path))
            health = predictor.check_model_health()
            results['sale_health'] = len(health.get('errors', [])) == 0
            if results['sale_health']:
                logger.info("✓ Sale model health check passed")
            else:
                logger.warning(f"✗ Sale model health check failed: {health.get('errors', [])}")
        except Exception as e:
            results['sale_health'] = False
            logger.error(f"✗ Sale model health check error: {e}")
    else:
        results['sale_health'] = None
        logger.info("⊘ Sale model not available for health check")
    
    # Check SentimentAnalyzer health
    try:
        analyzer = SentimentAnalyzer()
        health = analyzer.check_model_health()
        results['sentiment_health'] = len(health.get('errors', [])) == 0
        if results['sentiment_health']:
            logger.info("✓ Sentiment analyzer health check passed")
        else:
            logger.warning(f"✗ Sentiment analyzer health check failed: {health.get('errors', [])}")
    except Exception as e:
        results['sentiment_health'] = False
        logger.error(f"✗ Sentiment analyzer health check error: {e}")
    
    return results


def check_input_validation():
    """Check input validation works correctly."""
    logger.info("\n" + "=" * 60)
    logger.info("Checking Input Validation")
    logger.info("=" * 60)
    
    results = {}
    
    # Test SentimentAnalyzer validation
    try:
        analyzer = SentimentAnalyzer()
        
        # Test invalid inputs
        with np.testing.assert_raises(ValueError):
            analyzer._validate_text_input(None)
        with np.testing.assert_raises(ValueError):
            analyzer._validate_text_input("")
        
        results['sentiment_validation'] = True
        logger.info("✓ Sentiment input validation works")
    except Exception as e:
        results['sentiment_validation'] = False
        logger.error(f"✗ Sentiment validation failed: {e}")
    
    # Test EmotionDetector validation
    try:
        detector = EmotionDetector()
        
        # Test invalid inputs
        with np.testing.assert_raises(InvalidInputError):
            detector._validate_audio_features(None)
        with np.testing.assert_raises(InvalidInputError):
            detector._validate_audio_features({})
        with np.testing.assert_raises(InvalidInputError):
            detector._validate_audio_features({'mel_spectrogram': np.random.randn(64, 100)})
        
        results['emotion_validation'] = True
        logger.info("✓ Emotion input validation works")
    except Exception as e:
        results['emotion_validation'] = False
        logger.error(f"✗ Emotion validation failed: {e}")
    
    # Test SalePredictor validation
    try:
        predictor = SalePredictor()
        
        # Test invalid inputs
        with np.testing.assert_raises(InvalidInputError):
            predictor._validate_feature_vector(None)
        with np.testing.assert_raises(InvalidInputError):
            predictor._validate_feature_vector(np.array([]))
        with np.testing.assert_raises(InvalidInputError):
            predictor._validate_feature_vector(np.array([1, 2, 3]))  # Too few features
        
        results['sale_validation'] = True
        logger.info("✓ Sale prediction input validation works")
    except Exception as e:
        results['sale_validation'] = False
        logger.error(f"✗ Sale validation failed: {e}")
    
    # Test FeatureExtractor validation
    try:
        extractor = FeatureExtractor()
        
        # Test invalid inputs
        with np.testing.assert_raises(ValueError):
            extractor._validate_audio_file(None)
        with np.testing.assert_raises(ValueError):
            extractor._validate_audio_file("/nonexistent/file.wav")
        
        results['feature_validation'] = True
        logger.info("✓ Feature extraction input validation works")
    except Exception as e:
        results['feature_validation'] = False
        logger.error(f"✗ Feature validation failed: {e}")
    
    return results


def check_edge_cases():
    """Check edge case handling."""
    logger.info("\n" + "=" * 60)
    logger.info("Checking Edge Cases")
    logger.info("=" * 60)
    
    results = {}
    
    # Test empty text
    try:
        analyzer = SentimentAnalyzer()
        result = analyzer.analyze_sentiment("")
        results['empty_text'] = result['sentiment'] == 'neutral'
        logger.info("✓ Empty text handled correctly")
    except Exception as e:
        results['empty_text'] = False
        logger.error(f"✗ Empty text handling failed: {e}")
    
    # Test very long text
    try:
        analyzer = SentimentAnalyzer()
        long_text = "a" * 10001
        with np.testing.assert_raises(ValueError):
            analyzer._validate_text_input(long_text)
        results['long_text'] = True
        logger.info("✓ Long text validation works")
    except Exception as e:
        results['long_text'] = False
        logger.error(f"✗ Long text handling failed: {e}")
    
    # Test feature extraction with empty dict
    try:
        extractor = FeatureExtractor()
        features = extractor.extract_audio_features({})
        results['empty_features'] = len(features) > 0
        logger.info("✓ Empty features handled correctly")
    except Exception as e:
        results['empty_features'] = False
        logger.error(f"✗ Empty features handling failed: {e}")
    
    return results


def main():
    """Main validation function."""
    logger.info("=" * 60)
    logger.info("Production Readiness Validation")
    logger.info("=" * 60)
    
    all_results = {}
    
    # Run all checks
    all_results.update(check_models_load())
    all_results.update(check_model_health())
    all_results.update(check_input_validation())
    all_results.update(check_edge_cases())
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Validation Summary")
    logger.info("=" * 60)
    
    passed = sum(1 for v in all_results.values() if v is True)
    failed = sum(1 for v in all_results.values() if v is False)
    skipped = sum(1 for v in all_results.values() if v is None)
    
    for check_name, result in all_results.items():
        if result is True:
            status = "✓ PASSED"
        elif result is False:
            status = "✗ FAILED"
        else:
            status = "⊘ SKIPPED"
        logger.info(f"{check_name}: {status}")
    
    logger.info(f"\nTotal: {passed} passed, {failed} failed, {skipped} skipped")
    
    if failed == 0:
        logger.info("\n✓ All checks passed! System is production-ready.")
        return 0
    else:
        logger.error(f"\n✗ {failed} check(s) failed. Please review and fix issues.")
        return 1


if __name__ == '__main__':
    exit(main())

