"""
Unit tests for model forward passes and inference.
"""

import unittest
import numpy as np
import torch
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.call_analysis.models import (
    AcousticEmotionModel, 
    EmotionDetector, 
    SentimentAnalyzer, 
    SalePredictor,
    ModelNotTrainedError,
    InvalidInputError
)


class TestAcousticEmotionModel(unittest.TestCase):
    """Test AcousticEmotionModel forward pass."""
    
    def setUp(self):
        self.model = AcousticEmotionModel(n_mels=128, n_mfcc=40, num_classes=5, dropout=0.3)
        self.model.eval()
    
    def test_forward_pass_mel_only(self):
        """Test forward pass with Mel-Spectrogram only."""
        batch_size = 2
        mel_input = torch.randn(batch_size, 1, 128, 100)  # (batch, channel, n_mels, time)
        
        with torch.no_grad():
            output = self.model(mel_input, mfcc=None)
        
        self.assertEqual(output.shape, (batch_size, 5))
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())
    
    def test_forward_pass_with_mfcc(self):
        """Test forward pass with both Mel-Spectrogram and MFCC."""
        batch_size = 2
        mel_input = torch.randn(batch_size, 1, 128, 100)
        mfcc_input = torch.randn(batch_size, 40, 100)
        
        with torch.no_grad():
            output = self.model(mel_input, mfcc=mfcc_input)
        
        self.assertEqual(output.shape, (batch_size, 5))
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())
    
    def test_forward_pass_variable_length(self):
        """Test forward pass with variable length sequences."""
        batch_size = 2
        mel_input = torch.randn(batch_size, 1, 128, 100)
        lengths = torch.tensor([80, 100])  # Different lengths
        
        with torch.no_grad():
            output = self.model(mel_input, mfcc=None, lengths=lengths)
        
        self.assertEqual(output.shape, (batch_size, 5))


class TestEmotionDetector(unittest.TestCase):
    """Test EmotionDetector."""
    
    def setUp(self):
        self.detector = EmotionDetector()
    
    def test_model_not_trained_error(self):
        """Test that error is raised when model is not trained."""
        audio_features = {
            'mel_spectrogram': np.random.randn(128, 100),
            'mfcc': np.random.randn(40, 100)
        }
        
        with self.assertRaises(ModelNotTrainedError):
            self.detector.detect_emotion(audio_features)
    
    def test_validate_audio_features(self):
        """Test audio features validation."""
        # Test invalid input types
        with self.assertRaises(InvalidInputError):
            self.detector._validate_audio_features(None)
        
        with self.assertRaises(InvalidInputError):
            self.detector._validate_audio_features("not a dict")
        
        # Test missing mel_spectrogram
        with self.assertRaises(InvalidInputError):
            self.detector._validate_audio_features({})
        
        # Test invalid shape
        with self.assertRaises(InvalidInputError):
            self.detector._validate_audio_features({
                'mel_spectrogram': np.random.randn(64, 100)  # Wrong n_mels
            })
        
        # Test valid input
        try:
            self.detector._validate_audio_features({
                'mel_spectrogram': np.random.randn(128, 100)
            })
        except InvalidInputError:
            self.fail("Valid input should not raise InvalidInputError")
    
    def test_health_check(self):
        """Test model health check."""
        health = self.detector.check_model_health()
        
        self.assertIn('is_trained', health)
        self.assertIn('model_loaded', health)
        self.assertIn('inference_test', health)
        self.assertIn('errors', health)


class TestSentimentAnalyzer(unittest.TestCase):
    """Test SentimentAnalyzer."""
    
    def setUp(self):
        self.analyzer = SentimentAnalyzer()
    
    def test_validate_text_input(self):
        """Test text input validation."""
        # Test None
        with self.assertRaises(ValueError):
            self.analyzer._validate_text_input(None)
        
        # Test non-string
        with self.assertRaises(ValueError):
            self.analyzer._validate_text_input(123)
        
        # Test empty string
        with self.assertRaises(ValueError):
            self.analyzer._validate_text_input("")
        
        # Test too long
        with self.assertRaises(ValueError):
            self.analyzer._validate_text_input("a" * 10001)
        
        # Test valid input
        try:
            self.analyzer._validate_text_input("This is a test sentence.")
        except ValueError:
            self.fail("Valid input should not raise ValueError")
    
    def test_analyze_sentiment(self):
        """Test sentiment analysis."""
        result = self.analyzer.analyze_sentiment("I love this product!")
        
        self.assertIn('sentiment', result)
        self.assertIn('score', result)
        self.assertIn('confidence', result)
        self.assertIn(result['sentiment'], ['positive', 'negative', 'neutral'])
    
    def test_extract_key_phrases(self):
        """Test key phrase extraction."""
        text = "The customer service team was excellent and very helpful."
        phrases = self.analyzer.extract_key_phrases(text, top_n=5)
        
        self.assertIsInstance(phrases, list)
        if phrases:  # If spaCy is available
            self.assertLessEqual(len(phrases), 5)
            for phrase in phrases:
                self.assertIn('phrase', phrase)
                self.assertIn('sentiment_score', phrase)
    
    def test_health_check(self):
        """Test model health check."""
        health = self.analyzer.check_model_health()
        
        self.assertIn('sentiment_pipeline_loaded', health)
        self.assertIn('bert_model_loaded', health)
        self.assertIn('errors', health)


class TestSalePredictor(unittest.TestCase):
    """Test SalePredictor."""
    
    def setUp(self):
        self.predictor = SalePredictor()
    
    def test_validate_feature_vector(self):
        """Test feature vector validation."""
        # Test invalid input types
        with self.assertRaises(InvalidInputError):
            self.predictor._validate_feature_vector(None)
        
        with self.assertRaises(InvalidInputError):
            self.predictor._validate_feature_vector("not an array")
        
        # Test empty array
        with self.assertRaises(InvalidInputError):
            self.predictor._validate_feature_vector(np.array([]))
        
        # Test too few features
        with self.assertRaises(InvalidInputError):
            self.predictor._validate_feature_vector(np.array([1, 2, 3]))  # Only 3 features
        
        # Test valid input
        try:
            valid_features = np.random.randn(11)  # Minimum 11 features
            self.predictor._validate_feature_vector(valid_features)
        except InvalidInputError:
            self.fail("Valid input should not raise InvalidInputError")
    
    def test_model_not_trained_error(self):
        """Test that error is raised when model is not trained."""
        features = np.random.randn(11)
        
        with self.assertRaises(ModelNotTrainedError):
            self.predictor.predict_sale_probability(features)
    
    def test_create_fused_feature_vector(self):
        """Test fused feature vector creation."""
        sentiment_results = [
            {'score': 0.5},
            {'score': 0.7}
        ]
        emotion_results = [
            {'probabilities': {'neutral': 0.2, 'happiness': 0.5, 'anger': 0.1, 'sadness': 0.1, 'frustration': 0.1}}
        ]
        dynamics = {
            'silence_ratio': 0.1,
            'interruption_frequency': 0.05,
            'talk_listen_ratio': 1.2,
            'turn_taking_frequency': 0.3
        }
        
        features = self.predictor.create_fused_feature_vector(
            sentiment_results, emotion_results, dynamics
        )
        
        self.assertEqual(len(features), 11)  # 2 + 5 + 4
        self.assertIsInstance(features, np.ndarray)
    
    def test_health_check(self):
        """Test model health check."""
        health = self.predictor.check_model_health()
        
        self.assertIn('is_trained', health)
        self.assertIn('model_loaded', health)
        self.assertIn('errors', health)


if __name__ == '__main__':
    unittest.main()

