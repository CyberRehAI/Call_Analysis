"""
Integration tests for full pipeline.
"""

import unittest
import numpy as np
from pathlib import Path
import sys
import tempfile
import os

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.call_analysis.models import ConversationAnalyzer
from src.call_analysis.feature_extraction import FeatureExtractor


class TestIntegration(unittest.TestCase):
    """Integration tests for full pipeline."""
    
    def setUp(self):
        # Note: These tests require trained models
        # They will be skipped if models are not available
        self.models_dir = Path(__file__).parent.parent / 'models'
        self.emotion_model_path = self.models_dir / 'emotion_model.pth'
        self.sale_model_path = self.models_dir / 'sale_model.pkl'
    
    def test_conversation_analyzer_initialization(self):
        """Test ConversationAnalyzer initialization."""
        # Test without models (should work but models won't be trained)
        analyzer = ConversationAnalyzer()
        
        self.assertIsNotNone(analyzer.sentiment_analyzer)
        self.assertIsNotNone(analyzer.emotion_detector)
        self.assertIsNotNone(analyzer.sale_predictor)
    
    def test_feature_extraction_pipeline(self):
        """Test feature extraction pipeline."""
        extractor = FeatureExtractor()
        
        # Create dummy segments
        segments = [
            {
                'text': 'Hello, I am interested in your product.',
                'speaker': 'Customer',
                'start_time': 0.0,
                'end_time': 2.0
            },
            {
                'text': 'Great! Let me tell you about our features.',
                'speaker': 'Agent',
                'start_time': 2.0,
                'end_time': 5.0
            }
        ]
        
        # Test temporal feature extraction
        temporal_features = extractor.extract_temporal_features(segments)
        
        self.assertIsInstance(temporal_features, dict)
        self.assertIn('total_duration', temporal_features)
        self.assertIn('speaker_changes', temporal_features)
    
    def test_fused_feature_vector_creation(self):
        """Test fused feature vector creation."""
        from src.call_analysis.models import SalePredictor
        
        predictor = SalePredictor()
        
        sentiment_results = [
            {'score': 0.5, 'sentiment': 'positive'},
            {'score': 0.7, 'sentiment': 'positive'},
            {'score': -0.2, 'sentiment': 'negative'}
        ]
        
        emotion_results = [
            {
                'probabilities': {
                    'neutral': 0.2,
                    'happiness': 0.5,
                    'anger': 0.1,
                    'sadness': 0.1,
                    'frustration': 0.1
                }
            }
        ]
        
        dynamics = {
            'silence_ratio': 0.1,
            'interruption_frequency': 0.05,
            'talk_listen_ratio': 1.2,
            'turn_taking_frequency': 0.3
        }
        
        features = predictor.create_fused_feature_vector(
            sentiment_results, emotion_results, dynamics
        )
        
        self.assertEqual(len(features), 11)
        self.assertIsInstance(features, np.ndarray)
        
        # Verify feature ranges are reasonable
        self.assertTrue(np.all(np.isfinite(features)))


if __name__ == '__main__':
    unittest.main()

