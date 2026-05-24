"""
Unit tests for feature extraction.
"""

import unittest
import numpy as np
from pathlib import Path
import sys
import tempfile
import os

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.call_analysis.feature_extraction import FeatureExtractor


class TestFeatureExtractor(unittest.TestCase):
    """Test FeatureExtractor."""
    
    def setUp(self):
        self.extractor = FeatureExtractor()
    
    def test_validate_audio_file(self):
        """Test audio file validation."""
        # Test None
        with self.assertRaises(ValueError):
            self.extractor._validate_audio_file(None)
        
        # Test non-string
        with self.assertRaises(ValueError):
            self.extractor._validate_audio_file(123)
        
        # Test non-existent file
        with self.assertRaises(ValueError):
            self.extractor._validate_audio_file("/nonexistent/file.wav")
        
        # Test invalid extension
        with self.assertRaises(ValueError):
            with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
                temp_path = f.name
            try:
                self.extractor._validate_audio_file(temp_path)
            finally:
                os.unlink(temp_path)
    
    def test_extract_audio_features(self):
        """Test audio feature extraction."""
        # Create dummy audio features
        audio_features = {
            'mfcc': np.random.randn(40, 100),
            'mel_spectrogram': np.random.randn(128, 100),
            'chroma': np.random.randn(12, 100),
            'spectral_centroid': np.random.randn(1, 100),
            'spectral_rolloff': np.random.randn(1, 100),
            'zero_crossing_rate': np.random.randn(1, 100),
            'duration': 5.0,
            'sample_rate': 16000,
            'pitch': np.random.randn(100),
            'pitch_stats': {
                'mean': 200.0,
                'std': 50.0,
                'min': 100.0,
                'max': 300.0,
                'median': 200.0
            },
            'energy': np.random.randn(100),
            'speaking_rate': 3.5,
            'formants': [500.0, 1500.0, 2500.0]
        }
        
        features = self.extractor.extract_audio_features(audio_features)
        
        self.assertIsInstance(features, np.ndarray)
        self.assertGreater(len(features), 0)
        self.assertFalse(np.isnan(features).any())
        self.assertFalse(np.isinf(features).any())
    
    def test_extract_audio_features_empty(self):
        """Test feature extraction with empty input."""
        features = self.extractor.extract_audio_features({})
        
        self.assertIsInstance(features, np.ndarray)
        self.assertEqual(len(features), 100)  # Default size


if __name__ == '__main__':
    unittest.main()

