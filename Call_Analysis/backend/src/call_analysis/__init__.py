"""
Call Analysis System - AI-powered sentiment analysis and sale prediction
"""

__version__ = "0.1.0"
__author__ = "Call Analysis Team"

from .preprocessing import AudioProcessor, TextProcessor
from .feature_extraction import FeatureExtractor
from .models import SentimentAnalyzer, EmotionDetector, SalePredictor
from .dashboard import Dashboard
from .demo import DemoSystem

__all__ = [
    "AudioProcessor",
    "TextProcessor", 
    "FeatureExtractor",
    "SentimentAnalyzer",
    "EmotionDetector",
    "SalePredictor",
    "Dashboard",
    "DemoSystem"
]

def main():
    """Main entry point for the call analysis system"""
    demo = DemoSystem()
    demo.run_demo()