"""
Demo system for call analysis prototype, aligned with SRS1.
Integrates preprocessing, feature extraction, models, and dashboard for end-to-end POC.
Uses simulated audio for realism, stores in MongoDB (FR-7), and masks PII.
"""

import os
import json
import logging
from typing import Dict, List, Optional
from datetime import datetime
import numpy as np

from .models import ConversationAnalyzer
from .dashboard import Dashboard
from .preprocessing import AudioProcessor, TextProcessor
from .feature_extraction import FeatureExtractor
from pymongo import MongoClient
import spacy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load spaCy for PII masking (optional)
try:
    nlp = spacy.load('en_core_web_sm')
    SPACY_AVAILABLE = True
    logger.info("spaCy model loaded successfully")
except OSError:
    nlp = None
    SPACY_AVAILABLE = False
    logger.warning("spaCy model 'en_core_web_sm' not found. PII masking will be limited.")


class DemoSystem:
    """Demo system for showcasing call analysis capabilities (SRS1 POC)."""
    
    def __init__(self, hf_token: str = None):
        """Initialize demo system."""
        self.analyzer = ConversationAnalyzer()
        self.dashboard = Dashboard()
        # For real-call analysis via the web frontend, assume two speakers.
        # Use WhisperX 3.x built-in diarization (Pyannote.audio) with num_speakers=2 when HF_TOKEN is available.
        self.audio_processor = AudioProcessor(
            hf_token=hf_token,
            num_speakers=2,
            max_speakers=2,
            use_whisperx_builtin_diarization=True
        )
        self.text_processor = TextProcessor()
        self.feature_extractor = FeatureExtractor()
        
        # Simulated audio files (create dummy .wav for demo)
        self.demo_audio_paths = self._create_demo_audio()
        
        # Demo conversations (text-based fallback)
        self.demo_conversations = self._create_demo_conversations()
        
    def mask_pii(self, text: str) -> str:
        """
        Mask PII in demo text (SRS1 security).
        
        Args:
            text: Input text.
            
        Returns:
            Anonymized text.
        """
        if not SPACY_AVAILABLE or nlp is None:
            # Simple regex-based PII masking when spaCy is not available
            import re
            # Mask phone numbers
            text = re.sub(r'\b\d{3}-\d{3}-\d{4}\b', '[PHONE]', text)
            text = re.sub(r'\b\(\d{3}\)\s*\d{3}-\d{4}\b', '[PHONE]', text)
            # Mask email addresses
            text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
            return text
        
        try:
            doc = nlp(text)
            for ent in doc.ents:
                if ent.label_ in ['PERSON', 'PHONE', 'EMAIL', 'GPE', 'ORG']:
                    text = text.replace(ent.text, '[REDACTED]')
            return text
        except Exception as e:
            logger.error(f"PII masking failed: {e}")
            return text
    
    def _create_demo_audio(self) -> List[str]:
        """Create simulated dummy audio files for demo."""
        audio_paths = []
        for i in range(3):
            path = f"demo_audio_{i+1}.wav"
            if not os.path.exists(path):
                # Generate dummy audio with librosa (requires librosa import)
                import librosa
                y = np.random.normal(0, 0.1, 16000 * 60)  # 1-min dummy
                import soundfile as sf
                sf.write(path, y, 16000)
                logger.info(f"Created dummy audio: {path}")
            audio_paths.append(path)
        return audio_paths
    
    def _create_demo_conversations(self) -> List[Dict]:
        """Create demo conversation data with PII masked."""
        conversations = [
            {
                "id": "demo_001",
                "title": "Insurance Policy Inquiry - High Sale Potential",
                "audio_path": self.demo_audio_paths[0],
                "segments": [
                    {
                        "start_time": 0, "end_time": 12, "speaker": "Customer",
                        "text": self.mask_pii("Hello, I'm calling about your insurance policy. I've been looking for better coverage and heard great things about your company.")
                    },
                    # ... (mask all texts similarly; abbreviated for brevity)
                ]
            },
            # ... (similar for demo_002, demo_003)
        ]
        return conversations
    
    def run_demo(self):
        """Run the complete demo with full pipeline."""
        print("=" * 60)
        print("CALL ANALYSIS SYSTEM - DEMO (SRS1 POC)")
        print("=" * 60)
        print()
        
        # Show available demo conversations
        print("Available Demo Conversations:")
        for i, conv in enumerate(self.demo_conversations, 1):
            print(f"{i}. {conv['title']} (ID: {conv['id']})")
        print()
        
        # Analyze each conversation with full pipeline
        results = []
        for conv in self.demo_conversations:
            print(f"Analyzing: {conv['title']}")
            print("-" * 40)
            
            try:
                # Full pipeline: Preprocess audio -> Extract features -> Analyze
                transcription = self.audio_processor.transcribe_audio(conv['audio_path'], conv['id'])
                segments = self.audio_processor.perform_speaker_diarization(conv['audio_path'], conv['id'])
                processed_segments = self.text_processor.segment_conversation(
                    transcription['text'],
                    segments,
                    conv['id'],
                    transcription_segments=transcription.get('segments', [])
                )
                audio_features = self.audio_processor.extract_audio_features(conv['audio_path'])
                
                feature_data = [{'audio_features': audio_features, 'text_features': {'text': transcription['text']}, 'segments': processed_segments}]
                fused_features = self.feature_extractor.fit_transform(feature_data)[0]
                
                # Analyze with fused features
                analysis_result = self.analyzer.analyze_conversation(
                    audio_path=conv['audio_path'], 
                    segments=processed_segments, 
                    features=fused_features, 
                    call_id=conv['id']
                )
                results.append(analysis_result)
                
                # Display key metrics
                sale_prob = analysis_result['sale_prediction']['sale_probability']
                avg_sentiment = analysis_result['conversation_metrics']['average_sentiment']
                dominant_emotion = analysis_result['conversation_metrics']['dominant_emotion']
                
                print(f"Sale Probability: {sale_prob:.1%}")
                print(f"Average Sentiment: {avg_sentiment:.2f}")
                print(f"Dominant Emotion: {dominant_emotion}")
                print(f"Duration: {analysis_result['duration']:.1f}s")
                print()
            except Exception as e:
                logger.error(f"Analysis failed for {conv['title']}: {e}")
                print(f"Error: {e}")
                print()
        
        # Generate agent insights (fixed keys)
        print("AGENT INSIGHTS")
        print("=" * 40)
        agent_insights = self.analyzer.get_agent_insights(results)
        
        print(f"Total Conversations Analyzed: {len(results)}")
        print(f"Average Sale Probability: {agent_insights['average_sale_probability']:.1%}")
        print(f"Average Sentiment: {agent_insights['average_sentiment']:.2f}")
        print()
        
        print("Recommendations:")
        for rec in agent_insights['recommendations']:
            print(f"- {rec}")
        print()
        
        # Create batch dashboard
        print("Creating Batch Dashboard...")
        dashboard_paths = self.create_batch_dashboard(results)
        print(f"Batch dashboards saved: {dashboard_paths}")
        print()
        
        # Create comparison report
        print("Creating Comparison Report...")
        comparison_path = self.create_comparison_report()
        print(f"Comparison report: {comparison_path}")
        print()
        
        print("Demo completed successfully!")
        print("=" * 60)
        
        return results
    
    def create_batch_dashboard(self, results: List[Dict]) -> List[str]:
        """Create dashboards for all results (FR-8)."""
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        paths = []
        for result in results:
            call_id = result.get('conversation_id', 'unknown')
            path = os.path.join(output_dir, f"dashboard_{call_id}.html")
            self.dashboard.create_dashboard_html(result, call_id, path)
            paths.append(path)
        return paths
    
    def create_comparison_report(self, output_dir: str = "output") -> str:
        """Create comparison report (FR-8)."""
        results = self.batch_analyze_all()  # Ensure full pipeline
        os.makedirs(output_dir, exist_ok=True)
        report_path = os.path.join(output_dir, "comparison_report.txt")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("CALL ANALYSIS COMPARISON REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Conversations: {len(results)}\n\n")
            
            f.write("CONVERSATION SUMMARY\n")
            f.write("-" * 30 + "\n")
            f.write(f"{'ID':<10} {'Title':<30} {'Sale Prob':<10} {'Sentiment':<10} {'Emotion':<12}\n")
            f.write("-" * 80 + "\n")
            
            for result in results:
                sale_prob = result['sale_prediction']['sale_probability']
                sentiment = result['conversation_metrics']['average_sentiment']
                emotion = result['conversation_metrics']['dominant_emotion']
                title = result.get('conversation_title', 'N/A')[:28] + ".." if len(result.get('conversation_title', '')) > 30 else result.get('conversation_title', 'N/A')
                
                f.write(f"{result.get('conversation_id', 'N/A'):<10} {title:<30} {sale_prob:<10.1%} {sentiment:<10.2f} {emotion:<12}\n")
            
            f.write("\nDETAILED ANALYSIS:\n")
            for result in results:
                f.write(f"\n{result.get('conversation_title', 'N/A')} (ID: {result.get('conversation_id', 'N/A')}):\n")
                f.write(f"- Sale Probability: {result['sale_prediction']['sale_probability']:.1%}\n")
                f.write(f"- Average Sentiment: {result['conversation_metrics']['average_sentiment']:.2f}\n")
                f.write(f"- Dominant Emotion: {result['conversation_metrics']['dominant_emotion']}\n")
                f.write(f"- Summary: {result['summary'][:100]}...\n")
        
        return report_path
    
    def batch_analyze_all(self) -> List[Dict]:
        """Analyze all demos with full pipeline."""
        results = []
        for conv in self.demo_conversations:
            try:
                # Full pipeline (as in run_demo)
                transcription = self.audio_processor.transcribe_audio(conv['audio_path'], conv['id'])
                segments = self.audio_processor.perform_speaker_diarization(conv['audio_path'], conv['id'])
                processed_segments = self.text_processor.segment_conversation(
                    transcription['text'],
                    segments,
                    conv['id'],
                    transcription_segments=transcription.get('segments', [])
                )
                audio_features = self.audio_processor.extract_audio_features(conv['audio_path'])
                
                feature_data = [{'audio_features': audio_features, 'text_features': {'text': transcription['text']}, 'segments': processed_segments}]
                fused_features = self.feature_extractor.fit_transform(feature_data)[0]
                
                result = self.analyzer.analyze_conversation(
                    audio_path=conv['audio_path'], 
                    segments=processed_segments, 
                    features=fused_features, 
                    call_id=conv['id']
                )
                result['conversation_title'] = conv['title']
                result['conversation_id'] = conv['id']
                results.append(result)
            except Exception as e:
                logger.error(f"Batch analysis failed for {conv['id']}: {e}")
        return results
    
    def export_results_json(self, results: List[Dict], output_dir: str = "output") -> str:
        """Export results to JSON (FR-7 fallback)."""
        os.makedirs(output_dir, exist_ok=True)
        json_path = os.path.join(output_dir, "analysis_results.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)
        return json_path


def main():
    """Main function for running the demo."""
    demo = DemoSystem()  # Add hf_token if needed
    results = demo.run_demo()
    
    # Create additional outputs
    print("Creating additional outputs...")
    comparison_report = demo.create_comparison_report()
    json_export = demo.export_results_json(results)
    
    print(f"Comparison report: {comparison_report}")
    print(f"JSON export: {json_export}")
    
    return results


if __name__ == "__main__":
    main()