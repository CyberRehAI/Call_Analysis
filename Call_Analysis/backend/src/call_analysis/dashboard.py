"""
Visualization dashboard for call analysis results, aligned with SRS1.
Implements sentiment curves, emotion distributions, sale probability gauges,
conversation flow, agent performance, and feature importance (FR-8).
Saves results to MongoDB (FR-7) and ensures PII security.
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
from datetime import datetime
from pymongo import MongoClient
import spacy
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load spaCy for PII masking
# Load spaCy for PII masking (optional)
try:
    nlp = spacy.load('en_core_web_sm')
    SPACY_AVAILABLE = True
    logger.info("spaCy model loaded successfully")
except OSError:
    nlp = None
    SPACY_AVAILABLE = False
    logger.warning("spaCy model 'en_core_web_sm' not found. PII masking will be limited.")


class Dashboard:
    """Interactive dashboard for call analysis visualization (FR-8)."""
    
    def __init__(self):
        """Initialize dashboard."""
        self.colors = {
            'positive': '#2E8B57',
            'negative': '#DC143C',
            'neutral': '#4682B4',
            'sale': '#32CD32',
            'no_sale': '#FF6347',
            'customer': '#FF6B6B',
            'agent': '#4ECDC4'
        }
    
    def mask_pii(self, text: str) -> str:
        """
        Mask PII in text (SRS1 security requirement).
        
        Args:
            text: Input text.
            
        Returns:
            Anonymized text.
        """
        if not text or not text.strip():
            return text

        # If spaCy model is not available, fall back to simple regex-based masking
        if not SPACY_AVAILABLE or nlp is None:
            import re
            # Mask phone numbers (common patterns)
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
            logger.error(f"Error in PII masking: {e}")
            return text
    
    def validate_results(self, results: Dict):
        """
        Validate analysis results structure.
        
        Args:
            results: Analysis results dictionary.
            
        Raises:
            ValueError: If required keys are missing.
        """
        required_keys = ['sentiment_analysis', 'emotion_analysis', 'sale_prediction', 'segments', 'conversation_metrics']
        for key in required_keys:
            if key not in results:
                raise ValueError(f"Missing required key: {key}")
    
    def create_sentiment_timeline(self, sentiment_data: List[Dict]) -> go.Figure:
        """
        Create sentiment timeline visualization (FR-8).
        
        Args:
            sentiment_data: List of sentiment analysis results.
            
        Returns:
            Plotly figure object.
        """
        try:
            if not sentiment_data:
                return go.Figure()
            
            times = [s['start_time'] for s in sentiment_data]
            scores = [s['score'] for s in sentiment_data]
            sentiments = [s['sentiment'] for s in sentiment_data]
            speakers = [s['speaker'] for s in sentiment_data]
            texts = [self.mask_pii(s['text'])[:50] + '...' if len(s['text']) > 50 else self.mask_pii(s['text']) for s in sentiment_data]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=times,
                y=scores,
                mode='lines+markers',
                name='Sentiment Score',
                line=dict(color='blue', width=3),
                marker=dict(size=8),
                hovertemplate='<b>Time:</b> %{x}s<br><b>Sentiment:</b> %{y:.2f}<br><b>Text:</b> %{text}<extra></extra>',
                text=texts
            ))
            
            fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
            
            for i, (time, score, sentiment, speaker) in enumerate(zip(times, scores, sentiments, speakers)):
                color = self.colors.get(sentiment.lower(), '#808080')
                fig.add_trace(go.Scatter(
                    x=[time],
                    y=[score],
                    mode='markers',
                    marker=dict(color=color, size=12, symbol='circle'),
                    name=f'{sentiment.title()} ({speaker})',
                    showlegend=False,
                    hovertemplate=f'<b>Speaker:</b> {speaker}<br><b>Sentiment:</b> {sentiment}<br><b>Score:</b> {score:.2f}<extra></extra>'
                ))
            
            fig.update_layout(
                title='Sentiment Timeline',
                xaxis_title='Time (seconds)',
                yaxis_title='Sentiment Score',
                hovermode='closest',
                height=400,
                showlegend=True
            )
            return fig
        except Exception as e:
            logger.error(f"Sentiment timeline failed: {e}")
            return go.Figure()
    
    def create_emotion_distribution(self, emotion_data: List[Dict]) -> go.Figure:
        """
        Create emotion distribution pie chart (FR-8).
        
        Args:
            emotion_data: List of emotion analysis results.
            
        Returns:
            Plotly figure object.
        """
        try:
            if not emotion_data:
                return go.Figure()
            
            emotions = [e['emotion'] for e in emotion_data]
            emotion_counts = pd.Series(emotions).value_counts()
            
            fig = go.Figure(data=[go.Pie(
                labels=emotion_counts.index,
                values=emotion_counts.values,
                hole=0.3,
                textinfo='label+percent',
                textfont_size=12
            )])
            
            fig.update_layout(
                title='Emotion Distribution',
                height=400,
                showlegend=True
            )
            return fig
        except Exception as e:
            logger.error(f"Emotion distribution failed: {e}")
            return go.Figure()
    
    def create_sale_probability_gauge(self, sale_prediction: Dict) -> go.Figure:
        """
        Create sale probability gauge chart (FR-8).
        
        Args:
            sale_prediction: Sale prediction results.
            
        Returns:
            Plotly figure object.
        """
        try:
            probability = sale_prediction.get('sale_probability', 0)
            confidence = sale_prediction.get('confidence', 0)
            
            color = 'green' if probability > 0.7 else 'orange' if probability > 0.4 else 'red'
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=probability * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Sale Probability (%)"},
                delta={'reference': 50},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': color},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgray"},
                        {'range': [30, 70], 'color': "gray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            
            fig.update_layout(
                height=400,
                title=f'Sale Probability: {probability:.1%} (Confidence: {confidence:.1%})'
            )
            return fig
        except Exception as e:
            logger.error(f"Sale probability gauge failed: {e}")
            return go.Figure()
    
    def create_conversation_flow(self, segments: List[Dict]) -> go.Figure:
        """
        Create conversation flow visualization (FR-8).
        
        Args:
            segments: List of conversation segments.
            
        Returns:
            Plotly figure object.
        """
        try:
            if not segments:
                return go.Figure()
            
            times = [(s['start_time'] + s['end_time']) / 2 for s in segments]
            speakers = [s['speaker'] for s in segments]
            texts = [self.mask_pii(s['text'])[:30] + '...' if len(s['text']) > 30 else self.mask_pii(s['text']) for s in segments]
            
            fig = go.Figure()
            for i, (time, speaker, text) in enumerate(zip(times, speakers, texts)):
                color = self.colors.get(speaker.lower(), '#808080')
                fig.add_trace(go.Bar(
                    x=[time],
                    y=[1],
                    name=speaker,
                    marker_color=color,
                    text=text,
                    textposition='inside',
                    hovertemplate=f'<b>{speaker}</b><br>{text}<extra></extra>',
                    showlegend=False
                ))
            
            fig.update_layout(
                title='Conversation Flow',
                xaxis_title='Time (seconds)',
                yaxis_title='Speaker',
                height=200,
                barmode='stack'
            )
            return fig
        except Exception as e:
            logger.error(f"Conversation flow failed: {e}")
            return go.Figure()
    
    def create_agent_performance_chart(self, agent_data: List[Dict]) -> go.Figure:
        """
        Create agent performance comparison chart (FR-8).
        
        Args:
            agent_data: List of agent performance data.
            
        Returns:
            Plotly figure object.
        """
        try:
            if not agent_data:
                return go.Figure()
            
            agents = [a['agent_id'] for a in agent_data]
            sale_rates = [a['sale_rate'] for a in agent_data]
            avg_sentiment = [a['avg_sentiment'] for a in agent_data]
            
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Sale Rate by Agent', 'Average Sentiment by Agent')
            )
            
            fig.add_trace(
                go.Bar(x=agents, y=sale_rates, name='Sale Rate', marker_color='lightblue'),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Bar(x=agents, y=avg_sentiment, name='Avg Sentiment', marker_color='lightgreen'),
                row=1, col=2
            )
            
            fig.update_layout(
                title='Agent Performance Comparison',
                height=400,
                showlegend=False
            )
            return fig
        except Exception as e:
            logger.error(f"Agent performance chart failed: {e}")
            return go.Figure()
    
    def create_feature_importance_chart(self, feature_importance: Dict) -> go.Figure:
        """
        Create feature importance visualization (FR-8).
        
        Args:
            feature_importance: Feature importance data.
            
        Returns:
            Plotly figure object.
        """
        try:
            if not feature_importance or not feature_importance.get('importance'):
                return go.Figure()
            
            importance = feature_importance['importance']
            feature_names = feature_importance.get('feature_names', [f'Feature_{i}' for i in range(len(importance))])
            
            sorted_indices = np.argsort(importance)[::-1][:10]
            sorted_importance = [importance[i] for i in sorted_indices]
            sorted_names = [feature_names[i] for i in sorted_indices]
            
            fig = go.Figure(data=[
                go.Bar(x=sorted_importance, y=sorted_names, orientation='h')
            ])
            
            fig.update_layout(
                title='Top 10 Most Important Features',
                xaxis_title='Importance Score',
                yaxis_title='Features',
                height=500
            )
            return fig
        except Exception as e:
            logger.error(f"Feature importance chart failed: {e}")
            return go.Figure()
    
    def create_dashboard_html(self, analysis_results: Dict, call_id: str = None, output_path: str = None) -> str:
        """
        Create complete dashboard HTML string (FR-8).
        
        Args:
            analysis_results: Complete analysis results.
            call_id: Unique call identifier.
            output_path: Path to save HTML file (optional).
            
        Returns:
            HTML string.
        """
        try:
            self.validate_results(analysis_results)
            if call_id is None:
                call_id = analysis_results.get('conversation_id', 'N/A')
            
            sentiment_data = analysis_results.get('sentiment_analysis', [])
            emotion_data = analysis_results.get('emotion_analysis', [])
            sale_prediction = analysis_results.get('sale_prediction', {})
            segments = analysis_results.get('segments', [])
            conversation_metrics = analysis_results.get('conversation_metrics', {})
            feature_importance = analysis_results.get('sale_prediction', {}).get('feature_importance', {})
            
            sentiment_fig = self.create_sentiment_timeline(sentiment_data)
            emotion_fig = self.create_emotion_distribution(emotion_data)
            sale_fig = self.create_sale_probability_gauge(sale_prediction)
            flow_fig = self.create_conversation_flow(segments)
            feature_fig = self.create_feature_importance_chart({'importance': feature_importance or []})
            
            html_template = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Call Analysis Dashboard</title>
                <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
                <style>
                    body {{
                        font-family: Arial, sans-serif;
                        margin: 20px;
                        background-color: #f5f5f5;
                    }}
                    .header {{
                        background-color: #2c3e50;
                        color: white;
                        padding: 20px;
                        border-radius: 10px;
                        margin-bottom: 20px;
                    }}
                    .metrics {{
                        display: grid;
                        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                        gap: 20px;
                        margin-bottom: 20px;
                    }}
                    .metric-card {{
                        background-color: white;
                        padding: 20px;
                        border-radius: 10px;
                        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                        text-align: center;
                    }}
                    .metric-value {{
                        font-size: 2em;
                        font-weight: bold;
                        color: #2c3e50;
                    }}
                    .metric-label {{
                        color: #7f8c8d;
                        margin-top: 5px;
                    }}
                    .chart-container {{
                        background-color: white;
                        padding: 20px;
                        border-radius: 10px;
                        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                        margin-bottom: 20px;
                    }}
                    .summary {{
                        background-color: white;
                        padding: 20px;
                        border-radius: 10px;
                        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                    }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>Call Analysis Dashboard</h1>
                    <p>Conversation ID: {call_id} | 
                       Duration: {analysis_results.get('duration', 0):.1f}s | 
                       Segments: {len(segments)}</p>
                </div>
                
                <div class="metrics">
                    <div class="metric-card">
                        <div class="metric-value">{sale_prediction.get('sale_probability', 0):.1%}</div>
                        <div class="metric-label">Sale Probability</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{conversation_metrics.get('average_sentiment', 0):.2f}</div>
                        <div class="metric-label">Avg Sentiment</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{conversation_metrics.get('sentiment_trend', 0):.3f}</div>
                        <div class="metric-label">Sentiment Trend</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{conversation_metrics.get('dominant_emotion', 'N/A')}</div>
                        <div class="metric-label">Dominant Emotion</div>
                    </div>
                </div>
                
                <div class="chart-container">
                    <h3>Sale Probability</h3>
                    <div id="sale-chart"></div>
                </div>
                
                <div class="chart-container">
                    <h3>Sentiment Timeline</h3>
                    <div id="sentiment-chart"></div>
                </div>
                
                <div class="chart-container">
                    <h3>Emotion Distribution</h3>
                    <div id="emotion-chart"></div>
                </div>
                
                <div class="chart-container">
                    <h3>Conversation Flow</h3>
                    <div id="flow-chart"></div>
                </div>
                
                <div class="chart-container">
                    <h3>Feature Importance</h3>
                    <div id="feature-chart"></div>
                </div>
                
                <div class="summary">
                    <h3>Analysis Summary</h3>
                    <pre>{self.mask_pii(analysis_results.get('summary', 'No summary available'))}</pre>
                </div>
                
                <script>
                    Plotly.newPlot('sale-chart', {sale_fig.to_json()});
                    Plotly.newPlot('sentiment-chart', {sentiment_fig.to_json()});
                    Plotly.newPlot('emotion-chart', {emotion_fig.to_json()});
                    Plotly.newPlot('flow-chart', {flow_fig.to_json()});
                    Plotly.newPlot('feature-chart', {feature_fig.to_json()});
                </script>
            </body>
            </html>
            """
            
            if output_path:
                try:
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    with open(output_path, 'w', encoding='utf-8') as f:
                        f.write(html_template)
                    logger.info(f"Dashboard saved to {output_path}")
                except Exception as e:
                    logger.error(f"Failed to save dashboard: {e}")
            
            self.save_dashboard(analysis_results, call_id)  # FR-7
            return html_template
        except Exception as e:
            logger.error(f"Dashboard HTML creation failed: {e}")
            raise
    
    def generate_dashboard(self, analysis_results: Dict, call_id: str = None, output_path: str = None) -> str:
        """
        Generate dashboard HTML and save to disk (if output_path provided).
        """
        html = self.create_dashboard_html(analysis_results, call_id=call_id)
        if output_path is not None:
            try:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(html)
                logger.info(f"Dashboard saved to {output_path}")
            except Exception as e:
                logger.error(f"Failed to write dashboard HTML: {e}")
        return html
    
    def create_batch_dashboards(self, results_list: List[Dict], output_dir: str) -> List[str]:
        """
        Create dashboards for multiple conversations (performance optimization).
        
        Args:
            results_list: List of analysis results.
            output_dir: Directory to save HTML files.
            
        Returns:
            List of HTML file paths.
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            file_paths = []
            for i, results in enumerate(results_list):
                call_id = results.get('conversation_id', f'conv_{i}')
                output_path = os.path.join(output_dir, f'dashboard_{call_id}.html')
                self.create_dashboard_html(results, call_id, output_path)
                file_paths.append(output_path)
            return file_paths
        except Exception as e:
            logger.error(f"Batch dashboard creation failed: {e}")
            raise
    
    def save_dashboard(self, results: Dict, call_id: str):
        """
        Save dashboard metadata to MongoDB (FR-7).
        
        Args:
            results: Analysis results.
            call_id: Unique call identifier.
        """
        try:
            client = MongoClient('mongodb://localhost:27017/')
            db = client['call_center_db']
            collection = db['dashboards']
            collection.insert_one({
                'call_id': call_id,
                'results': results,
                'timestamp': datetime.now()
            })
            logger.info(f"Dashboard metadata saved for call_id: {call_id}")
        except Exception as e:
            logger.error(f"Error saving dashboard metadata: {e}")
    
    def create_summary_report(self, analysis_results: Dict) -> str:
        """
        Create text summary report (FR-8).
        
        Args:
            analysis_results: Complete analysis results.
            
        Returns:
            Formatted summary report.
        """
        try:
            self.validate_results(analysis_results)
            sale_prediction = analysis_results.get('sale_prediction', {})
            conversation_metrics = analysis_results.get('conversation_metrics', {})
            
            report = f"""
CALL ANALYSIS REPORT
===================

Conversation Overview:
- ID: {analysis_results.get('conversation_id', 'N/A')}
- Duration: {analysis_results.get('duration', 0):.1f} seconds
- Segments: {len(analysis_results.get('segments', []))}

Sale Prediction:
- Probability: {sale_prediction.get('sale_probability', 0):.1%}
- Prediction: {sale_prediction.get('prediction', 'N/A').upper()}
- Confidence: {sale_prediction.get('confidence', 0):.1%}

Sentiment Analysis:
- Average Sentiment: {conversation_metrics.get('average_sentiment', 0):.2f}
- Sentiment Trend: {conversation_metrics.get('sentiment_trend', 0):.3f}
- Customer Sentiment: {conversation_metrics.get('customer_sentiment', 0):.2f}
- Agent Sentiment: {conversation_metrics.get('agent_sentiment', 0):.2f}

Emotion Analysis:
- Dominant Emotion: {conversation_metrics.get('dominant_emotion', 'N/A')}
- Emotion Distribution: {conversation_metrics.get('emotion_distribution', {})}

Conversation Flow:
- Sentiment Volatility: {conversation_metrics.get('sentiment_volatility', 0):.3f}
- Flow Score: {conversation_metrics.get('conversation_flow_score', 0):.2f}

Recommendations:
{self.mask_pii(analysis_results.get('summary', 'No recommendations available'))}

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            """
            return report.strip()
        except Exception as e:
            logger.error(f"Summary report failed: {e}")
            raise