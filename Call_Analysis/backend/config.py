"""
Configuration settings for Call Analysis System
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Base configuration class"""
    
    # Hugging Face Configuration
    HF_TOKEN = os.getenv('HF_TOKEN', '')
    
    # Database Configuration
    MONGODB_URI = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/')
    MONGODB_DATABASE = os.getenv('MONGODB_DATABASE', 'call_center_db')
    
    # PostgreSQL Configuration
    POSTGRES_HOST = os.getenv('POSTGRES_HOST', 'localhost')
    POSTGRES_PORT = int(os.getenv('POSTGRES_PORT', 5432))
    POSTGRES_DB = os.getenv('POSTGRES_DB', 'call_analysis')
    POSTGRES_USER = os.getenv('POSTGRES_USER', '')
    POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD', '')
    
    # Flask Configuration
    SECRET_KEY = os.getenv('FLASK_SECRET_KEY', 'call-analysis-demo-2025')
    DEBUG = os.getenv('FLASK_DEBUG', 'True').lower() == 'true'
    HOST = os.getenv('FLASK_HOST', '0.0.0.0')
    PORT = int(os.getenv('FLASK_PORT', 5000))
    
    # Model Configuration
    WHISPER_MODEL_SIZE = os.getenv('WHISPER_MODEL_SIZE', 'base')
    BERT_MODEL_NAME = os.getenv('BERT_MODEL_NAME', 'distilbert-base-uncased')
    SENTIMENT_MODEL = os.getenv('SENTIMENT_MODEL', 'distilbert')  # Options: 'distilbert' or 'finbert'
    PYANNOTE_AUDIO_MODEL = os.getenv('PYANNOTE_AUDIO_MODEL', 'pyannote/speaker-diarization')
    
    # File Upload Configuration
    UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', 'uploads')
    # Handle MAX_CONTENT_LENGTH with proper parsing
    max_content_env = os.getenv('MAX_CONTENT_LENGTH', str(100 * 1024 * 1024))
    if isinstance(max_content_env, str) and max_content_env.endswith('MB'):
        MAX_CONTENT_LENGTH = int(max_content_env.replace('MB', '')) * 1024 * 1024
    else:
        MAX_CONTENT_LENGTH = int(max_content_env)
    ALLOWED_EXTENSIONS = set(os.getenv('ALLOWED_EXTENSIONS', 'wav,mp3,m4a,flac').split(','))
    
    # Logging Configuration
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE = os.getenv('LOG_FILE', 'logs/call_analysis.log')
    
    # Security Configuration
    PII_MASKING_ENABLED = os.getenv('PII_MASKING_ENABLED', 'True').lower() == 'true'
    DATA_RETENTION_DAYS = int(os.getenv('DATA_RETENTION_DAYS', 30))
    
    # Performance Configuration
    BATCH_SIZE = int(os.getenv('BATCH_SIZE', 10))
    MAX_WORKERS = int(os.getenv('MAX_WORKERS', 4))
    CACHE_TTL = int(os.getenv('CACHE_TTL', 3600))
    
    # Demo Configuration
    DEMO_MODE = os.getenv('DEMO_MODE', 'True').lower() == 'true'
    USE_SIMULATED_AUDIO = os.getenv('USE_SIMULATED_AUDIO', 'True').lower() == 'true'
    
    @staticmethod
    def init_app(app):
        """Initialize application with configuration"""
        app.config['SECRET_KEY'] = Config.SECRET_KEY
        app.config['UPLOAD_FOLDER'] = Config.UPLOAD_FOLDER
        app.config['MAX_CONTENT_LENGTH'] = Config.MAX_CONTENT_LENGTH
        
        # Create necessary directories
        os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(os.path.dirname(Config.LOG_FILE), exist_ok=True)
        os.makedirs('output', exist_ok=True)
        os.makedirs('templates', exist_ok=True)

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    DEMO_MODE = True
    USE_SIMULATED_AUDIO = True

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    DEMO_MODE = False
    USE_SIMULATED_AUDIO = False
    PII_MASKING_ENABLED = True

class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    DEMO_MODE = True
    USE_SIMULATED_AUDIO = True
    MONGODB_DATABASE = 'test_call_center_db'

# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}
