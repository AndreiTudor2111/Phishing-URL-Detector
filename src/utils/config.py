"""
Configuration management for the phishing detection system.
Uses relative paths to ensure portability across different systems.
"""

import os
from pathlib import Path


class Config:
    """Configuration class for managing file paths and settings."""
    
    # Base directories
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    DATA_DIR = BASE_DIR / "data"
    MODEL_DIR = BASE_DIR / "models"
    SRC_DIR = BASE_DIR / "src"
    
    # Data paths
    DATA_RAW_DIR = DATA_DIR / "raw"
    DATA_PROCESSED_DIR = DATA_DIR / "processed"
    
    # Datasets
    DATASET_DATNOU = DATA_RAW_DIR / "DataNou.csv"
    DATASET_MALICIOUS_PHISH = DATA_RAW_DIR / "malicious_phish.csv"
    DATASET_SMALL = DATA_RAW_DIR / "dataset.csv"
    BRAND_LIST = DATA_RAW_DIR / "allbrands.txt"
    
    # Model paths
    MODEL_PRODUCTION_DIR = MODEL_DIR / "production"
    MODEL_EXPERIMENTAL_DIR = MODEL_DIR / "experimental"
    
    # Production models
    MODEL_RANDOM_FOREST = MODEL_PRODUCTION_DIR / "best_random_forest.joblib"
    MODEL_XGBOOST = MODEL_PRODUCTION_DIR / "best_xgboost.joblib"
    MODEL_SCALER = MODEL_PRODUCTION_DIR / "scaler.pkl"
    
    # Model training parameters
    RANDOM_STATE = 42
    TEST_SIZE = 0.3
    CV_FOLDS = 5
    
    # Feature engineering parameters
    ENGLISH_CHAR_FREQUENCIES = {
        'a': 8.167, 'b': 1.492, 'c': 2.782, 'd': 4.253, 'e': 12.702,
        'f': 2.228, 'g': 2.015, 'h': 6.094, 'i': 6.966, 'j': 0.153,
        'k': 0.772, 'l': 4.025, 'm': 2.406, 'n': 6.749, 'o': 7.507,
        'p': 1.929, 'q': 0.095, 'r': 5.987, 's': 6.327, 't': 9.056,
        'u': 2.758, 'v': 0.978, 'w': 2.360, 'x': 0.150, 'y': 1.974,
        'z': 0.074
    }
    
    SUSPICIOUS_WORDS = [
        'confirm', 'account', 'secure', 'login', 'signin', 
        'submit', 'update', 'admin'
    ]
    
    PHISHING_HINTS = [
        'wp', 'login', 'includes', 'admin', 'content', 'site', 
        'images', 'js', 'alibaba', 'css', 'myaccount', 'dropbox', 
        'themes', 'plugins', 'signin', 'view'
    ]
    
    SUSPICIOUS_TLDS = [
        'zip', 'cricket', 'link', 'work', 'party', 'gq', 'kim', 
        'country', 'science', 'tk'
    ]
    
    @classmethod
    def ensure_directories(cls):
        """Create necessary directories if they don't exist."""
        directories = [
            cls.DATA_RAW_DIR,
            cls.DATA_PROCESSED_DIR,
            cls.MODEL_PRODUCTION_DIR,
            cls.MODEL_EXPERIMENTAL_DIR
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_brand_list(cls):
        """Load the brand list from file."""
        if cls.BRAND_LIST.exists():
            with open(cls.BRAND_LIST, 'r') as f:
                return [line.strip() for line in f.readlines()]
        return []
