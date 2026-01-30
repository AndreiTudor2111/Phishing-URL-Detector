"""
Prediction interface for phishing URL detection.
Loads trained models and makes predictions on URLs.
"""

import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from ..feature_extraction import extract_features
from ..utils.config import Config


class PhishingDetector:
    """Class for detecting phishing URLs using trained models."""
    
    def __init__(self, model_type='xgboost'):
        """
        Initialize the phishing detector.
        
        Args:
            model_type (str): Type of model to use ('xgboost' or 'random_forest')
        """
        self.model_type = model_type
        self.model = None
        self.scaler = None
        self._load_models()
    
    def _load_models(self):
        """Load the trained model and scaler."""
        try:
            if self.model_type.lower() == 'xgboost':
                model_path = Config.MODEL_XGBOOST
            elif self.model_type.lower() == 'random_forest':
                model_path = Config.MODEL_RANDOM_FOREST
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")
            
            if not model_path.exists():
                raise FileNotFoundError(f"Model not found at {model_path}")
            
            self.model = joblib.load(model_path)
            
            # Load scaler
            if Config.MODEL_SCALER.exists():
                self.scaler = joblib.load(Config.MODEL_SCALER)
            else:
                print("Warning: Scaler not found. Predictions may be inaccurate.")
                self.scaler = None
            
            print(f"✓ Loaded {self.model_type} model successfully")
            
        except Exception as e:
            raise Exception(f"Error loading models: {str(e)}")
    
    def predict(self, url):
        """
        Predict whether a single URL is phishing or legitimate.
        
        Args:
            url (str): URL to classify
            
        Returns:
            tuple: (prediction, probability) where prediction is 'Phishing' or 'Legitimate'
        """
        # Extract features
        features = extract_features(url)
        
        # Create DataFrame
        input_df = pd.DataFrame([features])
        
        # Handle missing or infinite values
        input_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        input_df.fillna(0, inplace=True)
        
        # Scale features
        if self.scaler is not None:
            input_scaled = self.scaler.transform(input_df)
        else:
            input_scaled = input_df.values
        
        # Make prediction
        prediction = self.model.predict(input_scaled)[0]
        
        # Get probability if available
        try:
            probability = self.model.predict_proba(input_scaled)[0]
            prob_phishing = probability[1]
        except:
            prob_phishing = 1.0 if prediction == 1 else 0.0
        
        result = "Phishing" if prediction == 1 else "Legitimate"
        return result, prob_phishing
    
    def predict_batch(self, urls):
        """
        Predict multiple URLs.
        
        Args:
            urls (list): List of URLs to classify
            
        Returns:
            list: List of tuples (url, prediction, probability)
        """
        results = []
        for url in urls:
            prediction, probability = self.predict(url)
            results.append((url, prediction, probability))
        return results
    
    def predict_from_file(self, file_path, url_column='url'):
        """
        Predict URLs from a CSV file.
        
        Args:
            file_path (str or Path): Path to CSV file
            url_column (str): Name of column containing URLs
            
        Returns:
            pd.DataFrame: DataFrame with predictions
        """
        df = pd.read_csv(file_path)
        
        if url_column not in df.columns:
            raise ValueError(f"Column '{url_column}' not found in file")
        
        predictions = []
        probabilities = []
        
        for url in df[url_column]:
            pred, prob = self.predict(url)
            predictions.append(pred)
            probabilities.append(prob)
        
        df['prediction'] = predictions
        df['phishing_probability'] = probabilities
        
        return df


def predict_url(url, model_type='xgboost'):
    """
    Convenience function to predict a single URL.
    
    Args:
        url (str): URL to classify
        model_type (str): Model type to use
        
    Returns:
        tuple: (prediction, probability)
    """
    detector = PhishingDetector(model_type=model_type)
    return detector.predict(url)
