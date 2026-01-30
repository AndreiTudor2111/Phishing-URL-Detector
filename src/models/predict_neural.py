"""
Neural Network Prediction Interface for Phishing Detection
Loads and uses PyTorch models for URL classification
"""

import torch
import torch.nn as nn
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import sys

parent_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(parent_path))

from src.feature_extraction import extract_features
from src.utils.config import Config


# ======================== Neural Network Architectures ========================
# These must match the architectures used in training

class FeedForwardNN(nn.Module):
    """Simple feedforward neural network."""
    
    def __init__(self, input_dim, hidden_dims=[256, 128, 64], dropout=0.3):
        super(FeedForwardNN, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class LSTMNN(nn.Module):
    """LSTM-based neural network."""
    
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.3):
        super(LSTMNN, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = x.unsqueeze(1)
        lstm_out, _ = self.lstm(x)
        out = lstm_out[:, -1, :]
        out = self.fc_layers(out)
        return out


class GRUNN(nn.Module):
    """GRU-based neural network."""
    
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.3):
        super(GRUNN, self).__init__()
        
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = x.unsqueeze(1)
        gru_out, _ = self.gru(x)
        out = gru_out[:, -1, :]
        out = self.fc_layers(out)
        return out


class CNNNN(nn.Module):
    """CNN-based neural network."""
    
    def __init__(self, input_dim, dropout=0.3):
        super(CNNNN, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
            
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
        )
        
        self.flattened_size = self._get_conv_output_size(input_dim)
        
        self.fc_layers = nn.Sequential(
            nn.Linear(self.flattened_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def _get_conv_output_size(self, input_dim):
        x = torch.zeros(1, 1, input_dim)
        x = self.conv_layers(x)
        return x.view(1, -1).size(1)
    
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x


# ======================== Prediction Class ========================

class NeuralNetworkDetector:
    """Class for detecting phishing URLs using PyTorch neural networks."""
    
    def __init__(self, model_type='lstm'):
        """
        Initialize the neural network detector.
        
        Args:
            model_type (str): Type of model ('feedforward', 'lstm', 'gru', 'cnn')
        """
        self.model_type = model_type.lower()
        self.device = torch.device('cpu')  # Use CPU for inference
        self.model = None
        self.scaler = None
        self.input_dim = 60  # Number of features
        
        self._load_model()
    
    def _load_model(self):
        """Load the trained model and scaler."""
        try:
            # Load scaler
            scaler_path = Config.MODEL_PRODUCTION_DIR / "neural_network_scaler.pkl"
            if not scaler_path.exists():
                raise FileNotFoundError(f"Scaler not found at {scaler_path}")
            
            self.scaler = joblib.load(scaler_path)
            
            # Initialize model architecture
            if self.model_type == 'feedforward':
                self.model = FeedForwardNN(self.input_dim)
                model_path = Config.MODEL_PRODUCTION_DIR / "neural_network_feedforward.pt"
            elif self.model_type == 'lstm':
                self.model = LSTMNN(self.input_dim)
                model_path = Config.MODEL_PRODUCTION_DIR / "neural_network_lstm.pt"
            elif self.model_type == 'gru':
                self.model = GRUNN(self.input_dim)
                model_path = Config.MODEL_PRODUCTION_DIR / "neural_network_gru.pt"
            elif self.model_type == 'cnn':
                self.model = CNNNN(self.input_dim)
                model_path = Config.MODEL_PRODUCTION_DIR / "neural_network_cnn.pt"
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")
            
            # Load model weights
            if not model_path.exists():
                raise FileNotFoundError(f"Model not found at {model_path}")
            
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            
            print(f"✓ Loaded {self.model_type.upper()} neural network successfully")
        
        except Exception as e:
            raise Exception(f"Error loading neural network model: {str(e)}")
    
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
        
        if features is None:
            return "Error", 0.0
        
        # Convert to DataFrame
        input_df = pd.DataFrame([features])
        
        # Handle missing or infinite values
        input_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        input_df.fillna(0, inplace=True)
        
        # Scale features
        input_scaled = self.scaler.transform(input_df.values)
        
        # Convert to tensor
        input_tensor = torch.FloatTensor(input_scaled).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            output = self.model(input_tensor).squeeze()
            probability = output.item()
        
        result = "Phishing" if probability >= 0.5 else "Legitimate"
        
        return result, probability
    
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
