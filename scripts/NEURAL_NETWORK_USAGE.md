# Neural Network Prediction Class

This module provides an interface for loading and using trained PyTorch neural networks
for phishing URL detection.

## Quick Start

```python
from scripts.neural_network_predict import NeuralNetworkDetector

# Initialize detector with specific model
detector = NeuralNetworkDetector(model_type='lstm')  # or 'gru', 'cnn', 'feedforward'

# Predict single URL
prediction, probability = detector.predict("http://suspicious-site.com")
print(f"Result: {prediction} (Confidence: {probability*100:.1f}%)")

# Batch prediction
urls = ["http://google.com", "http://phishing-site.tk"]
results = detector.predict_batch(urls)
```

## Model Types

- `feedforward` - Simple dense neural network
- `lstm` - LSTM-based sequential network
- `gru` - GRU-based sequential network  
- `cnn` - Convolutional neural network

## Requirements

- Trained model files in `models/production/neural_network_*.pt`
- Scaler file in `models/production/neural_network_scaler.pkl`
- PyTorch installed (`pip install torch`)
