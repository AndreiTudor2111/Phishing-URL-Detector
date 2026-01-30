# 🔐 Phishing URL Detection System

A machine learning-based system for detecting phishing URLs using advanced feature engineering and ensemble learning methods.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![ML](https://img.shields.io/badge/ML-XGBoost%20%7C%20Random%20Forest-green)
![DL](https://img.shields.io/badge/Deep%20Learning-PyTorch-red)
![Models](https://img.shields.io/badge/Models-6%20Available-orange)
![Accuracy](https://img.shields.io/badge/Accuracy-93%25%2B-brightgreen)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 📋 Overview

This project implements a comprehensive phishing URL detection system that analyzes **60 distinct features** from URLs to classify them as either phishing or legitimate. The system achieves **>93% accuracy** using both traditional machine learning (XGBoost, Random Forest) and deep learning models (LSTM, GRU, CNN, FeedForward).

### Key Features

- ✅ **60 Feature Analysis**: URL structure, statistical patterns, and suspicious indicators
- 🤖 **6 Model Options**: Traditional ML (XGBoost, Random Forest) + Neural Networks (LSTM, GRU, CNN, FeedForward)
- 🎯 **High Accuracy**: >93% accuracy across all models
- 🌐 **Interactive Web Interface**: User-friendly Streamlit application with model selection
- 📊 **Batch Processing**: Analyze multiple URLs from CSV files
- 🔧 **Modular Design**: Clean, maintainable code architecture
- 🧠 **Deep Learning**: PyTorch-based neural networks trained on 662K+ URLs

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Proiect
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

   **Note for PyTorch**: If you plan to use deep learning models or train neural networks, install PyTorch separately based on your system:
   ```bash
   # CPU only
   pip install torch torchvision
   
   # GPU (CUDA) - visit https://pytorch.org for specific command
   ```

### Usage

#### Web Interface (Streamlit)

Launch the interactive web application:

```bash
# From the project root directory
cd c:\Users\ostac\OneDrive\Desktop\Personal_projects\Proiect
streamlit run app/streamlit_app.py
```

Then open your browser to `http://localhost:8501`

**Note**: If you encounter `ModuleNotFoundError`, make sure you're running the command from the project root directory.

#### Python API

**Traditional ML Models:**

```python
from src.models.predict import PhishingDetector

# Initialize detector with XGBoost or Random Forest
detector = PhishingDetector(model_type='xgboost')  # or 'random_forest'

# Predict single URL
url = "http://example-suspicious-site.com/login"
prediction, probability = detector.predict(url)

print(f"Prediction: {prediction}")
print(f"Confidence: {probability*100:.1f}%")

# Batch prediction
urls = ["http://google.com", "http://suspicious-site.tk"]
results = detector.predict_batch(urls)
```

**Neural Network Models:**

```python
from src.models.predict_neural import NeuralNetworkDetector

# Initialize with specific neural network architecture
detector = NeuralNetworkDetector(model_type='lstm')  # 'lstm', 'gru', 'cnn', 'feedforward'

# Predict single URL
prediction, probability = detector.predict("http://example.com")

print(f"Prediction: {prediction}")
print(f"Confidence: {probability*100:.1f}%")
```

## 📁 Project Structure

```
Proiect/
├── README.md                          # This file
├── requirements.txt                    # Python dependencies
│
├── docs/                              # Documentation
│   ├── project_overview.md            # Technical overview
│   ├── model_performance.md           # Model metrics
│   ├── feature_engineering.md         # Feature documentation
│   └── research_papers/               # Research articles (8 PDFs)
│
├── data/                              # Datasets
│   ├── raw/                           # Original datasets
│   │   ├── DataNou.csv               # Primary dataset (43MB)
│   │   ├── malicious_phish.csv       # Alternative dataset (45MB)
│   │   ├── dataset.csv               # Smaller dataset (3.6MB)
│   │   └── allbrands.txt             # Brand list for detection
│   └── processed/                     # Processed data
│
├── models/                            # Trained models
│   ├── production/                    # Production-ready models
│   │   ├── best_random_forest.joblib # Fine-tuned RF model
│   │   ├── best_xgboost.joblib       # Fine-tuned XGBoost model
│   │   └── scaler.pkl                # Feature scaler
│   └── experimental/                  # Experimental models
│
├── src/                               # Source code
│   ├── feature_extraction/
│   │   ├── __init__.py
│   │   └── feature_extractor.py      # Feature engineering
│   ├── models/
│   │   ├── __init__.py
│   │   └── predict.py                # Prediction interface
│   └── utils/
│       ├── __init__.py
│       └── config.py                 # Configuration management
│
├── app/                              # Streamlit application
│   └── streamlit_app.py              # Web interface
│
└── scripts/                          # Utility scripts
    └── (training scripts)
```

## 🎯 Features Analyzed

The system extracts and analyzes **60 features** from each URL:

### Structural Features
- URL length, hostname length, path length
- Number of dots, slashes, hyphens
- Subdomain count
- Query parameter count

### Character-based Features
- Digit ratio and count
- Special character frequency
- Uppercase letter ratio
- Punctuation patterns

### Statistical Features
- URL entropy
- Character frequency distribution
- KL divergence from English
- Kolmogorov-Smirnov statistic
- Euclidean distance from normal patterns

### Pattern Recognition
- IP address detection
- Suspicious keywords (`login`, `signin`, `confirm`, etc.)
- URL shortening services
- Phishing hints (`admin`, `update`, `secure`, etc.)
- File extension analysis
- Suspicious TLDs (`.tk`, `.zip`, `.cricket`, etc.)

See [docs/feature_engineering.md](docs/feature_engineering.md) for complete details.

## 📊 Model Performance

### Traditional Machine Learning Models

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **Fine-Tuned XGBoost** | 93.3% | 93.5% | 93.1% | 93.3% |
| **Fine-Tuned Random Forest** | 93.5% | 93.8% | 93.2% | 93.5% |

Both models were trained on a dataset of **430,000+ URLs** and fine-tuned using GridSearchCV with 5-fold cross-validation.

### Deep Learning Models (PyTorch)

**Training Dataset:** 662,604 URLs (223,071 phishing + 439,533 legitimate)

| Model | Size | Speed | Architecture | Best For |
|-------|------|-------|--------------|----------|
| **FeedForward NN** | 243 KB | ⚡ Fastest | Dense layers | Quick inference |
| **LSTM Network** | 954 KB | 🔄 Fast | Sequential | Pattern learning |
| **GRU Network** | 725 KB | 🔄 Fast | Sequential | Efficient RNN |
| **CNN Network** | 1.12 MB | 🔄 Medium | Convolutional | Feature detection |

All neural networks are trained with:
- **60 features** per URL
- **Batch size:** 256
- **Optimizer:** Adam with learning rate scheduling
- **Early stopping** to prevent overfitting
- **GPU acceleration** supported (CUDA)

### Model Comparison & Recommendations

| Use Case | Recommended Model | Why |
|----------|------------------|-----|
| **Best Accuracy** | Random Forest | 93.5% accuracy, robust |
| **Fastest Inference** | FeedForward NN | 243 KB, lightweight |
| **Production API** | XGBoost | 93.3% accuracy, 4.8 MB, very fast |
| **Sequential Analysis** | LSTM/GRU | Deep learning, pattern recognition |
| **Research/Experiments** | Any Neural Network | Flexible, customizable |

### Training Your Own Models

To train deep learning models:

```bash
# Train all neural network models
python scripts/train_neural_network.py
```

This script trains 4 different neural network architectures:
- **LSTM** - Best for sequential pattern learning
- **GRU** - Efficient alternative to LSTM
- **CNN** - Excellent for pattern detection
- **FeedForward** - Fast and simple baseline

See [scripts/README.md](scripts/README.md) for detailed training documentation.

See [docs/model_performance.md](docs/model_performance.md) for detailed metrics and confusion matrices.

## 🔬 Research Foundation

This project is based on research from 8 academic papers covering:
- Recurrent Neural Networks for URL classification
- Convolutional Gated-Recurrent-Unit Neural Networks
- Statistical feature engineering for malicious URL detection
- Entropy-based phishing detection methods

All research papers are available in [docs/research_papers/](docs/research_papers/).

## 🛠️ Technical Stack

- **Language**: Python 3.8+
- **ML Libraries**: scikit-learn, XGBoost
- **Deep Learning**: PyTorch (optional, for neural networks)
- **Feature Engineering**: NumPy, SciPy, Pandas
- **Web Interface**: Streamlit
- **Model Serialization**: joblib, PyTorch (.pt files)

## 📝 Development

### Training New Models

```python
# (Training scripts coming soon)
```

### Adding New Features

1. Edit `src/feature_extraction/feature_extractor.py`
2. Add feature extraction logic to `extract_features()` function
3. Retrain models with new features

### Configuration

All paths and constants are managed in `src/utils/config.py`:
- Dataset locations
- Model paths
- Suspicious word lists
- Character frequency distributions

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## ⚠️ Important Notes

- This system analyzes URL characteristics only, not page content
- No detection system is 100% accurate
- Always exercise caution with suspicious links
- Never enter credentials on unverified websites

## 📄 License

This project is licensed under the MIT License.

## 👤 Author

**Ostache Andrei Tudor**
- Master's Program: Information Management and Protection 
- Institution: National University of Science and Technology Politehnica Bucharest

## 🙏 Acknowledgments

- Research papers from academic journals and conferences
- Open-source ML libraries and tools
- Phishing URL datasets from the cybersecurity community

---

**⭐ If you find this project helpful, please consider giving it a star!**
