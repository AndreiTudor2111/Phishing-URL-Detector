# Training Scripts

This directory contains scripts for training models for phishing URL detection.

## Available Scripts

### 1. train_neural_network.py

**Deep Learning Training Script** - Train neural network models using PyTorch

#### Models Included:
- **FeedForward Neural Network** - Simple dense network with BatchNorm and Dropout
- **LSTM Network** - Long Short-Term Memory for sequential pattern learning
- **GRU Network** - Gated Recurrent Unit for efficient sequence processing
- **CNN Network** - Convolutional Neural Network for pattern detection

#### Features:
- ✅ Automatic feature extraction from raw URLs
- ✅ Data preprocessing and scaling
- ✅ Training with early stopping
- ✅ Learning rate scheduling
- ✅ Model checkpointing (saves best model)
- ✅ Training history visualization
- ✅ Confusion matrix plots
- ✅ GPU support (CUDA)

#### Usage:

```bash
# Install PyTorch first (if not already installed)
pip install torch torchvision

# Run the training script
python scripts/train_neural_network.py
```

#### Configuration:

Edit the script to modify:
```python
BATCH_SIZE = 256        # Batch size for training
EPOCHS = 50             # Maximum epochs
LEARNING_RATE = 0.001   # Initial learning rate
```

#### Output:

Models will be saved to `models/production/`:
- `neural_network_feedforward.pt` - FeedForward model weights
- `neural_network_lstm.pt` - LSTM model weights
- `neural_network_gru.pt` - GRU model weights
- `neural_network_cnn.pt` - CNN model weights
- `neural_network_scaler.pkl` - Scaler for preprocessing

Visualizations:
- `training_history_*.png` - Training/validation loss curves
- `confusion_matrix_*.png` - Confusion matrices

#### Expected Performance:

| Model | Expected Accuracy |
|-------|-------------------|
| FeedForward | 92-94% |
| LSTM | 93-95% |
| GRU | 93-95% |
| CNN | 91-93% |

#### GPU Acceleration:

The script automatically detects and uses GPU if available:
- **CPU Training**: ~10-15 minutes per model
- **GPU Training**: ~2-3 minutes per model

#### Architecture Details:

**FeedForward:**
- Input → 256 → 128 → 64 → Output
- BatchNorm + ReLU + Dropout(0.3)

**LSTM:**
- 2-layer LSTM with hidden_dim=128
- Fully connected layers: 128 → 64 → 1

**GRU:**
- 2-layer GRU with hidden_dim=128
- Fully connected layers: 128 → 64 → 1

**CNN:**
- Conv1D layers: 1 → 64 → 128
- MaxPooling after each conv
- Fully connected: Flattened → 128 → 64 → 1

## Future Training Scripts

### Planned:
- [ ] `train_ensemble.py` - Ensemble of XGBoost + Neural Networks
- [ ] `train_transformer.py` - Transformer-based URL encoder
- [ ] `retrain_models.py` - Retrain existing models with new data
- [ ] `hyperparameter_tuning.py` - Automated hyperparameter search

## Training Tips

### Data Preparation
1. Make sure datasets are in `data/raw/`
2. Verify data quality (no corrupted URLs)
3. Check class balance (should be ~50/50)

### Model Selection
- **Quick Testing**: Use FeedForward network
- **Best Performance**: Try all models and compare
- **Production Deployment**: Select based on accuracy vs speed tradeoff

### Hyperparameter Tuning
- Increase epochs if loss is still decreasing
- Reduce learning rate if loss oscillates
- Increase batch size for faster training (if GPU memory allows)
- Add more dropout if overfitting occurs

### Monitoring Training
- Watch for overfitting (train loss << val loss)
- Early stopping prevents wasting time
- Learning rate scheduler helps convergence

## Troubleshooting

### Out of Memory Error
```bash
# Reduce batch size in script
BATCH_SIZE = 128  # or even 64
```

### Slow Training
```bash
# Check if GPU is being used
print(torch.cuda.is_available())

# Or reduce dataset size for testing
# Edit script to use smaller sample
```

### Poor Performance
- Check if features are extracted correctly
- Verify data preprocessing (scaling)
- Try different architectures
- Increase epochs
- Tune hyperparameters

## Contributing

To add new training scripts:
1. Create script in this directory
2. Follow the same structure (imports, config, main)
3. Save models to `models/production/` or `models/experimental/`
4. Update this README with documentation

---

*Last Updated: January 2026*
