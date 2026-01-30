"""
Deep Learning Training Script for Phishing URL Detection
Uses PyTorch to train neural network models (LSTM, GRU, CNN)

Author: Ostache Andrei Tudor
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path
parent_path = Path(__file__).parent.parent
sys.path.insert(0, str(parent_path))

from src.feature_extraction import extract_features
from src.utils.config import Config


# ======================== Dataset Class ========================
class URLDataset(Dataset):
    """PyTorch Dataset for URL features."""
    
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# ======================== Neural Network Models ========================

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
    """LSTM-based neural network for sequential URL character analysis."""
    
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
        # Reshape for LSTM: (batch, seq_len, features)
        x = x.unsqueeze(1)  # Add sequence dimension
        lstm_out, _ = self.lstm(x)
        # Take the last output
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
    """CNN-based neural network for pattern detection."""
    
    def __init__(self, input_dim, dropout=0.3):
        super(CNNNN, self).__init__()
        
        self.conv_layers = nn.Sequential(
            # Reshape will happen in forward
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
            
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
        )
        
        # Calculate flattened size
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
        """Calculate the output size after convolutions."""
        x = torch.zeros(1, 1, input_dim)
        x = self.conv_layers(x)
        return x.view(1, -1).size(1)
    
    def forward(self, x):
        # Reshape for Conv1d: (batch, channels, features)
        x = x.unsqueeze(1)  # Add channel dimension
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc_layers(x)
        return x


# ======================== Training Functions ========================

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    for features, labels in dataloader:
        features, labels = features.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(features).squeeze()
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    """Evaluate the model."""
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for features, labels in dataloader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features).squeeze()
            preds = (outputs >= 0.5).float()
            
            predictions.extend(preds.cpu().numpy())
            actuals.extend(labels.cpu().numpy())
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    accuracy = accuracy_score(actuals, predictions)
    precision = precision_score(actuals, predictions)
    recall = recall_score(actuals, predictions)
    f1 = f1_score(actuals, predictions)
    
    return accuracy, precision, recall, f1, predictions, actuals


def plot_training_history(history, save_path):
    """Plot training history."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(history['val_accuracy'], label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_confusion_matrix(cm, save_path, model_name):
    """Plot confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# ======================== Main Training Pipeline ========================

def main():
    """Main training pipeline."""
    
    print("=" * 60)
    print("Deep Learning Training Pipeline for Phishing URL Detection")
    print("=" * 60)
    
    # Configuration
    BATCH_SIZE = 256
    EPOCHS = 50
    LEARNING_RATE = 0.001
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\nDevice: {DEVICE}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Learning Rate: {LEARNING_RATE}")
    
    # Load data
    print("\n" + "=" * 60)
    print("Loading and preprocessing data...")
    print("=" * 60)
    
    data = pd.read_csv(Config.DATASET_DATNOU)
    print(f"Loaded {len(data)} URLs")
    
    # Extract features
    print("Extracting features...")
    feature_data = []
    skipped_urls = 0
    
    for idx, row in data.iterrows():
        if idx % 10000 == 0:
            print(f"Processed {idx}/{len(data)} URLs... (Skipped: {skipped_urls})")
        
        try:
            features = extract_features(row['url'])
            
            # Skip if feature extraction failed
            if features is None:
                skipped_urls += 1
                continue
            
            features['label'] = 1 if row['type'] == 'phishing' else 0
            feature_data.append(features)
        
        except Exception as e:
            # Skip any problematic URLs
            skipped_urls += 1
            continue
    
    print(f"\nTotal URLs processed: {len(data)}")
    print(f"Successfully extracted: {len(feature_data)}")
    print(f"Skipped (invalid/malformed): {skipped_urls}")
    
    features_df = pd.DataFrame(feature_data)
    
    # Handle missing values
    features_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    features_df.fillna(0, inplace=True)
    
    # Split features and labels
    X = features_df.drop('label', axis=1).values
    y = features_df['label'].values
    
    print(f"\nFeature shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Phishing URLs: {sum(y)} ({sum(y)/len(y)*100:.2f}%)")
    print(f"Legitimate URLs: {len(y)-sum(y)} ({(len(y)-sum(y))/len(y)*100:.2f}%)")
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=Config.RANDOM_STATE, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler
    scaler_path = Config.MODEL_PRODUCTION_DIR / "neural_network_scaler.pkl"
    joblib.dump(scaler, scaler_path)
    print(f"\nScaler saved to {scaler_path}")
    
    # Create datasets
    train_dataset = URLDataset(X_train_scaled, y_train)
    test_dataset = URLDataset(X_test_scaled, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Define models to train
    input_dim = X_train_scaled.shape[1]
    
    models_to_train = {
        'FeedForward': FeedForwardNN(input_dim),
        'LSTM': LSTMNN(input_dim),
        'GRU': GRUNN(input_dim),
        'CNN': CNNNN(input_dim)
    }
    
    # Train each model
    for model_name, model in models_to_train.items():
        print("\n" + "=" * 60)
        print(f"Training {model_name} Neural Network")
        print("=" * 60)
        
        model = model.to(DEVICE)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': []
        }
        
        best_val_accuracy = 0
        patience_counter = 0
        
        # Training loop
        for epoch in range(EPOCHS):
            train_loss = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
            
            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for features, labels in test_loader:
                    features, labels = features.to(DEVICE), labels.to(DEVICE)
                    outputs = model(features).squeeze()
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
            
            val_loss /= len(test_loader)
            scheduler.step(val_loss)
            
            # Evaluate
            val_acc, val_prec, val_rec, val_f1, _, _ = evaluate(model, test_loader, DEVICE)
            
            # Save history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_acc)
            
            # Print progress
            if (epoch + 1) % 5 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}] - "
                      f"Train Loss: {train_loss:.4f}, "
                      f"Val Loss: {val_loss:.4f}, "
                      f"Val Acc: {val_acc:.4f}")
            
            # Save best model
            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                model_path = Config.MODEL_PRODUCTION_DIR / f"neural_network_{model_name.lower()}.pt"
                torch.save(model.state_dict(), model_path)
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= 10:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Final evaluation
        print(f"\nFinal Evaluation - {model_name}")
        accuracy, precision, recall, f1, predictions, actuals = evaluate(model, test_loader, DEVICE)
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        
        # Plot training history
        history_path = Config.MODEL_PRODUCTION_DIR / f"training_history_{model_name.lower()}.png"
        plot_training_history(history, history_path)
        
        # Plot confusion matrix
        cm = confusion_matrix(actuals, predictions)
        cm_path = Config.MODEL_PRODUCTION_DIR / f"confusion_matrix_{model_name.lower()}.png"
        plot_confusion_matrix(cm, cm_path, model_name)
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"\nModels saved to: {Config.MODEL_PRODUCTION_DIR}")
    print("You can now use these models for prediction.")


if __name__ == "__main__":
    main()
