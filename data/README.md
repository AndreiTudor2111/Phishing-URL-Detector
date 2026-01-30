# Dataset Documentation

This directory contains the datasets used for training and testing the phishing URL detection models.

## Raw Datasets

### 1. DataNou.csv
- **Size**: ~43.7 MB
- **Records**: ~430,000 URLs
- **Source**: Consolidated phishing dataset
- **Columns**: `url`, `type`
- **Classes**: `phishing`, `legitimate`
- **Usage**: Primary training dataset
- **Class Distribution**: 
  - Phishing: ~50%
  - Legitimate: ~50%

### 2. malicious_phish.csv
- **Size**: ~45.6 MB
- **Records**: ~450,000 URLs
- **Source**: Alternative phishing dataset
- **Columns**: `url`, `type`
- **Classes**: `benign`, `defacement`, `phishing`, `malware`
- **Usage**: Alternative/validation dataset

### 3. dataset.csv
- **Size**: ~3.6 MB
- **Records**: ~36,000 URLs
- **Source**: Smaller curated dataset
- **Usage**: Quick testing and validation

### 4. allbrands.txt
- **Type**: Text file
- **Content**: List of known brand names
- **Usage**: Brand detection in URL paths and domains
- **Entries**: ~100 brand names

## Data Processing

All datasets are used with the feature extraction pipeline in `src/feature_extraction/feature_extractor.py`.

### Preprocessing Steps

1. **URL Parsing**: Extract hostname, path, query parameters
2. **Feature Extraction**: Calculate 50+ features
3. **Normalization**: Replace infinite/NaN values
4. **Scaling**: StandardScaler transformation
5. **Train/Test Split**: 70/30 split with stratification

## Data Quality

- All URLs are validated before feature extraction
- Missing values are handled appropriately
- Class balance is maintained through stratified sampling
- Outliers are detected and managed during feature scaling

## Usage Example

```python
import pandas as pd
from src.utils.config import Config

# Load primary dataset
df = pd.read_csv(Config.DATASET_DATNOU)

# View basic statistics
print(f"Total URLs: {len(df)}")
print(f"Phishing URLs: {(df['type'] == 'phishing').sum()}")
print(f"Legitimate URLs: {(df['type'] != 'phishing').sum()}")
```

## Citation

If using these datasets, please cite the original sources and this project.
