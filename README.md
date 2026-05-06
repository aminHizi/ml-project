# ML Retail Prediction System

A comprehensive machine learning system for retail customer analysis with clustering, churn prediction, and lifetime value estimation.

## Overview

This project combines three predictive models to provide actionable insights about customer behavior:
- **Clustering**: Segments customers into distinct groups
- **Churn Classification**: Predicts likelihood of customer departure
- **LTV Regression**: Estimates customer lifetime value

## Project Structure

```
├── notebooks/
│   └── exploration.py          # Data exploration and visualization
├── src/
│   ├── preprocessing.py        # Data cleaning and feature engineering
│   ├── train_model.py          # Model training pipeline
│   ├── predict.py             # Prediction inference
│   └── utils.py               # Helper functions
├── app/
│   └── app.py                 # Flask web application
├── data/
│   ├── raw/                   # Raw datasets
│   └── train_test/            # Processed datasets
├── models/                     # Trained model artifacts
├── reports/                    # Generated visualizations
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Setup

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare data** - Place your dataset at:
   ```
   data/raw/retail_customers_COMPLETE_CATEGORICAL.xlsx
   ```

## Workflow

### 1. Explore Data
```bash
python notebooks/exploration.py
```
Generates visualization reports in `reports/`

### 2. Preprocess Data
```bash
python src/preprocessing.py
```
- Handles missing values
- Encodes categorical features
- Scales numeric features
- Prevents data leakage with proper train-test split

### 3. Train Models
```bash
python src/train_model.py
```
Trains and saves three models to `models/`:
- `kmeans_model.pkl`
- `churn_classifier.pkl`
- `ltv_regressor.pkl`

### 4. Run Web App
```bash
python app/app.py
```
Access at `http://127.0.0.1:5000`

## Model Details

### Clustering (KMeans)
- 4 customer clusters
- Identifies customer segments

### Classification (Random Forest)
- Binary churn prediction
- Class weights balanced for imbalanced data
- Provides probability estimates

### Regression (Random Forest)
- Continuous LTV prediction
- Combines recency, frequency, and monetary value

## Key Features

- **No Data Leakage**: Train/test split before preprocessing
- **Robust Preprocessing**: Proper handling of categorical and numeric data
- **Feature Engineering**: RFM analysis and scaling
- **Web Interface**: User-friendly prediction interface
- **Modular Design**: Reusable preprocessing and prediction pipelines
