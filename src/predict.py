
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import joblib

sys.path.append(str(Path(__file__).parent))
from utils import load_model

BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / 'models'
DATA_TRAIN_TEST = BASE_DIR / 'data' / 'train_test'
DATA_RAW = BASE_DIR / 'data' / 'raw'


def preprocess_single(raw_values: dict) -> pd.DataFrame:
    """Preprocess a single customer record."""
    preprocessor = joblib.load(MODELS_DIR / 'preprocessor.pkl')
    columns = preprocessor['columns']
    scaler = preprocessor['scaler']

    # Load training set means for missing features
    X_train_means = pd.read_csv(DATA_TRAIN_TEST / 'X_train.csv').mean().to_dict()

    # Create full feature dict with defaults
    full_record = {col: X_train_means.get(col, 0.0) for col in columns}
    full_record.update({k: v for k, v in raw_values.items() if k in columns})

    # Convert to DataFrame and scale
    data_frame = pd.DataFrame([full_record])[columns]
    scaled_data = pd.DataFrame(scaler.transform(data_frame), columns=columns)
    return scaled_data


def predict_client(raw_values: dict) -> dict:
    """Make predictions for a customer."""
    scaled_data = preprocess_single(raw_values)

    # Load models
    kmeans = load_model('kmeans_model.pkl')
    classifier = load_model('churn_classifier.pkl')
    regressor = load_model('ltv_regressor.pkl')

    # Make predictions
    cluster = int(kmeans.predict(scaled_data)[0])
    churn_pred = int(classifier.predict(scaled_data)[0])
    churn_prob = float(classifier.predict_proba(scaled_data)[0][1])
    ltv = float(regressor.predict(scaled_data)[0])

    status = "LOYAL" if churn_pred == 0 else "CHURNED"

    print(f"\n1️⃣  Cluster  : Group {cluster}")
    print(f"2️⃣  Churn    : {status} ({churn_prob * 100:.1f}%)")
    print(f"3️⃣  LTV      : {ltv:,.0f}€")

    return {
        'cluster': cluster,
        'status': status,
        'proba': round(churn_prob * 100, 1),
        'ltv': round(ltv, 2),
    }


# Test predictions
if __name__ == "__main__":

    print("=" * 70)
    print("🧪 TEST PREDICTIONS")
    print("=" * 70)

    print("\n📊 TEST 1: Average Customer")
    predict_client({
        'Recency': 50,
        'Frequency': 8,
        'MonetaryTotal': 600,
        'MonetaryAvg': 75,
        'Age': 35,
    })

    print("\n\n📊 TEST 2: At-Risk Customer (high recency, low frequency)")
    predict_client({
        'Recency': 300,
        'Frequency': 1,
        'MonetaryTotal': 80,
        'MonetaryAvg': 80,
        'Age': 45,
    })

    print("\n\n📊 TEST 3: VIP Customer (high buyer, recent)")
    predict_client({
        'Recency': 5,
        'Frequency': 40,
        'MonetaryTotal': 8000,
        'MonetaryAvg': 200,
        'Age': 29,
    })
