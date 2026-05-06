import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    r2_score
)
import sys
import joblib

sys.path.append(str(Path(__file__).parent))
from utils import save_model, plot_feature_importance, plot_confusion_matrix, generate_report

BASE_DIR = Path(__file__).parent.parent
DATA_TRAIN_TEST = BASE_DIR / 'data' / 'train_test'

print("=" * 70)
print("🚀 MODEL TRAINING (NO DATA LEAKAGE)")
print("=" * 70)

# Load datasets
print("\n📂 Loading datasets...")
X_train = pd.read_csv(DATA_TRAIN_TEST / 'X_train.csv')
X_test = pd.read_csv(DATA_TRAIN_TEST / 'X_test.csv')
y_train = pd.read_csv(DATA_TRAIN_TEST / 'y_train.csv').squeeze()
y_test = pd.read_csv(DATA_TRAIN_TEST / 'y_test.csv').squeeze()

print(f"✓ Features: {X_train.shape[1]} columns")
print(f"  Examples: {list(X_train.columns[:5])}…")

# =============================================================================
# 1. CLUSTERING
# =============================================================================
print("\n" + "=" * 70)
print("1️⃣  CLUSTERING")
print("=" * 70)

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
clusters_train = kmeans.fit_predict(X_train)

print("✓ Cluster distribution:", pd.Series(clusters_train).value_counts().to_dict())
save_model(kmeans, 'kmeans_model.pkl')

# =============================================================================
# 2. CLASSIFICATION — Churn
# =============================================================================
print("\n" + "=" * 70)
print("2️⃣  CLASSIFICATION (Churn Prediction)")
print("=" * 70)

classifier = RandomForestClassifier(
    n_estimators=100,
    max_depth=8,
    class_weight='balanced',
    min_samples_leaf=10,
    random_state=42
)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
y_proba = classifier.predict_proba(X_test)[:, 1]
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = conf_matrix.ravel()
classification_metrics = classification_report(
    y_test,
    y_pred,
    target_names=['Loyal', 'Churned'],
    output_dict=True
)

print(f"✓ Accuracy: {accuracy:.3f}")
print("\nConfusion Matrix:")
print(f"  True Negative : {tn}")
print(f"  False Positive: {fp}")
print(f"  False Negative: {fn}")
print(f"  True Positive : {tp}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Loyal', 'Churned']))

test_client = X_test.iloc[0:1].copy()
pred = classifier.predict(test_client)[0]
proba = classifier.predict_proba(test_client)[0][1]
print(f"\n✓ Sample prediction from test set:")
print(f"  Result: {'CHURNED' if pred == 1 else 'LOYAL'} ({proba * 100:.1f}%)")

save_model(classifier, 'churn_classifier.pkl')
plot_feature_importance(classifier, X_train.columns, 'Feature Importance', 'feat_imp.png')
plot_confusion_matrix(
    conf_matrix,
    ['Loyal', 'Churned'],
    'Churn Confusion Matrix',
    'confusion_matrix.png'
)

# =============================================================================
# 3. REGRESSION — LTV Estimation
# =============================================================================
print("\n" + "=" * 70)
print("3️⃣  REGRESSION (LTV Estimation)")
print("=" * 70)

# Validate required columns
for dataset_name, dataset in [('X_train', X_train), ('X_test', X_test)]:
    missing = [c for c in ['MonetaryTotal', 'Frequency', 'Recency'] if c not in dataset.columns]
    if missing:
        raise ValueError(f"Missing columns in {dataset_name}: {missing}")

ltv_train = np.abs(X_train['MonetaryTotal'] * X_train['Frequency'] / (X_train['Recency'] + 1))
ltv_test = np.abs(X_test['MonetaryTotal'] * X_test['Frequency'] / (X_test['Recency'] + 1))

regressor = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
regressor.fit(X_train, ltv_train)

ltv_pred = regressor.predict(X_test)
mae = mean_absolute_error(ltv_test, ltv_pred)
r2 = r2_score(ltv_test, ltv_pred)

print(f"✓ MAE: {mae:.2f}")
print(f"✓ R²:  {r2:.3f}")

save_model(regressor, 'ltv_regressor.pkl')
report_metrics = {
    'clustering': {
        'n_clusters': 4,
        'cluster_distribution': pd.Series(clusters_train).value_counts().to_dict(),
    },
    'classification': {
        'accuracy': round(float(accuracy), 4),
        'confusion_matrix': {
            'true_negative': int(tn),
            'false_positive': int(fp),
            'false_negative': int(fn),
            'true_positive': int(tp),
        },
        'classification_report': classification_metrics,
        'sample_prediction': {
            'predicted_class': 'CHURNED' if pred == 1 else 'LOYAL',
            'churn_probability': round(float(proba), 4),
        },
    },
    'regression': {
        'mae': round(float(mae), 4),
        'r2': round(float(r2), 4),
    },
}
generate_report(report_metrics, 'training_report.json')

print("\n" + "=" * 70)
print("✅ 3 MODELS SAVED SUCCESSFULLY")
print("=" * 70)
