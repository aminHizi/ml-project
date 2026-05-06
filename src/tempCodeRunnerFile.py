
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import joblib
import warnings

warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).parent.parent
DATA_RAW = BASE_DIR / 'data' / 'raw'
DATA_TRAIN_TEST = BASE_DIR / 'data' / 'train_test'
MODELS_DIR = BASE_DIR / 'models'

for directory in [DATA_TRAIN_TEST, MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("🧹 PREPROCESSING (NO DATA LEAKAGE)")
print("=" * 70)

# Load data
print("\n📂 Loading data...")
data = pd.read_excel(DATA_RAW / 'retail_customers_COMPLETE_CATEGORICAL.xlsx')

# Fix mixed-type numeric columns safely
print("✓ Cleaning data types...")
for col in data.columns:
    if data[col].dtype == 'object':
        numeric_candidate = pd.to_numeric(data[col], errors='coerce')
        convertible_ratio = numeric_candidate.notna().mean()
        if convertible_ratio >= 0.8 or (col.startswith('Monetary') and convertible_ratio >= 0.5):
            data[col] = numeric_candidate

# Remove data leakage features
print("✓ Removing leakage features...")
leakage_features = [
    'ChurnRiskCategory', 'RFMSegment', 'CustomerType',
    'CustomerID', 'LastLoginIP', 'RegistrationDate', 'NewsletterSubscribed',
    'AccountStatus',
    'AgeCategory',
    'LoyaltyLevel',
    'WeekendPreference'
]
problematic_features = ['SpendingCategory', 'Country']
data = data.drop(columns=[f for f in leakage_features if f in data.columns])
data = data.drop(columns=[f for f in problematic_features if f in data.columns])

# Separate features and target
y = data['Churn']
X = data.drop('Churn', axis=1)

print(f"✓ Clean features: {X.shape[1]} columns")

categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

# Train-test split
print("✓ Creating train-test split...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Numeric imputation
print("✓ Imputing numeric features...")
imputer_numeric = SimpleImputer(strategy='median')
X_train[numeric_cols] = imputer_numeric.fit_transform(X_train[numeric_cols])
X_test[numeric_cols] = imputer_numeric.transform(X_test[numeric_cols])

# Categorical imputation and encoding
print("✓ Imputing and encoding categorical features...")
if categorical_cols:
    imputer_categorical = SimpleImputer(strategy='most_frequent')
    X_train[categorical_cols] = imputer_categorical.fit_transform(X_train[categorical_cols])
    X_test[categorical_cols] = imputer_categorical.transform(X_test[categorical_cols])

    X_train = pd.get_dummies(X_train, columns=categorical_cols, drop_first=True)
    X_test = pd.get_dummies(X_test, columns=categorical_cols, drop_first=True)
    X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

# Scaling (fit on train only)
print("✓ Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save datasets
print("✓ Saving datasets...")
processed_columns = list(X_train.columns)
pd.DataFrame(X_train_scaled, columns=processed_columns).to_csv(
    DATA_TRAIN_TEST / 'X_train.csv', index=False
)
pd.DataFrame(X_test_scaled, columns=processed_columns).to_csv(
    DATA_TRAIN_TEST / 'X_test.csv', index=False
)
y_train.to_csv(DATA_TRAIN_TEST / 'y_train.csv', index=False)
y_test.to_csv(DATA_TRAIN_TEST / 'y_test.csv', index=False)

# Save preprocessor
print("✓ Saving preprocessor...")
joblib.dump({
    'columns': processed_columns,
    'scaler': scaler,
    'imputer_numeric': imputer_numeric,
    'imputer_categorical': imputer_categorical if categorical_cols else None,
    'numeric_cols': numeric_cols,
    'categorical_cols': categorical_cols,
}, MODELS_DIR / 'preprocessor.pkl')

print("\n" + "=" * 70)
print("✅ PREPROCESSING COMPLETE")
print("=" * 70)
print(f"✓ {len(processed_columns)} processed features")
print(f"✓ Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")
print(f"✓ Columns: {processed_columns}")
