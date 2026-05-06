
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_RAW = BASE_DIR / 'data' / 'raw'
REPORTS_DIR = BASE_DIR / 'reports'
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("🔍 DATA EXPLORATION")
print("=" * 70)

# Load data
print("\n📂 Loading data...")
data = pd.read_excel(DATA_RAW / 'retail_customers_COMPLETE_CATEGORICAL.xlsx')

print(f"\n✓ Dataset: {data.shape[0]} rows × {data.shape[1]} columns")
print(f"\nColumns ({len(data.columns)}):")
for i, col in enumerate(data.columns, 1):
    print(f"   {i:2d}. {col}")

print(f"\n📊 Data types:")
print(data.dtypes)

print(f"\n❌ Missing values:")
missing = data.isnull().sum()
if missing.sum() > 0:
    print(missing[missing > 0])
else:
    print("   None")

print(f"\n📋 Data preview:")
print(data.head())

print(f"\n📈 Descriptive statistics (numeric):")
print(data.describe())

# Categorical analysis
categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
if categorical_cols:
    print(f"\n🏷️  Categorical variables ({len(categorical_cols)}):")
    for col in categorical_cols:
        print(f"\n   {col}:")
        print(data[col].value_counts().head())

# Churn distribution
if 'Churn' in data.columns:
    plt.figure(figsize=(8, 5))
    data['Churn'].value_counts().plot(kind='bar', color=['skyblue', 'salmon'])
    plt.title('Churn Distribution', fontweight='bold')
    plt.xlabel('Churn (0=Loyal, 1=Churned)')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / 'churn_distribution.png', dpi=300)
    print(f"\n✓ Churn distribution saved")

# Correlation matrix
numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
if len(numeric_cols) > 1:
    plt.figure(figsize=(12, 10))
    sns.heatmap(data[numeric_cols].corr(), annot=True, cmap='coolwarm',
                center=0, fmt='.2f')
    plt.title('Correlation Matrix', fontweight='bold')
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / 'correlation_matrix.png', dpi=300)
    print(f"✓ Correlation matrix saved")

print("\n" + "=" * 70)
print("✅ EXPLORATION COMPLETE")
print(f"📁 Reports saved to: {REPORTS_DIR}")
print("=" * 70)
