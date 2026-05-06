
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
import json

BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / 'models'
REPORTS_DIR = BASE_DIR / 'reports'


def save_model(model, filename):
    """Save a trained model to disk."""
    MODELS_DIR.mkdir(exist_ok=True)
    path = MODELS_DIR / filename
    joblib.dump(model, path)
    print(f"✓ Model saved: {path}")


def load_model(filename):
    """Load a trained model from disk."""
    path = MODELS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    return joblib.load(path)


def calculate_rfm_scores(data):
    """Calculate RFM (Recency, Frequency, Monetary) scores."""
    data = data.copy()

    # Recency: lower is better (inverted score)
    data['R_Score'] = pd.qcut(
        data['Recency'], 5, labels=[5, 4, 3, 2, 1], duplicates='drop'
    ).astype(int)

    # Frequency: higher is better
    data['F_Score'] = pd.qcut(
        data['Frequency'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5]
    ).astype(int)

    # Monetary: higher is better
    data['M_Score'] = pd.qcut(
        data['MonetaryTotal'], 5, labels=[1, 2, 3, 4, 5], duplicates='drop'
    ).astype(int)

    # Combined RFM score
    data['RFM_Score'] = (
        data['R_Score'].astype(str)
        + data['F_Score'].astype(str)
        + data['M_Score'].astype(str)
    )

    return data


def get_segment_label(row):
    """Assign segment label based on RFM scores."""
    r = int(row['R_Score'])
    f = int(row['F_Score'])

    if r >= 4 and f >= 4:
        return 'Champions'
    elif r >= 3 and f >= 3:
        return 'Loyal'
    elif r >= 4 and f <= 2:
        return 'New'
    elif r <= 2 and f >= 3:
        return 'At Risk'
    else:
        return 'Dormant'


def plot_feature_importance(model, feature_names, title, filename):
    """Plot and save feature importance visualization."""
    if not hasattr(model, 'feature_importances_'):
        print("✗ Model does not have feature_importances_")
        return

    importances = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False).head(10)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=importances, y='feature', x='importance',
                palette='viridis', ax=ax)
    ax.set_title(title, fontsize=14, fontweight='bold')
    fig.tight_layout()

    REPORTS_DIR.mkdir(exist_ok=True)
    fig.savefig(REPORTS_DIR / filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Plot saved: {filename}")


def plot_confusion_matrix(matrix, labels, title, filename):
    """Plot and save a confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt='d',
        cmap='Blues',
        cbar=False,
        xticklabels=labels,
        yticklabels=labels,
        ax=ax
    )
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    fig.tight_layout()

    REPORTS_DIR.mkdir(exist_ok=True)
    fig.savefig(REPORTS_DIR / filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Plot saved: {filename}")


def generate_report(metrics, filename='report.json'):
    """Generate and save a JSON report."""
    REPORTS_DIR.mkdir(exist_ok=True)
    path = REPORTS_DIR / filename
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)
    print(f"✓ Report saved: {path}")
