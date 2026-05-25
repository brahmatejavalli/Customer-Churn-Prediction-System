"""
train.py
--------
Trains, evaluates, and saves the best-performing
churn prediction model.
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score, roc_curve
)

sys.path.append(os.path.dirname(__file__))
from preprocess import run_preprocessing


def evaluate_model(model, X_test, y_test, model_name):
    """Print full evaluation report for a model."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print(f"\n{'='*45}")
    print(f"  Model: {model_name}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  ROC-AUC  : {auc:.4f}")
    print(f"{'='*45}")
    print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))

    return acc, auc, y_pred, y_prob


def plot_confusion_matrix(y_test, y_pred, model_name, save_path):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Churn', 'Churn'],
                yticklabels=['No Churn', 'Churn'])
    plt.title(f'Confusion Matrix — {model_name}', fontsize=14, fontweight='bold')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[INFO] Saved: {save_path}")


def plot_roc_curve(results, save_path):
    plt.figure(figsize=(8, 6))
    for name, y_prob, y_test in results:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[INFO] Saved: {save_path}")


def plot_feature_importance(model, feature_names, save_path, top_n=15):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
    else:
        return

    feat_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feat_df = feat_df.sort_values('Importance', ascending=False).head(top_n)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=feat_df, x='Importance', y='Feature', palette='viridis')
    plt.title(f'Top {top_n} Feature Importances', fontsize=14, fontweight='bold')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[INFO] Saved: {save_path}")


def train():
    # ── Load & preprocess ──────────────────────────────────────────
    DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'churn.csv')
    MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'model')
    os.makedirs(MODEL_DIR, exist_ok=True)

    X, y, feature_names, scaler = run_preprocessing(DATA_PATH)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n[INFO] Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"[INFO] Churn rate — Train: {y_train.mean():.1%}, Test: {y_test.mean():.1%}")

    # ── Define models ──────────────────────────────────────────────
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting':   GradientBoostingClassifier(n_estimators=100, random_state=42),
    }

    # ── Train and evaluate all models ─────────────────────────────
    best_model, best_auc, best_name = None, 0, ''
    roc_data = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        acc, auc, y_pred, y_prob = evaluate_model(model, X_test, y_test, name)

        # Cross-validation
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')
        print(f"  CV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        roc_data.append((name, y_prob, y_test))

        plot_confusion_matrix(
            y_test, y_pred, name,
            os.path.join(MODEL_DIR, f"cm_{name.replace(' ', '_').lower()}.png")
        )

        if auc > best_auc:
            best_auc = auc
            best_model = model
            best_name = name

    # ── Save best model ────────────────────────────────────────────
    print(f"\n[INFO] Best model: {best_name} (AUC={best_auc:.4f})")
    joblib.dump(best_model, os.path.join(MODEL_DIR, 'model.pkl'))
    joblib.dump(scaler,     os.path.join(MODEL_DIR, 'scaler.pkl'))
    joblib.dump(feature_names, os.path.join(MODEL_DIR, 'feature_names.pkl'))
    print(f"[INFO] Model saved to {MODEL_DIR}/model.pkl")

    # ── Plots ──────────────────────────────────────────────────────
    plot_roc_curve(roc_data, os.path.join(MODEL_DIR, 'roc_curve.png'))
    plot_feature_importance(
        best_model, feature_names,
        os.path.join(MODEL_DIR, 'feature_importance.png')
    )

    return best_model


if __name__ == "__main__":
    train()
