#!/usr/bin/env python3
"""
symptom_classifier.py
─────────────────────
Read the cleaned dataset from SQLite, train two models (LogReg + Random Forest),
pick the best, save it *and* the fitted LabelEncoder, and export a confusion-
matrix PNG.  ROC is skipped automatically if the task has >2 classes.

Run:  python symptom_classifier.py
"""

from pathlib import Path
import joblib
import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    classification_report,
)
import matplotlib.pyplot as plt
import numpy as np

# ── Config ─────────────────────────────────────────────────────────
DB_FILE      = "disease_symptoms.db"
TABLE        = "symptoms"
ARTIFACT_DIR = Path("artifacts")
ARTIFACT_DIR.mkdir(exist_ok=True)

# ── 1. Read data from SQLite ───────────────────────────────────────
engine = create_engine(f"sqlite:///{DB_FILE}")
df      = pd.read_sql(f"SELECT * FROM {TABLE};", engine)
print(f"Loaded {len(df):,} rows from {DB_FILE}")

X = df.drop(columns=["prognosis"])
y = df["prognosis"]

# Encode target but keep a reference so we can invert later
le    = LabelEncoder().fit(y)
y_enc = le.transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)

# ── 2. Train two models & pick best ────────────────────────────────
models = {
    "logreg": LogisticRegression(max_iter=1000, n_jobs=-1),
    "rf":     RandomForestClassifier(n_estimators=300,
                                     random_state=42,
                                     n_jobs=-1),
}

best_name, best_model, best_acc = None, None, 0.0
for name, model in models.items():
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"{name:7s} accuracy: {acc:.3f}")
    if acc > best_acc:
        best_name, best_model, best_acc = name, model, acc

print(f"\n  Best model: {best_name} ({best_acc:.3f})")
joblib.dump(best_model, "symptom_model.joblib")
joblib.dump(le,          "label_encoder.joblib")

# ── 3. Visuals ─────────────────────────────────────────────────────
ConfusionMatrixDisplay.from_estimator(
    best_model, X_test, y_test, xticks_rotation=45
)
plt.title(f"{best_name.upper()} – Confusion Matrix")
plt.savefig(ARTIFACT_DIR / "conf_matrix.png", bbox_inches="tight")
plt.clf()

n_classes = len(np.unique(y_train))
if n_classes == 2:
    RocCurveDisplay.from_estimator(best_model, X_test, y_test)
    plt.title(f"{best_name.upper()} – ROC Curve")
    plt.savefig(ARTIFACT_DIR / "roc_curve.png", bbox_inches="tight")
    plt.clf()
else:
    print(f"⚠️  Detected {n_classes} classes → skipping ROC "
          "(binary-only helper).")

print("Plots saved to", ARTIFACT_DIR / "")
print(classification_report(y_test,
                            best_model.predict(X_test),
                            target_names=le.classes_))


