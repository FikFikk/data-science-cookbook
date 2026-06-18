"""
Quick Demo: Imbalanced Classification
======================================
Versi cepat untuk demonstrasi konsep dasar
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

print("Quick Demo: Imbalanced Classification\n")

# Generate small imbalanced dataset
X, y = make_classification(
    n_samples=1000, n_features=10, n_classes=2,
    weights=[0.95, 0.05], random_state=42
)

print(f"Dataset: {len(y)} samples")
print(f"Class 0: {sum(y==0)} ({sum(y==0)/len(y)*100:.1f}%)")
print(f"Class 1: {sum(y==1)} ({sum(y==1)/len(y)*100:.1f}%)")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# Baseline
print("\n--- Baseline (No Treatment) ---")
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# With class_weight
print("\n--- With Class Weights ---")
model_w = LogisticRegression(class_weight='balanced', max_iter=1000)
model_w.fit(X_train, y_train)
y_pred_w = model_w.predict(X_test)
print(confusion_matrix(y_test, y_pred_w))
print(classification_report(y_test, y_pred_w))

print("\nLihat notebook.py untuk implementasi lengkap!")
