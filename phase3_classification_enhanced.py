import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs("outputs", exist_ok=True)
os.makedirs("plots", exist_ok=True)

df = pd.read_csv("outputs/phase2_clustered.csv")

# Encode Labels
label_map = {
    "Stable": 0,
    "At Risk": 1,
    "Vulnerable": 2
}

df['Target'] = df['Vulnerability_Label'].map(label_map)

X = df.drop(['Vulnerability_Label','Cluster','Target'], axis=1)
y = df['Target']

# Feature Scaling (NEW ENHANCEMENT)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print("Training set shape:", X_train.shape)
print("Test set shape:", X_test.shape)
print("Class distribution in training set:")
print(y_train.value_counts(normalize=True))

# ---------------------------
# XGBoost Model with Optimized Hyperparameters (ENHANCED VERSION)
# ---------------------------
print("\n" + "="*50)
print("ENHANCED XGBoost Model Training")
print("="*50)

# Use optimized hyperparameters based on typical best practices
# (In production, you would use RandomizedSearchCV, but this is faster for demo)
best_params = {
    'n_estimators': 200,
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'eval_metric': 'mlogloss',
    'use_label_encoder': False
}

print("Using optimized hyperparameters:")
for param, value in best_params.items():
    print(f"  {param}: {value}")

model = XGBClassifier(**best_params)

# Train the model
print("\nTraining XGBoost model...")
model.fit(X_train, y_train)

# Predictions
train_preds = model.predict(X_train)
test_preds = model.predict(X_test)

# Comprehensive Model Evaluation (ENHANCED)
# ---------------------------
print("\n" + "="*50)
print("COMPREHENSIVE MODEL EVALUATION")
print("="*50)

# Training metrics
train_accuracy = accuracy_score(y_train, train_preds)
train_precision = precision_score(y_train, train_preds, average='weighted', zero_division=0)
train_recall = recall_score(y_train, train_preds, average='weighted', zero_division=0)
train_f1 = f1_score(y_train, train_preds, average='weighted', zero_division=0)

print("TRAINING SET METRICS:")
print(f"Accuracy: {train_accuracy:.4f}")
print(f"Precision: {train_precision:.4f}")
print(f"Recall: {train_recall:.4f}")
print(f"F1-Score: {train_f1:.4f}")

# Test metrics
test_accuracy = accuracy_score(y_test, test_preds)
test_precision = precision_score(y_test, test_preds, average='weighted', zero_division=0)
test_recall = recall_score(y_test, test_preds, average='weighted', zero_division=0)
test_f1 = f1_score(y_test, test_preds, average='weighted', zero_division=0)

print("\nTEST SET METRICS:")
print(f"Accuracy: {test_accuracy:.4f}")
print(f"Precision: {test_precision:.4f}")
print(f"Recall: {test_recall:.4f}")
print(f"F1-Score: {test_f1:.4f}")

# Cross-validation scores (ENHANCED)
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1_weighted')
print(f"\nCross-validation F1 scores: {cv_scores}")
print(f"Mean CV F1: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Classification report
print("\n" + "="*50)
print("CLASSIFICATION REPORT (Test Set)")
print("="*50)
print(classification_report(y_test, test_preds, target_names=['Stable', 'At Risk', 'Vulnerable'], zero_division=0))

# Confusion Matrix Visualization (ENHANCED)
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, test_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Stable', 'At Risk', 'Vulnerable'],
            yticklabels=['Stable', 'At Risk', 'Vulnerable'])
plt.title('Confusion Matrix - Enhanced XGBoost Model')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig("plots/phase3_confusion_matrix.png", dpi=300, bbox_inches='tight')
plt.close()

# Save comprehensive results (ENHANCED)
with open("outputs/phase3_model_results.txt", "w") as f:
    f.write("Enhanced XGBoost Model Results\n")
    f.write("="*50 + "\n\n")
    f.write("HYPERPARAMETERS USED:\n")
    for param, value in best_params.items():
        f.write(f"  {param}: {value}\n")
    f.write("\n")

    f.write("TRAINING METRICS:\n")
    f.write(f"Accuracy: {train_accuracy:.4f}\n")
    f.write(f"Precision: {train_precision:.4f}\n")
    f.write(f"Recall: {train_recall:.4f}\n")
    f.write(f"F1-Score: {train_f1:.4f}\n\n")

    f.write("TEST METRICS:\n")
    f.write(f"Accuracy: {test_accuracy:.4f}\n")
    f.write(f"Precision: {test_precision:.4f}\n")
    f.write(f"Recall: {test_recall:.4f}\n")
    f.write(f"F1-Score: {test_f1:.4f}\n\n")

    f.write(f"Cross-validation F1: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})\n\n")

    f.write("CLASSIFICATION REPORT:\n")
    f.write(classification_report(y_test, test_preds, target_names=['Stable', 'At Risk', 'Vulnerable'], zero_division=0))

# Save model and scaler
joblib.dump(model, "outputs/phase3_model.pkl")
joblib.dump(scaler, "outputs/phase3_scaler.pkl")

# Save predictions
pd.DataFrame({
    "Actual": y_test,
    "Predicted": test_preds
}).to_csv("outputs/phase3_test_predictions.csv", index=False)

print("\n✅ Enhanced XGBoost model completed successfully!")
print("📁 Results saved to outputs/phase3_model_results.txt")
print("📊 Confusion matrix saved to plots/phase3_confusion_matrix.png")