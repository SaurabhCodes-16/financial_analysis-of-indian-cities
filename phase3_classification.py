import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
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

# Feature Scaling
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
# XGBoost Model with Hyperparameter Tuning
# ---------------------------
print("\n" + "="*50)
print("XGBoost Model Training with Hyperparameter Tuning")
print("="*50)

# Define parameter grid for RandomizedSearchCV (optimized for speed)
param_dist = {
    'n_estimators': [100, 200],
    'max_depth': [4, 6],
    'learning_rate': [0.1, 0.2],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

# Initialize XGBoost classifier
xgb_model = XGBClassifier(
    random_state=42,
    eval_metric='mlogloss',
    use_label_encoder=False
)

# Perform RandomizedSearchCV with 3-fold cross-validation (faster)
print("Performing hyperparameter tuning with RandomizedSearchCV...")
random_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_dist,
    n_iter=10,  # Reduced from 50 to 10 for speed
    cv=3,  # Reduced from 5 to 3 folds
    scoring='f1_weighted',
    n_jobs=-1,
    verbose=1,
    random_state=42
)

random_search.fit(X_train, y_train)

# Get best model
best_model = random_search.best_estimator_
print(f"\nBest parameters: {random_search.best_params_}")
print(f"Best cross-validation score: {random_search.best_score_:.4f}")

# Train final model with best parameters
best_model.fit(X_train, y_train)

# Comprehensive Model Evaluation
# ---------------------------
print("\n" + "="*50)
print("MODEL EVALUATION METRICS")
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

# Cross-validation scores
cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='f1_weighted')
print(f"\nCross-validation F1 scores: {cv_scores}")
print(f"Mean CV F1: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Classification report
print("\n" + "="*50)
print("CLASSIFICATION REPORT (Test Set)")
print("="*50)
print(classification_report(y_test, test_preds, target_names=['Stable', 'At Risk', 'Vulnerable'], zero_division=0))

# Confusion Matrix Visualization
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, test_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Stable', 'At Risk', 'Vulnerable'],
            yticklabels=['Stable', 'At Risk', 'Vulnerable'])
plt.title('Confusion Matrix - XGBoost Model')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig("plots/phase3_confusion_matrix.png", dpi=300, bbox_inches='tight')
plt.close()

# Save results to file
with open("outputs/phase3_model_results.txt", "w") as f:
    f.write("XGBoost Model Results\n")
    f.write("="*50 + "\n\n")
    f.write(f"Best Parameters: {random_search.best_params_}\n")
    f.write(f"Best CV Score: {random_search.best_score_:.4f}\n\n")

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
joblib.dump(best_model, "outputs/phase3_model.pkl")
joblib.dump(scaler, "outputs/phase3_scaler.pkl")

# Save predictions
pd.DataFrame({
    "Actual": y_test,
    "Predicted": test_preds
}).to_csv("outputs/phase3_test_predictions.csv", index=False)

print("\nModel and results saved successfully!")
print("Phase 3 Enhanced Classification Completed")

# ---------------------------
