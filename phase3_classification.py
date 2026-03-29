import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from xgboost import XGBClassifier
import os

os.makedirs("outputs", exist_ok=True)

df = pd.read_csv("outputs/phase2_clustered.csv")

label_map = {
    "Stable": 0,
    "At Risk": 1,
    "Vulnerable": 2
}

df['Target'] = df['Vulnerability_Label'].map(label_map)

X = df.drop(['Vulnerability_Label','Cluster','Target'], axis=1)
y = df['Target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    random_state=42
)

model.fit(X_train, y_train)

preds = model.predict(X_test)

acc = accuracy_score(y_test, preds)
precision = precision_score(y_test, preds, average='weighted')
recall = recall_score(y_test, preds, average='weighted')
f1 = f1_score(y_test, preds, average='weighted')

cm = confusion_matrix(y_test, preds)
report = classification_report(
    y_test, preds,
    target_names=["Stable", "At Risk", "Vulnerable"]
)

print("Accuracy:", acc)

joblib.dump(model, "outputs/phase3_model.pkl")

pd.DataFrame({
    "Actual": y_test,
    "Predicted": preds
}).to_csv("outputs/phase3_test_predictions.csv", index=False)

with open("outputs/model_results.txt", "w") as f:
    f.write("XGBOOST MODEL RESULTS\n")
    f.write("="*50 + "\n\n")

    f.write(f"Accuracy:  {acc:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall:    {recall:.4f}\n")
    f.write(f"F1 Score:  {f1:.4f}\n\n")

    f.write("Confusion Matrix:\n")
    f.write(str(cm) + "\n\n")

    f.write("Classification Report:\n")
    f.write(report)

print("Phase 3 Completed ✅")
print("Report saved at outputs/model_results.txt")