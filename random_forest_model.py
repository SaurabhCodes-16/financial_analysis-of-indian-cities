import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
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

print("Training Random Forest Model...")
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

preds = model.predict(X_test)

accuracy = accuracy_score(y_test, preds)
precision = precision_score(y_test, preds, average='weighted', zero_division=0)
recall = recall_score(y_test, preds, average='weighted', zero_division=0)
f1 = f1_score(y_test, preds, average='weighted', zero_division=0)

print("\n" + "="*50)
print("RANDOM FOREST MODEL PERFORMANCE")
print("="*50)
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, preds))
print("\nClassification Report:")
print(classification_report(y_test, preds, target_names=["Stable", "At Risk", "Vulnerable"]))

with open("outputs/random_forest_results.txt", "w") as f:
    f.write("RANDOM FOREST MODEL RESULTS\n")
    f.write("="*50 + "\n\n")
    f.write(f"Accuracy:  {accuracy:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall:    {recall:.4f}\n")
    f.write(f"F1 Score:  {f1:.4f}\n")
    f.write("\nConfusion Matrix:\n")
    f.write(str(confusion_matrix(y_test, preds)))
    f.write("\n\nClassification Report:\n")
    f.write(classification_report(y_test, preds, target_names=["Stable", "At Risk", "Vulnerable"]))

joblib.dump(model, "outputs/random_forest_model.pkl")

pd.DataFrame({
    "Actual": y_test,
    "Predicted": preds
}).to_csv("outputs/random_forest_predictions.csv", index=False)

print("\nResults saved to outputs/random_forest_results.txt")
