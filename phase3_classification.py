import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import os

os.makedirs("outputs", exist_ok=True)

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

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------
# XGBoost Model
# ---------------------------
model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    random_state=42
)

model.fit(X_train, y_train)

preds = model.predict(X_test)

acc = accuracy_score(y_test, preds)
print("Accuracy:", acc)

joblib.dump(model, "outputs/phase3_model.pkl")

pd.DataFrame({
    "Actual": y_test,
    "Predicted": preds
}).to_csv("outputs/phase3_test_predictions.csv", index=False)

print("Phase 3 Completed")
