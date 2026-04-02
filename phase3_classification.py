import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.gaussian_process import GaussianProcessClassifier
from xgboost import XGBClassifier
import lightgbm as lgb
import catboost as cb
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

# Define classifiers
classifiers = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'SVM': SVC(random_state=42),
    'KNN': KNeighborsClassifier(),
    'Naive Bayes': GaussianNB(),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'AdaBoost': AdaBoostClassifier(random_state=42),
    'Extra Trees': ExtraTreesClassifier(random_state=42),
    'XGBoost': XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42),
    'LightGBM': lgb.LGBMClassifier(random_state=42, verbosity=-1),
    'CatBoost': cb.CatBoostClassifier(random_state=42, verbose=False),
    'MLP Classifier': MLPClassifier(random_state=42, max_iter=1000),
    'Ridge Classifier': RidgeClassifier(random_state=42),
    'Linear SVC': LinearSVC(random_state=42, max_iter=10000)
}

# Results storage
results = []

print("Training and evaluating classifiers...")
print("="*60)

for name, clf in classifiers.items():
    try:
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)

        acc = accuracy_score(y_test, preds)
        precision = precision_score(y_test, preds, average='weighted')
        recall = recall_score(y_test, preds, average='weighted')
        f1 = f1_score(y_test, preds, average='weighted')

        results.append({
            'Classifier': name,
            'Accuracy': f"{acc:.4f}",
            'Precision': f"{precision:.4f}",
            'Recall': f"{recall:.4f}",
            'F1 Score': f"{f1:.4f}"
        })

        print(f"{name}: Accuracy = {acc:.4f}, F1 = {f1:.4f}")

    except Exception as e:
        print(f"Error with {name}: {str(e)}")
        results.append({
            'Classifier': name,
            'Accuracy': 'Error',
            'Precision': 'Error',
            'Recall': 'Error',
            'F1 Score': 'Error'
        })

# Save XGBoost model as before
xgb_model = classifiers['XGBoost']
joblib.dump(xgb_model, "outputs/phase3_model.pkl")

# Save predictions
xgb_preds = xgb_model.predict(X_test)
pd.DataFrame({
    "Actual": y_test,
    "Predicted": xgb_preds
}).to_csv("outputs/phase3_test_predictions.csv", index=False)

# Save results in tabular format
with open("outputs/classifier_comparison.txt", "w") as f:
    f.write("CLASSIFIER COMPARISON RESULTS\n")
    f.write("="*80 + "\n\n")

    # Create table header
    header = "| {:<20} | {:<10} | {:<10} | {:<10} | {:<10} |\n".format(
        "Classifier", "Accuracy", "Precision", "Recall", "F1 Score"
    )
    f.write(header)
    f.write("|{:<20}|{:<10}|{:<10}|{:<10}|{:<10}|\n".format("-"*20, "-"*10, "-"*10, "-"*10, "-"*10))

    # Write results
    for result in results:
        row = "| {:<20} | {:<10} | {:<10} | {:<10} | {:<10} |\n".format(
            result['Classifier'],
            result['Accuracy'],
            result['Precision'],
            result['Recall'],
            result['F1 Score']
        )
        f.write(row)

    f.write("\nNote: All metrics are weighted averages for multi-class classification.\n")

print("\nPhase 3 Completed ✅")
print("Classifier comparison saved at outputs/classifier_comparison.txt")
print("XGBoost model saved at outputs/phase3_model.pkl")