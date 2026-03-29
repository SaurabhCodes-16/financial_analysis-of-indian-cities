import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt
import os

os.makedirs("plots", exist_ok=True)

df = pd.read_csv("outputs/phase2_clustered.csv")
model = joblib.load("outputs/phase3_model.pkl")

X = df.drop(['Vulnerability_Label','Cluster'], axis=1)

X = X.astype(float)

X_sample = X.sample(500, random_state=42)

print("Creating SHAP Explainer...")

explainer = shap.Explainer(model.predict, X_sample)

print("Calculating SHAP Values...")
shap_values = explainer(X_sample)

import numpy as np
importance = np.abs(shap_values.values).mean(axis=0)
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importance
}).sort_values('Importance', ascending=False)

print("\n" + "="*50)
print("FEATURE IMPORTANCE (SHAP)")
print("="*50)
print(feature_importance.to_string(index=False))

with open("outputs/shap_feature_importance.txt", "w") as f:
    f.write("SHAP FEATURE IMPORTANCE RANKING\n")
    f.write("="*50 + "\n\n")
    f.write(feature_importance.to_string(index=False))
    f.write("\n\n")
    if hasattr(explainer, 'expected_value'):
        f.write("Base Value (Expected Model Output): {:.4f}\n".format(explainer.expected_value))
    else:
        f.write("Base Value (Expected Model Output): N/A (PermutationExplainer)\n")

with open("outputs/shap_prediction_details.txt", "w") as f:
    f.write("SHAP VALUE BREAKDOWN FOR FIRST 10 SAMPLES\n")
    f.write("="*50 + "\n\n")
    
    base_value = explainer.expected_value if hasattr(explainer, 'expected_value') else None
    
    for idx in range(min(10, len(X_sample))):
        f.write(f"Sample {idx + 1}:\n")
        if base_value is not None:
            f.write(f"  Base Value: {base_value:.4f}\n")
        
        sample_shap = pd.DataFrame({
            'Feature': X.columns,
            'SHAP_Value': shap_values.values[idx]
        }).sort_values('SHAP_Value', ascending=False, key=abs)
        
        f.write("  Feature Contributions:\n")
        for _, row in sample_shap.iterrows():
            f.write(f"    {row['Feature']}: {row['SHAP_Value']:+.6f}\n")
        f.write("\n")

print("\nSHAP text outputs saved to:")
print("  - outputs/shap_feature_importance.txt")
print("  - outputs/shap_prediction_details.txt")

print("\nSaving SHAP Plot...")

plt.figure()
shap.plots.beeswarm(shap_values, show=False)
plt.savefig("plots/shap_summary.png")
plt.close()

print("Phase 4 Completed")
