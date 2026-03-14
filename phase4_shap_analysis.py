import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt
import os

os.makedirs("plots", exist_ok=True)

df = pd.read_csv("outputs/phase2_clustered.csv")
model = joblib.load("outputs/phase3_model.pkl")

X = df.drop(['Vulnerability_Label','Cluster'], axis=1)

# 🔥 MOST IMPORTANT FIX
X = X.astype(float)

# Sample for speed
X_sample = X.sample(500, random_state=42)

print("Creating SHAP Explainer...")

explainer = shap.Explainer(model.predict, X_sample)

print("Calculating SHAP Values...")
shap_values = explainer(X_sample)

# ===========================
# TEXT OUTPUT SECTION
# ===========================

# Calculate feature importance (mean absolute SHAP values)
import numpy as np
importance = np.abs(shap_values.values).mean(axis=0)
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importance
}).sort_values('Importance', ascending=False)

# Print feature importance
print("\n" + "="*50)
print("FEATURE IMPORTANCE (SHAP)")
print("="*50)
print(feature_importance.to_string(index=False))

# Save feature importance to file
with open("outputs/shap_feature_importance.txt", "w") as f:
    f.write("SHAP FEATURE IMPORTANCE RANKING\n")
    f.write("="*50 + "\n\n")
    f.write(feature_importance.to_string(index=False))
    f.write("\n\n")
    if hasattr(explainer, 'expected_value'):
        f.write("Base Value (Expected Model Output): {:.4f}\n".format(explainer.expected_value))
    else:
        f.write("Base Value (Expected Model Output): N/A (PermutationExplainer)\n")

# Enhanced SHAP Analysis with Multiple Visualization Types
# ===========================
print("\n" + "="*60)
print("ENHANCED SHAP ANALYSIS - MULTIPLE VISUALIZATION TYPES")
print("="*60)

# 1. SHAP Summary Plot (Beeswarm)
plt.figure(figsize=(10, 6))
shap.plots.beeswarm(shap_values, show=False, max_display=15)
plt.title("SHAP Feature Importance Summary (Beeswarm Plot)")
plt.tight_layout()
plt.savefig("plots/shap_beeswarm.png", dpi=300, bbox_inches='tight')
plt.close()

# 2. SHAP Bar Plot (Global Feature Importance)
plt.figure(figsize=(10, 6))
shap.plots.bar(shap_values, show=False, max_display=15)
plt.title("SHAP Global Feature Importance (Bar Plot)")
plt.tight_layout()
plt.savefig("plots/shap_bar_importance.png", dpi=300, bbox_inches='tight')
plt.close()

# 3. Individual Prediction Explanations (Waterfall Plots)
print("\nGenerating individual prediction explanations...")
for idx in range(min(5, len(X_sample))):  # Top 5 samples
    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(shap_values[idx], show=False, max_display=10)
    plt.title(f"SHAP Waterfall Plot - Sample {idx + 1}")
    plt.tight_layout()
    plt.savefig(f"plots/shap_waterfall_sample_{idx+1}.png", dpi=300, bbox_inches='tight')
    plt.close()

# 4. SHAP Force Plot (first sample)
plt.figure(figsize=(12, 3))
shap.plots.force(explainer.expected_value, shap_values.values[0], X_sample.iloc[0],
                 feature_names=X.columns, show=False, matplotlib=True)
plt.title("SHAP Force Plot - Sample 1")
plt.tight_layout()
plt.savefig("plots/shap_force_plot_sample1.png", dpi=300, bbox_inches='tight')
plt.close()

# 5. Feature Interaction Analysis (if possible)
try:
    print("\nAnalyzing feature interactions...")
    # Calculate SHAP interaction values (more computationally intensive)
    shap_interaction_values = explainer.shap_interaction_values(X_sample)

    # Plot interaction heatmap for top features
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_interaction_values, X_sample, show=False, max_display=10)
    plt.title("SHAP Feature Interaction Summary")
    plt.tight_layout()
    plt.savefig("plots/shap_interaction_summary.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Save interaction analysis
    with open("outputs/shap_interaction_analysis.txt", "w") as f:
        f.write("SHAP FEATURE INTERACTION ANALYSIS\n")
        f.write("="*50 + "\n\n")
        f.write("Top feature interactions based on SHAP interaction values:\n\n")

        # Calculate mean absolute interaction values
        mean_interactions = np.abs(shap_interaction_values).mean(axis=0)
        interaction_df = pd.DataFrame(mean_interactions, index=X.columns, columns=X.columns)

        # Get top interactions (excluding self-interactions)
        interactions = []
        for i in range(len(X.columns)):
            for j in range(i+1, len(X.columns)):
                interactions.append({
                    'Feature1': X.columns[i],
                    'Feature2': X.columns[j],
                    'Interaction_Strength': interaction_df.iloc[i, j]
                })

        interaction_df = pd.DataFrame(interactions).sort_values('Interaction_Strength', ascending=False)
        f.write(interaction_df.head(10).to_string(index=False))

except Exception as e:
    print(f"Feature interaction analysis failed: {e}")
    with open("outputs/shap_interaction_analysis.txt", "w") as f:
        f.write("SHAP FEATURE INTERACTION ANALYSIS\n")
        f.write("="*50 + "\n\n")
        f.write("Feature interaction analysis was not performed due to computational constraints.\n")
        f.write("Consider using TreeExplainer for better interaction analysis if using tree-based models.\n")

# 6. Partial Dependence Plots for top features
print("\nGenerating partial dependence plots...")
top_features = feature_importance['Feature'].head(3).tolist()

for feature in top_features:
    try:
        plt.figure(figsize=(8, 5))
        shap.plots.partial_dependence(feature, model.predict, X_sample, ice=False,
                                    model_expected_value=explainer.expected_value, show=False)
        plt.title(f"Partial Dependence Plot - {feature}")
        plt.tight_layout()
        plt.savefig(f"plots/shap_partial_dependence_{feature.replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Partial dependence plot for {feature} failed: {e}")

# 7. SHAP Decision Plot
plt.figure(figsize=(10, 6))
shap.decision_plot(explainer.expected_value, shap_values.values[:10],
                  X_sample.iloc[:10], feature_names=X.columns, show=False)
plt.title("SHAP Decision Plot (First 10 Samples)")
plt.tight_layout()
plt.savefig("plots/shap_decision_plot.png", dpi=300, bbox_inches='tight')
plt.close()

# Enhanced text output with more detailed analysis
with open("outputs/shap_comprehensive_analysis.txt", "w") as f:
    f.write("COMPREHENSIVE SHAP ANALYSIS REPORT\n")
    f.write("="*60 + "\n\n")

    f.write("MODEL INFORMATION:\n")
    f.write(f"Model type: {type(model).__name__}\n")
    f.write(f"Dataset shape: {X.shape}\n")
    f.write(f"Sample size for analysis: {len(X_sample)}\n")
    f.write(f"SHAP Explainer: {type(explainer).__name__}\n")
    if hasattr(explainer, 'expected_value'):
        f.write(f"Expected value (base prediction): {explainer.expected_value:.4f}\n")
    f.write("\n")

    f.write("FEATURE IMPORTANCE RANKING:\n")
    f.write("-" * 40 + "\n")
    f.write(feature_importance.to_string(index=False))
    f.write("\n\n")

    f.write("KEY INSIGHTS:\n")
    f.write("-" * 40 + "\n")

    # Top positive and negative contributors
    top_positive = feature_importance[feature_importance['Importance'] > 0].head(3)
    top_negative = feature_importance[feature_importance['Importance'] < 0].tail(3)

    f.write("Top features contributing to higher predictions:\n")
    for _, row in top_positive.iterrows():
        f.write(f"  - {row['Feature']}: {row['Importance']:.4f}\n")

    f.write("\nTop features contributing to lower predictions:\n")
    for _, row in top_negative.iterrows():
        f.write(f"  - {row['Feature']}: {row['Importance']:.4f}\n")

    f.write("\nVISUALIZATIONS GENERATED:\n")
    f.write("-" * 40 + "\n")
    f.write("1. shap_beeswarm.png - Feature importance with feature values\n")
    f.write("2. shap_bar_importance.png - Global feature importance bars\n")
    f.write("3. shap_waterfall_sample_[1-5].png - Individual prediction explanations\n")
    f.write("4. shap_force_plot_sample1.png - Force plot for sample 1\n")
    f.write("5. shap_decision_plot.png - Decision plot for first 10 samples\n")
    f.write("6. shap_partial_dependence_[feature].png - Partial dependence plots\n")
    if 'shap_interaction_summary.png' in [f"plots/shap_interaction_summary.png"]:
        f.write("7. shap_interaction_summary.png - Feature interaction analysis\n")

print("\n" + "="*60)
print("SHAP ANALYSIS COMPLETED")
print("="*60)
print("Comprehensive SHAP analysis saved to:")
print("  - outputs/shap_comprehensive_analysis.txt")
print("  - outputs/shap_interaction_analysis.txt")
print("Multiple visualization plots saved to plots/")

print("\nPhase 4 Enhanced SHAP Analysis Completed")
