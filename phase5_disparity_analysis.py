import pandas as pd
import numpy as np
from scipy.stats import f_oneway
import matplotlib.pyplot as plt
import os

os.makedirs("plots", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

df = pd.read_csv("outputs/phase2_clustered.csv")

features = ['Income', 'Expense_Ratio', 'Savings_Gap', 'Recovery_Rate']

tier1 = df[(df['City_Tier_Tier_2'] == 0) & (df['City_Tier_Tier_3'] == 0)]
tier2 = df[df['City_Tier_Tier_2'] == 1]
tier3 = df[df['City_Tier_Tier_3'] == 1]

results = []

for feature in features:
    stat, p = f_oneway(
        tier1[feature],
        tier2[feature],
        tier3[feature]
    )
    results.append([feature, p])

results_df = pd.DataFrame(results, columns=['Feature', 'p-value'])
print("\nANOVA Results:\n")
print(results_df)

results_df.to_csv("outputs/disparity_results.csv", index=False)
results_df['p-value'] = results_df['p-value'].replace(0, 1e-300)
results_df['log_p'] = -np.log10(results_df['p-value'])

plt.figure()
plt.bar(results_df['Feature'], results_df['p-value'])
plt.yscale('log')
plt.title("Disparity Across City Tiers (p-values)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("plots/disparity_pvalues.png")
plt.close()

means = []

for feature in features:
    means.append([
        feature,
        tier1[feature].mean(),
        tier2[feature].mean(),
        tier3[feature].mean()
    ])

means_df = pd.DataFrame(means, columns=['Feature', 'Tier1', 'Tier2', 'Tier3'])
print("\nMean Comparison:\n")
print(means_df)

means_df.to_csv("outputs/feature_means_by_tier.csv", index=False)

means_df.set_index('Feature').plot(kind='bar')
plt.title("Feature Comparison Across City Tiers")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("plots/feature_comparison.png")
plt.close()

print("\nMulti-Attribute Disparity Analysis Completed Successfully ✅")
