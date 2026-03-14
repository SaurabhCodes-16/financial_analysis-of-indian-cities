import pandas as pd
from scipy.stats import f_oneway

df = pd.read_csv("outputs/phase2_clustered.csv")

tier_cols = [col for col in df.columns if "City_Tier" in col]

for col in tier_cols:
    groups = []

    for val in [0,1]:
        groups.append(df[df[col]==val]['Expense_Ratio'])

    if len(groups) == 2:
        stat, p = f_oneway(groups[0], groups[1])
        print(col, "p-value:", p)

print("Phase 5 Completed")
