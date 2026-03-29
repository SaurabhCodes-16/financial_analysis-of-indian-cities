import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs("plots", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

df = pd.read_csv("data/finance.csv")

print("Initial Shape:", df.shape)

df = df.fillna(df.median(numeric_only=True))

expense_cols = [
    'Rent', 'Groceries', 'Transport', 'Eating_Out',
    'Entertainment', 'Utilities', 'Healthcare', 'Education'
]

potential_cols = [col for col in df.columns if "Potential_Savings" in col]

df['Total_Expenses'] = df[expense_cols].sum(axis=1)
df['Total_Potential_Savings'] = df[potential_cols].sum(axis=1)

df['Savings_Gap'] = df['Desired_Savings'] - df['Disposable_Income']
df['Expense_Ratio'] = df['Total_Expenses'] / df['Income']
df['Recovery_Rate'] = df['Total_Potential_Savings'] / (np.abs(df['Savings_Gap']) + 1)

df = pd.get_dummies(df, columns=['City_Tier', 'Occupation'], drop_first=True)

plt.figure()
sns.histplot(df['Expense_Ratio'], kde=True)
plt.title("Expense Ratio Distribution")
plt.savefig("plots/expense_ratio_dist.png")
plt.close()

df.to_csv("outputs/phase1_processed.csv", index=False)

print("Phase 1 Completed")
